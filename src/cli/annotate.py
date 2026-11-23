"""Local 30-fps annotation tool for the cheating intervals of train videos (FAQ 30/40/50: manual human annotation of
train; FAQ 50.1/51: a model may assist the human). Each unannotated video opens with a proposal from our own
out-of-fold stack (trained on train only, no VLM): the 1 s windows with p_cheat >= THR; a human watches, corrects and
saves every interval, so the proposal is never written without review. --blind hides the curve, the proposal and
the onset line (fully manual annotation, e.g. a second annotator whose labels are later compared with the model);
--review orders the videos by how many OOF windows the model gets wrong (most first), for a blind second pass into a
new file. Serves the cached train frames on localhost only and saves every change to the intervals JSON
(`dataset.intervals`).

Usage:
    python -m cli.annotate --oof outputs/stack_a30/oof.csv --out configs/cheat_intervals.json
    python -m cli.annotate --blind --out configs/cheat_intervals_manual.json
    python -m cli.annotate --oof outputs/stack_gold/oof.csv --blind --review --out configs/cheat_intervals_v2.json
    # then open http://localhost:8765 (keys: space play, <- -> frame, shift+<- -> 0.5 s, S start, E end, ...)
"""

import argparse
import json
import re
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from dataset.intervals import load_intervals, merge_spans, save_intervals
from dataset.labels import CHEAT_BEHAVIORS

THR, WINDOW = 0.43, 30
ORDER = ["Repeated_Peeking", "Using_Smartwatch", "Reading_Hidden_Notes", "Looking_Under_the_Desk"]


def video_table(videos: pd.DataFrame, oof: pd.DataFrame, onsets: pd.Series,
                intervals: dict[str, list[list[int]]], review: bool = False) -> list[dict]:
    """Cheat videos in annotation order (hardest behaviour first), each with its frame count, onset, model curve
    [[window start, p_cheat], ...], proposal (merged windows with p_cheat >= THR) and saved intervals (None = not
    annotated yet). With `review`, ordered by the number of windows where the OOF prediction (p_cheat >= THR)
    disagrees with the OOF label, most first."""
    curves = {v: g.sort_values("start")[["start", "p_cheat"]].round(3).values.tolist()
              for v, g in oof.groupby("video_id")}
    cheat = videos[videos.behavior.isin(CHEAT_BEHAVIORS)].copy()
    cheat["rank"] = cheat.behavior.map(ORDER.index)
    if review:
        wrong = ((oof.p_cheat >= THR) != oof.label).groupby(oof.video_id).sum()
        cheat["rank"] = -cheat.video_id.map(wrong).fillna(0)
    return [{"video_id": r.video_id, "behavior": r.behavior, "view": r.view, "n_frames": int(r.n_frames),
             "onset": int(onsets.get(r.video_id, -1)), "curve": curves.get(r.video_id, []),
             "proposal": merge_spans([[s, min(s + WINDOW, r.n_frames)] for s, p in curves.get(r.video_id, [])
                                      if p >= THR]),
             "intervals": intervals.get(r.video_id)}
            for r in cheat.sort_values(["rank", "subject", "view"]).itertuples()]


class Handler(BaseHTTPRequestHandler):
    """GET / (page), /api/videos, /frames/<video_id>/<n>.jpg; POST /api/save {video_id, intervals}."""

    def __init__(self, *args, frames: Path, out: Path, table: list[dict], **kwargs) -> None:
        self.frames, self.out, self.table = frames, out, table
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:
        pass

    def send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send((Path(__file__).parent / "annotate.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/videos":
            self.send(json.dumps(self.table).encode(), "application/json")
        elif m := re.fullmatch(r"/frames/([\w-]+)/(\d{5})\.jpg", self.path):
            p = self.frames / m[1] / f"{m[2]}.jpg"
            if m[1] in {v["video_id"] for v in self.table} and p.exists():
                self.send(p.read_bytes(), "image/jpeg")
            else:
                self.send(b"not found", "text/plain", 404)
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        row = next((v for v in self.table if v["video_id"] == req["video_id"]), None)
        if self.path != "/api/save" or row is None:
            self.send(b"bad request", "text/plain", 400)
            return
        intervals = load_intervals(self.out)
        intervals[row["video_id"]] = [[max(0, int(s)), min(row["n_frames"], int(e))] for s, e in req["intervals"]]
        save_intervals(self.out, intervals)
        row["intervals"] = load_intervals(self.out)[row["video_id"]]
        self.send(json.dumps(row["intervals"]).encode(), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--oof", default="outputs/stack_a30/oof.csv")
    parser.add_argument("--onsets", default="configs/onsets.csv")
    parser.add_argument("--out", default="configs/cheat_intervals.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--blind", action="store_true", help="no model curve or proposal")
    parser.add_argument("--review", action="store_true", help="most OOF errors first")
    args = parser.parse_args()
    cache = Path(args.cache)
    onsets = pd.read_csv(args.onsets).set_index("video_id").onset
    table = video_table(pd.read_csv(cache / "videos.csv"), pd.read_csv(args.oof), onsets, load_intervals(args.out),
                        args.review)
    for v in table if args.blind else []:
        v["curve"], v["proposal"], v["onset"] = [], [], -1
    handler = partial(Handler, frames=cache / "frames", out=Path(args.out), table=table)
    print(f"{len(table)} cheat videos, {sum(v['intervals'] is not None for v in table)} annotated -> "
          f"http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()