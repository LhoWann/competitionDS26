"""Manual 30-fps cheating intervals of train videos (`cli.annotate`): JSON {video_id: [[start, end), ...]} in frames.
A video present with an empty list was watched and shows no cheating. A window is cheating when the marked frames are
the majority of its frames, the test rule (FAQ 44.3/63.1: the dominant behaviour visible in the 30 frames)."""

import json
from pathlib import Path


def load_intervals(path: str | Path) -> dict[str, list[list[int]]]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def merge_spans(spans: list[list[int]]) -> list[list[int]]:
    """Sorted spans with overlapping or touching ones merged; empty spans dropped."""
    merged: list[list[int]] = []
    for s, e in sorted((int(s), int(e)) for s, e in spans if e > s):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def save_intervals(path: str | Path, intervals: dict[str, list[list[int]]]) -> None:
    """Writes sorted, merged intervals atomically (the annotation server saves after every change)."""
    clean = {vid: merge_spans(spans) for vid, spans in sorted(intervals.items())}
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=1))
    tmp.replace(path)


def marked_fraction(spans: list[list[int]], start: int, length: int) -> float:
    """Fraction of the frames [start, start + length) inside the marked intervals."""
    covered = sum(max(0, min(e, start + length) - max(s, start)) for s, e in spans)
    return covered / length