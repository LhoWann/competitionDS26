"""Builds the train video index and a frame cache that reproduces the organizers' test pipeline.

Test JPEGs are 640x360, libjpeg quality 95, 4:2:0, no ffmpeg comment marker, i.e. `cv2.VideoCapture` ->
`cv2.resize` -> `cv2.imwrite`. Train videos go through the same steps so both splits share one domain.

CLI: python -m cli.prepare --config configs/base.yaml [--set data.max_subjects=4]
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import Config
from dataset.labels import BEHAVIORS

TEST_SIZE = (640, 360)
JPEG_QUALITY = 95


def build_index(data_dir: Path, n_folds: int, seed: int, max_subjects: int = 0) -> pd.DataFrame:
    """Lists train videos with labels and a subject-grouped fold id.

    Args:
        data_dir: Competition data root containing `train/{cheat,normal}/subject_*/{view}_{behavior}.MOV`.
        n_folds: Number of subject-disjoint folds.
        seed: Seed for the subject shuffle.
        max_subjects: If > 0, keeps only this many subjects (for quick local runs).
    """
    rows = []
    for p in sorted(Path(data_dir, "train").glob("*/*/*.MOV")):
        view, behavior = p.stem.split("_", 1)
        rows.append({"video_id": f"{p.parent.name}_{p.stem}", "path": str(p), "subject": p.parent.name, "view": view,
                     "behavior": behavior, "behavior_id": BEHAVIORS.index(behavior),
                     "label": int(p.parent.parent.name == "cheat")})
    df = pd.DataFrame(rows)
    subjects = np.random.default_rng(seed).permutation(sorted(df.subject.unique()))
    if max_subjects > 0:
        subjects = subjects[:max_subjects]
        df = df[df.subject.isin(subjects)]
    fold_of = {s: i % n_folds for i, s in enumerate(subjects)}
    return df.assign(fold=df.subject.map(fold_of)).reset_index(drop=True)


def extract_video(path: str, out_dir: Path) -> int:
    """Writes every frame of a video as test-style JPEG; returns the frame count. Skips finished videos."""
    done = out_dir / "n_frames.txt"
    if done.exists():
        return int(done.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(path)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, TEST_SIZE)
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        n += 1
    cap.release()
    done.write_text(str(n))
    return n


def prepare(cfg: Config, workers: int) -> pd.DataFrame:
    """Builds `{cache_dir}/videos.csv` and `{cache_dir}/frames/{video_id}/*.jpg`."""
    cache = Path(cfg.paths.cache_dir)
    df = build_index(Path(cfg.paths.data_dir), cfg.data.n_folds, cfg.train.seed, cfg.data.max_subjects)
    outs = [cache / "frames" / v for v in df.video_id]
    with ProcessPoolExecutor(workers) as ex:
        counts = list(tqdm(ex.map(extract_video, df.path, outs), total=len(df), desc="extract"))
    df["n_frames"] = counts
    df.to_csv(cache / "videos.csv", index=False)
    return df