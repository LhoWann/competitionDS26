"""Frozen YOLO-pose feature cache: one keypoint vector per frame (`features.backbone=yolo*-pose`, see `features.pose`).

Writes `{cfg.feature_dir}/train/{video_id}.npy` (every `stride`-th cached frame) and `{cfg.feature_dir}/test/{seq}.npy`
(every `stride`-th of the 30 test frames), each float16 of shape (frames, tokens, dim).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.config import Config, FeaturesConfig
from dataset.clips import load_frames
from features.pose import PoseEncoder


class FrameDataset(Dataset):
    """Flat list of JPEG paths; each item is one resized uint8 frame (3, H, W)."""

    def __init__(self, paths: list[Path], size_hw: list[int]) -> None:
        self.paths = paths
        self.size_hw = size_hw

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return load_frames([self.paths[idx]], self.size_hw)[0]


def encode_to_files(encoder: nn.Module, jobs: list[tuple[Path, list[Path]]], cfg: FeaturesConfig,
                    workers: int) -> None:
    """Encodes the frames of every (output file, frame paths) job, streaming one .npy per job in order."""
    device = next(encoder.buffers()).device
    flat = [p for _, paths in jobs for p in paths]
    loader = DataLoader(FrameDataset(flat, cfg.img_size), batch_size=cfg.batch_size, num_workers=workers)
    queue, buf = list(reversed(jobs)), None
    for frames in tqdm(loader, desc="encode", unit_scale=cfg.batch_size):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = encoder(frames.to(device, non_blocking=True)).cpu().half().numpy()
        buf = out if buf is None else np.concatenate([buf, out])
        while queue and len(buf) >= len(queue[-1][1]):
            path, paths = queue.pop()
            np.save(path, buf[:len(paths)])
            buf = buf[len(paths):]


def extract(cfg: Config, workers: int, pretrained: bool = True) -> Path:
    """Fills the feature cache for all train videos (`videos.csv`; none without one, e.g. in the test-only inference
    notebook) and test sequences; skips existing files."""
    fc, root, cache = cfg.features, cfg.feature_dir, Path(cfg.paths.cache_dir)
    index = cache / "videos.csv"
    videos = pd.read_csv(index) if index.exists() else pd.DataFrame({"video_id": [], "n_frames": []})
    jobs = [(root / "train" / f"{v}.npy", [cache / "frames" / v / f"{i:05d}.jpg" for i in range(0, n, fc.stride)])
            for v, n in zip(videos.video_id, videos.n_frames, strict=True)]
    test_dir = Path(cfg.paths.data_dir) / "test"
    if test_dir.exists():
        jobs += [(root / "test" / f"{s.name}.npy", sorted(s.glob("*.jpg"))[::fc.stride])
                 for s in sorted(p for p in test_dir.iterdir() if p.is_dir())]
    for split in ("train", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)
    jobs = [j for j in jobs if not j[0].exists()]
    if jobs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encode_to_files(PoseEncoder(fc, cache / "weights", pretrained).to(device), jobs, fc, workers)
    return root