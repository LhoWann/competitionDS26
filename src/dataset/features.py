"""Clip datasets over the frozen-feature cache (`features.extract`), using the same windows as `dataset.clips`."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.config import Config


class FeatureClipDataset(Dataset):
    """Train/val clips as (clip_len // stride, F) feature windows of cached videos, all held in memory."""

    def __init__(self, videos: pd.DataFrame, cfg: Config, train: bool) -> None:
        self.videos = videos.reset_index(drop=True)
        self.clip_len, self.stride = cfg.data.clip_len, cfg.features.stride
        self.window = self.clip_len // self.stride
        root = cfg.feature_dir / "train"
        self.feats = [torch.from_numpy(np.load(root / f"{v}.npy")[:, 0]) for v in self.videos.video_id]
        self.rng = np.random.default_rng()
        if train:
            self.items = [(i, -1) for i in range(len(self.videos)) for _ in range(cfg.data.clips_per_video)]
        else:
            self.items = [(i, s) for i, n in enumerate(self.videos.n_frames)
                          for s in range(0, n - self.clip_len + 1, self.clip_len)]

    @property
    def dim(self) -> int:
        return self.feats[0].shape[1]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        vi, start = self.items[idx]
        row = self.videos.iloc[vi]
        if start < 0:
            start = int(self.rng.integers(0, row.n_frames - self.clip_len + 1))
        i0 = start // self.stride
        return {"inputs": self.feats[vi][i0:i0 + self.window], "behavior": int(row.behavior_id),
                "label": int(row.label), "video": vi, "start": start}


class FeatureSequenceDataset(Dataset):
    """Test sequences from `{feature_dir}/test/*.npy`, ordered like `dataset.clips.SequenceDataset`."""

    def __init__(self, cfg: Config) -> None:
        files = sorted(Path(cfg.feature_dir / "test").glob("*.npy"))
        self.ids = [p.stem for p in files]
        self.feats = [torch.from_numpy(np.load(p)[:, 0]) for p in files]

    @property
    def dim(self) -> int:
        return self.feats[0].shape[1]

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {"inputs": self.feats[idx], "index": idx}