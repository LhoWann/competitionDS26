"""Clip datasets: train/val clips from the `dataset.prepare` frame cache, test clips from the test folders."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.config import DataConfig
from dataset.transforms import build_augment


def frame_indices(start: int, clip_len: int, num_frames: int, jitter: bool, rng: np.random.Generator) -> np.ndarray:
    """Picks `num_frames` evenly spaced frames from a `clip_len` window, optionally jittered within each segment."""
    edges = np.linspace(0, clip_len, num_frames + 1)
    pos = rng.uniform(edges[:-1], edges[1:]) if jitter else (edges[:-1] + edges[1:]) / 2
    return start + np.minimum(pos.astype(int), clip_len - 1)


def load_frames(paths: list[Path], size_hw: list[int], box: tuple[float, ...] | None = None) -> torch.Tensor:
    """Reads JPEGs into a uint8 tensor (T, 3, H, W), optionally cropped to a fractional (x1, y1, x2, y2) box, resized
    with area interpolation."""
    h, w = size_hw
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if box is not None:
            ih, iw = img.shape[:2]
            img = img[int(box[1] * ih):int(box[3] * ih), int(box[0] * iw):int(box[2] * iw)]
        frames.append(cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def person_box(poses: np.ndarray, margin: float, out_hw: list[int], frame_hw: tuple[int, int] = (360, 640),
               min_conf: float = 0.3) -> tuple[float, ...] | None:
    """Fractional crop around the union of one clip's detected person boxes (`features.pose` vectors (N, 56)),
    enlarged by `margin` and widened to the model input aspect; None if nobody was detected."""
    det = poses[poses[:, -1] > min_conf]
    if not len(det):
        return None
    bw, bh, cx, cy = det[:, -5], det[:, -4], det[:, -3], det[:, -2]
    x1, x2, y1, y2 = (cx - bw / 2).min(), (cx + bw / 2).max(), (cy - bh / 2).min(), (cy + bh / 2).max()
    w, h = (x2 - x1) * margin, (y2 - y1) * margin
    aspect = out_hw[1] / out_hw[0] * frame_hw[0] / frame_hw[1]
    w, h = min(max(w, h * aspect), 1.0), min(max(h, w / aspect), 1.0)
    x0 = float(np.clip((x1 + x2 - w) / 2, 0, 1 - w))
    y0 = float(np.clip((y1 + y2 - h) / 2, 0, 1 - h))
    return x0, y0, x0 + w, y0 + h


# Head (twice as wide as tall) and right / left wrist boxes as (COCO keypoints, rows, cols) units.
ROI_PARTS = (([0, 1, 2, 3, 4], 1, 2), ([10], 1, 1), ([9], 1, 1))


def roi_boxes(poses: np.ndarray, size: float, frame_hw: tuple[int, int] = (360, 640),
              min_conf: float = 0.3) -> list[tuple[float, ...]] | None:
    """Fractional boxes around the head and both wrists over one clip (`features.pose` vectors (N, 56)): `size`
    person-box heights per unit of `ROI_PARTS` plus the keypoints' travel, at that aspect; None if nobody was detected.
    A keypoint never seen falls back to the person box centre."""
    det = poses[poses[:, -1] > min_conf]
    if not len(det):
        return None
    fh, fw = frame_hw
    bh = det[:, -4:-3]
    kx = (det[:, -3:-2] + det[:, 0:34:2] * bh * fh / fw) * fw
    ky = (det[:, -2:-1] + det[:, 1:34:2] * bh) * fh
    vis = det[:, 34:51] > min_conf
    side = size * float(np.median(bh)) * fh
    boxes = []
    for kpts, rows, cols in ROI_PARTS:
        m = vis[:, kpts]
        xs, ys = (kx[:, kpts][m], ky[:, kpts][m]) if m.any() else (det[:, -3] * fw, det[:, -2] * fh)
        w, h = side * cols + np.ptp(xs), side * rows + np.ptp(ys)
        w, h = max(w, h * cols / rows), max(h, w * rows / cols)
        w, h = min(w / fw, 1.0), min(h / fh, 1.0)
        x0 = float(np.clip((xs.min() + xs.max()) / 2 / fw - w / 2, 0, 1 - w))
        y0 = float(np.clip((ys.min() + ys.max()) / 2 / fh - h / 2, 0, 1 - h))
        boxes.append((x0, y0, x0 + w, y0 + h))
    return boxes


def load_clip(paths: list[Path], poses: np.ndarray | None, cfg: DataConfig) -> torch.Tensor:
    """One clip's frames, cropped to the person box of the clip's own pose vectors when given (`data.person_crop`)."""
    return load_frames(paths, cfg.img_size, None if poses is None else person_box(poses, cfg.person_crop, cfg.img_size))


class VideoClipDataset(Dataset):
    """Clips from cached train videos.

    In training mode each video yields `clips_per_video` random windows per epoch. In eval mode each video is
    split into non-overlapping `clip_len` windows, mirroring how the test sequences were cut.
    """

    def __init__(self, videos: pd.DataFrame, frames_dir: Path, cfg: DataConfig, train: bool,
                 crop_dir: Path | None = None) -> None:
        self.videos = videos.reset_index(drop=True)
        self.frames_dir = Path(frames_dir)
        self.cfg = cfg
        self.train = train
        self.poses = None if crop_dir is None else [np.load(crop_dir / "train" / f"{v}.npy")[:, 0].astype(np.float32)
                                                   for v in self.videos.video_id]
        self.augment = build_augment(cfg) if train else None
        self.rng = np.random.default_rng()
        if train:
            self.items = [(i, -1) for i in range(len(self.videos)) for _ in range(cfg.clips_per_video)]
        else:
            self.items = [(i, s) for i, n in enumerate(self.videos.n_frames)
                          for s in range(0, n - cfg.clip_len + 1, cfg.clip_len)]

    def __len__(self) -> int:
        return len(self.items)

    def clip_poses(self, vi: int, start: int, length: int) -> np.ndarray | None:
        """Pose vectors (strided cache) of the frames [start, start + length) of video `vi`, None without a cache."""
        if self.poses is None:
            return None
        step = max(1, round(self.videos.n_frames[vi] / len(self.poses[vi])))
        return self.poses[vi][start // step:(start + length - 1) // step + 1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        vi, start = self.items[idx]
        row = self.videos.iloc[vi]
        length = self.cfg.clip_len
        if start < 0:
            j = self.cfg.speed_jitter
            length = min(row.n_frames, round(length * self.rng.uniform(1 - j, 1 + j)))
            start = int(self.rng.integers(0, row.n_frames - length + 1))
        ids = frame_indices(start, length, self.cfg.num_frames, self.train, self.rng)
        paths = [self.frames_dir / row.video_id / f"{i:05d}.jpg" for i in ids]
        clip = load_clip(paths, self.clip_poses(vi, start, length), self.cfg)
        return {"inputs": clip if self.augment is None else self.augment(clip), "behavior": int(row.behavior_id),
                "label": int(row.label), "video": vi, "start": start}


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: gives each worker copy of the dataset its own RNG stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


class SequenceDataset(Dataset):
    """Test sequences: one folder of `clip_len` JPEG frames per sample."""

    def __init__(self, test_dir: Path, cfg: DataConfig, crop_dir: Path | None = None) -> None:
        self.seqs = sorted(p for p in Path(test_dir).iterdir() if p.is_dir())
        self.cfg = cfg
        self.crop_dir = crop_dir

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        frames = sorted(self.seqs[idx].glob("*.jpg"))
        ids = frame_indices(0, len(frames), self.cfg.num_frames, False, np.random.default_rng())
        poses = None
        if self.crop_dir is not None:
            poses = np.load(self.crop_dir / "test" / f"{self.seqs[idx].name}.npy")[:, 0].astype(np.float32)
        return {"inputs": load_clip([frames[i] for i in ids], poses, self.cfg), "index": idx}

    @property
    def ids(self) -> list[str]:
        return [p.name for p in self.seqs]