"""Memory-light EDA helpers: videos are streamed frame by frame, test JPEGs are decoded in small batches on GPU."""

from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.io import decode_jpeg, read_file

from dataset.prepare import TEST_SIZE, build_index

SMALL = (160, 90)


def video_table(data_dir: Path) -> pd.DataFrame:
    """Train index enriched with container metadata (no decoding)."""
    df = build_index(Path(data_dir), n_folds=5, seed=42)
    props = {"fps": cv2.CAP_PROP_FPS, "n_frames": cv2.CAP_PROP_FRAME_COUNT, "width": cv2.CAP_PROP_FRAME_WIDTH,
             "height": cv2.CAP_PROP_FRAME_HEIGHT}
    meta = []
    for p in df.path:
        cap = cv2.VideoCapture(p)
        meta.append({k: cap.get(v) for k, v in props.items()})
        cap.release()
    meta = pd.DataFrame(meta).astype({"n_frames": int, "width": int, "height": int})
    df = pd.concat([df, meta], axis=1)
    return df.assign(duration=df.n_frames / df.fps)


def read_frame(path: str, index: int) -> np.ndarray:
    """One RGB frame at test resolution."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    _, frame = cap.read()
    cap.release()
    return cv2.cvtColor(cv2.resize(frame, TEST_SIZE), cv2.COLOR_BGR2RGB)


def motion_curve(path: str) -> np.ndarray:
    """Per-frame motion energy (mean absolute grayscale difference at 160x90), streamed without holding the video."""
    cap = cv2.VideoCapture(path)
    prev, out = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            out.append(np.abs(g - prev).mean())
        prev = g
    cap.release()
    return np.array(out)


@torch.no_grad()
def test_seq_stats(test_dir: Path, device: str = "cuda", batch: int = 16) -> pd.DataFrame:
    """Per test sequence: brightness, saturation and motion energy, decoded `batch` sequences at a time on GPU."""
    seqs = sorted(p for p in Path(test_dir).iterdir() if p.is_dir())
    rows = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        files = [read_file(str(f)) for s in chunk for f in sorted(s.glob("*.jpg"))]
        imgs = torch.stack(decode_jpeg(files, device=device)).float()
        imgs = imgs.view(len(chunk), -1, *imgs.shape[1:])
        small = torch.nn.functional.interpolate(imgs.flatten(0, 1), size=SMALL[::-1], mode="area")
        gray = (small * torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)).sum(1)
        gray = gray.view(len(chunk), -1, *gray.shape[1:])
        motion = (gray[:, 1:] - gray[:, :-1]).abs().mean((1, 2, 3))
        mx, mn = imgs.amax(2), imgs.amin(2)
        sat = ((mx - mn) / mx.clamp(min=1)).mean((1, 2, 3)) * 255
        for s, b, st, m in zip(chunk, imgs.mean((1, 2, 3, 4)).tolist(), sat.tolist(), motion.tolist(), strict=True):
            rows.append({"id": s.name, "brightness": b, "saturation": st, "motion": m})
    return pd.DataFrame(rows)


@torch.no_grad()
def chain_test_sequences(test_dir: Path, thr: float, device: str = "cuda") -> list[list[int]]:
    """Links test sequences whose first frame continues another's last frame (greedy, cycle-free).

    Returns chains of sequence indices (sorted test folder order). Used to show that sequences are consecutive
    1-second cuts of longer videos.
    """
    seqs = sorted(p for p in Path(test_dir).iterdir() if p.is_dir())

    def thumbs(name_fn: Callable[[Path], Path]) -> torch.Tensor:
        out = []
        for i in range(0, len(seqs), 256):
            imgs = decode_jpeg([read_file(str(name_fn(s))) for s in seqs[i:i + 256]], device=device)
            g = torch.stack(imgs).float().mean(1, keepdim=True)
            out.append(torch.nn.functional.interpolate(g, size=(36, 64), mode="area").flatten(1))
        return torch.cat(out)

    first = thumbs(lambda s: sorted(s.glob("*.jpg"))[0])
    last = thumbs(lambda s: sorted(s.glob("*.jpg"))[-1])
    dist = (torch.cdist(last, first) / first.shape[1] ** 0.5).fill_diagonal_(float("inf")).cpu().numpy()
    n = len(seqs)
    succ, pred = -np.ones(n, int), -np.ones(n, int)
    for flat in np.argsort(dist, axis=None):
        i, j = divmod(int(flat), n)
        if dist[i, j] > thr:
            break
        if succ[i] >= 0 or pred[j] >= 0:
            continue
        k = j
        while succ[k] >= 0:
            k = succ[k]
        if k == i:
            continue
        succ[i], pred[j] = j, i
    chains = []
    for s in np.flatnonzero(pred < 0):
        c = [int(s)]
        while succ[c[-1]] >= 0:
            c.append(int(succ[c[-1]]))
        chains.append(c)
    return chains


def show_grid(images: list[np.ndarray], titles: list[str], cols: int = 4, size: float = 3.2) -> plt.Figure:
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 0.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, t in zip(np.ravel(axes), images, titles, strict=False):
        ax.imshow(img)
        ax.set_title(t, fontsize=8)
    fig.tight_layout()
    return fig