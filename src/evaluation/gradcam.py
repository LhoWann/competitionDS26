"""Grad-CAM check of a VideoMAE fold model on its validation clips: the share of the decision map on the head and
wrists (pose `roi_boxes`) for right vs wrong clips, plus overlays of the most confident errors. A share close to the
boxes' area fraction (lift ~1) means the model looks at those body parts no more than at random places."""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from core.config import Config
from dataset.clips import VideoClipDataset, person_box, roi_boxes
from dataset.labels import BEHAVIORS
from models.clip_classifier import VideoMAEClassifier, cheat_log_probs
from training.engine import build_model, make_loader


def token_cam(model: VideoMAEClassifier, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM over the last-layer tubelet tokens for each clip's predicted behavior: (B, 8) logits and
    (B, T / tubelet, H / patch, W / patch) maps."""
    tokens = model.tokens(pixels.float().requires_grad_())
    tokens.retain_grad()
    logits = model.head(model.norm(tokens.mean(1)))
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    cam = (tokens * tokens.grad.mean(1, keepdim=True)).sum(-1).relu()
    c = model.backbone.config
    t, h, w = pixels.shape[1] // c.tubelet_size, pixels.shape[-2] // c.patch_size, pixels.shape[-1] // c.patch_size
    return logits.detach().float(), cam.detach().float().view(len(cam), t, h, w)


def roi_mask(rois: list[tuple[float, ...]], crop: tuple[float, ...] | None, grid_hw: tuple[int, int]) -> np.ndarray:
    """(h, w) token mask of the frame-fraction `rois` inside the `crop` box the model sees (None: full frame)."""
    h, w = grid_hw
    x0, y0, x1, y1 = crop or (0.0, 0.0, 1.0, 1.0)
    fx, fy = x0 + (np.arange(w) + 0.5) / w * (x1 - x0), y0 + (np.arange(h) + 0.5) / h * (y1 - y0)
    m = np.zeros((h, w), bool)
    for a, b, c, d in rois:
        m |= ((fy >= b) & (fy <= d))[:, None] & ((fx >= a) & (fx <= c))[None]
    return m


def overlay(clip: torch.Tensor, cam: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    """Middle frame of a (T, 3, H, W) uint8 clip with its Grad-CAM slice as a heatmap and the ROI tokens outlined."""
    frame = cv2.cvtColor(clip[len(clip) // 2].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    heat = cam[len(cam) // 2] / max(cam.max(), 1e-8)
    heat = cv2.applyColorMap(cv2.resize((heat * 255).astype(np.uint8), (w, h)), cv2.COLORMAP_JET)
    img = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    edge = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
    cv2.putText(img, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def gradcam(cfg: Config, fold: int, ckpt: Path, out_dir: Path, thr: float = 0.5, roi: float = 0.5, stride: int = 1,
            n_show: int = 16, batch_size: int = 8) -> dict[str, dict[str, float]]:
    """Writes `gradcam.csv` (one row per clip), `summary.json` and the `wrong.jpg` / `right.jpg` overlay grids of
    every `stride`-th validation clip of `fold`; returns the summary."""
    videos = pd.read_csv(Path(cfg.paths.cache_dir) / "videos.csv")
    va = videos[videos.fold == fold].reset_index(drop=True)
    ds = VideoClipDataset(va, Path(cfg.paths.cache_dir) / "frames", cfg.data, train=False, crop_dir=cfg.crop_dir)
    ds.items = ds.items[::stride]
    model = build_model(replace(cfg, model=replace(cfg.model, pretrained=False)))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval().requires_grad_(False)
    rows, cams, masks = [], [], []
    for batch in make_loader(ds, batch_size, cfg.data.num_workers, train=False):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits, cam = token_cam(model, batch["inputs"].to(device))
        p_cheat = cheat_log_probs(logits.cpu())[1].exp().numpy()
        for i, (vi, start) in enumerate(zip(batch["video"].tolist(), batch["start"].tolist(), strict=True)):
            poses = ds.clip_poses(vi, start, cfg.data.clip_len)
            crop = None if poses is None or cfg.data.person_crop <= 0 else person_box(
                poses, cfg.data.person_crop, cfg.data.img_size)
            rois = None if poses is None else roi_boxes(poses, roi)
            c = cam[i].cpu().numpy()
            mask = np.zeros(c.shape[1:], bool) if rois is None else roi_mask(rois, crop, c.shape[1:])
            total = c.sum()
            rows.append({"video_id": va.video_id[vi], "start": start, "behavior": int(batch["behavior"][i]),
                         "label": int(batch["label"][i]), "pred_behavior": int(logits[i].argmax()),
                         "p_cheat": float(p_cheat[i]), "roi_mass": float((c * mask).sum() / total) if total else np.nan,
                         "roi_area": float(mask.mean()) if rois is not None else np.nan})
            cams.append(c)
            masks.append(mask)
    df = pd.DataFrame(rows)
    df["correct"] = (df.p_cheat >= thr).astype(int) == df.label
    df["lift"] = df.roi_mass / df.roi_area
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "gradcam.csv", index=False)
    summary = {name: {"n": int(len(g)), "roi_mass": float(g.roi_mass.mean()), "roi_area": float(g.roi_area.mean()),
                      "lift": float(g.lift.mean())}
               for name, g in (("right", df[df.correct]), ("wrong", df[~df.correct]))}
    summary["per_behavior_lift"] = {BEHAVIORS[b]: float(g.lift.mean()) for b, g in df.groupby("behavior")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    conf = (df.p_cheat - thr).abs()
    for name, sel in (("wrong", ~df.correct), ("right", df.correct)):
        idx = conf[sel].sort_values(ascending=False).index[:n_show]
        tiles = [overlay(ds[i]["inputs"], cams[i], masks[i], f"{BEHAVIORS[df.behavior[i]][:14]}>"
                         f"{BEHAVIORS[df.pred_behavior[i]][:14]} {df.p_cheat[i]:.2f}") for i in idx]
        if tiles:
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
            cv2.imwrite(str(out_dir / f"{name}.jpg"), np.vstack([np.hstack(tiles[r:r + 4])
                                                                for r in range(0, len(tiles), 4)]))
    return summary