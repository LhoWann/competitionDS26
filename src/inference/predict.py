"""Test inference: averages every fold checkpoint of a run and writes a submission. Video models also write the
flip-TTA variant (`*_flip.csv`, thresholded by cv_flip.json), so the variant chosen on OOF is used unchanged on test."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

from core.config import Config
from dataset.clips import SequenceDataset, VideoClipDataset
from dataset.features import FeatureSequenceDataset
from training.cv import load_videos, summarize
from training.engine import build_model, make_loader, predictions_frame, run_inference


def predict_oof(cfg: Config) -> dict[str, float] | None:
    """Re-scores every fold checkpoint of a video run on its held-out subjects with the current data settings (e.g.
    `data.eval_passes`, `data.crop_features`) and rewrites the fold and pooled OOF files and cv.json / cv_flip.json,
    so that stacking and thresholds see the same inference as the test clips. Returns cv.json on rank 0."""
    if cfg.model.kind == "features":
        raise ValueError("predict_oof re-scores video runs; retrain feature heads instead")
    videos = load_videos(cfg)
    acc = Accelerator(mixed_precision=cfg.train.mixed_precision)
    for ckpt in sorted(cfg.run_dir.glob("fold*/best.pt")):
        fold = int(ckpt.parent.name.removeprefix("fold"))
        va = videos[videos.fold == fold].reset_index(drop=True)
        ds = VideoClipDataset(va, Path(cfg.paths.cache_dir) / "frames", cfg.data, train=False, crop_dir=cfg.crop_dir)
        model = build_model(replace(cfg, model=replace(cfg.model, pretrained=False)))
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        dl = acc.prepare(make_loader(ds, cfg.train.batch_size * 2, cfg.data.num_workers, train=False))
        res = run_inference(acc.prepare(model), dl, acc, ["behavior", "label", "video", "start"], hflip=True)
        meta = {"fold": fold, "video_id": va.video_id.to_numpy()[res["video"]],
                "subject": va.subject.to_numpy()[res["video"]]}
        if acc.is_main_process:
            predictions_frame(res).assign(**meta).to_csv(ckpt.parent / "oof.csv", index=False)
            flip = predictions_frame(res | {"logits": res["logits_flip"]})
            flip.assign(**meta).to_csv(ckpt.parent / "oof_flip.csv", index=False)
        acc.print(f"{ckpt.parent.name}: {len(ds)} clips re-scored")
        acc.free_memory(model)
    return summarize(cfg.run_dir) if acc.is_main_process else None


def predict(cfg: Config) -> pd.DataFrame | None:
    """Writes `{run_dir}/test_probs.csv` and `{run_dir}/submission.csv` (+ `_flip` variants for video models); returns
    the plain probabilities on rank 0."""
    ckpts = sorted(cfg.run_dir.glob("fold*/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {cfg.run_dir}")
    ckpt_folds = sorted(c.parent.name for c in ckpts)
    oof_folds = sorted(p.parent.name for p in cfg.run_dir.glob("fold*/oof.csv"))
    if ckpt_folds != oof_folds:
        raise RuntimeError(f"checkpoints of {ckpt_folds} but finished OOF of {oof_folds}: a fold stopped mid-training "
                           "or lost its checkpoint upload; retrain it first")
    acc = Accelerator(mixed_precision=cfg.train.mixed_precision)
    feats = cfg.model.kind == "features"
    test_dir = Path(cfg.paths.data_dir) / "test"
    ds = FeatureSequenceDataset(cfg) if feats else SequenceDataset(test_dir, cfg.data, cfg.crop_dir)
    dl = acc.prepare(make_loader(ds, cfg.train.batch_size * 2, cfg.data.num_workers, train=False))
    variants = {"": "logits"} | ({} if feats else {"_flip": "logits_flip"})
    probs: dict[str, list[np.ndarray]] = {s: [] for s in variants}
    for ckpt in ckpts:
        model = build_model(replace(cfg, model=replace(cfg.model, pretrained=False)), ds.dim if feats else 0)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        res = run_inference(acc.prepare(model), dl, acc, ["index"], hflip=not feats)
        for suffix, key in variants.items():
            df = predictions_frame(res | {"logits": res[key]}).sort_values("index")
            probs[suffix].append(df.filter(like="p_").to_numpy())
            acc.print(f"{ckpt.parent.name}{suffix}: mean p_cheat {df.p_cheat.mean():.3f}")
        acc.free_memory(model)
    if not acc.is_main_process:
        return None
    cols = df.filter(like="p_").columns
    for suffix, p in probs.items():
        out = pd.DataFrame(np.mean(p, axis=0), columns=cols).assign(id=ds.ids)
        cv_path = cfg.run_dir / f"cv{suffix}.json"
        if suffix and not cv_path.exists():
            print(f"no {cv_path.name} (run trained before flip-TTA OOF): {suffix} variant skipped")
            continue
        thr = json.loads(cv_path.read_text())["thr"] if cv_path.exists() else 0.5
        out.to_csv(cfg.run_dir / f"test_probs{suffix}.csv", index=False)
        sub = pd.DataFrame({"id": out.id, "label": (out.p_cheat >= thr).astype(int)})
        sub.to_csv(cfg.run_dir / f"submission{suffix}.csv", index=False)
        print(f"{suffix or 'plain'}: threshold {thr:.2f} | predicted cheat ratio {sub.label.mean():.3f} | "
              f"-> {cfg.run_dir / f'submission{suffix}.csv'}")
    return pd.read_csv(cfg.run_dir / "test_probs.csv")