"""Test inference: averages every fold checkpoint of a run (frame models with hflip TTA) and writes a submission."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

from core.config import Config
from dataset.clips import SequenceDataset
from dataset.features import FeatureSequenceDataset
from training.engine import build_model, make_loader, predictions_frame, run_inference


def predict(cfg: Config, hflip_tta: bool = True) -> pd.DataFrame | None:
    """Writes `{run_dir}/test_probs.csv` and `{run_dir}/submission.csv`; returns the probabilities on rank 0."""
    acc = Accelerator(mixed_precision=cfg.train.mixed_precision)
    feats = cfg.model.kind == "features"
    test_dir = Path(cfg.paths.data_dir) / "test"
    ds = FeatureSequenceDataset(cfg) if feats else SequenceDataset(test_dir, cfg.data, cfg.crop_dir)
    dl = acc.prepare(make_loader(ds, cfg.train.batch_size * 2, cfg.data.num_workers, train=False))
    ckpts = sorted(cfg.run_dir.glob("fold*/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {cfg.run_dir}")
    probs = []
    for ckpt in ckpts:
        model = build_model(replace(cfg, model=replace(cfg.model, pretrained=False)), ds.dim if feats else 0)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        res = run_inference(acc.prepare(model), dl, acc, ["index"], hflip_tta and not feats)
        df = predictions_frame(res).sort_values("index")
        probs.append(df.filter(like="p_").to_numpy())
        acc.print(f"{ckpt.parent.name}: mean p_cheat {df.p_cheat.mean():.3f}")
        acc.free_memory(model)
    if not acc.is_main_process:
        return None
    cols = df.filter(like="p_").columns
    out = pd.DataFrame(np.mean(probs, axis=0), columns=cols).assign(id=ds.ids)
    cv_path = cfg.run_dir / "cv.json"
    thr = json.loads(cv_path.read_text())["thr"] if cv_path.exists() else 0.5
    out.to_csv(cfg.run_dir / "test_probs.csv", index=False)
    sub = pd.DataFrame({"id": out.id, "label": (out.p_cheat >= thr).astype(int)})
    sub.to_csv(cfg.run_dir / "submission.csv", index=False)
    print(f"threshold {thr:.2f} | predicted cheat ratio {sub.label.mean():.3f} | -> {cfg.run_dir / 'submission.csv'}")
    return out