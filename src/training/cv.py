"""Subject-grouped cross-validation over all configured folds."""

import json
from pathlib import Path

import pandas as pd
import yaml
from accelerate import Accelerator

from core.config import Config
from evaluation.metrics import evaluate_clips
from training.engine import train_fold


def summarize(run_dir: Path) -> dict[str, float]:
    """Pools every fold's OOF predictions in `run_dir` and stores CV metrics (incl. tuned threshold) in cv.json."""
    oof = pd.concat([pd.read_csv(p) for p in sorted(run_dir.glob("fold*/oof.csv"))], ignore_index=True)
    oof["video"] = oof.video_id
    cv = evaluate_clips(oof)
    oof.to_csv(run_dir / "oof.csv", index=False)
    (run_dir / "cv.json").write_text(json.dumps(cv, indent=2))
    return cv


def run(cfg: Config) -> dict[str, float] | None:
    acc = Accelerator(mixed_precision=cfg.train.mixed_precision, gradient_accumulation_steps=cfg.train.grad_accum)
    videos = pd.read_csv(Path(cfg.paths.cache_dir) / "videos.csv")
    if acc.is_main_process:
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        (cfg.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    for fold in cfg.train.folds:
        train_fold(cfg, fold, videos, acc)
    acc.wait_for_everyone()
    if not acc.is_main_process:
        return None
    cv = summarize(cfg.run_dir)
    print("CV (pooled OOF):", json.dumps({k: round(v, 4) for k, v in cv.items()}))
    return cv