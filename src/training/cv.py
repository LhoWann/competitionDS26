"""Subject-grouped cross-validation over all configured folds."""

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from accelerate import Accelerator

from core.config import Config
from evaluation.metrics import evaluate_clips
from training.engine import train_fold


def summarize(run_dir: Path) -> dict[str, float]:
    """Pools every fold's OOF predictions in `run_dir` into oof.csv and stores CV metrics (incl. tuned threshold) in
    cv.json; the flip-TTA predictions of video models likewise into oof_flip.csv / cv_flip.json. Returns cv.json."""
    plain = sorted(run_dir.glob("fold*/oof.csv"))
    flip = sorted(run_dir.glob("fold*/oof_flip.csv"))
    if flip and [p.parent for p in flip] != [p.parent for p in plain]:
        raise ValueError(f"{run_dir}: oof_flip.csv only in {[p.parent.name for p in flip]}; retrain the other folds")
    out = {}
    for suffix, files in (("", plain), ("_flip", flip)):
        if not files:
            continue
        oof = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        oof["video"] = oof.video_id
        out[suffix] = evaluate_clips(oof)
        oof.to_csv(run_dir / f"oof{suffix}.csv", index=False)
        (run_dir / f"cv{suffix}.json").write_text(json.dumps(out[suffix], indent=2))
    return out[""]


def recipe(cfg: dict[str, Any]) -> dict[str, Any]:
    """The parts of a config dict that change what a fold learns: everything but paths, loader workers, folds, resume
    and how the effective batch is split into micro-batch x grad_accum."""
    d = copy.deepcopy(cfg)
    d.pop("paths", None)
    d["data"].pop("num_workers", None)
    t = d["train"]
    t["effective_batch"] = t.pop("batch_size") * t.pop("grad_accum", 1)
    for key in ("folds", "resume"):
        t.pop(key, None)
    return d


def pending_folds(cfg: Config) -> list[int]:
    """Folds of `train.folds` to train. Raises if `run_dir` holds a run of another recipe (its folds would be pooled
    with the new ones); with `train.resume`, skips folds that already have both oof.csv and best.pt."""
    saved = cfg.run_dir / "config.yaml"
    if saved.exists() and recipe(yaml.safe_load(saved.read_text())) != recipe(cfg.to_dict()):
        raise ValueError(f"{saved} has another recipe than the current config: use a new run name")
    done = [f for f in cfg.train.folds if all((cfg.run_dir / f"fold{f}" / n).exists() for n in ("oof.csv", "best.pt"))]
    return [f for f in cfg.train.folds if not (cfg.train.resume and f in done)]


def run(cfg: Config) -> dict[str, float] | None:
    acc = Accelerator(mixed_precision=cfg.train.mixed_precision, gradient_accumulation_steps=cfg.train.grad_accum)
    videos = pd.read_csv(Path(cfg.paths.cache_dir) / "videos.csv")
    folds = pending_folds(cfg)
    if folds != list(cfg.train.folds):
        acc.print(f"resume: folds {sorted(set(cfg.train.folds) - set(folds))} already finished, training {folds}")
    if acc.is_main_process:
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        (cfg.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    for fold in folds:
        train_fold(cfg, fold, videos, acc)
    acc.wait_for_everyone()
    if not acc.is_main_process:
        return None
    cv = summarize(cfg.run_dir)
    print("CV (pooled OOF):", json.dumps({k: round(v, 4) for k, v in cv.items()}))
    if (cfg.run_dir / "cv_flip.json").exists():
        flip = json.loads((cfg.run_dir / "cv_flip.json").read_text())
        print("CV with flip TTA:", json.dumps({k: round(v, 4) for k, v in flip.items()}))
    return cv