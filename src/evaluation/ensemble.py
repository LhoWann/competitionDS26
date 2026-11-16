"""Clip-level ensembles of runs: weighted geometric mean of behavior probabilities, with weights and threshold chosen
on the pooled subject-grouped OOF and then applied unchanged to every test clip on its own."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.labels import BEHAVIORS, CHEAT_IDS
from evaluation.metrics import evaluate_clips

PROB_COLS = [f"p_beh{i}" for i in range(len(BEHAVIORS))]


def blend(frames: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    """Aligned frames (same rows) -> copy of the first with blended `p_beh*`, `p_cheat` and `pred_behavior`."""
    lp = sum(w * np.log(f[PROB_COLS].clip(lower=1e-6).to_numpy()) for f, w in zip(frames, weights, strict=True))
    p = np.exp(lp - lp.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    out = frames[0].copy()
    out[PROB_COLS] = p
    if "pred_behavior" in out:
        out["pred_behavior"] = p.argmax(1)
    out["p_cheat"] = p[:, CHEAT_IDS].sum(1)
    return out


def search_weights(oofs: list[pd.DataFrame], step: float = 0.1) -> tuple[list[float], dict[str, float]]:
    """Grid over the weight simplex; keeps the blend with the best OOF F1-macro at its tuned threshold (then AUC)."""
    n, k = len(oofs), round(1 / step)
    best, best_w, best_m = (-1.0, -1.0), [], {}
    for c in itertools.product(range(k + 1), repeat=n):
        if sum(c) != k:
            continue
        w = [x / k for x in c]
        m = evaluate_clips(blend(oofs, w))
        if (m["f1_best"], m["auc"]) > best:
            best, best_w, best_m = (m["f1_best"], m["auc"]), w, m
    return best_w, best_m


def ensemble(run_dirs: list[Path], out_dir: Path, step: float = 0.1) -> dict[str, float]:
    """Writes `oof.csv`, `cv.json` (with weights), `test_probs.csv` and `submission.csv` of the best blend."""
    oofs = [pd.read_csv(d / "oof.csv").sort_values(["video_id", "start"], ignore_index=True) for d in run_dirs]
    for d, o in zip(run_dirs[1:], oofs[1:], strict=True):
        if not o[["video_id", "start"]].equals(oofs[0][["video_id", "start"]]):
            raise ValueError(f"{d} OOF clips differ from {run_dirs[0]}")
    weights, cv = search_weights(oofs, step)
    tests = [pd.read_csv(d / "test_probs.csv").sort_values("id", ignore_index=True) for d in run_dirs]
    test = blend(tests, weights)
    out_dir.mkdir(parents=True, exist_ok=True)
    blend(oofs, weights).to_csv(out_dir / "oof.csv", index=False)
    cv |= {"weights": dict(zip([d.name for d in run_dirs], weights, strict=True))}
    (out_dir / "cv.json").write_text(json.dumps(cv, indent=2))
    test.to_csv(out_dir / "test_probs.csv", index=False)
    pd.DataFrame({"id": test.id, "label": (test.p_cheat >= cv["thr"]).astype(int)}).to_csv(
        out_dir / "submission.csv", index=False)
    return cv