"""Competition metric (binary F1-macro) and diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def binary_metrics(p_cheat: np.ndarray, y: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    """Accuracy, F1-macro, per-class F1 and AUC for cheat probabilities."""
    pred = (p_cheat >= thr).astype(int)
    f1 = f1_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    auc = roc_auc_score(y, p_cheat) if len(np.unique(y)) == 2 else float("nan")
    return {"acc": accuracy_score(y, pred), "f1_macro": f1.mean(), "f1_normal": f1[0], "f1_cheat": f1[1], "auc": auc}


def best_threshold(p_cheat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Grid-searches the threshold maximizing F1-macro; returns (threshold, f1_macro)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [binary_metrics(p_cheat, y, t)["f1_macro"] for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def evaluate_clips(df: pd.DataFrame) -> dict[str, float]:
    """Clip-level metrics plus video-level F1 (mean clip probability per video), from columns
    `p_cheat`, `label`, `behavior`, `pred_behavior`, `video`."""
    out = binary_metrics(df.p_cheat.to_numpy(), df.label.to_numpy())
    out["thr"], out["f1_best"] = best_threshold(df.p_cheat.to_numpy(), df.label.to_numpy())
    out["beh_acc"] = float((df.behavior == df.pred_behavior).mean())
    vid = df.groupby("video").agg(p_cheat=("p_cheat", "mean"), label=("label", "first"))
    out["vid_f1"] = binary_metrics(vid.p_cheat.to_numpy(), vid.label.to_numpy())["f1_macro"]
    return out