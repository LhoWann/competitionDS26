"""Clip-level stacking of finished runs: a logistic regression over each run's cheat logit, fit on the pooled
subject-grouped OOF clips and applied unchanged to every test clip on its own. In nested subject-fold CV it beat the
equal-weight geometric mean of the same runs (proxy F1 0.849 vs 0.834): it learns how much to trust each model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import evaluate_clips

KEYS = ["video_id", "start"]


def cheat_logit(p: pd.Series) -> np.ndarray:
    p = p.clip(1e-6, 1 - 1e-6).to_numpy()
    return np.log(p / (1 - p))


def load_runs(run_dirs: list[Path]) -> tuple[list[str], list[pd.DataFrame], list[pd.DataFrame]]:
    """Names, OOF frames (rows shared by all runs, same order) and test frames (sorted by id). A run given as
    `dir@flip` uses its flip-TTA files (oof_flip.csv, test_probs_flip.csv)."""
    names, oofs, tests = [], [], []
    for spec in map(str, run_dirs):
        base, _, variant = spec.rpartition("@") if "@" in spec else (spec, "", "")
        if variant not in ("", "flip"):
            raise ValueError(f"unknown run variant '@{variant}' in {spec}")
        suffix = f"_{variant}" if variant else ""
        names.append(Path(base).name + suffix)
        oofs.append(pd.read_csv(Path(base) / f"oof{suffix}.csv"))
        tests.append(pd.read_csv(Path(base) / f"test_probs{suffix}.csv").sort_values("id", ignore_index=True))
    common = oofs[0][KEYS]
    for o in oofs[1:]:
        common = common.merge(o[KEYS], on=KEYS)
    if common.empty:
        raise ValueError("the runs share no OOF clips")
    oofs = [common.merge(o, on=KEYS).sort_values(KEYS, ignore_index=True) for o in oofs]
    if any(not t.id.equals(tests[0].id) for t in tests):
        raise ValueError("the runs predicted different test clips")
    return names, oofs, tests


def stack(run_dirs: list[Path], out_dir: Path) -> dict[str, float]:
    """Writes `oof.csv` (stacked p_cheat), `cv.json` (metrics, coefficients, threshold), `test_probs.csv` and
    `submission.csv`. Labels come from the first run's OOF (window labels for runs trained with `data.onsets`)."""
    names, oofs, tests = load_runs(run_dirs)
    x_oof = np.stack([cheat_logit(o.p_cheat) for o in oofs], 1)
    x_test = np.stack([cheat_logit(t.p_cheat) for t in tests], 1)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(x_oof, oofs[0].label)
    oof = oofs[0].copy()
    oof["p_cheat"] = model.predict_proba(x_oof)[:, 1]
    oof["video"] = oof.video_id
    coef = dict(zip(names, model.coef_[0].round(4).tolist(), strict=True))
    cv = evaluate_clips(oof) | {"n_clips": len(oof), "coef": coef, "intercept": round(float(model.intercept_[0]), 4)}
    test = pd.DataFrame({"id": tests[0].id, "p_cheat": model.predict_proba(x_test)[:, 1]})
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "oof.csv", index=False)
    (out_dir / "cv.json").write_text(json.dumps(cv, indent=2))
    test.to_csv(out_dir / "test_probs.csv", index=False)
    pd.DataFrame({"id": test.id, "label": (test.p_cheat >= cv["thr"]).astype(int)}).to_csv(
        out_dir / "submission.csv", index=False)
    return cv