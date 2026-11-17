"""Clip-level ensemble of finished runs (each needs `oof.csv` and `test_probs.csv`).

Usage: python -m cli.ensemble --runs outputs/convnext_t_t8 outputs/feat_pose_y11s --out outputs/ens_x [--step 0.1]
       [--weights 0.5 0.5]  (fixed weights: only the threshold is tuned on OOF); `outputs/run@flip` = flip-TTA files
"""

import argparse
import json
from pathlib import Path

from evaluation.ensemble import ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()
    cv = ensemble(args.runs, args.out, args.step, args.weights)
    print(json.dumps({k: v if isinstance(v, dict) else round(v, 4) for k, v in cv.items()}))


if __name__ == "__main__":
    main()