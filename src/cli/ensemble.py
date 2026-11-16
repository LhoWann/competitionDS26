"""Clip-level ensemble of finished runs (each needs `oof.csv` and `test_probs.csv`).

Usage: python -m cli.ensemble --runs outputs/convnext_t_t8 outputs/feat_pose_y11s --out outputs/ens_x [--step 0.1]
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
    args = parser.parse_args()
    cv = ensemble(args.runs, args.out, args.step)
    print(json.dumps({k: v if isinstance(v, dict) else round(v, 4) for k, v in cv.items()}))


if __name__ == "__main__":
    main()