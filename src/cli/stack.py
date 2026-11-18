"""Stacks finished runs (each needs `oof.csv` and `test_probs.csv`) into one submission.

Usage: python -m cli.stack --runs outputs/vjepa2_l_crop_win@flip outputs/videomae_b_crop_win@flip
       outputs/feat_pose_y26s_win --out outputs/stack_win   (`run@flip` = the run's flip-TTA files)
"""

import argparse
import json
from pathlib import Path

from evaluation.stack import stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv = stack(args.runs, args.out)
    print(json.dumps({k: v if isinstance(v, dict) else round(v, 4) for k, v in cv.items()}))


if __name__ == "__main__":
    main()