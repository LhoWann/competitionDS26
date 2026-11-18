"""Writes the per-video action onsets (`dataset.onsets`) used as `data.onsets`.

Usage: python -m cli.onsets --oof outputs/ens_vjs2f_vms2f_pose26/oof.csv --out configs/onsets.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from dataset.onsets import estimate_onsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    onsets = estimate_onsets(pd.read_csv(args.oof))
    args.out.write_text(onsets.to_csv(index=False, lineterminator="\n").rstrip("\n"))
    secs = (onsets.onset // 30).value_counts().sort_index().to_dict()
    print(f"{len(onsets)} videos -> {args.out}; onset seconds: {secs}")


if __name__ == "__main__":
    main()