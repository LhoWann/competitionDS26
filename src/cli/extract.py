"""Frozen-backbone feature cache for the `model.kind=features` head (needs `cli.prepare` first).

Usage: python -m cli.extract --config configs/feat_pose.yaml [--set features.backbone=yolo11s-pose] [--workers 4]
"""

from cli import config_parser
from core.config import load_config
from features.extract import extract


def main() -> None:
    parser = config_parser(__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print("features ->", extract(load_config(args.config, args.set), args.workers))


if __name__ == "__main__":
    main()