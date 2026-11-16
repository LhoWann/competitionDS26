"""Builds the train index and test-style frame cache.

Usage: python -m cli.prepare --config configs/base.yaml [--set data.max_subjects=4] [--workers 4]
"""

from cli import config_parser
from core.config import load_config
from dataset.prepare import prepare


def main() -> None:
    parser = config_parser(__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    df = prepare(load_config(args.config, args.set), args.workers)
    print(df.groupby(["fold", "label"]).size().unstack())


if __name__ == "__main__":
    main()