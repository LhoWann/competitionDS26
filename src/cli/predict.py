"""Fold-ensemble test inference and submission writing.

Usage:
    python -m cli.predict --config configs/debug.yaml
    accelerate launch --multi_gpu --num_processes 2 -m cli.predict --config configs/base.yaml
"""

from cli import config_parser
from core.config import load_config
from inference.predict import predict


def main() -> None:
    parser = config_parser(__doc__)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    predict(load_config(args.config, args.set), hflip_tta=not args.no_tta)


if __name__ == "__main__":
    main()