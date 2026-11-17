"""Fold-ensemble test inference and submission writing.

Usage:
    python -m cli.predict --config configs/debug.yaml
    accelerate launch --multi_gpu --num_processes 2 -m cli.predict --config configs/colab_videomae.yaml
"""

from cli import config_parser
from core.config import load_config
from inference.predict import predict


def main() -> None:
    args = config_parser(__doc__).parse_args()
    predict(load_config(args.config, args.set))


if __name__ == "__main__":
    main()