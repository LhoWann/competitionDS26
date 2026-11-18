"""Cross-validated training.

Usage:
    python -m cli.train --config configs/debug.yaml
    accelerate launch --multi_gpu --num_processes 2 -m cli.train --config configs/colab_videomae_win.yaml
"""

from cli import config_parser
from core.config import load_config
from training.cv import run


def main() -> None:
    args = config_parser(__doc__).parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()