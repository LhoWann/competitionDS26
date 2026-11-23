"""Fold-ensemble test inference and submission writing.

Usage:
    python -m cli.predict --config configs/debug.yaml
    accelerate launch --multi_gpu --num_processes 2 -m cli.predict --config configs/colab_videomae_win.yaml
    # Re-score the fold checkpoints on their OOF clips first (e.g. with every frame: data.eval_passes=2):
    python -m cli.predict --config outputs/<run>/config.yaml --set data.eval_passes=2 --oof
"""

from cli import config_parser
from core.config import load_config
from inference.predict import predict, predict_oof


def main() -> None:
    parser = config_parser(__doc__)
    parser.add_argument("--oof", action="store_true", help="re-score the OOF clips of every fold checkpoint only")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if args.oof:
        predict_oof(cfg)
    else:
        predict(cfg)


if __name__ == "__main__":
    main()