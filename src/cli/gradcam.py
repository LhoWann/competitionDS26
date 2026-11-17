"""Grad-CAM check of a VideoMAE fold checkpoint on its validation clips (head / wrist share of the decision map).

Usage: python -m cli.gradcam --config outputs/videomae_b_crop/config.yaml --set paths.cache_dir=cache
           --fold 0 --ckpt outputs/videomae_b_crop/fold0/best.pt --out outputs/gradcam_vm_f0 [--thr 0.54]
"""

import json
from pathlib import Path

from cli import config_parser
from core.config import load_config
from evaluation.gradcam import gradcam


def main() -> None:
    parser = config_parser(__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--roi", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    print(json.dumps(gradcam(cfg, args.fold, args.ckpt, args.out, args.thr, args.roi, args.stride), indent=1))


if __name__ == "__main__":
    main()