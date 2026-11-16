"""YOLO-pose frame encoder: the most confident person's 17 COCO keypoints as a subject-invariant frame vector.

Each frame becomes (1, 56): keypoint xy relative to the person box centre in units of box height (34), keypoint
confidences (17), box width, height and centre as frame fractions (4) and detection confidence (1); zeros if nobody
is found. The box also drives the person crop of `dataset.clips` (`data.crop_box`).
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

from core.config import FeaturesConfig

POSE_DIM = 56


def pose_vector(box: np.ndarray, conf: float, kpts: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """(4,) xyxy box, its confidence and (17, 3) keypoints (x, y, conf) in pixels -> (56,) frame vector."""
    h, w = hw
    bh = max(box[3] - box[1], 1.0)
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    xy = (kpts[:, :2] - centre) / bh
    box_vec = [(box[2] - box[0]) / w, bh / h, centre[0] / w, centre[1] / h, conf]
    return np.concatenate([xy.ravel(), kpts[:, 2], box_vec]).astype(np.float32)


class PoseEncoder(nn.Module):
    """Wraps an Ultralytics pose model; `features.backbone` names the weights (e.g. `yolo11s-pose`), downloaded
    into `weights_dir`. Without `pretrained` the architecture is built untrained (tests)."""

    def __init__(self, cfg: FeaturesConfig, weights_dir: Path, pretrained: bool = True) -> None:
        super().__init__()
        weights_dir.mkdir(parents=True, exist_ok=True)
        self.yolo = YOLO(str(weights_dir / f"{cfg.backbone}.pt") if pretrained else f"{cfg.backbone}.yaml")
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        imgs = list(frames.flip(1).permute(0, 2, 3, 1).cpu().numpy())
        out = np.zeros((len(imgs), 1, POSE_DIM), np.float32)
        for i, r in enumerate(self.yolo(imgs, verbose=False, device=self.anchor.device)):
            if len(r.boxes) and r.keypoints is not None:
                k = int(r.boxes.conf.argmax())
                out[i, 0] = pose_vector(r.boxes.xyxy[k].cpu().numpy(), r.boxes.conf[k].item(),
                                        r.keypoints.data[k].cpu().numpy(), imgs[i].shape[:2])
        return torch.from_numpy(out)