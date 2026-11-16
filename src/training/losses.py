"""Training objective."""


import torch
import torch.nn.functional as F

from core.config import TrainConfig
from models.clip_classifier import cheat_log_probs


def clip_loss(logits: torch.Tensor, behavior: torch.Tensor, label: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Behavior cross-entropy plus binary NLL on the marginal cheat probability."""
    logits = logits.float()
    ce = F.cross_entropy(logits, behavior, label_smoothing=cfg.label_smoothing)
    log_n, log_c = cheat_log_probs(logits)
    y = label.float()
    return ce + cfg.binary_weight * -(y * log_c + (1 - y) * log_n).mean()