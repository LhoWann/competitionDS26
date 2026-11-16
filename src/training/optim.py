"""Optimizers (AdamW, AdamW + Muon hybrid) and LR schedule."""

import math

import torch
from torch import nn

from core.config import TrainConfig


class HybridOptimizer(torch.optim.Optimizer):
    """Steps several optimizers as one, so AMP unscaling, grad clipping and LR scheduling see a single object.

    `param_groups` are the children's own group dicts, so scheduler LR updates propagate to them.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        super().__init__([g for o in optimizers for g in o.param_groups], {})

    @torch.no_grad()
    def step(self, closure: None = None) -> None:
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW for the pretrained backbone (lower LR), if any. For the new head: AdamW, or with `optimizer=muon`, Muon on
    the temporal transformer's 2D hidden matrices and AdamW on the rest (biases, norms, positional embedding, head).
    """
    wd = cfg.weight_decay
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head = [(n, p) for n, p in model.named_parameters() if not n.startswith("backbone.")]
    muon = [p for n, p in head if cfg.optimizer == "muon" and n.startswith("temporal.") and p.ndim == 2]
    muon_ids = {id(p) for p in muon}
    adam_head = [p for _, p in head if id(p) not in muon_ids]
    groups = [
        {"params": [p for p in bb if p.ndim > 1], "lr": cfg.backbone_lr, "weight_decay": wd},
        {"params": [p for p in bb if p.ndim <= 1], "lr": cfg.backbone_lr, "weight_decay": 0.0},
        {"params": [p for p in adam_head if p.ndim > 1], "lr": cfg.lr, "weight_decay": wd},
        {"params": [p for p in adam_head if p.ndim <= 1], "lr": cfg.lr, "weight_decay": 0.0},
    ]
    adamw = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.999))
    if not muon:
        return adamw
    return HybridOptimizer([adamw, torch.optim.Muon(muon, lr=cfg.lr, weight_decay=wd, adjust_lr_fn="match_rms_adamw")])


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)