"""Clip classifiers -> 8 behavior logits: end-to-end timm frame backbone, VideoMAE video transformer, or a head over
cached frozen features."""


import timm
import torch
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel

from core.config import ModelConfig
from dataset.labels import BEHAVIORS, CHEAT_IDS


class ClipClassifier(nn.Module):
    """Takes uint8 clips (B, T, 3, H, W); normalization runs on-device so loaders ship compact uint8 tensors."""

    def __init__(self, cfg: ModelConfig, num_frames: int, img_size: list[int]) -> None:
        super().__init__()
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0,
                                          drop_path_rate=cfg.drop_path)
        if hasattr(self.backbone, "set_input_size"):
            self.backbone.set_input_size(img_size=tuple(img_size))
        self.backbone.set_grad_checkpointing(cfg.grad_checkpointing)
        data_cfg = timm.data.resolve_data_config({}, model=self.backbone)
        self.register_buffer("mean", torch.tensor(data_cfg["mean"]).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("std", torch.tensor(data_cfg["std"]).view(1, 3, 1, 1) * 255, persistent=False)
        dim = self.backbone.num_features
        self.pos = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.temporal = temporal_encoder(dim, cfg)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(dim, len(BEHAVIORS)))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        b, t = pixels.shape[:2]
        x = (pixels.flatten(0, 1).float() - self.mean) / self.std
        feats = self.backbone(x).view(b, t, -1)
        feats = self.temporal(feats + self.pos)
        return self.head(self.norm(feats.mean(1)))


class VideoMAEClassifier(nn.Module):
    """HF VideoMAE (e.g. `MCG-NJU/videomae-base-finetuned-kinetics`) over uint8 clips (B, T, 3, H, W): tubelet
    tokens -> mean pool -> 8 logits. Position embeddings are fixed sinusoids, so any even T and H, W divisible by 16
    keep the pretrained weights valid."""

    def __init__(self, cfg: ModelConfig, num_frames: int, img_size: list[int]) -> None:
        super().__init__()
        size = {"num_frames": num_frames, "image_size": tuple(img_size)}
        if cfg.pretrained:
            self.backbone = VideoMAEModel.from_pretrained(cfg.backbone, attn_implementation="sdpa", **size)
        else:
            self.backbone = VideoMAEModel(VideoMAEConfig(**size))
        if cfg.grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        self.register_buffer("mean", torch.tensor(timm.data.IMAGENET_DEFAULT_MEAN).view(1, 1, 3, 1, 1) * 255,
                             persistent=False)
        self.register_buffer("std", torch.tensor(timm.data.IMAGENET_DEFAULT_STD).view(1, 1, 3, 1, 1) * 255,
                             persistent=False)
        dim = self.backbone.config.hidden_size
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(dim, len(BEHAVIORS)))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=(pixels.float() - self.mean) / self.std).last_hidden_state
        return self.head(self.norm(tokens.mean(1)))


class FeatureClipClassifier(nn.Module):
    """Head over cached frozen-backbone frame vectors (B, T, F): projection -> temporal transformer -> 8 logits."""

    def __init__(self, cfg: ModelConfig, in_dim: int, num_frames: int) -> None:
        super().__init__()
        dim = cfg.hidden
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(cfg.dropout), nn.Linear(in_dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.temporal = temporal_encoder(dim, cfg)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(dim, len(BEHAVIORS)))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.temporal(self.proj(feats.float()) + self.pos)
        return self.head(self.norm(x.mean(1)))


def temporal_encoder(dim: int, cfg: ModelConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, cfg.temporal_heads, dim * 2, cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, cfg.temporal_layers, enable_nested_tensor=False)


def cheat_log_probs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (log p(normal), log p(cheat)) by marginalizing the behavior softmax over each group."""
    log_z = logits.logsumexp(-1)
    normal_ids = [i for i in range(logits.shape[-1]) if i not in CHEAT_IDS]
    return logits[:, normal_ids].logsumexp(-1) - log_z, logits[:, CHEAT_IDS].logsumexp(-1) - log_z