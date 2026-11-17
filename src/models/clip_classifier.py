"""Clip classifiers -> 8 behavior logits: VideoMAE or V-JEPA 2 video transformer, or a head over cached frozen pose
vectors."""


import timm
import torch
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel, VJEPA2Config, VJEPA2ForVideoClassification

from core.config import ModelConfig
from dataset.labels import BEHAVIORS, CHEAT_IDS


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
        freeze_prefix(self.backbone.embeddings, self.backbone.encoder.layer, cfg.train_blocks)
        self.register_buffer("mean", imagenet_stat(timm.data.IMAGENET_DEFAULT_MEAN), persistent=False)
        self.register_buffer("std", imagenet_stat(timm.data.IMAGENET_DEFAULT_STD), persistent=False)
        dim = self.backbone.config.hidden_size
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(dim, len(BEHAVIORS)))

    def tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) clips -> last-layer tubelet tokens (B, T / tubelet * H / patch * W / patch, dim)."""
        return self.backbone(pixel_values=(pixels.float() - self.mean) / self.std).last_hidden_state

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.tokens(pixels).mean(1)))


class VJEPA2Classifier(nn.Module):
    """HF V-JEPA 2 video classifier (e.g. `facebook/vjepa2-vitl-fpc16-256-ssv2`): ViT encoder (RoPE, so any T and
    H, W divisible by the patch size) -> its pretrained attentive pooler -> new 8-way linear head. With
    `model.train_blocks=0` the encoder stays frozen and only pooler + head learn (the paper's attentive-probe setup)."""

    def __init__(self, cfg: ModelConfig, num_frames: int, img_size: list[int]) -> None:
        super().__init__()
        if cfg.pretrained:
            full = VJEPA2ForVideoClassification.from_pretrained(cfg.backbone, attn_implementation="sdpa")
        else:
            full = VJEPA2ForVideoClassification(VJEPA2Config(crop_size=img_size[0], frames_per_clip=num_frames))
        self.backbone = full.vjepa2
        self.backbone.predictor = nn.Identity()
        if cfg.grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        freeze_prefix(self.backbone.encoder.embeddings, self.backbone.encoder.layer, cfg.train_blocks)
        self.pooler = full.pooler
        self.register_buffer("mean", imagenet_stat(timm.data.IMAGENET_DEFAULT_MEAN), persistent=False)
        self.register_buffer("std", imagenet_stat(timm.data.IMAGENET_DEFAULT_STD), persistent=False)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(full.config.hidden_size, len(BEHAVIORS)))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = (pixels.float() - self.mean) / self.std
        tokens = self.backbone(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        return self.head(self.pooler(tokens))


def imagenet_stat(values: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(values).view(1, 1, 3, 1, 1) * 255


def freeze_prefix(embeddings: nn.Module, layers: nn.ModuleList, train_blocks: int) -> None:
    """Freezes the embeddings and all but the last `train_blocks` transformer blocks; -1 keeps everything trainable."""
    if train_blocks < 0:
        return
    for module in [embeddings, *layers[:len(layers) - train_blocks]]:
        module.requires_grad_(False)


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