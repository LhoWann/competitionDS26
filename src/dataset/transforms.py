"""Clip-level augmentation."""


from torchvision.transforms import v2

from core.config import DataConfig


def build_augment(cfg: DataConfig) -> v2.Compose:
    """Clip-consistent augmentation: v2 transforms share random params across the T frames of one tensor."""
    h, w = cfg.img_size
    j = cfg.color_jitter
    return v2.Compose([
        v2.RandomResizedCrop((h, w), scale=tuple(cfg.crop_scale), ratio=(w / h * 0.8, w / h * 1.25), antialias=True),
        v2.RandomHorizontalFlip(cfg.hflip),
        v2.RandomApply([v2.ColorJitter(j, j, j, cfg.hue)], p=0.8),
        v2.RandomGrayscale(cfg.grayscale),
    ])