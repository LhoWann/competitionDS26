"""Experiment configuration: typed dataclasses loaded from YAML with dotted CLI overrides."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data"
    cache_dir: str = "cache"
    output_dir: str = "outputs"


@dataclass
class DataConfig:
    clip_len: int = 30
    num_frames: int = 8
    img_size: list[int] = field(default_factory=lambda: [224, 384])
    clips_per_video: int = 16
    n_folds: int = 5
    max_subjects: int = 0
    num_workers: int = 4
    hflip: float = 0.5
    crop_scale: list[float] = field(default_factory=lambda: [0.6, 1.0])
    color_jitter: float = 0.3
    grayscale: float = 0.0
    speed_jitter: float = 0.0
    person_crop: float = 0.0
    crop_features: str = "yolo11s-pose_360x640_s2"


@dataclass
class FeaturesConfig:
    backbone: str = "vit_base_patch16_dinov3.lvd1689m"
    img_size: list[int] = field(default_factory=lambda: [224, 384])
    stride: int = 2
    grid: list[int] = field(default_factory=lambda: [2, 4])
    pool: str = "cls_mean"
    batch_size: int = 32

    @property
    def tag(self) -> str:
        return f"{self.backbone}_{self.img_size[0]}x{self.img_size[1]}_s{self.stride}"


@dataclass
class ModelConfig:
    kind: str = "frames"
    hidden: int = 256
    backbone: str = "convnext_tiny.fb_in22k"
    pretrained: bool = True
    drop_path: float = 0.1
    temporal_layers: int = 1
    temporal_heads: int = 8
    dropout: float = 0.2
    grad_checkpointing: bool = False


@dataclass
class TrainConfig:
    folds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    epochs: int = 10
    batch_size: int = 8
    lr: float = 3e-4
    backbone_lr: float = 3e-5
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    label_smoothing: float = 0.1
    binary_weight: float = 1.0
    mixed_precision: str = "fp16"
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    patience: int = 4
    select: str = "f1_macro"
    ema: float = 0.0
    seed: int = 42


@dataclass
class Config:
    name: str = "debug"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.output_dir) / self.name

    @property
    def feature_dir(self) -> Path:
        return Path(self.paths.cache_dir) / "features" / self.features.tag

    @property
    def crop_dir(self) -> Path | None:
        """Pose cache whose person boxes crop the frames (`data.person_crop` > 0), else None."""
        return Path(self.paths.cache_dir) / "features" / self.data.crop_features if self.data.person_crop > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(obj: Any, values: dict[str, Any]) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Builds a Config from defaults, an optional YAML file and `section.key=value` overrides.

    Args:
        path: YAML file whose nested keys mirror the dataclass sections.
        overrides: Strings like `train.epochs=2`; values are parsed as YAML scalars/lists.
    """
    cfg = Config()
    if path is not None:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides or []:
        dotted, raw = item.split("=", 1)
        nested: dict[str, Any] = {}
        node = nested
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
        _merge(cfg, nested)
    return cfg