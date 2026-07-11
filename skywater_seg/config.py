"""
Configuration management for training and inference.

Uses dataclass + YAML for clean, typed configuration.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class DataConfig:
    """Dataset configuration."""

    # Paths
    image_dir: str = "data/images"
    mask_dir: str = "data/masks"
    train_split: str = ""  # Path to train.txt (one filename per line), or ""
    val_split: str = ""  # Path to val.txt, or ""

    # If split files not provided, use random split
    val_ratio: float = 0.15

    # Image settings
    image_size: Tuple[int, int] = (512, 512)  # (height, width)
    num_classes: int = 3  # 0=background, 1=sky, 2=water (set to 4 for +person)
    ignore_index: int = 255  # Value to ignore in loss computation

    # Class remapping: raw mask pixel value → target class index
    # e.g. for ADE20K: {3: 1, 13: 3, 22: 2, 27: 2, 61: 2, 110: 2, 114: 2, 129: 2}
    # None means no remapping (masks already use 0, 1, 2, ... directly)
    class_mapping: Optional[Dict[int, int]] = None

    # Cityscapes mode: auto-detect city subdirectory layout
    cityscapes: bool = False

    # Augmentation
    augmentation: bool = True
    # Color
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    # Geometric
    random_crop: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0  # Don't flip vertically (sky is up)
    rotation: float = 10.0  # Max rotation degrees
    scale: Tuple[float, float] = (0.8, 1.2)

    # Normalization (ImageNet stats)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture
    name: str = "deeplabv3plus"  # Architecture name
    encoder_name: str = "timm-mobilenetv3_large_100"
    encoder_weights: str = "imagenet"  # Pretrained weights for encoder
    in_channels: int = 3
    classes: int = 3  # Number of output classes

    # DeepLabV3+ specific
    encoder_output_stride: int = 16
    decoder_channels: int = 256
    decoder_atrous_rates: Tuple[int, ...] = (12, 24, 36)

    # MobileNetV3 specific
    activation: str = "hard_swish"  # MobileNetV3 uses h-swish


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimization
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw, sgd, adam
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW

    # Learning rate schedule
    scheduler: str = "cosine"  # cosine, poly, step, plateau
    lr_min: float = 1e-6
    lr_warmup_epochs: int = 5
    lr_warmup_start: float = 1e-6
    poly_power: float = 0.9  # For poly scheduler

    # Loss
    loss: str = "dice_ce"  # dice_ce, ce, focal, jaccard, lovasz
    ce_weight: float = 0.5  # Weight of CE in combined loss
    dice_weight: float = 0.5  # Weight of Dice in combined loss
    class_weights: Optional[List[float]] = None  # Per-class CE weights

    # Training
    mixed_precision: bool = True  # AMP
    gradient_accumulation: int = 1  # Accumulate N batches before step
    grad_clip: float = 1.0  # Max gradient norm
    num_workers: int = 4
    pin_memory: bool = True

    # Validation
    val_every: int = 1  # Validate every N epochs
    save_every: int = 10  # Save checkpoint every N epochs
    early_stopping_patience: int = 20

    # Logging
    log_every: int = 20  # Log every N batches
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "skywater-seg"

    # Resume
    resume_from: str = ""  # Path to checkpoint to resume from


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source (used in multi-dataset mode)."""

    name: str = ""                           # e.g. "ade20k", "cityscapes"
    image_dir: str = ""
    mask_dir: str = ""
    image_size: Optional[Tuple[int, int]] = None  # override, None = use Config.data.image_size
    num_classes: Optional[int] = None              # override, None = use Config.data.num_classes
    class_mapping: Optional[Dict[int, int]] = None
    augmentation: Optional[bool] = None            # None = use Config.data.augmentation
    cityscapes: bool = False                       # Use Cityscapes directory layout
    split: str = "train"                           # "train" or "val"


@dataclass
class Config:
    """Master configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Multi-dataset mode: list of dataset sources (overrides data.image_dir etc.)
    datasets: List[DatasetConfig] = field(default_factory=list)
    mix_weights: Optional[List[float]] = None  # per-dataset sampling weights (None = uniform)

    # Experiment
    experiment_name: str = "skywater-seg"
    output_dir: str = "./checkpoints"
    seed: int = 42
    device: str = "cuda"

    def save(self, path: str):
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict:
        """Convert to plain dict for YAML serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "__dataclass_fields__"):
                result[key] = {
                    k: v for k, v in value.__dict__.items()
                }
            else:
                result[key] = value
        return result

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from dict, with nested dataclass construction."""
        config = cls()

        if "data" in data:
            config.data = DataConfig(**{
                k: v for k, v in data["data"].items()
                if k in DataConfig.__dataclass_fields__
            })
        if "model" in data:
            config.model = ModelConfig(**{
                k: v for k, v in data["model"].items()
                if k in ModelConfig.__dataclass_fields__
            })
        if "train" in data:
            config.train = TrainConfig(**{
                k: v for k, v in data["train"].items()
                if k in TrainConfig.__dataclass_fields__
            })

        # Parse multi-dataset entries
        if "datasets" in data:
            config.datasets = []
            for ds_data in data["datasets"]:
                config.datasets.append(DatasetConfig(**{
                    k: v for k, v in ds_data.items()
                    if k in DatasetConfig.__dataclass_fields__
                }))
        if "mix_weights" in data:
            config.mix_weights = data["mix_weights"]

        # Top-level fields
        for key in ["experiment_name", "output_dir", "seed", "device"]:
            if key in data:
                setattr(config, key, data[key])

        return config


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()
