"""
Configuration management for training and inference.

Dataclasses define the typed schema.  OmegaConf handles YAML loading,
saving, and CLI-override merging — so ``!!python/tuple`` tags are gone
and dot-notation overrides are type-coerced automatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

# ═══════════════════════════════════════════════════════════════════════
# Dataclass schemas (single source of truth for fields + defaults)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DataConfig:
    """Dataset configuration."""

    # Paths
    image_dir: str = "data/images"
    mask_dir: str = "data/masks"
    train_split: str = ""
    val_split: str = ""

    # If split files not provided, use random split
    val_ratio: float = 0.15

    # Image settings
    image_size: Tuple[int, int] = (512, 512)
    num_classes: int = 3
    ignore_index: int = 255

    # Class remapping: raw mask pixel value → target class index
    class_mapping: Optional[Dict[int, int]] = None

    # Cityscapes mode
    cityscapes: bool = False

    # Augmentation
    augmentation: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    random_crop: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotation: float = 10.0
    scale: Tuple[float, float] = (0.8, 1.2)

    # Normalisation (ImageNet stats)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "deeplabv3plus"
    encoder_name: str = "timm-mobilenetv3_large_100"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 3

    # DeepLabV3+ specific
    encoder_output_stride: int = 16
    decoder_channels: int = 256
    decoder_atrous_rates: Tuple[int, ...] = (12, 24, 36)

    # MobileNetV3 specific
    activation: str = "hard_swish"


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)

    scheduler: str = "cosine"
    lr_min: float = 1e-6
    lr_warmup_epochs: int = 5
    lr_warmup_start: float = 1e-6
    poly_power: float = 0.9

    loss: str = "dice_ce"
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    class_weights: Optional[List[float]] = None

    mixed_precision: bool = True
    gradient_accumulation: int = 1
    grad_clip: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True

    val_every: int = 1
    save_every: int = 10
    early_stopping_patience: int = 20

    log_every: int = 20
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "skywater-seg"

    resume_from: str = ""


@dataclass
class DatasetConfig:
    """Single dataset source (multi-dataset mode)."""

    name: str = ""
    image_dir: str = ""
    mask_dir: str = ""
    image_size: Optional[Tuple[int, int]] = None
    num_classes: Optional[int] = None
    class_mapping: Optional[Dict[int, int]] = None
    augmentation: Optional[bool] = None
    cityscapes: bool = False
    split: str = "train"


@dataclass
class Config:
    """Master configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    datasets: List[DatasetConfig] = field(default_factory=list)
    mix_weights: Optional[List[float]] = None

    experiment_name: str = "skywater-seg"
    output_dir: str = "./checkpoints"
    seed: int = 42
    device: str = "cuda"

    # ── OmegaConf-powered I/O ──────────────────────────────────────

    def to_omegaconf(self) -> DictConfig:
        """Return an OmegaConf ``DictConfig`` view of this config."""
        return OmegaConf.structured(self)

    def save(self, path: str) -> None:
        """Save config to a clean YAML file (no ``!!python/tuple`` tags)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.to_omegaconf(), p)

    def to_dict(self) -> dict:
        """Convert to plain dict (tuples become lists for YAML compat)."""
        return OmegaConf.to_container(self.to_omegaconf(), resolve=True)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML, merging with defaults for missing keys."""
        return cls.from_dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from a plain dict (backward-compatible)."""
        schema = OmegaConf.structured(cls())
        merged = OmegaConf.merge(schema, data)
        return OmegaConf.to_object(merged)


def get_default_config() -> Config:
    """Return a Config populated with all default values."""
    return Config()


def cli_to_dotlist(raw: list) -> list:
    """Convert ``--key=val`` / ``--key val`` to OmegaConf dotlist.

    >>> cli_to_dotlist(["--train.batch_size=8", "--train.epochs", "50", "--amp"])
    ['train.batch_size=8', 'train.epochs=50', 'amp=true']
    """
    out, skip = [], False
    for i, a in enumerate(raw):
        if skip:
            skip = False
            continue
        a = a[2:] if a.startswith("--") else a
        if "=" in a:
            out.append(a)
        elif i + 1 < len(raw) and not raw[i + 1].startswith("--"):
            out.append(f"{a}={raw[i + 1]}")
            skip = True
        else:
            out.append(f"{a}=true")
    return out
