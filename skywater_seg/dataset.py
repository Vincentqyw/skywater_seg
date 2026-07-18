"""
PyTorch Dataset for sky/water/person segmentation.

Supports:
  - Single dataset (image + mask pairs with class remapping)
  - Multi-dataset mixed training (combine multiple sources via config)
  - ADE20K, Cityscapes, and custom datasets
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from skywater_seg.config import Config


# ── Robust image/mask loading (PIL-based, avoid OpenCV JP2 decode issues) ──

def _load_image_robust(path: str) -> np.ndarray:
    """Load image as RGB numpy array using PIL. Returns None on failure."""
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _load_mask_robust(path: str) -> np.ndarray:
    """Load mask as grayscale numpy array using PIL. Returns None on failure."""
    try:
        from PIL import Image
        return np.array(Image.open(path), dtype=np.uint8)
    except Exception:
        return None


# ── Cityscapes-specific helpers ──

def _find_cityscapes_pairs(image_dir: str, split: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Scan Cityscapes directory structure and return (image_path, mask_path) pairs."""
    pairs = []
    # Cityscapes uses "train"/"val", but try alternative names too
    split_names = {"train": ["train", "training"], "val": ["val", "validation"]}
    candidates = split_names.get(split, [split])

    img_root = None
    msk_root = None
    for s in candidates:
        if (Path(image_dir) / s).exists():
            img_root = Path(image_dir) / s
            msk_root = Path(mask_dir) / s
            break

    if img_root is None:
        return pairs

    if not img_root.exists():
        return pairs

    for city_dir in sorted(img_root.iterdir()):
        if not city_dir.is_dir():
            continue
        for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
            # Derive mask filename: aachen_000000_000019_leftImg8bit.png -> ..._gtFine_labelIds.png
            stem = img_path.stem.replace("_leftImg8bit", "")
            mask_path = msk_root / city_dir.name / f"{stem}_gtFine_labelIds.png"
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))

    return pairs


# ── Single Dataset ──

class SkyWaterDataset(Dataset):
    """Dataset for sky/water/person segmentation.

    Supports:
      - Flat directory: image_dir/*.jpg + mask_dir/*_mask.png
      - Split-file mode: file_list with one image name per line
      - Subdirectory mode: image_dir/training/*.jpg + mask_dir/training/*.png
      - Cityscapes mode: auto-detect city subdirectories (via cityscapes=True)
      - Class remapping: class_mapping dict maps raw→target class indices
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 4,
        augmentation: bool = False,
        config: Optional["DataConfig"] = None,
        file_list: Optional[List[str]] = None,
        class_mapping: Optional[Dict[int, int]] = None,
        cityscapes: bool = False,
        split: str = "train",
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.config = config
        self.class_mapping = class_mapping
        self._cityscapes = cityscapes
        self._split = split

        # ── Gather images ──
        if cityscapes:
            self._pairs = _find_cityscapes_pairs(str(image_dir), split, str(mask_dir))
            self.images = [p[0] for p in self._pairs]
            self._masks = {p[0]: p[1] for p in self._pairs}
        elif file_list is not None:
            self.images = file_list
            self._pairs = None
            self._masks = {}
        elif (Path(image_dir) / split).exists() or \
             (Path(image_dir) / f"{split}ing").exists() or \
             (Path(image_dir) / ("" if split != "val" else "validation")).exists():
            # Subdirectory mode: handles "train"/"val" (Cityscapes),
            # "training"/"validation" (ADE20K)
            candidates = [split]
            if split == "train":
                candidates.append("training")
            elif split == "val":
                candidates.extend(["validation", "val"])
            for sub_name in candidates:
                if (Path(image_dir) / sub_name).exists():
                    sub_dir = Path(image_dir) / sub_name
                    break
            else:
                sub_dir = Path(image_dir) / split
            extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
            self.images = sorted([
                str(f.relative_to(Path(image_dir)))  # e.g. "training/ADE_001.jpg"
                for f in sub_dir.rglob("*")
                if f.suffix.lower() in extensions
            ])
            self._pairs = None
            self._masks = {}
        else:
            extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
            self.images = sorted([
                f.name for f in self.image_dir.glob("*")
                if f.suffix.lower() in extensions
            ])
            self._pairs = None
            self._masks = {}

        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # ── Augmentation transforms ──
        if augmentation and config is not None:
            self.transform = self._build_transforms(config)
        else:
            self.transform = None

    def _build_transforms(self, cfg: "DataConfig"):
        import albumentations as A
        h, w = self.image_size
        transforms = [A.Resize(height=h, width=w)]
        if cfg.horizontal_flip > 0:
            transforms.append(A.HorizontalFlip(p=cfg.horizontal_flip))
        if cfg.vertical_flip > 0:
            transforms.append(A.VerticalFlip(p=cfg.vertical_flip))
        if cfg.rotation > 0:
            transforms.append(A.Rotate(limit=cfg.rotation, border_mode=cv2.BORDER_CONSTANT, p=0.5))
        transforms.append(A.ColorJitter(
            brightness=cfg.brightness, contrast=cfg.contrast,
            saturation=cfg.saturation, hue=cfg.hue, p=0.5))
        transforms.append(A.Normalize(mean=cfg.mean, std=cfg.std))
        return A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self._get_image_path(idx)

        # Load image
        image = _load_image_robust(image_path)
        if image is None:
            h, w = self.image_size
            return {"image": torch.zeros((3, h, w), dtype=torch.float32),
                    "mask": torch.zeros((h, w), dtype=torch.long),
                    "name": Path(image_path).name}

        # Load mask
        mask = self._load_mask(idx)

        # Class remapping
        if self.class_mapping is not None:
            remapped = np.zeros_like(mask, dtype=np.uint8)
            for raw_val, target_val in self.class_mapping.items():
                remapped[mask == raw_val] = target_val
            mask = remapped

        # Resize
        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            mean = self.config.mean if self.config else [0.485, 0.456, 0.406]
            std = self.config.std if self.config else [0.229, 0.224, 0.225]
            image = image.astype(np.float32) / 255.0
            image = (image - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        mask = torch.clamp(mask, 0, self.num_classes - 1)

        return {"image": image, "mask": mask, "name": Path(image_path).name}

    def _get_image_path(self, idx: int) -> str:
        name = self.images[idx]
        if self._masks:
            return name  # Cityscapes mode: absolute paths stored
        return str(self.image_dir / name)

    def _load_mask(self, idx: int) -> np.ndarray:
        name = self.images[idx]

        # Cityscapes mode: mask path stored alongside image
        if self._masks and name in self._masks:
            mask = _load_mask_robust(self._masks[name])
            if mask is not None:
                return mask

        # Standard mode: derive mask from image name
        stem = Path(name).stem

        # Extract subdirectory from image name (e.g. "training/ADE_001.jpg" → "training")
        subdir = Path(name).parent if name else Path(".")

        candidates = []
        if str(subdir) != ".":
            candidates += [
                self.mask_dir / subdir / f"{stem}_mask.png",
                self.mask_dir / subdir / f"{stem}.png",
            ]
        candidates += [
            self.mask_dir / f"{stem}_mask.png",
            self.mask_dir / f"{stem}.png",
            self.mask_dir / f"{stem}_mask.jpg",
        ]

        for path in candidates:
            if path.exists():
                mask = _load_mask_robust(str(path))
                if mask is not None:
                    return mask

        # No mask found — return zeros
        h, w = self.image_size
        return np.zeros((h, w), dtype=np.uint8)


# ── Multi-Dataset Wrapper ──

class MultiDataset(Dataset):
    """Concatenates multiple SkyWaterDataset instances for mixed training.

    Supports weighted sampling: each dataset can have a `weight` to control
    how often samples are drawn from it.
    """

    def __init__(self, datasets: List[SkyWaterDataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        if weights is None:
            weights = [1.0] * len(datasets)
        self.weights = weights

        self._lengths = [len(d) for d in datasets]
        self._offsets = []
        offset = 0
        for l in self._lengths:
            self._offsets.append(offset)
            offset += l

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Weighted random sampling if weights differ
        if len(set(self.weights)) > 1:
            ds_idx = np.random.choice(len(self.datasets), p=np.array(self.weights) / sum(self.weights))
            local_idx = np.random.randint(0, self._lengths[ds_idx])
        else:
            # Uniform: find which dataset owns this index
            for ds_idx in range(len(self.datasets) - 1, -1, -1):
                if idx >= self._offsets[ds_idx]:
                    local_idx = idx - self._offsets[ds_idx]
                    break
            else:
                ds_idx, local_idx = 0, 0

        return self.datasets[ds_idx][local_idx]

    @property
    def image_size(self):
        return self.datasets[0].image_size

    @property
    def num_classes(self):
        return self.datasets[0].num_classes


# ── Dataloader Factory ──

def create_dataloaders(config: Config) -> Tuple[torch.utils.data.DataLoader,
                                                   torch.utils.data.DataLoader]:
    """Create train and validation dataloaders from config.

    Supports:
      1. Single dataset (config.data)
      2. Multi-dataset (config.datasets list)
      3. Subdirectory split (training/validation subdirs)
      4. Explicit split files
      5. Cityscapes mode
    """
    from torch.utils.data import DataLoader

    # ── Multi-dataset mode ──
    if config.datasets:
        train_datasets = []
        val_datasets = []

        for ds_cfg in config.datasets:
            aug = ds_cfg.augmentation if ds_cfg.augmentation is not None else config.data.augmentation

            train_ds = SkyWaterDataset(
                image_dir=ds_cfg.image_dir,
                mask_dir=ds_cfg.mask_dir,
                image_size=tuple(ds_cfg.image_size or config.data.image_size),
                num_classes=ds_cfg.num_classes or config.data.num_classes,
                augmentation=aug,
                config=config.data if aug else None,
                class_mapping=ds_cfg.class_mapping,
                cityscapes=ds_cfg.cityscapes,
                split="train",
            )
            val_ds = SkyWaterDataset(
                image_dir=ds_cfg.image_dir,
                mask_dir=ds_cfg.mask_dir,
                image_size=tuple(ds_cfg.image_size or config.data.image_size),
                num_classes=ds_cfg.num_classes or config.data.num_classes,
                augmentation=False,
                config=None,
                class_mapping=ds_cfg.class_mapping,
                cityscapes=ds_cfg.cityscapes,
                split="val",
            )
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        train_dataset = MultiDataset(train_datasets, config.mix_weights)
        val_dataset = MultiDataset(val_datasets, None)  # uniform val sampling

    # ── Single dataset: Mode 1 (explicit split files) ──
    elif config.data.train_split and os.path.exists(config.data.train_split):
        with open(config.data.train_split, encoding="utf-8") as f:
            train_files = [line.strip() for line in f if line.strip()]
        with open(config.data.val_split, encoding="utf-8") as f:
            val_files = [line.strip() for line in f if line.strip()]

        train_dataset = SkyWaterDataset(
            image_dir=config.data.image_dir, mask_dir=config.data.mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=config.data.augmentation,
            config=config.data if config.data.augmentation else None,
            file_list=train_files,
            class_mapping=config.data.class_mapping,
            cityscapes=config.data.cityscapes,
            split="train",
        )
        val_dataset = SkyWaterDataset(
            image_dir=config.data.image_dir, mask_dir=config.data.mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=False, config=None,
            file_list=val_files,
            class_mapping=config.data.class_mapping,
            cityscapes=config.data.cityscapes,
            split="val",
        )

    # ── Single dataset: Mode 2 (subdirectory split) ──
    elif (Path(config.data.image_dir) / "training").exists():
        train_dataset = SkyWaterDataset(
            image_dir=str(Path(config.data.image_dir) / "training"),
            mask_dir=str(Path(config.data.mask_dir) / "training"),
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=config.data.augmentation,
            config=config.data if config.data.augmentation else None,
            class_mapping=config.data.class_mapping,
            cityscapes=config.data.cityscapes,
            split="train",
        )
        val_dataset = SkyWaterDataset(
            image_dir=str(Path(config.data.image_dir) / "validation"),
            mask_dir=str(Path(config.data.mask_dir) / "validation"),
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=False, config=None,
            class_mapping=config.data.class_mapping,
            cityscapes=config.data.cityscapes,
            split="val",
        )

    # ── Single dataset: Mode 3 (random split) ──
    else:
        full_dataset = SkyWaterDataset(
            image_dir=config.data.image_dir, mask_dir=config.data.mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=False, config=None,
            class_mapping=config.data.class_mapping,
            cityscapes=config.data.cityscapes,
            split="train",
        )
        from torch.utils.data import random_split
        n_val = max(1, int(len(full_dataset) * config.data.val_ratio))
        n_train = len(full_dataset) - n_val
        train_base, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.seed),
        )
        train_dataset = _AugmentedWrapper(train_base, full_dataset, config.data)

    # ── DataLoaders ──
    pin_memory = config.train.pin_memory and not torch.backends.mps.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size,
        shuffle=True, num_workers=config.train.num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.train.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    return train_loader, val_loader


# ── Internal helpers ──

class _AugmentedWrapper(Dataset):
    """Wraps a random_split subset with augmentation enabled."""

    def __init__(self, dataset, base_dataset, config: "DataConfig"):
        self.dataset = dataset
        self.base = base_dataset
        self.image_size = base_dataset.image_size
        self.num_classes = base_dataset.num_classes
        self.transform = base_dataset._build_transforms(config) if config.augmentation else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        indices = self.dataset.indices
        original_idx = indices[idx]
        image_name = self.base.images[original_idx]
        stem = Path(image_name).stem

        image_path = self.base.image_dir / image_name
        image = _load_image_robust(str(image_path))
        if image is None:
            h, w = self.image_size
            return {"image": torch.zeros((3, h, w), dtype=torch.float32),
                    "mask": torch.zeros((h, w), dtype=torch.long),
                    "name": Path(image_path).name}

        mask = self.base._load_mask(original_idx)
        if self.base.class_mapping is not None:
            remapped = np.zeros_like(mask, dtype=np.uint8)
            for raw_val, target_val in self.base.class_mapping.items():
                remapped[mask == raw_val] = target_val
            mask = remapped

        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        return {"image": image, "mask": mask, "name": image_name}
