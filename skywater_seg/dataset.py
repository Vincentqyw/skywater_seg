"""
PyTorch Dataset for sky/water segmentation.

Supports:
  - Image + mask pairs (PNG masks with class values 0, 1, 2)
  - Custom data splits (train/val .txt files)
  - Albumentations-based augmentation pipeline
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from skywater_seg.config import DataConfig


def _load_image_robust(path: str) -> np.ndarray:
    """Load image as RGB numpy array using PIL only.

    OpenCV has JPEG decoding issues on Windows; PIL is more reliable.
    Returns None on failure.
    """
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _load_mask_robust(path: str) -> np.ndarray:
    """Load mask as grayscale numpy array using PIL only."""
    try:
        from PIL import Image
        return np.array(Image.open(path), dtype=np.uint8)
    except Exception:
        return None


class SkyWaterDataset(Dataset):
    """Dataset for sky/water/person segmentation.

    Expects:
      - image_dir: directory with RGB images
      - mask_dir: directory with single-channel PNG masks
        (0=background, 1=sky, 2=water, 3=person... or custom via class_mapping)

    Mask files are matched by: {image_stem}_mask.png, or {image_stem}.png
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 3,
        augmentation: bool = False,
        config: Optional[DataConfig] = None,
        file_list: Optional[List[str]] = None,
        class_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            image_dir: Path to image directory
            mask_dir: Path to mask directory
            image_size: Target (height, width)
            num_classes: Number of classes (including background)
            augmentation: Whether to apply augmentations
            config: Full DataConfig for augmentation parameters
            file_list: Optional list of image filenames (basenames with extension)
            class_mapping: Optional dict mapping raw mask values → target class indices.
                e.g. {3: 1, 13: 3, 22: 2} for ADE20K sky/person/water.
                Pixels not in the mapping become 0 (background).
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.config = config
        self.class_mapping = class_mapping

        # Gather image files
        if file_list is not None:
            self.images = file_list
        else:
            extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
            self.images = sorted([
                f.name for f in self.image_dir.glob("*")
                if f.suffix.lower() in extensions
            ])

        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Set up augmentations
        if augmentation and config is not None:
            import albumentations as A
            self.transform = self._build_transforms(config)
        else:
            self.transform = None

    def _build_transforms(self, cfg: DataConfig):
        import albumentations as A

        h, w = self.image_size

        transforms = []

        # Resize to target size first
        transforms.append(A.Resize(height=h, width=w))

        # Geometric
        if cfg.horizontal_flip > 0:
            transforms.append(A.HorizontalFlip(p=cfg.horizontal_flip))
        if cfg.vertical_flip > 0:
            transforms.append(A.VerticalFlip(p=cfg.vertical_flip))
        if cfg.rotation > 0:
            transforms.append(
                A.Rotate(limit=cfg.rotation, border_mode=cv2.BORDER_CONSTANT, p=0.5)
            )

        # Color
        transforms.append(
            A.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue,
                p=0.5,
            )
        )

        # Random scale + crop
        if cfg.random_crop:
            transforms.append(
                A.RandomResizedCrop(
                    size=(h, w),
                    scale=(0.5, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5,
                )
            )

        # Normalization (applied at the end)
        transforms.append(
            A.Normalize(mean=cfg.mean, std=cfg.std)
        )

        return A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_name = self.images[idx]
        stem = Path(image_name).stem

        # Load image (use PIL first — more reliable on Windows; OpenCV fallback)
        image_path = self.image_dir / image_name
        image = _load_image_robust(str(image_path))
        if image is None:
            # Corrupted image — return a blank sample instead of crashing
            h, w = self.image_size
            image = np.zeros((h, w, 3), dtype=np.float32)
            mask = np.zeros((h, w), dtype=np.uint8)
            return {
                "image": torch.from_numpy(image).permute(2, 0, 1).float(),
                "mask": torch.from_numpy(mask).long(),
                "name": image_name,
            }
        # image is already RGB

        # Load mask
        mask = self._load_mask(stem)

        # Apply class remapping (e.g. ADE20K class indices → our class indices)
        if self.class_mapping is not None:
            mask = self._remap_mask(mask)

        # Resize to target size first (always)
        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply transforms (augmentation + normalize for train, normalize-only for val)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Validation: apply only normalization (no augmentation)
            mean = self.config.mean if self.config else [0.485, 0.456, 0.406]
            std = self.config.std if self.config else [0.229, 0.224, 0.225]
            image = image.astype(np.float32) / 255.0
            image = (image - np.array(mean, dtype=np.float32)) / \
                    np.array(std, dtype=np.float32)

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()  # (H, W)

        # Ensure mask values are valid
        mask = torch.clamp(mask, 0, self.num_classes - 1)

        return {
            "image": image,
            "mask": mask,
            "name": image_name,
        }

    def _load_mask(self, stem: str) -> np.ndarray:
        """Load mask file, trying multiple naming conventions.

        Priority:
          1. {stem}_mask.png
          2. {stem}.png  (in mask_dir)
          3. {stem}_mask.jpg
        """
        candidates = [
            self.mask_dir / f"{stem}_mask.png",
            self.mask_dir / f"{stem}.png",
            self.mask_dir / f"{stem}_mask.jpg",
        ]

        for path in candidates:
            if path.exists():
                mask = _load_mask_robust(str(path))
                if mask is not None:
                    return mask

        # No mask found → return all-zeros
        h, w = self.image_size
        # Try to read image to get original size
        image_path = self.image_dir / f"{stem}.jpg"
        if not image_path.exists():
            image_path = self.image_dir / f"{stem}.png"
        if image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]

        return np.zeros((h, w), dtype=np.uint8)

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remap raw mask pixel values to target class indices.

        Uses self.class_mapping dict. Pixels not found in the mapping
        become 0 (background).

        Args:
            mask: (H, W) uint8 array with raw pixel values

        Returns:
            (H, W) uint8 array with remapped class indices
        """
        remapped = np.zeros_like(mask, dtype=np.uint8)
        for raw_val, target_val in self.class_mapping.items():
            remapped[mask == raw_val] = target_val
        return remapped


def create_dataloaders(
    config: "Config",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders from config.

    Supports three split modes (in priority order):
      1. Explicit split files (train_split / val_split paths)
      2. ADE20K-style subdirectory split (image_dir/training + image_dir/validation)
      3. Random split of a flat image directory

    Args:
        config: Full Config object

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split

    class_mapping = config.data.class_mapping

    # Determine file lists
    train_files = None
    val_files = None

    if config.data.train_split and os.path.exists(config.data.train_split):
        with open(config.data.train_split, encoding="utf-8") as f:
            train_files = [line.strip() for line in f if line.strip()]
    if config.data.val_split and os.path.exists(config.data.val_split):
        with open(config.data.val_split, encoding="utf-8") as f:
            val_files = [line.strip() for line in f if line.strip()]

    if train_files is not None and val_files is not None:
        # Mode 1: Explicit split files
        train_dataset = SkyWaterDataset(
            image_dir=config.data.image_dir,
            mask_dir=config.data.mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=config.data.augmentation,
            config=config.data,
            file_list=train_files,
            class_mapping=class_mapping,
        )
        val_dataset = SkyWaterDataset(
            image_dir=config.data.image_dir,
            mask_dir=config.data.mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=False,
            config=None,
            file_list=val_files,
            class_mapping=class_mapping,
        )
    elif (Path(config.data.image_dir) / "training").exists() and \
         (Path(config.data.image_dir) / "validation").exists():
        # Mode 2: ADE20K-style subdirectory split
        train_image_dir = str(Path(config.data.image_dir) / "training")
        val_image_dir = str(Path(config.data.image_dir) / "validation")
        train_mask_dir = str(Path(config.data.mask_dir) / "training")
        val_mask_dir = str(Path(config.data.mask_dir) / "validation")

        train_dataset = SkyWaterDataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=config.data.augmentation,
            config=config.data,
            class_mapping=class_mapping,
        )
        val_dataset = SkyWaterDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=False,
            config=None,
            class_mapping=class_mapping,
        )
    else:
        # Mode 3: Random split of flat directory
        full_dataset = SkyWaterDataset(
            image_dir=config.data.image_dir,
            mask_dir=config.data.mask_dir,
            image_size=tuple(config.data.image_size),
            num_classes=config.data.num_classes,
            augmentation=False,  # Base dataset, train gets its own
            config=None,
            file_list=None,
            class_mapping=class_mapping,
        )

        n_val = max(1, int(len(full_dataset) * config.data.val_ratio))
        n_train = len(full_dataset) - n_val

        train_base, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.seed),
        )

        # Wrap train split with augmentation
        train_dataset = _AugmentedWrapper(
            dataset=train_base,
            base_dataset=full_dataset,
            config=config.data,
        )

    # MPS does not support pin_memory
    import torch
    pin_memory = config.train.pin_memory and not torch.backends.mps.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


class _AugmentedWrapper(Dataset):
    """Wraps a subset of a dataset with augmentation enabled."""

    def __init__(self, dataset, base_dataset, config: DataConfig):
        self.dataset = dataset
        self.base = base_dataset
        self.aug = base_dataset.augmentation
        self.transform = base_dataset._build_transforms(config) if config.augmentation else None

        # Need access to files for proper loading
        self.image_size = base_dataset.image_size
        self.num_classes = base_dataset.num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the item WITHOUT augmentation from base first
        # This is a simplified wrapper; in practice, use the full dataset
        # with augmentation flag per-split

        # For proper random_split support, we need the underlying indices
        indices = self.dataset.indices
        original_idx = indices[idx]

        # Manually load with augmentation
        image_name = self.base.images[original_idx]
        stem = Path(image_name).stem

        image_path = self.base.image_dir / image_name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.base._load_mask(stem)

        # Apply class remapping if configured
        if self.base.class_mapping is not None:
            mask = self.base._remap_mask(mask)

        # Resize to target size
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
