#!/usr/bin/env python3
"""
Prepare ADEChallengeData2016 for sky/water/person training.

Filters images that contain at least one target class (sky/water/person),
generates train.txt and val.txt split files.

Target classes:
  sky:    3
  water:  22, 27, 61, 105, 110, 114, 129
  person: 13

Usage:
  python scripts/prepare_ade20k_person.py
"""

import os, sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ADE_ROOT = Path("E:/datasets/ADEChallengeData2016")
OUT_DIR = Path("data/ade20k_person")

TARGET_CLASSES = {
    3,      # sky
    13,     # person
    22,     # water
    27,     # sea
    61,     # river
    105,    # fountain
    110,    # swimming pool
    114,    # waterfall
    129,    # lake
}


def filter_split(split_name: str) -> list:
    """Return list of image filenames that contain target classes."""
    msk_dir = ADE_ROOT / "annotations" / split_name
    img_dir = ADE_ROOT / "images" / split_name

    valid = []
    no_target = 0
    total = 0

    for f in tqdm(sorted(os.listdir(msk_dir)), desc=f"Scanning {split_name}"):
        if not f.endswith(".png"):
            continue
        total += 1

        mask = cv2.imread(str(msk_dir / f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        unique = set(np.unique(mask))
        if unique & TARGET_CLASSES:
            # Found at least one target class
            stem = Path(f).stem
            # Check image exists
            for ext in [".jpg", ".jpeg", ".png"]:
                if (img_dir / f"{stem}{ext}").exists():
                    valid.append(f"{stem}{ext}")
                    break
        else:
            no_target += 1

    print(f"  {split_name}: {len(valid)}/{total} images have targets "
          f"({len(valid)/total*100:.1f}%), skipped {no_target}")
    return valid


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_images = filter_split("training")
    val_images = filter_split("validation")

    # Write split files
    train_path = OUT_DIR / "train.txt"
    with open(train_path, "w") as f:
        for name in sorted(train_images):
            f.write(f"{name}\n")
    print(f"\ntrain.txt: {len(train_images)} images -> {train_path}")

    val_path = OUT_DIR / "val.txt"
    with open(val_path, "w") as f:
        for name in sorted(val_images):
            f.write(f"{name}\n")
    print(f"val.txt:   {len(val_images)} images -> {val_path}")

    print(f"\n[DONE] Ready for training!")
    print(f"  uv run python train.py --config configs/ade20k_person.yaml")


if __name__ == "__main__":
    main()
