#!/usr/bin/env python3
"""
Extract sky+water segmentation masks from ADE20K 2021 dataset.

ADE20K 2021 format:
  - Each image has a .json file listing objects with `name`, `name_ndx`, `instance_mask`
  - Instance masks are grayscale PNGs (255=foreground, 0=background)
  - The JSON tells us which instance is sky, which is water, etc.

Output format (compatible with skywater_seg training pipeline):
  - data/images/{name}.jpg      — RGB image
  - data/masks/{name}_mask.png  — single-channel PNG (0=bg, 1=sky, 2=water)

Usage:
  uv run python scripts/extract_ade20k.py \
      --ade-root /Users/realcat/datasets/ADE20K_2021_17_01 \
      --out-dir data/ade20k_skywater \
      --splits training validation
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# Water-related class names to match in ADE20K JSON
# ============================================================
# ADE20K uses WordNet names; we match against these patterns.
# The JSON `name` field contains the WordNet concept name.

WATER_NAMES: Set[str] = {
    "water", "sea", "river", "lake", "pond", "ocean",
    "waterfall, falls", "stream, watercourse", "canal",
    "reservoir", "bayou", "lagoon", "harbor, harbour",
    "inlet", "cove", "fjord", "loch", "marsh", "swamp",
    "wetland", "brook", "creek", "estuary", "fountain",
    "pool", "puddle", "basin", "dam", "ditch",
    "water surf", "whitewater", "waterway",
    "fish farm water", "water ditch",
    "pond water", "cistern, water tank",
}

SKY_NAMES: Set[str] = {
    "sky",
}


def is_water_name(name: str) -> bool:
    """Check if an ADE20K object name is water-related."""
    name_lower = name.lower().strip()
    for wn in WATER_NAMES:
        if wn in name_lower or name_lower in wn:
            return True
    # Also match compound names like "sea water", "river water"
    if any(w in name_lower for w in ["water", "sea", "river", "lake", "pond",
                                       "ocean", "stream", "canal"]):
        return True
    return False


def is_sky_name(name: str) -> bool:
    """Check if an ADE20K object name is sky."""
    name_lower = name.lower().strip()
    return "sky" in name_lower


def _mask_to_color_vis(mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask (0=bg, 1=sky, 2=water) to BGR color image for human viewing."""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[mask == 1] = (0, 140, 255)    # Sky: orange
    vis[mask == 2] = (255, 200, 0)    # Water: cyan
    return vis


def read_instance_mask(mask_path: str, threshold: int = 128) -> np.ndarray:
    """Read a binary instance mask from ADE20K.

    Instance masks have values [0, 128, 255] where:
      0   = background
      128 = boundary pixels (included as foreground)
      255 = foreground

    Returns boolean array (True = foreground).
    """
    if not os.path.exists(mask_path):
        return None
    mask = np.array(Image.open(mask_path))
    return mask >= threshold


def extract_from_json(
    json_path: str,
    image_dir: str,
) -> Tuple[Optional[str], Optional[np.ndarray], Dict]:
    """Extract sky+water mask from one ADE20K JSON annotation.

    Args:
        json_path: Path to the .json annotation file
        image_dir: Base directory for resolving relative mask paths

    Returns:
        (image_path, combined_mask, stats) or (None, None, {}) on failure
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ann = data["annotation"]
    image_file = ann["filename"]  # e.g., "ADE_frame_00000041.jpg"
    json_dir = os.path.dirname(json_path)

    # Full path to the image
    image_path = os.path.join(json_dir, image_file)
    if not os.path.exists(image_path):
        return None, None, {"error": f"Image not found: {image_path}"}

    # Read image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        return None, None, {"error": f"Cannot read image: {image_path}"}
    h, w = img.shape[:2]

    # Find sky and water objects
    sky_instances = []
    water_instances = []

    for obj in ann.get("object", []):
        name = obj.get("name", "")
        inst_rel_path = obj.get("instance_mask", "")

        if not inst_rel_path:
            continue

        inst_full_path = os.path.join(json_dir, inst_rel_path)

        if is_sky_name(name):
            sky_instances.append((name, inst_full_path))
        elif is_water_name(name):
            water_instances.append((name, inst_full_path))

    # Build combined mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    sky_count = 0
    water_count = 0

    for name, inst_path in sky_instances:
        inst_mask = read_instance_mask(inst_path)
        if inst_mask is not None:
            combined_mask[inst_mask] = 1
            sky_count += 1

    for name, inst_path in water_instances:
        inst_mask = read_instance_mask(inst_path)
        if inst_mask is not None:
            combined_mask[inst_mask] = 2
            water_count += 1

    stats = {
        "image": image_file,
        "sky_instances": sky_count,
        "water_instances": water_count,
        "sky_names": [n for n, _ in sky_instances],
        "water_names": [n for n, _ in water_instances],
        "sky_pixels": int(np.sum(combined_mask == 1)),
        "water_pixels": int(np.sum(combined_mask == 2)),
    }

    return image_path, combined_mask, stats


def process_split(
    ade_root: str,
    split: str,
    out_image_dir: str,
    out_mask_dir: str,
    min_coverage: float = 0.001,  # at least 0.1% of image must be sky/water
    max_images: int = 0,
) -> List[Dict]:
    """Process one split (training/validation) of ADE20K.

    Args:
        ade_root: Path to ADE20K_2021_17_01 root
        split: "training" or "validation"
        out_image_dir: Directory to save images
        out_mask_dir: Directory to save masks
        min_coverage: Minimum fraction of image that must be sky/water
        max_images: Maximum images to process (0 = all)

    Returns:
        List of stats dicts for each processed image
    """
    images_root = os.path.join(ade_root, "images", "ADE", split)
    out_img_dir = Path(out_image_dir)
    out_msk_dir = Path(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files recursively
    json_files = sorted(Path(images_root).rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {split}")

    results = []
    skipped_no_sky_water = 0
    skipped_low_coverage = 0
    errors = 0
    processed = 0

    pbar = tqdm(json_files, desc=f"Processing {split}", unit="img")

    for json_path in pbar:
        if max_images > 0 and processed >= max_images:
            break

        try:
            image_path, mask, stats = extract_from_json(
                str(json_path), images_root
            )
        except Exception as e:
            errors += 1
            pbar.set_postfix({"err": str(e)[:30]})
            continue

        if image_path is None or mask is None:
            errors += 1
            continue

        # Check if there's anything useful
        total_sky_water = stats["sky_pixels"] + stats["water_pixels"]
        h, w = mask.shape

        if total_sky_water == 0:
            skipped_no_sky_water += 1
            continue

        coverage = total_sky_water / (h * w)
        if coverage < min_coverage:
            skipped_low_coverage += 1
            continue

        # Generate output filename (preserve original name + scene path)
        rel_path = os.path.relpath(str(json_path.parent), images_root)
        safe_name = rel_path.replace("/", "_") + "_" + Path(stats["image"]).stem
        safe_name = safe_name.replace(" ", "_")

        # Copy image
        out_img_path = out_img_dir / f"{safe_name}.jpg"
        if not out_img_path.exists():
            img = cv2.imread(image_path)
            if img is not None:
                cv2.imwrite(str(out_img_path), img)

        # Save mask
        out_mask_path = out_msk_dir / f"{safe_name}_mask.png"
        cv2.imwrite(str(out_mask_path), mask)

        # Also save a colorized visualization (values 0,1,2 are invisible in grayscale)
        vis = _mask_to_color_vis(mask)
        vis_path = out_msk_dir / f"{safe_name}_vis.png"
        cv2.imwrite(str(vis_path), vis)

        stats["output_name"] = safe_name
        stats["coverage"] = round(coverage, 4)
        results.append(stats)
        processed += 1

        pbar.set_postfix({
            "ok": processed,
            "no_sw": skipped_no_sky_water,
            "low": skipped_low_coverage,
        })

    print(f"\n{split} summary:")
    print(f"  JSON files scanned:  {len(json_files)}")
    print(f"  Images extracted:    {processed}")
    print(f"  Skipped (no sky/water): {skipped_no_sky_water}")
    print(f"  Skipped (low coverage): {skipped_low_coverage}")
    print(f"  Errors:              {errors}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract sky+water masks from ADE20K 2021 dataset"
    )
    parser.add_argument(
        "--ade-root", type=str,
        default="/Users/realcat/datasets/ADE20K_2021_17_01",
        help="Path to ADE20K_2021_17_01 root directory",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/ade20k_skywater",
        help="Output directory for extracted dataset",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["training", "validation"],
        help="Which ADE20K splits to process",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.001,
        help="Minimum fraction of image that must be sky/water (0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Maximum images per split (0 = all)",
    )
    parser.add_argument(
        "--train-split-ratio", type=float, default=0.85,
        help="Fraction of extracted images to use for training",
    )
    parser.add_argument(
        "--save-file-lists", action="store_true", default=True,
        help="Generate train.txt and val.txt split files",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    all_results = {}

    for split in args.splits:
        img_dir = out_dir / "images"
        msk_dir = out_dir / "masks"

        results = process_split(
            ade_root=args.ade_root,
            split=split,
            out_image_dir=str(img_dir),
            out_mask_dir=str(msk_dir),
            min_coverage=args.min_coverage,
            max_images=args.max_images,
        )
        all_results[split] = results

    # ---- Generate train.txt / val.txt ----
    if args.save_file_lists:
        # Combine all splits and shuffle
        all_images = []
        for split, results in all_results.items():
            for r in results:
                all_images.append((r["output_name"], split))

        # Use training split images for train, validation for val
        train_images = []
        val_images = []

        for name, split in all_images:
            if split == "training":
                train_images.append(name)
            else:
                val_images.append(name)

        # Write train.txt
        train_path = out_dir / "train.txt"
        with open(train_path, "w") as f:
            for name in sorted(train_images):
                f.write(f"{name}.jpg\n")
        print(f"\ntrain.txt: {len(train_images)} images -> {train_path}")

        # Write val.txt
        val_path = out_dir / "val.txt"
        with open(val_path, "w") as f:
            for name in sorted(val_images):
                f.write(f"{name}.jpg\n")
        print(f"val.txt:   {len(val_images)} images -> {val_path}")

    # ---- Summary ----
    total = sum(len(r) for r in all_results.values())
    print(f"\n{'='*60}")
    print(f"✅ ADE20K extraction complete!")
    print(f"   Total images extracted: {total}")
    print(f"   Output directory: {out_dir}")
    print(f"   Images: {out_dir / 'images'}")
    print(f"   Masks:  {out_dir / 'masks'}")
    print(f"\n   Ready for training:")
    print(f"   uv run python train.py --config configs/default.yaml \\")
    print(f"       --data.image_dir {out_dir / 'images'} \\")
    print(f"       --data.mask_dir {out_dir / 'masks'} \\")
    print(f"       --data.train_split {out_dir / 'train.txt'} \\")
    print(f"       --data.val_split {out_dir / 'val.txt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
