#!/usr/bin/env python3
"""
Minimal ONNX FP16 inference + visualisation example for skywater_seg.

Dependencies: onnxruntime, numpy, Pillow, huggingface_hub
(no PyTorch, no skywater_seg package needed)

Usage:
    # Auto-download FP16 model from HuggingFace Hub (default)
    uv run python demo/python/minimal_onnx_infer.py -i <image.jpg>

    # Auto-download FP32 model instead
    uv run python demo/python/minimal_onnx_infer.py -i <image.jpg> --fp32

    # Specify a local ONNX file
    uv run python demo/python/minimal_onnx_infer.py -i <image.jpg> -m path/to/model.onnx

    # GPU inference
    uv run python demo/python/minimal_onnx_infer.py -i <image.jpg> --provider cuda

    # Custom output directory
    uv run python demo/python/minimal_onnx_infer.py -i <image.jpg> -o output_dir/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HF_REPO = "Realcat/skywater_seg"
HF_FILENAME_FP16 = "skywater_segformer_b2_fp16.onnx"
HF_FILENAME_FP32 = "skywater_segformer_b2_fp32.onnx"

# ═══════════════════════════════════════════════════════════════════════
# Constants — must match training config
# ═══════════════════════════════════════════════════════════════════════

INPUT_SIZE = (384, 384)
NUM_CLASSES = 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["Background", "Sky", "Water", "Person"]
CLASS_COLORS_RGB = {
    0: (0, 0, 0),        # background: black
    1: (255, 140, 0),    # sky: orange
    2: (0, 200, 255),    # water: cyan
    3: (255, 60, 60),    # person: red
}


# ═══════════════════════════════════════════════════════════════════════
# Core pipeline
# ═══════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """Load image as RGB uint8 (H, W, 3)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Resize → ImageNet normalise → NCHW.  Returns (1, 3, H, W) float32."""
    img = np.array(
        Image.fromarray(img_rgb).resize(INPUT_SIZE[::-1], Image.BILINEAR),
        dtype=np.float32,
    )
    img /= 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis, ...]


def postprocess(logits: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Bilinear-resize logits → argmax.  Returns (H, W) uint8 mask."""
    # logits shape: (1, 4, h, w)
    mask = np.argmax(logits[0], axis=0).astype(np.uint8)  # (h, w)
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
    return np.array(mask_img, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Class-index mask → RGB colour image (H, W, 3) uint8."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_RGB.items():
        rgb[mask == cls_id] = color
    return rgb


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend colourised mask over the original RGB image."""
    h, w = mask.shape
    if image_rgb.shape[:2] != (h, w):
        image_rgb = np.array(
            Image.fromarray(image_rgb).resize((w, h), Image.BILINEAR)
        )
    color = colorize_mask(mask)
    blended = (image_rgb * (1 - alpha) + color * alpha).astype(np.uint8)
    return blended


def save_side_by_side(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    out_path: str,
) -> None:
    """Save a 1x3 strip: Input | Mask | Overlay."""
    overlay = overlay_mask(image_rgb, mask)
    color = colorize_mask(mask)

    # Ensure same height
    h = image_rgb.shape[0]
    color_resized = np.array(Image.fromarray(color).resize(
        (image_rgb.shape[1], h), Image.NEAREST
    ))
    overlay_resized = np.array(Image.fromarray(overlay).resize(
        (image_rgb.shape[1], h), Image.NEAREST
    ))

    strip = np.hstack([image_rgb, color_resized, overlay_resized])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(strip).save(out_path)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Minimal ONNX FP16 inference + visualisation"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Local path to ONNX model (overrides auto-download from HF Hub)",
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Use FP32 ONNX model instead of FP16 (only when auto-downloading)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="demo/output",
        help="Output directory (default: demo/output)",
    )
    parser.add_argument(
        "--provider", default="cpu",
        choices=["cpu", "cuda", "tensorrt", "dml", "coreml", "rocm"],
        help="ONNX Runtime execution provider (default: cpu)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HF Hub cache directory (default: uses HF_HOME / system default)",
    )
    args = parser.parse_args()

    # ── Check input ──────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"ERROR: input image not found: {args.input}")
        return 1

    # ── Resolve model path ───────────────────────────────────────────
    if args.model is not None:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"ERROR: model not found: {model_path}")
            return 1
        print(f"Using local model: {model_path}")
    else:
        filename = HF_FILENAME_FP32 if args.fp32 else HF_FILENAME_FP16
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("ERROR: huggingface_hub required for auto-download. "
                  "Install with: uv add huggingface_hub")
            return 1

        print(f"Downloading {filename} from {HF_REPO} ...")
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            cache_dir=args.cache_dir,
        )
        print(f"  Cached at: {model_path}")

    # ── Load ONNX session ────────────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: uv sync")
        return 1

    PROVIDER_MAP = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "dml": "DmlExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "rocm": "ROCMExecutionProvider",
    }
    ep = PROVIDER_MAP[args.provider]
    available = ort.get_available_providers()
    if ep not in available:
        print(f"WARNING: '{ep}' not available (available: {available}), "
              f"falling back to CPU")
        ep = "CPUExecutionProvider"

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [ep] if ep == "CPUExecutionProvider" else [ep, "CPUExecutionProvider"]

    print(f"Loading: {model_path}")
    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"  Provider: {session.get_providers()}")
    print(f"  Input:    {input_name}  {session.get_inputs()[0].shape}")
    print(f"  Output:   {output_name}  {session.get_outputs()[0].shape}")

    # ── Load & preprocess ────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    img_rgb = load_image(args.input)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f"  Original size: {orig_w}x{orig_h}")

    np_in = preprocess(img_rgb)
    print(f"  Preprocessed:  {np_in.shape}  {np_in.dtype}")

    # ── Inference ────────────────────────────────────────────────────
    print("Running inference...")
    logits = session.run([output_name], {input_name: np_in})[0]
    mask = postprocess(logits, orig_h, orig_w)
    print(f"  Logits shape:  {logits.shape}")
    print(f"  Mask shape:    {mask.shape}")

    # ── Class stats ──────────────────────────────────────────────────
    print("\nClass distribution:")
    total = mask.size
    for cls_id in range(NUM_CLASSES):
        count = int((mask == cls_id).sum())
        print(f"  {CLASS_NAMES[cls_id]:>10}: {count / total * 100:5.1f}%  "
              f"({count:>8,} px)")

    # ── Save outputs ─────────────────────────────────────────────────
    name = Path(args.input).stem
    out_dir = args.output_dir

    # 1. Raw class-index mask (uint8 PNG — each pixel is 0-3)
    mask_path = os.path.join(out_dir, f"{name}_mask.png")
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    print(f"\nSaved mask:       {mask_path}")

    # 2. Colour visualisation
    color_path = os.path.join(out_dir, f"{name}_color.png")
    Image.fromarray(colorize_mask(mask)).save(color_path)
    print(f"Saved color:      {color_path}")

    # 3. Overlay on original
    overlay_path = os.path.join(out_dir, f"{name}_overlay.png")
    Image.fromarray(overlay_mask(img_rgb, mask)).save(overlay_path)
    print(f"Saved overlay:    {overlay_path}")

    # 4. Side-by-side comparison strip
    strip_path = os.path.join(out_dir, f"{name}_comparison.png")
    save_side_by_side(img_rgb, mask, strip_path)
    print(f"Saved comparison: {strip_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
