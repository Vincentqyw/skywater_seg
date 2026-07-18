#!/usr/bin/env python3
"""
Sky/Water/Person segmentation inference CLI.

Usage:
  # From HuggingFace (auto-downloads)
  uv run python inference.py --hf -i test.jpg

  # From local checkpoint
  uv run python inference.py --checkpoint model.pth -i test.jpg

  # ONNX Runtime (no PyTorch needed)
  uv run python inference.py --onnx model.onnx -i test.jpg

  # Export ONNX
  uv run python inference.py --checkpoint model.pth --export-onnx model.onnx
"""

import argparse
from pathlib import Path

from loguru import logger

from skywater_seg.visualization import draw_overlay


def main():
    parser = argparse.ArgumentParser(description="Sky/Water Segmentation Inference")
    parser.add_argument("--hf", action="store_true",
                        help="Download model from HuggingFace (Realcat/skywater_seg)")
    parser.add_argument("--checkpoint", "-c", type=str,
                        help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--onnx", type=str,
                        help="Path to ONNX model (no PyTorch needed)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image or directory")
    parser.add_argument("--output", "-o", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, cpu, coreml")
    parser.add_argument("--export-onnx", type=str, default=None,
                        help="Export to ONNX and exit")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip visualization overlay")
    parser.add_argument("--crf", action="store_true",
                        help="Apply CRF post-processing (requires pydensecrf)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    if args.hf:
        logger.info("Loading from HuggingFace: Realcat/skywater_seg")
        from skywater_seg.inference import load_model, segment
        model = load_model(args.device)
        class _HF:
            def predict(self, path, **_):
                m = segment(path, model)
                return {"mask": m, "sky_mask": m == 1, "water_mask": m == 2}
        infer = _HF()

    elif args.onnx:
        logger.info(f"Loading ONNX model: {args.onnx}")
        from skywater_seg.inference import ONNXRuntimeInference
        infer = ONNXRuntimeInference(args.onnx, provider=args.device)

    elif args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        from skywater_seg.inference import SegmentationInference
        infer = SegmentationInference(args.checkpoint, device=args.device)

        if args.export_onnx:
            infer.export_onnx(args.export_onnx)
            logger.info("ONNX export complete. Exiting.")
            return
    else:
        logger.error("Provide --hf, --checkpoint, or --onnx")
        return

    # ---- Process images ----
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    paths = sorted([str(f) for f in input_path.glob("*") if f.suffix.lower() in exts]) if input_path.is_dir() else [str(input_path)]

    logger.info(f"Processing {len(paths)} images...")
    import cv2
    for img_path in paths:
        stem = Path(img_path).stem
        result = infer.predict(img_path, apply_crf=args.crf)
        cv2.imwrite(str(output_path / f"{stem}_mask.png"), result["mask"])
        if not args.no_overlay:
            cv2.imwrite(str(output_path / f"{stem}_vis.jpg"), draw_overlay(img_path, result["mask"]))
        logger.info(f"[OK] {stem}: sky={result['sky_mask'].sum():,}px, water={result['water_mask'].sum():,}px")

    logger.info(f"Done! {len(paths)} images -> {output_path}")


if __name__ == "__main__":
    main()
