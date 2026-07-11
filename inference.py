#!/usr/bin/env python3
"""
Convenience inference script for sky/water segmentation.

Usage:
  # Single image
  python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth --input test.jpg

  # Directory of images
  python inference.py --checkpoint model.pth --input data/images/ --output results/

  # With ONNX export
  python inference.py --checkpoint model.pth --export-onnx skywater_seg.onnx

  # Using ONNX model directly (no PyTorch needed)
  python inference.py --onnx skywater_seg.onnx --input test.jpg
"""

import argparse
from pathlib import Path

from skywater_seg.inference import (
    ONNXRuntimeInference,
    SegmentationInference,
    draw_overlay,
    run_inference_cli,
)


def main():
    parser = argparse.ArgumentParser(description="Sky/Water Segmentation Inference")
    parser.add_argument("--checkpoint", "-c", type=str,
                        help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--onnx", type=str,
                        help="Path to ONNX model (use ONNX Runtime, no PyTorch)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image or directory")
    parser.add_argument("--output", "-o", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config YAML (only needed for .pth checkpoint)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--export-onnx", type=str, default=None,
                        help="Export to ONNX and exit")
    parser.add_argument("--export-coreml", type=str, default=None,
                        help="Export ONNX→CoreML .mlpackage and exit (macOS only)")
    parser.add_argument("--export-torchscript", type=str, default=None,
                        help="Export to TorchScript .pt and exit")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip saving visualization overlay")
    parser.add_argument("--crf", action="store_true",
                        help="Apply CRF post-processing (requires pydensecrf)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    if args.onnx:
        print(f"Loading ONNX model: {args.onnx}")
        infer = ONNXRuntimeInference(args.onnx)
    elif args.checkpoint:
        from skywater_seg.config import Config
        config = Config.from_yaml(args.config) if Path(args.config).exists() else Config()

        print(f"Loading checkpoint: {args.checkpoint}")
        infer = SegmentationInference(args.checkpoint, config, device=args.device)

        # Export ONNX if requested
        if args.export_onnx:
            infer.export_onnx(args.export_onnx)
            print("ONNX export complete. Exiting.")
            return

        # Export CoreML if requested (macOS only)
        if args.export_coreml:
            from skywater_seg.coreml_export import export_coreml
            onnx_path = args.export_onnx or str(Path(args.checkpoint).with_suffix(".onnx"))
            if not Path(onnx_path).exists():
                infer.export_onnx(onnx_path)
            export_coreml(onnx_path, args.export_coreml)
            print("CoreML export complete. Exiting.")
            return

        # Export TorchScript if requested
        if args.export_torchscript:
            infer.export_torchscript(args.export_torchscript)
            print("TorchScript export complete. Exiting.")
            return
    else:
        print("Error: Provide --checkpoint or --onnx")
        return

    # ---- Process images ----
    extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    if input_path.is_file():
        image_paths = [str(input_path)]
    else:
        image_paths = sorted([
            str(f) for f in input_path.glob("*")
            if f.suffix.lower() in extensions
        ])

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        stem = Path(img_path).stem

        result = infer.predict(
            img_path,
            apply_crf=args.crf,
            return_probabilities=False,
        )

        # Save mask
        mask_path = output_path / f"{stem}_mask.png"
        import cv2
        cv2.imwrite(str(mask_path), result["mask"])

        # Save overlay
        if not args.no_overlay:
            vis = draw_overlay(img_path, result["mask"])
            vis_path = output_path / f"{stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis)

        print(f"  ✓ {stem} → sky={result['sky_mask'].sum():,}px, water={result['water_mask'].sum():,}px")

    print(f"\nDone! {len(image_paths)} images → {output_path}")


if __name__ == "__main__":
    main()
