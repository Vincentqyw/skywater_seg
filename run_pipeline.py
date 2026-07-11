#!/usr/bin/env python3
"""
End-to-End Sky-Water Segmentation Pipeline (MacBook Optimized)
================================================================
Runs the complete workflow:
  Phase 1: Auto-annotate with Grounding DINO + SAM (MPS-accelerated)
  Phase 2: Train lightweight DeepLabV3+ MobileNetV3 model (MPS-accelerated)
  Phase 3: Export to ONNX + CoreML for super-fast inference on MacBook

Usage:
  # Full pipeline
  python run_pipeline.py --image-dir data/images

  # Annotation only
  python run_pipeline.py --image-dir data/images --annotate-only

  # Training only (requires existing masks)
  python run_pipeline.py --image-dir data/images --train-only

  # Skip to export from existing checkpoint
  python run_pipeline.py --export-only --checkpoint checkpoints/skywater-seg/best_model.pth

Environment (uv):
  uv sync                    # Install all deps
  uv run python run_pipeline.py --image-dir data/images
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(name: str, cmd: list, env=None):
    """Run a pipeline step with pretty printing."""
    print(f"\n{'='*60}")
    print(f"▶ {name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ Step failed after {elapsed:.0f}s: {name}")
        sys.exit(1)

    print(f"\n✅ {name} — completed in {elapsed:.0f}s")
    return result


def detect_mac():
    """Check if running on Apple Silicon Mac."""
    import platform
    is_mac = platform.system() == "Darwin"
    is_apple_silicon = is_mac and (
        "arm" in platform.machine().lower()
        or "Apple" in platform.processor()
    )
    return is_mac, is_apple_silicon


def main():
    is_mac, is_apple_silicon = detect_mac()

    parser = argparse.ArgumentParser(
        description="🖼️  Sky-Water Segmentation — End-to-End Pipeline"
    )

    # Data
    parser.add_argument("--image-dir", type=str, default="data/images",
                        help="Directory with input images")
    parser.add_argument("--mask-dir", type=str, default="data/masks",
                        help="Output for auto-generated masks")
    parser.add_argument("--output-dir", type=str, default="checkpoints/skywater-seg",
                        help="Output for training checkpoints")

    # Phase control
    parser.add_argument("--annotate-only", action="store_true",
                        help="Only run auto-annotation (Phase 1)")
    parser.add_argument("--train-only", action="store_true",
                        help="Only run training (Phase 2)")
    parser.add_argument("--export-only", action="store_true",
                        help="Only run export (Phase 3)")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to existing checkpoint (skips training if provided for export)")

    # Model selection (MacBook-optimized defaults)
    parser.add_argument("--gdino-model", type=str,
                        default="tiny" if is_apple_silicon else "base",
                        choices=["tiny", "base"],
                        help="Grounding DINO model")
    parser.add_argument("--sam-model", type=str,
                        default="mobile" if is_apple_silicon else "vit_h",
                        choices=["vit_h", "vit_l", "vit_b", "mobile", "efficient"],
                        help="SAM model (default: mobile on Mac for max speed)")
    parser.add_argument("--fast", action="store_true", default=is_apple_silicon,
                        help="Enable all speed optimizations: fp16 + fast sam + 768px")

    # Precision
    parser.add_argument("--precision", type=str,
                        default="fp16" if is_apple_silicon else "fp32",
                        choices=["fp32", "fp16"],
                        help="FP16 gives ~2x speedup (default: fp16 on Mac)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int,
                        default=8 if is_apple_silicon else 16,
                        help="Batch size (default: 8 on Mac, 16 on CUDA)")

    # Export
    parser.add_argument("--export-coreml", action="store_true", default=is_apple_silicon,
                        help="Export to CoreML for Apple Neural Engine (macOS only)")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="Export to ONNX")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("🌊 Sky-Water Segmentation Pipeline")
    print(f"   Platform: {'🍎 Apple Silicon' if is_apple_silicon else '💻 ' + ('macOS' if is_mac else 'Linux/Win')}")
    if is_apple_silicon:
        print(f"   Acceleration: MPS (GPU) + ANE (CoreML export)")
    print(f"   Image dir:  {args.image_dir}")
    print(f"{'='*60}")

    project_root = Path(__file__).parent
    env = dict(**__import__("os").environ)

    # ---- Phase 1: Auto-Annotation ----
    if not args.train_only and not args.export_only:
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "auto_annotate.py"),
            "-i", args.image_dir,
            "-o", args.mask_dir,
            "--gdino-model", args.gdino_model,
            "--sam-model", args.sam_model,
            "--precision", args.precision,
            "--image-size", "768" if is_apple_silicon else "1024",
        ]
        if args.fast:
            cmd.append("--fast")
        run_step("Phase 1: Auto-Annotation (Grounding DINO + SAM)", cmd)

        if args.annotate_only:
            print("\n✅ Annotation complete. Masks saved to:", args.mask_dir)
            print("   Review masks, then run:")
            print(f"   uv run python run_pipeline.py --image-dir {args.image_dir} --train-only")
            return

    # ---- Phase 2: Training ----
    checkpoint_path = args.checkpoint

    if not args.annotate_only and not args.export_only:
        if not checkpoint_path:
            run_step(
                "Phase 2: Training (DeepLabV3+ MobileNetV3)",
                [
                    sys.executable,
                    str(project_root / "train.py"),
                    "--config", str(project_root / "configs" / "default.yaml"),
                    "--data.image_dir", args.image_dir,
                    "--data.mask_dir", args.mask_dir,
                    "--train.epochs", str(args.epochs),
                    "--train.batch_size", str(args.batch_size),
                    "--output_dir", str(Path(args.output_dir).parent),
                ],
            )
            checkpoint_path = str(Path(args.output_dir) / "best_model.pth")
        else:
            print(f"\n⏭️  Using existing checkpoint: {checkpoint_path}")

        if args.train_only:
            print(f"\n✅ Training complete. Model: {checkpoint_path}")
            return

    # ---- Phase 3: Export ----
    if checkpoint_path and Path(checkpoint_path).exists():
        onnx_path = str(Path(args.output_dir) / "skywater_seg.onnx")

        # ONNX export
        if args.export_onnx:
            run_step(
                "Phase 3a: ONNX Export",
                [
                    sys.executable,
                    str(project_root / "inference.py"),
                    "--checkpoint", checkpoint_path,
                    "--export-onnx", onnx_path,
                ],
            )

        # CoreML export (macOS only, super fast inference)
        if args.export_coreml and is_mac:
            coreml_path = str(Path(args.output_dir) / "skywater_seg.mlpackage")
            run_step(
                "Phase 3b: CoreML Export (Apple Neural Engine)",
                [
                    sys.executable, "-c",
                    f"""
import sys; sys.path.insert(0, "{project_root}")
from skywater_seg.coreml_export import export_coreml
export_coreml("{onnx_path}", "{coreml_path}", compute_units="all")
""",
                ],
            )
            print(f"\n🚀 CoreML model for Apple Neural Engine: {coreml_path}")
    else:
        if args.export_only and not checkpoint_path:
            print("⚠️  No checkpoint provided for export. Use --checkpoint PATH")

    # ---- Done ----
    print(f"\n{'='*60}")
    print("🎉 Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  Masks:      {args.mask_dir}")
    print(f"  Model:      {checkpoint_path}")
    if args.export_onnx:
        print(f"  ONNX:       {Path(args.output_dir) / 'skywater_seg.onnx'}")
    if args.export_coreml and is_mac:
        print(f"  CoreML:     {Path(args.output_dir) / 'skywater_seg.mlpackage'}")
    print()
    print("📋 Quick Inference:")
    print(f"  uv run python inference.py --checkpoint {checkpoint_path} -i <image>")
    if args.export_onnx:
        print(f"  uv run python inference.py --onnx {onnx_path} -i <image>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
