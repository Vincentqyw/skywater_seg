"""
MacBook-optimized inference: CoreML export + Apple Neural Engine inference.

CoreML delivers the fastest inference on Apple Silicon (M1/M2/M3/M4):
  - Uses Apple Neural Engine (ANE) for ML acceleration
  - Zero-copy sharing between CPU/GPU/ANE
  - Typically 2-5x faster than ONNX Runtime on MacBook

Conversion path:
  PyTorch → ONNX → CoreML (via coremltools)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def export_coreml(
    onnx_path: str,
    output_path: str,
    compute_units: str = "all",
    minimum_deployment_target: Optional[str] = None,
) -> str:
    """Convert ONNX model to CoreML for maximum MacBook inference speed.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to output .mlpackage or .mlmodelc
        compute_units: "all" (CPU+GPU+ANE), "cpu_and_gpu", "cpu_only", "all_but_remote"
        minimum_deployment_target: e.g. "macOS13" or "iOS16"

    Returns:
        Path to the CoreML model
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools is required for CoreML export.\n"
            "Install with: uv pip install coremltools\n"
            "Or: pip install coremltools>=7.0"
        )

    print(f"Converting ONNX → CoreML...")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {output_path}")

    # Load ONNX model
    onnx_model = ct.converters.onnx.load(onnx_path)

    # Convert to CoreML
    mlmodel = ct.converters.onnx.convert(
        model=onnx_model,
        minimum_deployment_target=minimum_deployment_target
        or ct.target.macOS13,
        compute_units=getattr(ct.ComputeUnit, compute_units.upper(), ct.ComputeUnit.ALL),
    )

    # Add batch dimension info
    mlmodel.input_description["input"] = "RGB image (1, 3, H, W) normalized with ImageNet stats"
    mlmodel.output_description["output"] = "Per-class logits (1, C, H, W)"

    # Save
    if output_path.endswith(".mlpackage"):
        mlmodel.save(output_path)
    elif output_path.endswith(".mlmodelc"):
        # Save as compiled model (ready for deployment)
        pkg_path = output_path.replace(".mlmodelc", ".mlpackage")
        mlmodel.save(pkg_path)
        ct.models.MLModel(pkg_path).get_compiled_model_path()
        # Rename compiled model
        compiled_src = pkg_path.replace(".mlpackage", ".mlmodelc")
        if os.path.exists(compiled_src):
            os.rename(compiled_src, output_path)
        os.remove(pkg_path)  # Clean up intermediate
    else:
        mlmodel.save(output_path)

    size_mb = _dir_size(output_path) / (1024 ** 2)
    print(f"  ✅ CoreML model saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  🚀 Compute units: {compute_units}")
    print(f"  💡 On Apple Silicon, CoreML runs on Neural Engine for max speed")

    return output_path


def export_coreml_from_torch(
    model,
    output_path: str,
    input_shape: Tuple[int, int] = (512, 512),
    compute_units: str = "all",
) -> str:
    """Direct PyTorch → CoreML conversion (tracing).

    This skips the ONNX intermediate step and can produce
    more optimized CoreML models for simple architectures.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools is required. Install: pip install coremltools>=7.0")

    import torch

    model.eval()
    model.cpu()

    h, w = input_shape
    example_input = torch.randn(1, 3, h, w)

    # Trace the model
    traced = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input",
            shape=(1, 3, h, w),
        )],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=getattr(ct.ComputeUnit, compute_units.upper(), ct.ComputeUnit.ALL),
    )

    mlmodel.save(output_path)
    size_mb = _dir_size(output_path) / (1024 ** 2)
    print(f"✅ PyTorch → CoreML: {output_path} ({size_mb:.1f} MB)")

    return output_path


class CoreMLInference:
    """Super-fast inference on MacBook using CoreML / Apple Neural Engine.

    Usage:
        infer = CoreMLInference("skywater_seg.mlpackage")
        result = infer.predict("image.jpg")
    """

    def __init__(self, model_path: str):
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("coremltools required: pip install coremltools>=7.0")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CoreML model not found: {model_path}")

        self.model = ct.models.MLModel(model_path)
        self._warmup()

    def _warmup(self):
        """Warm up the model (first inference is always slower)."""
        import torch
        dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)
        try:
            self.model.predict({"input": dummy})
        except Exception:
            pass  # Warmup may fail with random data, ignore

    def predict(self, image: Union[str, np.ndarray]) -> dict:
        """Run inference on a single image.

        Args:
            image: Path to image or RGB numpy array (H, W, 3)

        Returns:
            dict with mask, sky_mask, water_mask
        """
        import cv2
        import torch
        import torch.nn.functional as F

        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image[:, :, :3]

        orig_h, orig_w = img.shape[:2]

        # Preprocess: resize to 512x512, normalize with ImageNet stats
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 512, 512)

        # CoreML inference
        result = self.model.predict({"input": img})
        logits = result["output"]  # (1, C, 512, 512) or (1, C, H, W)

        # Convert to torch tensor for resize
        logits_t = torch.from_numpy(logits)
        logits_t = F.interpolate(
            logits_t, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )
        mask = torch.argmax(logits_t, dim=1)[0].numpy().astype(np.uint8)

        return {
            "mask": mask,
            "sky_mask": (mask == 1),
            "water_mask": (mask == 2),
        }


def _dir_size(path: str) -> int:
    """Get total size of a file or directory."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
