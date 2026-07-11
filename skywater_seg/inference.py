"""
Inference and model export for sky/water segmentation.

Features:
  - Single image inference
  - Batch inference
  - ONNX export
  - ONNX Runtime inference (no PyTorch dependency)
  - TensorRT export instructions
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from skywater_seg.config import Config
from skywater_seg.model import create_model
from skywater_seg.utils import get_device


class SegmentationInference:
    """Inference wrapper for trained sky/water segmentation model."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint_path: Path to .pth checkpoint
            config: Config object (if None, uses default)
            device: Device for inference
        """
        self.device = get_device(device)

        # Create model
        if config is None:
            config = Config()
        self.config = config

        self.model = create_model(config)
        self.model.to(self.device)

        # Load weights
        if checkpoint_path.endswith(".pth"):
            state_dict = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            # Handle both full checkpoint and state_dict-only files
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

        self.model.eval()
        self.num_classes = config.model.classes
        self.image_size = tuple(config.data.image_size)

        print(f"Model loaded on {self.device}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Input size: {self.image_size}")

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray],
        return_probabilities: bool = False,
        apply_crf: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run inference on a single image.

        Args:
            image: Path to image file or RGB numpy array (H, W, 3)
            return_probabilities: If True, also return per-class probability maps
            apply_crf: If True, apply CRF post-processing (requires pydensecrf)

        Returns:
            Dict with keys:
              - "mask": (H, W) uint8 class indices
              - "sky_mask": (H, W) bool (class 1)
              - "water_mask": (H, W) bool (class 2)
              - "probs": (C, H, W) float32 probabilities (if return_probabilities)
        """
        # Load image
        if isinstance(image, str):
            image_bgr = cv2.imread(image)
            if image_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            if image_rgb.shape[-1] == 4:
                image_rgb = image_rgb[:, :, :3]

        orig_h, orig_w = image_rgb.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(image_rgb)
        input_tensor = input_tensor.to(self.device)

        # Inference
        logits = self.model(input_tensor)  # (1, C, H, W)

        # Resize back to original size
        logits = F.interpolate(
            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

        probs = F.softmax(logits, dim=1)  # (1, C, H, W)
        mask = torch.argmax(probs, dim=1)[0]  # (H, W)

        # Convert to numpy
        mask_np = mask.cpu().numpy().astype(np.uint8)
        probs_np = probs[0].cpu().numpy().astype(np.float32)  # (C, H, W)

        # CRF post-processing (optional)
        if apply_crf:
            mask_np = self._apply_crf(image_rgb, probs_np)

        result = {
            "mask": mask_np,
            "sky_mask": (mask_np == 1),
            "water_mask": (mask_np == 2),
        }

        if return_probabilities:
            result["probs"] = probs_np

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8,
    ) -> List[Dict[str, np.ndarray]]:
        """Run inference on a batch of images.

        Args:
            images: List of image paths or numpy arrays
            batch_size: Batch size for inference

        Returns:
            List of result dicts (same format as predict())
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Load and preprocess
            batch_tensors = []
            orig_sizes = []
            for img in batch_images:
                if isinstance(img, str):
                    img_bgr = cv2.imread(img)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img[:, :, :3]
                orig_sizes.append(img_rgb.shape[:2])
                batch_tensors.append(self._preprocess(img_rgb))

            input_batch = torch.cat(batch_tensors, dim=0).to(self.device)

            # Forward
            logits = self.model(input_batch)

            # Process each image
            for j, (h, w) in enumerate(orig_sizes):
                logits_j = F.interpolate(
                    logits[j:j+1], size=(h, w), mode="bilinear", align_corners=False
                )
                probs = F.softmax(logits_j, dim=1)
                mask = torch.argmax(probs, dim=1)[0]

                mask_np = mask.cpu().numpy().astype(np.uint8)
                results.append({
                    "mask": mask_np,
                    "sky_mask": (mask_np == 1),
                    "water_mask": (mask_np == 2),
                })

        return results

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input:
        Resize, normalize (ImageNet stats), convert to tensor.
        """
        h, w = self.image_size

        # Resize
        image = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # To tensor (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor

    @staticmethod
    def _apply_crf(
        image_rgb: np.ndarray,
        probs: np.ndarray,
        n_iterations: int = 5,
    ) -> np.ndarray:
        """Apply Dense CRF post-processing to refine mask boundaries.

        Requires: pip install pydensecrf
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            print("Warning: pydensecrf not installed. Skipping CRF.")
            return np.argmax(probs, axis=0).astype(np.uint8)

        h, w = image_rgb.shape[:2]

        # Ensure contiguous
        probs_c = np.ascontiguousarray(probs)
        unary = unary_from_softmax(probs_c)

        d = dcrf.DenseCRF2D(w, h, probs.shape[0])
        d.setUnaryEnergy(unary)

        # Pairwise potentials
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=40, srgb=13, rgbim=image_rgb, compat=10)

        Q = d.inference(n_iterations)
        result = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)
        return result

    # ================================================================
    # ONNX Export
    # ================================================================

    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 17,
        dynamic_batch: bool = True,
        simplify: bool = True,
    ) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Path to save .onnx file
            opset_version: ONNX opset version
            dynamic_batch: Support variable batch size
            simplify: Run onnx-simplifier to optimize graph

        Returns:
            Path to exported ONNX model
        """
        print(f"Exporting to ONNX: {output_path}")

        h, w = self.image_size
        self.model.eval()
        self.model.to("cpu")  # ONNX export on CPU

        dummy_input = torch.randn(1, 3, h, w)

        # Dynamic axes
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        } if dynamic_batch else {}

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        # Simplify
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                onnx_model = onnx.load(output_path)
                model_simp, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(model_simp, output_path)
                    print("  ONNX model simplified ✓")
                else:
                    print("  ⚠️ ONNX simplification check failed, using original")
            except ImportError:
                print("  ℹ️ onnx-simplifier not installed, skipping simplification")
                print("    Install with: pip install onnx-simplifier")

        # Verify
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✅ ONNX model verified: {output_path}")

        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"  📦 Model size: {size_mb:.1f} MB")

        # Print TensorRT conversion hint
        print(f"\n  💡 To convert to TensorRT engine:")
        print(f"     trtexec --onnx={output_path} --fp16 \\")
        print(f"         --saveEngine={Path(output_path).stem}.trt")

        return output_path

    def export_torchscript(self, output_path: str) -> str:
        """Export model to TorchScript format."""
        h, w = self.image_size
        self.model.eval()
        self.model.to("cpu")

        dummy_input = torch.randn(1, 3, h, w)

        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(output_path)

        size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"TorchScript model saved: {output_path} ({size_mb:.1f} MB)")
        return output_path


class ONNXRuntimeInference:
    """Lightweight inference using ONNX Runtime (no PyTorch dependency).

    Usage:
        infer = ONNXRuntimeInference("model.onnx")
        result = infer.predict("image.jpg")
    """

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        # Input shape: (batch, 3, height, width) or (batch, 3, "height", "width")
        self.input_h = input_shape[2] if isinstance(input_shape[2], int) else 512
        self.input_w = input_shape[3] if isinstance(input_shape[3], int) else 512

        print(f"ONNX Runtime session ready")
        print(f"  Input: {self.input_name} {input_shape}")
        print(f"  Providers: {self.session.get_providers()}")

    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference on a single image."""
        # Load image
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image[:, :, :3]
        orig_h, orig_w = img_rgb.shape[:2]

        # Preprocess
        img = cv2.resize(
            img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        # Inference
        logits = self.session.run([self.output_name], {self.input_name: img})[0]

        # Resize to original
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = F.interpolate(
            logits_tensor, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )
        mask = torch.argmax(logits_tensor, dim=1)[0].numpy().astype(np.uint8)

        return {
            "mask": mask,
            "sky_mask": (mask == 1),
            "water_mask": (mask == 2),
        }


# ============================================================
# CLI inference function
# ============================================================

def run_inference_cli(
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
    save_overlay: bool = True,
    export_onnx: Optional[str] = None,
):
    """CLI entry point for inference."""
    config = Config.from_yaml(config_path) if config_path else Config()

    infer = SegmentationInference(checkpoint_path, config, device)

    # Optional ONNX export
    if export_onnx:
        infer.export_onnx(export_onnx)

    # Process images
    input_path = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        images = [str(input_path)]
    else:
        extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        images = sorted([
            str(f) for f in input_path.glob("*") if f.suffix.lower() in extensions
        ])

    print(f"Processing {len(images)} images...")

    for img_path in images:
        result = infer.predict(img_path)

        stem = Path(img_path).stem

        # Save mask
        mask_path = output_path / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), result["mask"])

        # Save overlay
        if save_overlay:
            vis = draw_overlay(img_path, result["mask"])
            vis_path = output_path / f"{stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis)

    print(f"Done! Results saved to {output_dir}")


def draw_overlay(
    image: Union[str, np.ndarray],
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw segmentation mask overlay on image.

    Returns BGR image.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = mask.shape
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))

    overlay = np.zeros_like(img)
    overlay[mask == 1] = [255, 140, 0]   # Sky: orange
    overlay[mask == 2] = [0, 200, 255]   # Water: cyan

    vis = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return vis
