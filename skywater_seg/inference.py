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

from loguru import logger

from skywater_seg.config import Config
from skywater_seg.model import create_model
from skywater_seg.utils import class_colors_bgr, configure_backend, get_device


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
        configure_backend(self.device)

        # Load checkpoint first to extract metadata
        if checkpoint_path.endswith(".pth"):
            state_dict = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

        # Auto-detect config from checkpoint metadata, fall back to provided config
        config = config if config is not None else Config()
        meta = state_dict.get("model_meta") if isinstance(state_dict, dict) else None
        if meta is not None:
            config = _config_from_meta(meta, config)
            logger.info(f"[Auto] Config extracted from checkpoint: {meta['encoder_name']}")

        self.config = config
        self.model = create_model(config)
        self.model.to(self.device)

        # Load weights
        model_weights = state_dict.get("model_state_dict", state_dict)
        self.model.load_state_dict(model_weights)

        self.model.eval()
        self.num_classes = config.model.classes
        self.image_size = tuple(config.data.image_size)

        logger.info(f"Model loaded on {self.device}")
        logger.info(f"  Classes: {self.num_classes}")
        logger.info(f"  Input size: {self.image_size}")

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
        """Resize, normalize (ImageNet stats), convert to NCHW tensor."""
        h, w = self.image_size
        image = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        image = (image.astype(np.float32) / 255.0 - ONNXRuntimeInference.MEAN) / ONNXRuntimeInference.STD
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
            logger.warning("pydensecrf not installed. Skipping CRF.")
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
        logger.info(f"Exporting to ONNX: {output_path}")

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
                    logger.info("[OK] ONNX simplified")
                else:
                    logger.warning("ONNX simplification check failed, using original")
            except ImportError:
                logger.info("onnx-simplifier not installed, skipping simplification")

        # Verify
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model verified: {output_path}")

        size_mb = os.path.getsize(output_path) / (1024 ** 2)
        logger.info(f"  Model size: {size_mb:.1f} MB")
        logger.info(f"  To convert to TensorRT: trtexec --onnx={output_path} --fp16 "
                    f"--saveEngine={Path(output_path).stem}.trt")

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
        logger.info(f"TorchScript model saved: {output_path} ({size_mb:.1f} MB)")
        return output_path


class ONNXRuntimeInference:
    """ONNX Runtime inference with explicit GPU/CPU provider control.

    Supports FP32 and FP16 ONNX models.  For FP16 models, pass
    ``provider="cuda"`` and the CUDAExecutionProvider will use FP16 math
    automatically when the model contains FP16 weights.

    Usage::

        # NVIDIA GPU (CUDA / TensorRT)
        infer = ONNXRuntimeInference("model.onnx", provider="cuda")

        # Apple Silicon (CoreML / Neural Engine)
        infer = ONNXRuntimeInference("model.onnx", provider="coreml")

        # AMD GPU
        infer = ONNXRuntimeInference("model.onnx", provider="rocm")

        # Intel CPU/GPU
        infer = ONNXRuntimeInference("model.onnx", provider="openvino")

        # Windows GPU (DirectML)
        infer = ONNXRuntimeInference("model.onnx", provider="dml")

        # CPU only
        infer = ONNXRuntimeInference("model.onnx", provider="cpu")

        # Single image
        result = infer.predict("image.jpg")         # str path or np.ndarray
        mask   = result["mask"]                     # (H, W) uint8
        sky    = result["sky_mask"]                 # (H, W) bool

        # Batch inference
        results = infer.predict_batch(["a.jpg", "b.jpg"], batch_size=4)
    """

    _PROVIDER_MAP = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "coreml": "CoreMLExecutionProvider",      # Apple Silicon / Neural Engine
        "rocm": "ROCMExecutionProvider",           # AMD GPU
        "openvino": "OpenVINOExecutionProvider",   # Intel CPU/GPU
        "dml": "DmlExecutionProvider",             # DirectML (Windows GPU)
        "acl": "ACLExecutionProvider",             # ARM Compute Library
    }

    # ImageNet normalisation (same as training)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: str,
        provider: str = "cuda",
        *,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """Create an ONNX Runtime inference session.

        Args:
            onnx_path: Path to ``.onnx`` file (FP32 or FP16).
            provider: ``"cuda"`` | ``"cpu"`` | ``"tensorrt"`` | ``"coreml"`` |
                      ``"rocm"`` | ``"openvino"`` | ``"dml"`` | ``"acl"``.
                      Falls back to CPU if the requested provider is
                      not available.
            input_size: Override the (height, width) inferred from
                        the ONNX graph (rarely needed).
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )
        ep = self._PROVIDER_MAP.get(provider, provider)
        available = ort.get_available_providers()
        if ep not in available:
            logger.warning(
                f"Provider '{ep}' not available (available: {available}). "
                f"Falling back to CPU."
            )
            ep = "CPUExecutionProvider"

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [ep] if ep == "CPUExecutionProvider" else [ep, "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
        self._providers = self.session.get_providers()

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        in_shape = self.session.get_inputs()[0].shape

        if input_size is not None:
            self.input_h, self.input_w = input_size
        else:
            self.input_h = int(in_shape[2]) if isinstance(in_shape[2], int) else 512
            self.input_w = int(in_shape[3]) if isinstance(in_shape[3], int) else 512

        logger.info("ONNX Runtime session ready")
        logger.info(f"  Input: {self.input_name} {in_shape}")
        logger.info(f"  Providers: {self._providers}")

    # -- public properties -------------------------------------------------

    @property
    def providers(self) -> List[str]:
        """Active execution providers for this session."""
        return list(self._providers)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Model input size as ``(height, width)``."""
        return (self.input_h, self.input_w)

    # -- preprocessing -----------------------------------------------------

    @staticmethod
    def preprocess(
        image_rgb: np.ndarray,
        input_size: Tuple[int, int] = (384, 384),
    ) -> np.ndarray:
        """Resize + ImageNet-normalise an RGB image to NCHW numpy array.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 RGB image.
            input_size: ``(height, width)`` to resize to.

        Returns:
            ``(1, 3, H, W)`` float32 numpy array.
        """
        h, w = input_size
        img = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - ONNXRuntimeInference.MEAN) / ONNXRuntimeInference.STD
        return img.transpose(2, 0, 1)[np.newaxis, ...]

    # -- inference ---------------------------------------------------------

    def predict(
        self,
        image: Union[str, np.ndarray],
        return_probabilities: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run inference on a single image.

        Args:
            image: Path to an image file, or an RGB/BGR numpy array
                   of shape ``(H, W, 3)``.
            return_probabilities: If True, include ``"probs"`` key with
                                  ``(C, H, W)`` float32 probability map.

        Returns:
            Dict with keys ``mask``, ``sky_mask``, ``water_mask``,
            and optionally ``probs``.
        """
        # -- load ----------------------------------------------------------
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
            if img_rgb.shape[-1] == 4:
                img_rgb = img_rgb[:, :, :3]
        orig_h, orig_w = img_rgb.shape[:2]

        # -- preprocess + forward ------------------------------------------
        np_in = self.preprocess(img_rgb, (self.input_h, self.input_w))
        logits = self.session.run([self.output_name], {self.input_name: np_in})[0]

        # -- resize to original --------------------------------------------
        logits_t = torch.from_numpy(logits)
        logits_t = F.interpolate(
            logits_t, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )
        # argmax(logits) == argmax(softmax(logits)) — skip softmax for speed
        mask = torch.argmax(logits_t, dim=1)[0].numpy().astype(np.uint8)

        result: Dict[str, np.ndarray] = {
            "mask": mask,
            "sky_mask": (mask == 1),
            "water_mask": (mask == 2),
        }
        if return_probabilities:
            probs = F.softmax(logits_t, dim=1)
            result["probs"] = probs[0].numpy().astype(np.float32)
        return result

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8,
    ) -> List[Dict[str, np.ndarray]]:
        """Run inference on a batch of images.

        Args:
            images: List of image paths or numpy arrays.
            batch_size: Number of images to process at once.

        Returns:
            List of result dicts (same format as :meth:`predict`).
        """
        results: List[Dict[str, np.ndarray]] = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]

            # Load + preprocess each image
            batch_arrs = []
            orig_sizes = []
            for img in batch_imgs:
                if isinstance(img, str):
                    img_bgr = cv2.imread(img)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img[:, :, :3]
                orig_sizes.append(img_rgb.shape[:2])
                batch_arrs.append(self.preprocess(img_rgb, (self.input_h, self.input_w)))

            np_batch = np.concatenate(batch_arrs, axis=0)  # (B, 3, H, W)
            logits = self.session.run([self.output_name], {self.input_name: np_batch})[0]

            # Process each image in the batch
            for j, (oh, ow) in enumerate(orig_sizes):
                lt = torch.from_numpy(logits[j:j+1])
                lt = F.interpolate(lt, size=(oh, ow), mode="bilinear",
                                   align_corners=False)
                mask = torch.argmax(lt, dim=1)[0].numpy().astype(np.uint8)
                results.append({
                    "mask": mask,
                    "sky_mask": (mask == 1),
                    "water_mask": (mask == 2),
                })

        return results

    def run_raw(self, np_input: np.ndarray) -> np.ndarray:
        """Low-level ONNX forward pass — returns raw logits.

        Args:
            np_input: ``(B, 3, H, W)`` preprocessed float32 array.

        Returns:
            ``(B, C, H, W)`` logits array.
        """
        return self.session.run([self.output_name], {self.input_name: np_input})[0]


# ═══════════════════════════════════════════════════════════════════════
# Standalone ONNX export helpers
# ═══════════════════════════════════════════════════════════════════════

def export_onnx(
    model: torch.nn.Module,
    image_size: Tuple[int, int],
    output_path: str,
    *,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> str:
    """Export a PyTorch segmentation model to ONNX FP32.

    This is a standalone function — it does **not** require
    :class:`SegmentationInference`.  Use it when you already have a
    model object::

        cfg = Config.from_yaml("config.yaml")
        model = create_model(cfg)
        model.load_state_dict(torch.load("ckpt.pth")["model_state_dict"])
        export_onnx(model, cfg.data.image_size, "model.onnx")

    Args:
        model: PyTorch ``nn.Module`` in eval mode.
        image_size: ``(height, width)`` input size.
        output_path: Where to write the ``.onnx`` file.
        opset_version: ONNX opset (≥ 17 recommended for SegFormer).
        dynamic_batch: If True, support dynamic batch / height / width.
        simplify: Run ``onnx-simplifier`` to fold constants and
                  remove redundant ops.

    Returns:
        ``output_path``.
    """
    logger.info(f"Exporting ONNX → {output_path}")

    h, w = image_size
    model.eval()
    model.to("cpu")

    dummy = torch.randn(1, 3, h, w)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model, dummy, output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    import onnx

    # Simplify
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            m = onnx.load(output_path)
            m_simp, ok = onnx_simplify(m)
            if ok:
                onnx.save(m_simp, output_path)
                logger.info("  ONNX simplified")
            else:
                logger.warning("  Simplification check failed; keeping original")
        except ImportError:
            logger.info("  onnx-simplifier not installed; skipping")

    # Verify
    onnx.checker.check_model(onnx.load(output_path))
    sz = os.path.getsize(output_path) / 1e6
    logger.info(f"  Model: {output_path} ({sz:.1f} MB)")
    return output_path


def convert_onnx_fp16(
    onnx_fp32_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Convert an FP32 ONNX model to FP16.

    Uses ``onnxconverter-common`` to cast weights and activations to
    float16 while keeping I/O types as float32 (so callers don't need
    to change their preprocessing).

    Args:
        onnx_fp32_path: Path to the FP32 ``.onnx`` file.
        output_path: Destination path.  Defaults to
                     ``{stem}_fp16.onnx``.

    Returns:
        Path to the FP16 ``.onnx`` file.
    """
    if output_path is None:
        p = Path(onnx_fp32_path)
        output_path = str(p.parent / f"{p.stem}_fp16.onnx")

    import onnx

    try:
        from onnxconverter_common import float16 as oc_f16
    except ImportError:
        raise ImportError(
            "onnxconverter-common is required for FP16 conversion. "
            "Install with: pip install onnxconverter-common"
        )

    logger.info(f"Converting to FP16 → {output_path}")
    m = onnx.load(onnx_fp32_path)
    m_fp16 = oc_f16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m_fp16, output_path)
    sz = os.path.getsize(output_path) / 1e6
    logger.info(f"  FP16 model: {output_path} ({sz:.1f} MB)")
    return output_path



# Mapping from meta keys → (config_section, attribute_name, transform_fn)
_META_FIELDS = [
    ("image_size",       "data",  "image_size",      tuple),
    ("num_classes",      "data",  "num_classes",     None),
    ("mean",             "data",  "mean",            None),
    ("std",              "data",  "std",             None),
    ("class_mapping",    "data",  "class_mapping",   lambda v: {int(k): x for k, x in v.items()} if v else None),
    ("model_name",       "model", "name",            None),
    ("encoder_name",     "model", "encoder_name",    None),
    ("encoder_weights",  "model", "encoder_weights", None),
    ("classes",          "model", "classes",         None),
    ("in_channels",      "model", "in_channels",     None),
]


def _config_from_meta(meta: dict, fallback: Optional[Config] = None) -> Config:
    """Build a Config from checkpoint metadata, merging with optional fallback."""
    cfg = fallback if fallback is not None else Config()
    for meta_key, section, attr, xform in _META_FIELDS:
        if meta_key in meta and meta[meta_key] is not None:
            val = meta[meta_key]
            setattr(getattr(cfg, section), attr, xform(val) if xform else val)
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Simple high-level API
# ═══════════════════════════════════════════════════════════════════════

def load_model(device: str = "cuda") -> "SkyWaterSegModel":
    """Load the SegFormer B2 model from HuggingFace.

    Args:
        device: ``"cuda"``, ``"cpu"``, or ``"mps"``.

    Returns:
        Model in eval mode on the requested device.
    """
    from skywater_seg.model import SkyWaterSegModel
    model = SkyWaterSegModel.from_pretrained("Realcat/skywater_seg")
    return model.eval().to(device)


def segment(
    image: Union[str, np.ndarray],
    model: Optional["SkyWaterSegModel"] = None,
    device: str = "cuda",
) -> np.ndarray:
    """Segment an image and return the class-index mask.

    Args:
        image: Path to image file, or RGB numpy array ``(H, W, 3)``.
        model: Pre-loaded model (if None, loads from HuggingFace).
        device: Device for inference.

    Returns:
        ``(H, W)`` uint8 mask: 0=background, 1=sky, 2=water, 3=person.
    """
    if model is None:
        model = load_model(device)
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) if isinstance(image, str) else image[:, :, :3]
    h, w = img.shape[:2]
    t = torch.from_numpy(
        (cv2.resize(img, model.image_size[::-1]).astype(np.float32) / 255.0
         - ONNXRuntimeInference.MEAN) / ONNXRuntimeInference.STD
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = F.interpolate(model(t), size=(h, w), mode="bilinear")
    return torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)