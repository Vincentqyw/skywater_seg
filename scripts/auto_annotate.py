#!/usr/bin/env python3
"""
Auto-Annotation Script: Grounding DINO + SAM (Segment Anything Model)
===========================================================================
Automatically generates pixel-level sky/water segmentation masks using
text-prompted open-vocabulary detection (Grounding DINO) combined with
high-quality mask generation (SAM).

Pipeline:
  1. Grounding DINO detects sky/water regions from text prompts
  2. SAM refines each detection into a precise pixel mask
  3. Output: multi-class mask (0=background, 1=sky, 2=water)

Usage:
  # Annotate a directory of images
  python scripts/auto_annotate.py -i data/images -o data/masks

  # Annotate a single image
  python scripts/auto_annotate.py -i image.jpg -o masks/

  # With custom class definitions
  python scripts/auto_annotate.py -i data/images -o data/masks --classes my_classes.json

  # Using lighter models for faster processing
  python scripts/auto_annotate.py -i data/images -o data/masks \\
      --gdino-model tiny --sam-model vit_b --image-size 768

Requirements:
  pip install torch transformers segment-anything opencv-python tqdm huggingface_hub

Model downloads (automatic on first run):
  - Grounding DINO: ~700MB (base) or ~350MB (tiny) from HuggingFace
  - SAM ViT-H: ~2.4GB, ViT-L: ~1.2GB, ViT-B: ~350MB from HuggingFace

Author: SkyWater Segmentation Pipeline
License: MIT
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

# ============================================================
# Default class definitions
# ============================================================
# Format: {class_name: {prompts: [...], mask_value: int}}
# Prompts are matched by Grounding DINO independently (separated by ".")
# mask_value is written into the output mask

DEFAULT_CLASSES = {
    "sky": {
        "prompts": [
            "sky",
            "cloudy sky",
            "clear sky",
            "overcast sky",
            "sky region",
        ],
        "mask_value": 1,
        "color": [255, 180, 0],  # BGR for visualization
    },
    "water": {
        "prompts": [
            "water",
            "lake",
            "river",
            "ocean",
            "sea",
            "pond",
            "stream",
            "water surface",
            "canal",
            "reservoir",
        ],
        "mask_value": 2,
        "color": [0, 180, 255],  # BGR for visualization
    },
}

# ============================================================
# Model registry
# ============================================================

GROUNDING_DINO_MODELS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}

SAM_MODELS = {
    "vit_h": {
        "repo": "facebook/sam-vit-huge",
        "filename": "sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "repo": "facebook/sam-vit-large",
        "filename": "sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "repo": "facebook/sam-vit-base",
        "filename": "sam_vit_b_01ec64.pth",
    },
}


# ============================================================
# Grounding DINO Detector
# ============================================================


class GroundingDINODetector:
    """Open-vocabulary object detection via Grounding DINO.

    Uses HuggingFace transformers for easy model loading and inference.
    Supports FP16 precision for ~2x speedup on MPS / CUDA.
    """

    def __init__(self, model_size: str = "base", device: str = "cuda", precision: str = "fp32"):
        if torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        elif torch.backends.mps.is_available() and device == "cuda":
            self.device = "mps"
        else:
            self.device = "cpu"

        self.precision = precision
        if precision == "fp16" and self.device == "cpu":
            logger.warning("  ⚠️  FP16 not supported on CPU, falling back to FP32")
            self.precision = "fp32"

        self._use_amp = self.precision == "fp16" and self.device in ("cuda", "mps")

        if model_size not in GROUNDING_DINO_MODELS:
            raise ValueError(
                f"Unknown model size '{model_size}'. "
                f"Choose from: {list(GROUNDING_DINO_MODELS.keys())}"
            )

        model_id = GROUNDING_DINO_MODELS[model_size]

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers>=4.40.0 is required. Install with: pip install transformers>=4.40.0"
            )

        logger.info(f"  Loading Grounding DINO ({model_size}, {self.precision}) from {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        if self._use_amp:
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"  Grounding DINO loaded on {self.device} ({self.precision})")

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> List[Dict]:
        """Detect objects in image matching the text prompt.

        Args:
            image: PIL RGB image
            text_prompt: Text queries separated by ". " (e.g. "sky. water.")
            box_threshold: Minimum confidence for bounding boxes
            text_threshold: Minimum confidence for text-phrase matching

        Returns:
            List of dicts with keys: box [x1,y1,x2,y2], score, label (text phrase)
        """
        h, w = image.height, image.width

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )

        # Move to device, handling nested dicts
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }

        with torch.no_grad():
            # FP16 autocast for ~2x speedup on MPS/CUDA
            from contextlib import nullcontext

            amp_ctx = (
                torch.autocast(device_type=self.device, dtype=torch.float16)
                if self._use_amp
                else nullcontext()
            )
            with amp_ctx:
                outputs = self.model(**inputs)

        # Post-process: convert model outputs to boxes + labels
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]

        detections = []
        for box, score, label in zip(
            results.get("boxes", []),
            results.get("scores", []),
            results.get("labels", []),
        ):
            detections.append(
                {
                    "box": [float(c) for c in box],  # [x1, y1, x2, y2]
                    "score": float(score),
                    "label": str(label).strip().lower(),
                }
            )

        return detections


# ============================================================
# SAM / MobileSAM Mask Generator
# ============================================================

# Extended model registry including fast variants
SAM_MODELS_FAST = {
    **SAM_MODELS,
    # MobileSAM: ~9MB, ~60x smaller than ViT-H, ~5x faster than ViT-B
    "mobile": {
        "repo": "dhkim2810/MobileSAM",
        "filename": "mobile_sam.pt",
    },
    # EfficientSAM: ~15MB, excellent speed/quality trade-off
    "efficient": {
        "repo": "yformer/EfficientSAM",
        "filename": "efficientsam_s_gpu.jit",
    },
}


class SAMMaskGenerator:
    """High-quality mask generation via SAM / MobileSAM.

    Supports:
      - Standard SAM: vit_h, vit_l, vit_b  (highest quality)
      - MobileSAM:     mobile              (~60x smaller, ~5x faster)
      - EfficientSAM:  efficient           (~40x smaller, ~3x faster)
      - FP16 precision for ~2x additional speedup on MPS/CUDA
    """

    def __init__(self, model_type: str = "vit_h", device: str = "cuda", precision: str = "fp32"):
        if torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        elif torch.backends.mps.is_available() and device == "cuda":
            self.device = "mps"
        else:
            self.device = "cpu"

        self.precision = precision
        if precision == "fp16" and self.device == "cpu":
            logger.warning("  ⚠️  FP16 not supported on CPU, falling back to FP32")
            self.precision = "fp32"

        self._use_amp = self.precision == "fp16" and self.device in ("cuda", "mps")

        if model_type not in SAM_MODELS_FAST:
            raise ValueError(
                f"Unknown SAM model '{model_type}'. Choose from: {list(SAM_MODELS_FAST.keys())}"
            )

        self.model_type = model_type
        info = SAM_MODELS_FAST[model_type]

        # ---- MobileSAM path (fastest, ~9MB) ----
        if model_type == "mobile":
            self._load_mobile_sam(info)
            return

        # ---- EfficientSAM path (fast, ~15MB) ----
        if model_type == "efficient":
            self._load_efficient_sam(info)
            return

        # ---- Standard SAM path ----
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything is required. Install with: pip install segment-anything"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        logger.info(
            f"  Downloading SAM checkpoint ({model_type}, ~{self._size_str(model_type)})..."
        )
        checkpoint_path = hf_hub_download(
            repo_id=info["repo"],
            filename=info["filename"],
        )

        logger.info(f"  Loading SAM ({model_type}, {self.precision})...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        if self._use_amp:
            sam = sam.half()

        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        logger.info(f"  SAM loaded on {self.device} ({self.precision})")

    def _load_mobile_sam(self, info: dict):
        """Load MobileSAM — ~9MB, ~60x smaller than ViT-H, drop-in SAM API."""
        try:
            from segment_anything import SamPredictor
        except ImportError:
            raise ImportError("segment-anything is required: pip install segment-anything")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required: pip install huggingface_hub")

        logger.info("  Downloading MobileSAM checkpoint (~9MB)...")
        checkpoint_path = hf_hub_download(
            repo_id=info["repo"],
            filename=info["filename"],
        )

        logger.info(f"  Loading MobileSAM ({self.precision})...")

        # MobileSAM uses a lightweight ViT-Tiny encoder + SAM's mask decoder
        # Try the mobile_sam package first, fall back to manual loading
        try:
            from mobile_sam import SamPredictor as MobileSamPredictor
            from mobile_sam import sam_model_registry

            sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
            predictor_cls = MobileSamPredictor
        except ImportError:
            # Fallback: MobileSAM checkpoint is compatible with standard SAM API
            # when using vit_t architecture
            from segment_anything import SamPredictor, sam_model_registry

            try:
                sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
            except Exception:
                # Last resort: load with standard build_sam
                from segment_anything import build_sam

                sam = build_sam(checkpoint=checkpoint_path)
            predictor_cls = SamPredictor

        if self._use_amp:
            sam = sam.half()

        sam.to(self.device)
        sam.eval()
        self.predictor = predictor_cls(sam)
        self._sam_model = sam
        logger.info(
            f"  MobileSAM loaded on {self.device} ({self.precision}) — ⚡ ~5x faster than ViT-B"
        )

    def _load_efficient_sam(self, info: dict):
        """Load EfficientSAM — ~15MB, good speed/quality trade-off."""
        try:
            from segment_anything import SamPredictor
        except ImportError:
            raise ImportError("segment-anything is required: pip install segment-anything")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required: pip install huggingface_hub")

        logger.info("  Downloading EfficientSAM checkpoint (~15MB)...")
        checkpoint_path = hf_hub_download(
            repo_id=info["repo"],
            filename=info["filename"],
        )

        logger.info(f"  Loading EfficientSAM ({self.precision})...")

        # EfficientSAM exports as TorchScript; use custom wrapper
        try:
            # Load the TorchScript model as the image encoder
            efficient_encoder = torch.jit.load(checkpoint_path)
            efficient_encoder.to(self.device)
            if self._use_amp:
                efficient_encoder = efficient_encoder.half()
            efficient_encoder.eval()

            # Wrap in a compatible interface
            self._efficient_encoder = efficient_encoder
            self._use_efficient = True

            # We still need SAM's prompt encoder + mask decoder
            # Load a lightweight SAM for the decoder parts
            from huggingface_hub import hf_hub_download
            from segment_anything import sam_model_registry

            vit_b_info = SAM_MODELS["vit_b"]
            vit_b_path = hf_hub_download(
                repo_id=vit_b_info["repo"], filename=vit_b_info["filename"]
            )
            sam_full = sam_model_registry["vit_b"](checkpoint=vit_b_path)

            # Replace the image encoder with EfficientSAM
            self._efficient_sam = sam_full
            if self._use_amp:
                self._efficient_sam = self._efficient_sam.half()
            self._efficient_sam.to(self.device)
            self._efficient_sam.eval()

            # Create our own predictor that uses EfficientSAM encoder + SAM decoder
            self.predictor = _EfficientSAMPredictor(
                efficient_encoder, self._efficient_sam, self.device
            )
            logger.info(
                f"  EfficientSAM loaded on {self.device} ({self.precision}) — ⚡ ~3x faster than ViT-B"
            )

        except Exception as e:
            logger.warning(f"  ⚠️  EfficientSAM loading failed ({e}), falling back to MobileSAM...")
            self._load_mobile_sam(SAM_MODELS_FAST["mobile"])

    @staticmethod
    def _size_str(model_type: str) -> str:
        sizes = {
            "vit_h": "2.4GB",
            "vit_l": "1.2GB",
            "vit_b": "350MB",
            "mobile": "9MB",
            "efficient": "15MB",
        }
        return sizes.get(model_type, "?")

    def generate_masks(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        min_mask_score: float = 0.7,
    ) -> List[Tuple[np.ndarray, float]]:
        """Generate precise masks from bounding boxes.

        Args:
            image: RGB numpy array (H, W, 3), uint8
            boxes: List of [x1, y1, x2, y2] boxes in pixel coordinates
            min_mask_score: Minimum IoU stability score for SAM masks

        Returns:
            List of (mask, score) tuples, where mask is (H, W) bool array
        """
        if not boxes:
            return []

        # Set the image (SAM encoder runs only once per image)
        self.predictor.set_image(image)

        results = []
        for box in boxes:
            x1, y1, x2, y2 = box

            # Validate box
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue  # Box too small

            input_box = np.array([x1, y1, x2, y2])

            # FP16 autocast for mask decoder (~2x speedup)
            from contextlib import nullcontext

            amp_ctx = (
                torch.autocast(device_type=self.device, dtype=torch.float16)
                if self._use_amp
                else nullcontext()
            )
            with amp_ctx:
                mask, score, _ = self.predictor.predict(
                    box=input_box,
                    multimask_output=False,
                )

            mask = mask[0]  # (H, W) boolean
            score = float(score[0])

            if score >= min_mask_score:
                results.append((mask, score))

        return results


# ============================================================
# Auto Annotator
# ============================================================


class _EfficientSAMPredictor:
    """Adapter: uses EfficientSAM lightweight encoder + SAM prompt encoder & mask decoder.

    This is the key to EfficientSAM's speed: replace ViT-H (636M params)
    with EfficientSAM's tiny encoder while keeping SAM's proven decoder.
    """

    def __init__(self, efficient_encoder, sam_full, device):
        self.encoder = efficient_encoder
        self.sam = sam_full
        self.device = device
        self._features = None
        self._orig_size = None
        self._input_size = None

    def set_image(self, image: np.ndarray):
        """Run EfficientSAM encoder to get image embeddings."""
        import cv2

        self._orig_size = image.shape[:2]

        # SAM expects 1024x1024 input
        target_size = (1024, 1024)
        self._input_size = target_size

        # Preprocess
        img = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        img = img.to(self.device)

        # Normalize with SAM's stats
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
        img = (img - pixel_mean) / pixel_std

        with torch.no_grad():
            # EfficientSAM encoder forward
            if hasattr(self.encoder, "forward"):
                self._features = self.encoder(img)
            else:
                self._features = self.encoder(img)

        # Store in sam's format for compatibility
        self.sam.image_encoder = None  # We'll intercept in predict

    def predict(self, box: np.ndarray, multimask_output: bool = False):
        """Use SAM's prompt encoder + mask decoder with EfficientSAM features."""
        with torch.no_grad():
            # Prepare SAM inputs
            from segment_anything.utils.transforms import ResizeLongestSide

            transform = ResizeLongestSide(1024)

            # Transform box to 1024x1024 space
            box_t = transform.apply_boxes(box, self._orig_size)
            box_t = torch.as_tensor(box_t, dtype=torch.float32, device=self.device)
            box_t = box_t.unsqueeze(0)  # (1, 4)

            # Run SAM's prompt encoder
            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=None,
                boxes=box_t,
                masks=None,
            )

            # Run SAM's mask decoder with EfficientSAM image features
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=self._features,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=multimask_output,
            )

            # Post-process: upscale masks to original resolution
            from segment_anything.utils.transforms import ResizeLongestSide

            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=self._input_size,
                original_size=self._orig_size,
            )

            return masks[0].cpu().numpy(), iou_predictions[0].cpu().numpy(), None


class AutoAnnotator:
    """Main pipeline: Grounding DINO detection → SAM mask refinement → merged mask.

    Usage:
        annotator = AutoAnnotator(gdino_model="base", sam_model="vit_h")
        mask, info = annotator.annotate("image.jpg")
        # mask: (H, W) uint8, 0=bg, 1=sky, 2=water
    """

    def __init__(
        self,
        gdino_model: str = "base",
        sam_model: str = "vit_h",
        device: str = "cuda",
        precision: str = "fp32",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        sam_mask_threshold: float = 0.7,
        image_size: int = 1024,
        fast_mode: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            gdino_model: "tiny" or "base"
            sam_model: "vit_h", "vit_l", "vit_b", "mobile", "efficient"
            device: "cuda" or "cpu"
            precision: "fp32" or "fp16" — FP16 gives ~2x speedup on MPS/CUDA
            box_threshold: Grounding DINO box confidence threshold
            text_threshold: Grounding DINO text-phrase matching threshold
            sam_mask_threshold: SAM mask quality (IoU stability) threshold
            image_size: Max image dimension for Grounding DINO processing
            fast_mode: Enable all speed optimizations (fp16 + mobile sam + 512px)
            verbose: Print detailed progress
        """
        # Fast mode overrides
        if fast_mode:
            precision = "fp16"
            if sam_model in ("vit_h", "vit_l"):
                sam_model = "mobile"
            if image_size > 768:
                image_size = 768
            if verbose:
                logger.info("⚡ FAST MODE enabled: fp16 + mobile sam + 768px")

        self.precision = precision
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.sam_mask_threshold = sam_mask_threshold
        self.image_size = image_size
        self.verbose = verbose

        # Determine device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "cuda" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if verbose:
            logger.info("=" * 60)
            logger.info("Initializing Auto-Annotator")
            logger.info(f"  Device:    {self.device}")
            logger.info(f"  Precision: {self.precision}")
            logger.info(f"  GDINO:     {gdino_model}")
            logger.info(f"  SAM:       {sam_model}")
            logger.info(f"  Img size:  {image_size}px")
            logger.info(f"  Fast mode: {'ON ⚡' if fast_mode else 'off'}")
            logger.info("=" * 60)

        # Load models
        self.detector = GroundingDINODetector(
            model_size=gdino_model,
            device=self.device,
            precision=self.precision,
        )
        self.mask_generator = SAMMaskGenerator(
            model_type=sam_model,
            device=self.device,
            precision=self.precision,
        )

        if verbose:
            logger.info("\n✅ Auto-Annotator ready!\n")

    def annotate(
        self,
        image_path: str,
        classes: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Annotate a single image.

        Args:
            image_path: Path to input image
            classes: Class definitions dict (uses DEFAULT_CLASSES if None)

        Returns:
            mask: (H, W) uint8 numpy array (0=background, 1=sky, 2=water, ...)
            info: Dict with detection/mask statistics
        """
        if classes is None:
            classes = DEFAULT_CLASSES

        # ---- Load image ----
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        # ---- Resize for Grounding DINO (it's expensive on large images) ----
        scale = min(self.image_size / max(orig_w, orig_h), 1.0)
        if scale < 1.0:
            proc_w, proc_h = int(orig_w * scale), int(orig_h * scale)
            image_proc = image_pil.resize((proc_w, proc_h), Image.BILINEAR)
        else:
            proc_w, proc_h = orig_w, orig_h
            image_proc = image_pil
            scale = 1.0

        # ---- Build text prompts ----
        # Grounding DINO treats "." as a phrase separator
        # Each phrase is independently matched against the image
        all_phrases = []
        phrase_to_class = {}  # maps lowercase phrase → class_name

        for class_name, class_info in classes.items():
            for prompt in class_info["prompts"]:
                phrase_lower = prompt.strip().lower()
                all_phrases.append(prompt)
                phrase_to_class[phrase_lower] = class_name

        # Join phrases with ". " as Grounding DINO expects
        # Grounding DINO format: "phrase1. phrase2. phrase3."
        text_prompt = ". ".join(all_phrases) + "."

        # ---- Step 1: Grounding DINO detection ----
        detections = self.detector.detect(
            image_proc,
            text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # ---- Step 2: Group boxes by class ----
        # Scale boxes from processing size back to original size
        boxes_by_class: Dict[str, List[List[float]]] = {}
        info_detections = []

        inv_scale_x = orig_w / proc_w
        inv_scale_y = orig_h / proc_h

        for det in detections:
            phrase = det["label"].strip().lower()
            class_name = phrase_to_class.get(phrase)

            if class_name is None:
                # Try fuzzy match: check if any prompt is a substring
                for prompt, cls in phrase_to_class.items():
                    if prompt in phrase or phrase in prompt:
                        class_name = cls
                        break

            if class_name is None:
                continue

            # Scale box to original image coordinates
            box = det["box"]
            box_scaled = [
                box[0] * inv_scale_x,
                box[1] * inv_scale_y,
                box[2] * inv_scale_x,
                box[3] * inv_scale_y,
            ]

            if class_name not in boxes_by_class:
                boxes_by_class[class_name] = []
            boxes_by_class[class_name].append(box_scaled)

            info_detections.append(
                {
                    "box": [round(c, 1) for c in box_scaled],
                    "score": round(det["score"], 4),
                    "class": class_name,
                    "prompt": phrase,
                }
            )

        # ---- Step 3: SAM mask generation ----
        image_np = np.array(image_pil)  # Original resolution, RGB
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Process water first, then sky (so sky can override if needed,
        # or vice versa — water gets higher value which may be preferred)
        # Actually: process in mask_value order so higher values overwrite
        class_order = sorted(classes.items(), key=lambda x: x[1]["mask_value"])

        for class_name, class_info in class_order:
            mask_value = class_info["mask_value"]
            boxes = boxes_by_class.get(class_name, [])

            if not boxes:
                continue

            # Generate SAM masks
            mask_results = self.mask_generator.generate_masks(
                image_np, boxes, min_mask_score=self.sam_mask_threshold
            )

            # Write masks into final mask
            for mask_bool, sam_score in mask_results:
                # Apply morphological cleanup
                mask_bool = self._clean_mask(mask_bool)
                final_mask[mask_bool] = mask_value

        # ---- Step 4: Compile info ----
        info = {
            "image_path": image_path,
            "image_size": [orig_w, orig_h],
            "num_detections": len(detections),
            "detections": info_detections,
            "mask_stats": {
                "sky_pixels": int(np.sum(final_mask == 1)),
                "water_pixels": int(np.sum(final_mask == 2)),
                "sky_pct": round(float(np.mean(final_mask == 1)) * 100, 2),
                "water_pct": round(float(np.mean(final_mask == 2)) * 100, 2),
                "total_coverage_pct": round(float(np.mean(final_mask > 0)) * 100, 2),
            },
        }

        return final_mask, info

    def annotate_directory(
        self,
        input_dir: str,
        output_dir: str,
        classes: Optional[Dict] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"),
        save_visualization: bool = True,
        skip_existing: bool = True,
    ) -> Dict:
        """Annotate all images in a directory.

        Args:
            input_dir: Directory containing images
            output_dir: Directory to save masks
            classes: Class definitions
            extensions: File extensions to process
            save_visualization: Whether to save overlay visualizations
            skip_existing: Skip images that already have mask files

        Returns:
            Summary dict with statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Gather image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))

        if not image_files:
            raise FileNotFoundError(f"No images found in {input_dir}")

        # Filter already-processed
        if skip_existing:
            remaining = []
            for f in image_files:
                mask_file = output_path / f"{f.stem}_mask.png"
                if not mask_file.exists():
                    remaining.append(f)
            skipped = len(image_files) - len(remaining)
            if skipped > 0:
                logger.warning(f"Skipping {skipped} already-annotated images")
            image_files = remaining

        if not image_files:
            logger.info("All images already annotated!")
            return {"total": 0, "skipped": skipped}

        logger.info(f"Processing {len(image_files)} images...\n")

        results = []
        start_time = time.time()

        for img_file in tqdm(image_files, desc="Annotating", unit="img"):
            try:
                mask, info = self.annotate(str(img_file), classes)

                # Save mask as PNG (0, 1, 2 values visually interpretable)
                mask_path = output_path / f"{img_file.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask)

                # Save visualization overlay
                if save_visualization:
                    vis = self._visualize(str(img_file), mask, classes)
                    vis_path = output_path / f"{img_file.stem}_vis.jpg"
                    cv2.imwrite(str(vis_path), vis)

                info["output_mask"] = str(mask_path)
                info["status"] = "success"
                results.append(info)

            except Exception as e:
                import traceback

                error_msg = f"{e}\n{traceback.format_exc()}"
                logger.warning(f"\n⚠️  Error on {img_file.name}: {e}")
                results.append(
                    {
                        "image_path": str(img_file),
                        "status": "error",
                        "error": str(e),
                    }
                )

        elapsed = time.time() - start_time

        # ---- Summary ----
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]

        summary = {
            "total": len(image_files),
            "successful": len(successful),
            "failed": len(failed),
            "elapsed_seconds": round(elapsed, 1),
            "avg_seconds_per_image": round(elapsed / max(len(image_files), 1), 1),
        }

        if successful:
            coverages = [r["mask_stats"]["total_coverage_pct"] for r in successful]
            summary["avg_coverage_pct"] = round(np.mean(coverages), 2)
            summary["min_coverage_pct"] = round(np.min(coverages), 2)
            summary["max_coverage_pct"] = round(np.max(coverages), 2)

        # Save summary
        summary_path = output_path / "annotation_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"summary": summary, "details": results}, f, indent=2)

        # Print
        logger.info(f"\n{'=' * 60}")
        logger.info("📊 Annotation Complete!")
        logger.info(f"  Images processed: {summary['total']}")
        logger.info(f"  ✅ Successful:     {summary['successful']}")
        logger.info(f"  ❌ Failed:         {summary['failed']}")
        logger.info(
            f"  ⏱️  Time:           {summary['elapsed_seconds']:.0f}s "
            f"({summary['avg_seconds_per_image']:.1f}s/img)"
        )
        if successful:
            logger.info(f"  📐 Avg coverage:   {summary['avg_coverage_pct']:.1f}%")
        logger.info(f"  📁 Output:         {output_path}")
        logger.info(f"{'=' * 60}")

        return summary

    # ---- Internal helpers ----

    @staticmethod
    def _clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply morphological opening to remove small noise, then closing
        to fill small holes."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Opening: remove small islands
        mask = mask.astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Closing: fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask.astype(bool)

    @staticmethod
    def _visualize(
        image_path: str,
        mask: np.ndarray,
        classes: Optional[Dict] = None,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Create a visualization overlay of the mask on the original image.

        Returns BGR image suitable for cv2.imwrite.
        """
        if classes is None:
            classes = DEFAULT_CLASSES

        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        h, w = mask.shape
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))

        overlay = np.zeros_like(image)

        # Build color map from class definitions
        for class_name, class_info in classes.items():
            mv = class_info["mask_value"]
            color = class_info.get("color", [128, 128, 128])
            overlay[mask == mv] = color

        vis = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        # Add legend
        y_offset = 30
        for class_name, class_info in classes.items():
            color = class_info.get("color", [128, 128, 128])
            pct = float(np.mean(mask == class_info["mask_value"])) * 100
            label = f"{class_name}: {pct:.1f}%"
            cv2.putText(
                vis,
                label,
                (12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            y_offset += 28

        return vis


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="🖼️  Auto-annotate sky/water regions with Grounding DINO + SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate a folder of images
  python auto_annotate.py -i ./data/images -o ./data/masks

  # Single image, faster models
  python auto_annotate.py -i test.jpg -o ./output --gdino-model tiny --sam-model vit_b

  # Custom classes via JSON config
  python auto_annotate.py -i ./images -o ./masks --classes my_classes.json

  # Lower thresholds for more detections
  python auto_annotate.py -i ./images -o ./masks --box-threshold 0.15 --text-threshold 0.10
        """,
    )
    # I/O
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input image file or directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data/masks",
        help="Output directory for masks (default: ./data/masks)",
    )
    # Model selection
    parser.add_argument(
        "--gdino-model",
        type=str,
        default="base",
        choices=["tiny", "base"],
        help="Grounding DINO model: tiny (faster) or base (more accurate)",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b", "mobile", "efficient"],
        help="SAM model: mobile/efficient (fastest), vit_b (balanced), vit_h/l (best)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="FP16 gives ~2x speedup on MPS/CUDA (default: fp32)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable all speed optimizations: fp16 + mobile sam + 768px",
    )
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda (GPU) or cpu",
    )
    # Thresholds
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO box confidence (0-1, lower = more detections)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Grounding DINO text matching confidence (0-1)",
    )
    parser.add_argument(
        "--sam-mask-threshold",
        type=float,
        default=0.7,
        help="SAM mask quality threshold (0-1)",
    )
    # Processing
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Max image dimension for Grounding DINO (768-2048)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip saving visualization overlays",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-annotate images that already have masks",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Path to JSON file with custom class definitions",
    )

    args = parser.parse_args()

    # ---- Load class definitions ----
    classes = DEFAULT_CLASSES
    if args.classes:
        with open(args.classes, "r") as f:
            classes = json.load(f)
        # Validate
        for name, info in classes.items():
            assert "prompts" in info, f"Class '{name}' missing 'prompts'"
            assert "mask_value" in info, f"Class '{name}' missing 'mask_value'"
        logger.info(f"Loaded custom classes: {list(classes.keys())}")

    # ---- Initialize annotator ----
    annotator = AutoAnnotator(
        gdino_model=args.gdino_model,
        sam_model=args.sam_model,
        device=args.device,
        precision=args.precision,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        sam_mask_threshold=args.sam_mask_threshold,
        image_size=args.image_size,
        fast_mode=args.fast,
    )

    # ---- Process ----
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image mode
        logger.info(f"Processing single image: {input_path.name}")
        mask, info = annotator.annotate(str(input_path), classes)

        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save mask
        mask_file = output_path / f"{input_path.stem}_mask.png"
        cv2.imwrite(str(mask_file), mask)
        logger.info(f"Mask saved: {mask_file}")

        # Save visualization
        if not args.no_viz:
            vis = annotator._visualize(str(input_path), mask, classes)
            vis_file = output_path / f"{input_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_file), vis)
            logger.info(f"Visualization saved: {vis_file}")

        # Print stats
        logger.info("\nDetection stats:")
        for det in info.get("detections", []):
            logger.info(f"  [{det['class']}] score={det['score']:.3f}")
        logger.info(f"Mask stats: {json.dumps(info['mask_stats'], indent=2)}")

    else:
        # Directory mode
        annotator.annotate_directory(
            input_dir=str(input_path),
            output_dir=args.output,
            classes=classes,
            save_visualization=not args.no_viz,
            skip_existing=not args.no_skip,
        )


if __name__ == "__main__":
    main()
