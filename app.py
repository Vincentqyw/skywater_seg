#!/usr/bin/env python3
"""
Gradio UI for sky-water-person segmentation inference.

Usage (local):
    uv run python app.py                          # HF Hub model (default)
    uv run python app.py --local checkpoint.pth   # local checkpoint
    uv run python app.py --onnx model.onnx        # ONNX Runtime
    uv run python app.py --share                  # public link

HuggingFace Spaces:
    Detected automatically via SPACE_ID env var — no CLI needed.
    Uses @spaces.GPU for GPU acceleration.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
from loguru import logger

import gradio as gr

# HF Spaces GPU support (no-op when running locally without the package)
try:
    import spaces

    _HAS_SPACES = True
except ImportError:
    spaces = None  # type: ignore[assignment]
    _HAS_SPACES = False

# Project imports
from skywater_seg.inference import segment_skywater, SkyWaterSegModel
from skywater_seg.visualization import (
    CLASS_NAMES,
    CLASS_COLORS_RGB,
    mask_to_color,
    overlay_mask,
)

# ── Detect runtime environment ─────────────────────────────────────────
_ON_SPACES = bool(os.environ.get("SPACE_ID"))

# ── Globals (set in main) ──────────────────────────────────────────────
_MODEL = None
_ONNX_INFER = None
_USE_ONNX = False


# ═══════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Strip alpha channel if present, return (H, W, 3) uint8 RGB copy."""
    if image.shape[-1] == 4:
        return image[:, :, :3].copy()
    return image


def run_inference(image: np.ndarray | None) -> np.ndarray | None:
    """Run segmentation, return uint8 mask (H, W). Called once per image."""
    if image is None:
        return None
    image = _ensure_rgb(image)
    h, w = image.shape[:2]
    t0 = time.perf_counter()

    if _USE_ONNX and _ONNX_INFER is not None:
        result = _ONNX_INFER.predict(image)
        mask = result["mask"]
        backend = f"ONNX ({_ONNX_INFER.providers[0]})"
    else:
        mask = segment_skywater(image, model=_MODEL, device="cuda" if _MODEL else "cpu")
        backend = "PyTorch"

    elapsed = (time.perf_counter() - t0) * 1000
    total_px = mask.size
    pcts = ", ".join(
        f"{CLASS_NAMES[i]}={(mask == i).sum() / total_px * 100:.1f}%"
        for i in range(len(CLASS_NAMES))
    )
    logger.info(f"[inference] [{backend}] {w}x{h}  |  {elapsed:.0f} ms  |  {pcts}")
    return mask


# Wrap with HF Spaces GPU decorator when available
if _HAS_SPACES:
    run_inference = spaces.GPU(run_inference)


# ═══════════════════════════════════════════════════════════════════════
# Tab renderers — all take (image, mask, ...), never call inference
# ═══════════════════════════════════════════════════════════════════════


def process_overlay(
    image: np.ndarray | None,
    mask: np.ndarray | None,
    alpha: float,
    draw_contours: bool,
) -> np.ndarray | None:
    if image is None or mask is None:
        return None
    image = _ensure_rgb(image)
    vis = overlay_mask(image, mask, alpha=alpha, draw_contours=draw_contours)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def process_mask(
    image: np.ndarray | None,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    if image is None or mask is None:
        return None
    return mask_to_color(mask)


def process_per_class(
    image: np.ndarray | None,
    mask: np.ndarray | None,
) -> tuple:
    """Return (bg, sky, water, person) RGBA masks as a 2x2 grid."""
    if image is None or mask is None:
        return None, None, None, None
    return (
        _binary_to_colored(mask == 0, CLASS_COLORS_RGB[0]),
        _binary_to_colored(mask == 1, CLASS_COLORS_RGB[1]),
        _binary_to_colored(mask == 2, CLASS_COLORS_RGB[2]),
        _binary_to_colored(mask == 3, CLASS_COLORS_RGB[3]),
    )


def _binary_to_colored(binary: np.ndarray, color: tuple) -> np.ndarray:
    """Convert a (H, W) bool mask to an RGBA image with the given color on transparent bg."""
    h, w = binary.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[binary, 0] = color[0]
    rgba[binary, 1] = color[1]
    rgba[binary, 2] = color[2]
    rgba[binary, 3] = 255
    return rgba


def process_stats(
    image: np.ndarray | None,
    mask: np.ndarray | None,
):
    """Return a tuple of 4 percentage strings, one per class Label."""
    if image is None or mask is None:
        return tuple("—" for _ in CLASS_NAMES)
    total = mask.size
    return tuple(
        f"{100 * (mask == i).sum() / total:.1f}%" for i in range(len(CLASS_NAMES))
    )


def process_compare_slider(
    image: np.ndarray | None,
    mask: np.ndarray | None,
    alpha: float,
) -> tuple | None:
    """Return (original, overlay) tuple for gr.ImageSlider native comparison slider."""
    if image is None or mask is None:
        return None
    image = _ensure_rgb(image)
    overlay_bgr = overlay_mask(image, mask, alpha=alpha, draw_contours=True)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return (image, overlay_rgb)


# ═══════════════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════════════

HEADER_MD = """
<div align="center" style="margin-bottom: 24px;">

# 🌤️ Sky-Water-Person Segmentation

Upload an image to segment **sky**, **water**, and **person** regions—designed to eliminate their interference with **SfM** and image-matching pipelines.

[![GitHub](https://img.shields.io/badge/GitHub-Vincentqyw%2Fskywater_seg-24292e?logo=github&logoColor=white&style=flat-square)](https://github.com/Vincentqyw/skywater_seg)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Model-Realcat%2Fskywater__seg-ff9a00?style=flat-square)](https://huggingface.co/Realcat/skywater_seg)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Dataset-Realcat%2Fskywater-3b82f6?style=flat-square)](https://huggingface.co/datasets/Realcat/skywater)
[![Hugging Face Space](https://img.shields.io/badge/🤗-Live_Demo-blue.svg)](https://huggingface.co/spaces/Realcat/skywater_seg)

**SegFormer MiT-B2** (24.7M) &nbsp;|&nbsp; 384x384 &nbsp;|&nbsp; mIoU 88.1% &nbsp;|&nbsp; ONNX · CoreML · PyTorch

<p style="margin-bottom: 160px;">🟠 Sky &nbsp; 🔵 Water &nbsp; 🔴 Person &nbsp; ⚫ Background</p>

</div>
"""

GRADIO_CSS = """
footer { display: none !important; }

/* ── Input panel card ── */
#input-col {
    background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%) !important;
    border: 1px solid #313244 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* ── Settings accordion ── */
#settings-accordion {
    border: 1px solid #45475a !important;
    border-radius: 8px !important;
}

/* ── Run button glow ── */
#run-btn {
    background: linear-gradient(135deg, #f5a623 0%, #f76b1c 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
#run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(245, 166, 35, 0.35) !important;
}

/* ── Tab underline accent ── */
[class*="tab"] button[aria-selected="true"],
button[aria-selected="true"] {
    border-bottom: 2px solid #f5a623 !important;
}

/* ── Examples gallery: 3 per row ── */
#input-col > div > div[class*="grid"] {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 8px !important;
}
#input-col img[class*="example"],
#input-col [class*="gallery"] img,
#input-col div[class*="grid"] img {
    height: 80px !important;
    width: 100% !important;
    object-fit: cover !important;
    border-radius: 6px !important;
}
"""


# Asset base path — local on dev, GitHub raw URLs on HF Spaces (no binary storage)
_ASSET_BASE = (
    "https://raw.githubusercontent.com/Vincentqyw/skywater_seg/main/assets"
    if _ON_SPACES
    else "assets"
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SkyWater Segmentation") as demo:
        gr.Markdown(HEADER_MD)

        with gr.Row():
            with gr.Column(scale=1, elem_id="input-col"):
                input_img = gr.Image(label="Input Image", type="numpy", height=320)

                with gr.Accordion("⚙️ Settings", open=False, elem_id="settings-accordion"):
                    alpha_slider = gr.Slider(
                        0.0, 1.0, value=0.45, step=0.05, label="Overlay Alpha",
                    )
                    contours_chk = gr.Checkbox(value=True, label="Draw Contours")

                btn = gr.Button(
                    "🚀 Run Segmentation", variant="primary", size="lg",
                    elem_id="run-btn",
                )

                gr.Examples(
                    examples=[
                        f"{_ASSET_BASE}/photos/DSCF5825.jpg",
                        f"{_ASSET_BASE}/photos/DSCF5827.jpg",
                        f"{_ASSET_BASE}/photos/DSCF6021.jpg",
                        f"{_ASSET_BASE}/photos/DSCF6049.jpg",
                        f"{_ASSET_BASE}/photos/DSCF8778.jpg",
                        f"{_ASSET_BASE}/photos/DSCF8781.jpg",
                        f"{_ASSET_BASE}/photos/DSCF8790.jpg",
                        f"{_ASSET_BASE}/photos/DSCF8858.jpg",
                        f"{_ASSET_BASE}/photos/DSCF8917.jpg",
                        f"{_ASSET_BASE}/photos/DSCF8980.jpg",
                        f"{_ASSET_BASE}/photos/DSCF8993.jpg",
                        f"{_ASSET_BASE}/photos/DSCF9075.jpg",
                        f"{_ASSET_BASE}/photos/hk.jpg",
                        f"{_ASSET_BASE}/photos/IMG_2613.jpg",
                        f"{_ASSET_BASE}/photos/IMG_2844.jpg",
                        f"{_ASSET_BASE}/photos/IMG_2955.jpg",
                        f"{_ASSET_BASE}/photos/IMG_3619.jpg",
                        f"{_ASSET_BASE}/photos/IMG_3621.jpg",
                        f"{_ASSET_BASE}/photos/IMG_4764.jpg",
                        f"{_ASSET_BASE}/photos/IMG_4954.jpg",
                        f"{_ASSET_BASE}/photos/IMG_5303.jpg",
                        f"{_ASSET_BASE}/photos/IMG_5464.jpg",
                        f"{_ASSET_BASE}/photos/IMG_6318.jpg",
                        f"{_ASSET_BASE}/photos/IMG_7054.jpg",
                        f"{_ASSET_BASE}/photos/IMG_7055.jpg",
                        f"{_ASSET_BASE}/photos/IMG_7084.jpg",
                        f"{_ASSET_BASE}/photos/IMG_7086.jpg",
                        f"{_ASSET_BASE}/photos/IMG_7507.jpg",
                        f"{_ASSET_BASE}/photos/IMG_7679.jpg",
                        f"{_ASSET_BASE}/photos/IMG_7714.jpg",
                        f"{_ASSET_BASE}/photos/IMG_8612.jpg",
                        f"{_ASSET_BASE}/photos/IMG_8617.jpg",
                    ],
                    inputs=input_img,
                    label="📸 Examples",
                    examples_per_page=15,
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Overlay"):
                        overlay_out = gr.Image(label="Overlay on Original", height=540)
                        with gr.Row():
                            stats_labels = []
                            for name in CLASS_NAMES:
                                lbl = gr.Label(
                                    label=name,
                                    value="—",
                                    container=True,
                                )
                                stats_labels.append(lbl)

                    with gr.Tab("Colorized Mask"):
                        mask_out = gr.Image(label="Colorized Mask", height=540)

                    with gr.Tab("Compare"):
                        compare_out = gr.ImageSlider(label="Original ↔ Overlay", height=540)

                    with gr.Tab("Per-Class"):
                        with gr.Row():
                            bg_out = gr.Image(label="Background", height=310)
                            sky_out = gr.Image(label="Sky", height=310)
                        with gr.Row():
                            water_out = gr.Image(label="Water", height=310)
                            person_out = gr.Image(label="Person", height=310)

        # ── Hidden state: mask flows through the .then() chain ────────
        mask_state = gr.State()

        # ── Event binding ────────────────────────────────────────────
        btn.click(
            fn=run_inference, inputs=[input_img], outputs=[mask_state],
        ).then(
            fn=process_overlay,
            inputs=[input_img, mask_state, alpha_slider, contours_chk],
            outputs=overlay_out,
        ).then(
            fn=process_mask,
            inputs=[input_img, mask_state],
            outputs=mask_out,
        ).then(
            fn=process_compare_slider,
            inputs=[input_img, mask_state, alpha_slider],
            outputs=compare_out,
        ).then(
            fn=process_per_class,
            inputs=[input_img, mask_state],
            outputs=[bg_out, sky_out, water_out, person_out],
        ).then(
            fn=process_stats,
            inputs=[input_img, mask_state],
            outputs=stats_labels,
        )

        # Update overlay + compare when alpha / contours change
        alpha_slider.change(
            fn=process_overlay,
            inputs=[input_img, mask_state, alpha_slider, contours_chk],
            outputs=overlay_out,
        ).then(
            fn=process_compare_slider,
            inputs=[input_img, mask_state, alpha_slider],
            outputs=compare_out,
        )
        contours_chk.change(
            fn=process_overlay,
            inputs=[input_img, mask_state, alpha_slider, contours_chk],
            outputs=overlay_out,
        ).then(
            fn=process_compare_slider,
            inputs=[input_img, mask_state, alpha_slider],
            outputs=compare_out,
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _load_model(local_path: str | None = None, onnx_path: str | None = None):
    """Load the segmentation model from HF Hub, local checkpoint, or ONNX."""
    global _MODEL, _ONNX_INFER, _USE_ONNX

    if onnx_path:
        logger.info(f"Loading ONNX model: {onnx_path}")
        from skywater_seg.inference import ONNXRuntimeInference

        _ONNX_INFER = ONNXRuntimeInference(onnx_path, provider="cuda")
        _USE_ONNX = True
        logger.info(f"ONNX Runtime ready — providers: {_ONNX_INFER.providers}")
    elif local_path:
        logger.info(f"Loading local checkpoint: {local_path}")
        from skywater_seg.inference import SegmentationInference

        _MODEL = SegmentationInference(local_path).model
    else:
        logger.info("Loading model from HuggingFace Hub: Realcat/skywater_seg")
        _MODEL = SkyWaterSegModel.from_pretrained("Realcat/skywater_seg")
        _MODEL.eval()
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL.to(device)
            logger.info(f"Model on: {device}")
        except Exception:
            pass


def main():
    _load_model()

    demo = build_ui()
    demo.launch(
        show_error=True,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css=GRADIO_CSS,
    )


def main_cli():
    """Local CLI entry point with argparse (not used on HF Spaces)."""
    parser = argparse.ArgumentParser(
        description="Gradio UI for Sky-Water-Person Segmentation",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--local", type=str, metavar="CHECKPOINT",
        help="Path to a local .pth checkpoint",
    )
    group.add_argument(
        "--onnx", type=str, metavar="MODEL",
        help="Path to a local .onnx model (uses ONNX Runtime)",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()

    _load_model(local_path=args.local, onnx_path=args.onnx)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Ember(primary_hue="orange", secondary_hue="blue"),
        css=GRADIO_CSS,
    )


if __name__ == "__main__":
    if _ON_SPACES:
        # HuggingFace Spaces: no CLI, Gradio handles host/port automatically
        logger.info("🚀 Running on HuggingFace Spaces")
        main()
    else:
        main_cli()
