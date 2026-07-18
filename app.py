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
# Inference helper
# ═══════════════════════════════════════════════════════════════════════

# Cache to avoid redundant inference in the .then() chain
_CACHE_HASH: int | None = None
_CACHE_MASK: np.ndarray | None = None


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Strip alpha channel if present, return (H, W, 3) uint8 RGB copy."""
    if image.shape[-1] == 4:
        return image[:, :, :3].copy()
    return image


def run_inference(image: np.ndarray) -> np.ndarray:
    """Run segmentation on an RGB image, return uint8 mask (H, W).

    Caches the result by image hash so the .then() chain doesn't
    re-run the model for each output tab.
    """
    global _CACHE_HASH, _CACHE_MASK

    image = _ensure_rgb(image)
    img_hash = hash(image.tobytes())

    if img_hash == _CACHE_HASH and _CACHE_MASK is not None:
        return _CACHE_MASK

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

    print(f"[inference] [{backend}] {w}×{h}  |  {elapsed:.0f} ms  |  {pcts}")

    _CACHE_HASH = img_hash
    _CACHE_MASK = mask
    return mask


# Wrap with HF Spaces GPU decorator when available
if _HAS_SPACES:
    run_inference = spaces.GPU(run_inference)


# ═══════════════════════════════════════════════════════════════════════
# Tab: Overlay
# ═══════════════════════════════════════════════════════════════════════


def process_overlay(
    image: np.ndarray | None,
    alpha: float,
    draw_contours: bool,
) -> np.ndarray | None:
    if image is None:
        return None
    image = _ensure_rgb(image)
    mask = run_inference(image)
    vis = overlay_mask(image, mask, alpha=alpha, draw_contours=draw_contours)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════
# Tab: Colorized Mask
# ═══════════════════════════════════════════════════════════════════════


def process_mask(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    image = _ensure_rgb(image)
    mask = run_inference(image)
    return mask_to_color(mask)


# ═══════════════════════════════════════════════════════════════════════
# Tab: Per-Class Binary Masks
# ═══════════════════════════════════════════════════════════════════════


def process_per_class(image: np.ndarray | None) -> tuple:
    """Return (bg, sky, water, person) RGBA masks as a 2×2 grid."""
    if image is None:
        return None, None, None, None
    image = _ensure_rgb(image)
    mask = run_inference(image)
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


# ═══════════════════════════════════════════════════════════════════════
# Tab: Statistics
# ═══════════════════════════════════════════════════════════════════════


def process_stats(image: np.ndarray | None):
    """Return a tuple of 4 percentage strings, one per class Label."""
    if image is None:
        return tuple("—" for _ in CLASS_NAMES)
    image = _ensure_rgb(image)
    mask = run_inference(image)
    total = mask.size
    return tuple(
        f"{100 * (mask == i).sum() / total:.1f}%" for i in range(len(CLASS_NAMES))
    )


# ═══════════════════════════════════════════════════════════════════════
# Tab: Compare (Gradio native ImageSlider)
# ═══════════════════════════════════════════════════════════════════════


def process_compare_slider(
    image: np.ndarray | None,
    alpha: float,
) -> tuple | None:
    """Return (original, overlay) tuple for gr.ImageSlider native comparison slider."""
    if image is None:
        return None
    image = _ensure_rgb(image)
    mask = run_inference(image)
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

<div style="display: flex; justify-content: center; gap: 6px; flex-wrap: nowrap;">
<a href="https://github.com/Vincentqyw/skywater_seg" target="_blank"><img src="https://img.shields.io/badge/GitHub-Vincentqyw%2Fskywater_seg-24292e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a><a href="https://huggingface.co/Realcat/skywater_seg" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Model-Realcat%2Fskywater__seg-ff9a00?style=flat-square" alt="HF Model"></a><a href="https://huggingface.co/datasets/Realcat/skywater" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Dataset-Realcat%2Fskywater-3b82f6?style=flat-square" alt="HF Dataset"></a>
</div>

**SegFormer MiT-B2** (24.7M) &nbsp;|&nbsp; 384×384 &nbsp;|&nbsp; mIoU 88.1% &nbsp;|&nbsp; ONNX · CoreML · PyTorch

<p style="margin-bottom: 160px;">🟠 Sky &nbsp; 🔵 Water &nbsp; 🔴 Person &nbsp; ⚫ Background</p>

</div>
"""

GRADIO_CSS = """
footer { display: none !important; }

/* Input panel card */
#input-col {
    background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%) !important;
    border: 1px solid #313244 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Settings accordion */
#settings-accordion {
    border: 1px solid #45475a !important;
    border-radius: 8px !important;
}

/* Run button glow */
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

/* Tab underline accent */
.tabs > .tab-nav > button.selected {
    border-bottom-color: #f5a623 !important;
}

/* Examples grid: 3 per row, uniform thumbnails */
#input-col .grid.gap-4 {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 8px !important;
}
#input-col .grid.gap-4 img {
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
                        f"{_ASSET_BASE}/hk.jpg",
                        f"{_ASSET_BASE}/264489593_6de914a0ab_o.jpg",
                        f"{_ASSET_BASE}/3134760025_0aaa4fdc8b_o.jpg",
                        f"{_ASSET_BASE}/331810308_2fe422b1ec_o.jpg",
                        f"{_ASSET_BASE}/525678483_c9b1a3665a_o.jpg",
                        f"{_ASSET_BASE}/981256188_8f690e95b1_o.jpg",
                        f"{_ASSET_BASE}/0015_096.jpg",
                        f"{_ASSET_BASE}/ade_ADE_val_00000590.jpg",
                        f"{_ASSET_BASE}/ade_ADE_val_00001354.jpg",
                        f"{_ASSET_BASE}/ade_ADE_val_00001674.jpg",
                    ],
                    inputs=input_img,
                    label="📸 Examples",
                    examples_per_page=10,
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

        # ── Event binding ────────────────────────────────────────────
        overlay_inputs = [input_img, alpha_slider, contours_chk]
        compare_inputs = [input_img, alpha_slider]

        btn.click(
            fn=process_overlay, inputs=overlay_inputs, outputs=overlay_out,
        ).then(fn=process_mask, inputs=[input_img], outputs=mask_out).then(
            fn=process_compare_slider,
            inputs=compare_inputs,
            outputs=compare_out,
        ).then(
            fn=process_per_class,
            inputs=[input_img],
            outputs=[bg_out, sky_out, water_out, person_out],
        ).then(
            fn=process_stats, inputs=[input_img], outputs=stats_labels,
        )

        # Update overlay + compare when alpha changes
        alpha_slider.change(
            fn=process_overlay, inputs=overlay_inputs, outputs=overlay_out,
        ).then(
            fn=process_compare_slider,
            inputs=compare_inputs,
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
        print(f"Loading ONNX model: {onnx_path}")
        from skywater_seg.inference import ONNXRuntimeInference

        _ONNX_INFER = ONNXRuntimeInference(onnx_path, provider="cuda")
        _USE_ONNX = True
        print(f"  ONNX Runtime ready — providers: {_ONNX_INFER.providers}")
    elif local_path:
        print(f"Loading local checkpoint: {local_path}")
        from skywater_seg.inference import SegmentationInference

        _MODEL = SegmentationInference(local_path).model
    else:
        print("Loading model from HuggingFace Hub: Realcat/skywater_seg")
        _MODEL = SkyWaterSegModel.from_pretrained("Realcat/skywater_seg")
        _MODEL.eval()
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL.to(device)
            print(f"  Model on: {device}")
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
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css=GRADIO_CSS,
    )


if __name__ == "__main__":
    if _ON_SPACES:
        # HuggingFace Spaces: no CLI, Gradio handles host/port automatically
        print("🚀 Running on HuggingFace Spaces")
        main()
    else:
        main_cli()
