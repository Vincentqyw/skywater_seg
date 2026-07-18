"""
Visualization helpers for segmentation results.

All visualisation functions live here — colour conversion, overlays,
composite grids, and matplotlib charts.

Plot functions require ``matplotlib`` (optional — graceful fallback).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from loguru import logger

# ── Shared colour palette (single source of truth) ─────────────────────

CLASS_NAMES = ["Background", "Sky", "Water", "Person"]
"""Human-readable class names matching the 4-class scheme."""

NUM_CLASSES = len(CLASS_NAMES)

CLASS_COLORS_RGB = {
    0: (0, 0, 0),          # background: black
    1: (255, 140, 0),      # sky: orange
    2: (0, 200, 255),      # water: cyan
    3: (255, 60, 60),      # person: red
}
"""RGB colour palette keyed by class index."""


_CLASS_COLORS_BGR = {k: (b, g, r) for k, (r, g, b) in CLASS_COLORS_RGB.items()}


def class_colors_bgr() -> Dict[int, Tuple[int, int, int]]:
    """Return the shared palette in BGR order for OpenCV functions."""
    return _CLASS_COLORS_BGR


# ═══════════════════════════════════════════════════════════════════════
# Mask ↔ colour conversion
# ═══════════════════════════════════════════════════════════════════════

def overlay_mask(
    image: Union[str, np.ndarray],
    mask: np.ndarray,
    alpha: float = 0.45,
    *,
    draw_contours: bool = True,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Draw a colour-blended overlay with optional contour outlines.

    Args:
        image: Path to an image file, or an RGB/BGR numpy array
               of shape ``(H, W, 3)``.
        mask: ``(H, W)`` uint8 class-index mask.
        alpha: Blend strength (0 = original, 1 = fully colourised).
        draw_contours: If True, draw per-class boundary contours.
        contour_thickness: Line width for contours.

    Returns:
        BGR numpy array suitable for ``cv2.imwrite`` / ``cv2.imshow``.
    """
    # -- load image --------------------------------------------------------
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
    else:
        img = image
        if img.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img

    h, w = mask.shape
    if img_bgr.shape[:2] != (h, w):
        img_bgr = cv2.resize(img_bgr, (w, h))

    colors_bgr = class_colors_bgr()

    # -- build overlay ------------------------------------------------------
    overlay = np.zeros_like(img_bgr)
    for cls_id, color in colors_bgr.items():
        if cls_id == 0:
            continue
        overlay[mask == cls_id] = color

    vis = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)

    # -- contours on top of blended image -----------------------------------
    if draw_contours:
        for cls_id, color in colors_bgr.items():
            if cls_id == 0:
                continue
            binary = (mask == cls_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(vis, contours, -1, color, contour_thickness)

    return vis


# ═══════════════════════════════════════════════════════════════════════
# Composite grids
# ═══════════════════════════════════════════════════════════════════════

def make_comparison_grid(
    samples: List[dict],
    out_path: str,
    *,
    dpi: int = 200,
) -> str:
    """Save an N-row × 4-column comparison figure.

    Columns: **Input | Ground Truth | Prediction A | Prediction B**

    Each sample dict must have::

        {
            "image":   np.ndarray,   # (H, W, 3) RGB
            "gt":      np.ndarray,   # (H, W) uint8 mask
            "pred_a":  np.ndarray,   # (H, W) uint8 mask
            "pred_b":  np.ndarray,   # (H, W) uint8 mask
            "label_a": str,          # column title
            "label_b": str,          # column title
        }

    Args:
        samples: List of sample dicts.
        out_path: Where to save the PNG.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, s in enumerate(samples):
        axes[row, 0].imshow(s["image"])
        axes[row, 0].set_title("Input", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(colorize_mask(s["gt"]))
        axes[row, 1].set_title("Ground Truth", fontsize=11)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(colorize_mask(s["pred_a"]))
        axes[row, 2].set_title(s.get("label_a", "Prediction A"), fontsize=11)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(colorize_mask(s["pred_b"]))
        axes[row, 3].set_title(s.get("label_b", "Prediction B"), fontsize=11)
        axes[row, 3].axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Comparison grid saved: {out_path}")
    return out_path


def make_overlay_grid(
    samples: List[dict],
    out_path: str,
    *,
    dpi: int = 200,
) -> str:
    """Save an N-row × 5-column overlay grid.

    Columns: **Input | GT | Overlay A | Overlay B | Difference map**

    Each sample dict must have::

        {
            "image":   np.ndarray,   # (H, W, 3) RGB
            "gt":      np.ndarray,   # (H, W) uint8 mask
            "pred_a":  np.ndarray,   # (H, W) uint8 mask
            "pred_b":  np.ndarray,   # (H, W) uint8 mask
            "label_a": str,
            "label_b": str,
        }

    Returns:
        ``out_path``.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    n = min(len(samples), 8)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[None, :]

    titles = ["Input", "Ground Truth", "Overlay A", "Overlay B", "Difference"]

    for row, s in enumerate(samples[:n]):
        axes[row, 0].imshow(s["image"])
        axes[row, 0].set_title(titles[0], fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(colorize_mask(s["gt"]))
        axes[row, 1].set_title(titles[1], fontsize=10)
        axes[row, 1].axis("off")

        ov_a = cv2.cvtColor(overlay_mask(s["image"], s["pred_a"]), cv2.COLOR_BGR2RGB)
        axes[row, 2].imshow(ov_a)
        axes[row, 2].set_title(f"{titles[2]}\n{s.get('label_a', 'A')}", fontsize=10)
        axes[row, 2].axis("off")

        ov_b = cv2.cvtColor(overlay_mask(s["image"], s["pred_b"]), cv2.COLOR_BGR2RGB)
        axes[row, 3].imshow(ov_b)
        axes[row, 3].set_title(f"{titles[3]}\n{s.get('label_b', 'B')}", fontsize=10)
        axes[row, 3].axis("off")

        diff = (s["pred_a"] != s["pred_b"])
        dm = np.zeros_like(s["image"])
        dm[diff] = [255, 60, 60]
        axes[row, 4].imshow(dm)
        nd = diff.sum()
        pct = 100 * nd / diff.size
        axes[row, 4].set_title(f"{titles[4]}\n{nd:,} px ({pct:.3f}%)", fontsize=10)
        axes[row, 4].axis("off")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Overlay grid saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# Plot / chart functions (require matplotlib)
# ═══════════════════════════════════════════════════════════════════════

def plot_speed_comparison(
    latencies: Dict[str, float],
    out_path: str,
    *,
    title: str = "Inference Speed Comparison",
    dpi: int = 200,
) -> str:
    """Horizontal bar chart comparing inference latency across methods.

    Args:
        latencies: ``{method_name: latency_ms}`` dict.
        out_path: Where to save the PNG.
        title: Chart title.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    names = list(latencies.keys())
    vals = list(latencies.values())
    palette = ["#6baed6", "#fd8d3c", "#31a354", "#756bb1", "#d62728"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(names, vals, color=palette[:len(names)], edgecolor="white",
                   linewidth=1.0, height=0.55)

    best = min(vals)
    ax.axvline(x=best, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)

    for bar, v in zip(bars, vals):
        speedup = best / v if v > 0 else 0
        label = f"  {v:.1f} ms"
        if v != best and speedup > 0.01:
            label += f"  ({speedup:.2f}×)"
        else:
            label += "  ← fastest"
        ax.text(bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=11,
                fontweight="bold" if v == best else "normal")

    ax.set_xlabel("Latency (ms) — lower is better", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.45)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Speed chart saved: {out_path}")
    return out_path


def plot_iou_comparison(
    results: Dict[str, dict],
    out_path: str,
    *,
    class_names: Optional[List[str]] = None,
    title: str = "Per-Class IoU Comparison",
    dpi: int = 200,
) -> str:
    """Grouped bar chart of per-class IoU across inference backends.

    Args:
        results: ``{method: {"iou_Sky": 92.1, "iou_Water": 79.4, ...}}``.
                 Keys are matched via ``f"iou_{class_name}"``.
        out_path: Where to save the PNG.
        class_names: Class names to display (default: :data:`CLASS_NAMES`).
        title: Chart title.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if class_names is None:
        class_names = CLASS_NAMES

    methods = list(results.keys())
    x = np.arange(len(class_names))
    w = 0.8 / len(methods)
    palette = ["#6baed6", "#31a354", "#756bb1", "#fd8d3c"]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, meth in enumerate(methods):
        ious = [results[meth].get(f"iou_{cn}", 0) for cn in class_names]
        bars = ax.bar(x + i * w, ious, w, label=meth,
                      color=palette[i % len(palette)],
                      edgecolor="white", linewidth=0.4)
        for bar, iou in zip(bars, ious):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{iou:.1f}", ha="center", fontsize=8, rotation=90)

    ax.set_ylabel("IoU (%)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  IoU chart saved: {out_path}")
    return out_path


def plot_summary_table(
    results: Dict[str, dict],
    out_path: str,
    *,
    rows: Optional[List[Tuple[str, str]]] = None,
    dpi: int = 200,
) -> str:
    """Render a styled summary table as a PNG.

    Args:
        results: ``{method: {metric_key: value, ...}}``.
        out_path: Where to save the PNG.
        rows: List of ``(label, metric_key)`` pairs to include.
              Default: mIoU(fg), mIoU(all), PixelAcc, IoU per class, Latency.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if rows is None:
        rows = [
            ("mIoU (fg)", "miou_foreground"), ("mIoU (all)", "miou_all"),
            ("Pixel Acc.", "pixel_accuracy"),
            ("IoU Sky", "iou_Sky"), ("IoU Water", "iou_Water"),
            ("IoU Person", "iou_Person"), ("Latency", "latency_ms"),
        ]

    methods = list(results.keys())
    row_labels = [r[0] for r in rows]

    cell_text = []
    for _, key in rows:
        cell_text.append([
            f"{results[m].get(key, 0):.1f}{' ms' if key == 'latency_ms' else '%'}"
            for m in methods
        ])

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=methods,
        cellLoc="center", rowLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for j in range(len(methods)):
        table[0, j].set_facecolor("#404040")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        table[i + 1, 0].set_facecolor("#f0f0f0")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Summary table saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# Tensor utilities (moved from utils.py)
# ═══════════════════════════════════════════════════════════════════════

def tensor_to_image(
    tensor: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    """Convert a normalised tensor back to a displayable RGB image.

    Args:
        tensor: ``(C, H, W)`` or ``(B, C, H, W)`` normalised tensor.
        mean: Channel-wise mean (default: ImageNet).
        std: Channel-wise std (default: ImageNet).

    Returns:
        ``(H, W, 3)`` uint8 numpy array.
    """
    if mean is None:
        from skywater_seg.inference import ONNXRuntimeInference
        mean = ONNXRuntimeInference.MEAN
    if std is None:
        from skywater_seg.inference import ONNXRuntimeInference
        std = ONNXRuntimeInference.STD

    if tensor.dim() == 4:
        tensor = tensor[0]

    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# Pre-built LUT for vectorised mask→RGB conversion
_MASK_LUT = np.array([list(CLASS_COLORS_RGB[i]) for i in range(NUM_CLASSES)], dtype=np.uint8)


def mask_to_color(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a class-index mask to an RGB colour image via LUT lookup.

    Accepts both PyTorch tensors and numpy arrays for convenience.

    Args:
        mask: ``(H, W)`` class indices as torch.Tensor or numpy array.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 3:
            mask = mask[0]
        mask = mask.cpu().numpy().astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return _MASK_LUT[mask]


# ═══════════════════════════════════════════════════════════════════════
# Overlay (canonical version — was in inference.py)
# ═══════════════════════════════════════════════════════════════════════


colorize_mask = mask_to_color  # alias -- both convert mask to RGB
def draw_overlay(
    image: Union[str, np.ndarray],
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw segmentation mask overlay with contour outlines.

    Delegates to :func:`overlay_mask` with ``draw_contours=True``
    and ``contour_thickness=2``.  Kept for backward compatibility.

    Args:
        image: Path to image file, or RGB/BGR numpy array ``(H, W, 3)``.
        mask: ``(H, W)`` uint8 class-index mask.
        alpha: Blend strength (0 = original image, 1 = fully colourised).

    Returns:
        BGR numpy array suitable for ``cv2.imwrite`` / ``cv2.imshow``.
    """
    return overlay_mask(image, mask, alpha=alpha,
                        draw_contours=True, contour_thickness=2)


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot functions. "
            "Install with: pip install matplotlib"
        )
