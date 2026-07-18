"""SkyWater Segmentation Package — inference, ONNX export, and visualization."""

__version__ = "0.3.0"

# ── Config ────────────────────────────────────────────────────────────
from skywater_seg.config import Config, get_default_config, cli_to_dotlist

# ── Model ─────────────────────────────────────────────────────────────
from skywater_seg.model import SkyWaterSegModel, create_model, get_model_info

# ── Inference ─────────────────────────────────────────────────────────
from skywater_seg.inference import (
    SegmentationInference,
    ONNXRuntimeInference,
    export_onnx,
    convert_onnx_fp16,
    load_model,
    segment,
)

# ── Visualization ─────────────────────────────────────────────────────
from skywater_seg.visualization import (
    CLASS_COLORS_RGB,
    CLASS_NAMES,
    NUM_CLASSES,
    class_colors_bgr,
    colorize_mask,
    mask_to_color,
    draw_overlay,
    overlay_mask,
    tensor_to_image,
    make_comparison_grid,
    make_overlay_grid,
    plot_speed_comparison,
    plot_iou_comparison,
    plot_summary_table,
)

# ── Utilities ─────────────────────────────────────────────────────────
from skywater_seg.utils import (
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    get_device,
    configure_backend,
    set_seed,
)

__all__ = [
    # config
    "Config",
    "get_default_config",
    "cli_to_dotlist",
    # model
    "SkyWaterSegModel",
    "create_model",
    "get_model_info",
    # inference
    "SegmentationInference",
    "ONNXRuntimeInference",
    "export_onnx",
    "convert_onnx_fp16",
    "load_model",
    "segment",
    # visualization
    "CLASS_COLORS_RGB",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "class_colors_bgr",
    "colorize_mask",
    "mask_to_color",
    "draw_overlay",
    "overlay_mask",
    "tensor_to_image",
    "make_comparison_grid",
    "make_overlay_grid",
    "plot_speed_comparison",
    "plot_iou_comparison",
    "plot_summary_table",
    # utils
    "compute_iou",
    "compute_dice",
    "compute_pixel_accuracy",
    "get_device",
    "configure_backend",
    "set_seed",
]
