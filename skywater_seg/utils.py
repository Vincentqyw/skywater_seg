"""
Utility functions: metrics, visualization, logging, device management.
"""

import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Shared class-color palette (single source of truth)
# ============================================================

CLASS_COLORS_RGB = {
    0: (0, 0, 0),         # background: black
    1: (255, 140, 0),     # sky: orange
    2: (0, 200, 255),     # water: cyan
    3: (255, 60, 60),     # person: red
}


def class_colors_bgr() -> dict:
    """Return the shared palette in BGR order for OpenCV functions."""
    return {k: (b, g, r) for k, (r, g, b) in CLASS_COLORS_RGB.items()}


# ============================================================
# Metrics
# ============================================================

def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """Compute per-class and mean IoU.

    Args:
        pred: (B, H, W) predicted class indices
        target: (B, H, W) ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        Dict with "iou_class_X" and "miou" (mean IoU over foreground classes)
    """
    ious = {}
    mask = (target != ignore_index)

    for c in range(num_classes):
        pred_c = (pred == c) & mask
        target_c = (target == c) & mask

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            ious[f"iou_class_{c}"] = round((intersection / union).item(), 4)
        else:
            ious[f"iou_class_{c}"] = float("nan")  # class absent from batch

    # Mean IoU over foreground classes (skip NaN = absent from batch)
    fg_vals = [v for c, v in ious.items()
               if c.startswith("iou_class_") and c != "iou_class_0" and not np.isnan(v)]
    ious["miou"] = round(np.mean(fg_vals) if fg_vals else 0.0, 4)

    return ious


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """Compute per-class and mean Dice coefficient."""
    dices = {}
    mask = (target != ignore_index)

    for c in range(num_classes):
        pred_c = (pred == c) & mask
        target_c = (target == c) & mask

        intersection = (pred_c & target_c).sum().float()
        total = pred_c.sum().float() + target_c.sum().float()

        if total > 0:
            dices[f"dice_class_{c}"] = round((2.0 * intersection / total).item(), 4)
        else:
            dices[f"dice_class_{c}"] = float("nan")

    fg_vals = [v for c, v in dices.items()
               if c.startswith("dice_class_") and c != "dice_class_0" and not np.isnan(v)]
    dices["mdice"] = round(np.mean(fg_vals) if fg_vals else 0.0, 4)

    return dices


def compute_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """Compute pixel-wise accuracy (ignoring the ignore_index)."""
    mask = (target != ignore_index)
    correct = ((pred == target) & mask).sum().float()
    total = mask.sum().float()
    return float((correct / (total + 1e-6)).item())


# ============================================================
# Visualization
# ============================================================

def tensor_to_image(tensor: torch.Tensor, mean=None, std=None) -> np.ndarray:
    """Convert normalized tensor to displayable RGB image.

    Args:
        tensor: (C, H, W) or (B, C, H, W) normalized tensor
        mean: Normalization mean
        std: Normalization std

    Returns:
        (H, W, 3) uint8 numpy array
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if tensor.dim() == 4:
        tensor = tensor[0]

    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    img = torch.clamp(img, 0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


def mask_to_color(mask: torch.Tensor) -> np.ndarray:
    """Convert class-index mask to color visualization.

    Color scheme (RGB):
        0 (background): black  (0, 0, 0)
        1 (sky):         orange (255, 140, 0)
        2 (water):       cyan   (0, 200, 255)
        3 (person):      red    (255, 60, 60)

    Args:
        mask: (H, W) class indices

    Returns:
        (H, W, 3) uint8 RGB image
    """
    if mask.dim() == 3:
        mask = mask[0]

    mask_np = mask.cpu().numpy().astype(np.uint8)

    h, w = mask_np.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_RGB.items():
        if cls_id == 0:
            continue
        vis[mask_np == cls_id] = color
    return vis


# ============================================================
# Device management
# ============================================================

def get_device(device_str: str = "cuda") -> torch.device:
    """Get the best available device."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "cuda" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def configure_backend(device: torch.device):
    """Apply PyTorch backend optimizations for the given device.

    CUDA: enables Flash Attention / memory-efficient SDP (PyTorch 2.0+).
    Called once at process start; idempotent.
    """
    if device.type != "cuda":
        return
    if not hasattr(torch.backends.cuda, "enable_flash_sdp"):
        return
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def to_device(data, device: torch.device):
    """Recursively move tensors to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    return data


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Checkpoint management
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    path: str,
    is_best: bool = False,
    model_meta: Optional[Dict] = None,
):
    """Save training checkpoint with model metadata for self-contained inference."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if model_meta:
        checkpoint["model_meta"] = model_meta

    torch.save(checkpoint, path)
    if is_best:
        best_path = str(Path(path).parent / "best_model.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# ============================================================
# Learning rate schedulers
# ============================================================

def create_scheduler(optimizer, config, steps_per_epoch: int):
    """Create learning rate scheduler from config."""
    total_steps = steps_per_epoch * config.train.epochs
    warmup_steps = steps_per_epoch * config.train.lr_warmup_epochs

    if config.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
            eta_min=config.train.lr_min,
        )
    elif config.train.scheduler == "poly":
        # Polynomial decay: lr = (lr - lr_min) * (1 - iter/max_iter)^power + lr_min
        lambda_poly = lambda step: (
            (1 - step / (total_steps - warmup_steps)) ** config.train.poly_power
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_poly)
    elif config.train.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1,
        )
    elif config.train.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.train.scheduler}")

    return scheduler
