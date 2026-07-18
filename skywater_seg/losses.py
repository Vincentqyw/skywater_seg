from __future__ import annotations

"""
Loss functions for semantic segmentation.

Provides standard and combined losses suitable for sky/water segmentation:
  - CrossEntropy (with optional class weights)
  - Dice Loss
  - Focal Loss
  - Combined Dice + CE Loss
  - Jaccard (IoU) Loss
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """

    def __init__(
        self,
        mode: str = "multiclass",
        smooth: float = 1.0,
        ignore_index: int = 255,
    ):
        """
        Args:
            mode: "binary", "multiclass", or "multilabel"
            smooth: Laplace smoothing factor
            ignore_index: Class index to ignore in computation
        """
        super().__init__()
        self.mode = mode
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        Returns:
            scalar loss
        """
        if self.mode == "multiclass":
            pred = F.softmax(pred, dim=1)

            # One-hot encode target, handle ignore_index
            num_classes = pred.shape[1]
            mask = target != self.ignore_index
            target_one_hot = (
                F.one_hot(torch.clamp(target, 0, num_classes - 1), num_classes)
                .permute(0, 3, 1, 2)
                .float()
            )  # (B, C, H, W)

            # Apply mask
            mask = mask.unsqueeze(1).float()  # (B, 1, H, W)
            pred = pred * mask
            target_one_hot = target_one_hot * mask

            intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))
            union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

            dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
            # Exclude background (class 0) from loss if desired
            dice = dice_per_class[1:].mean()  # Average over foreground classes
        else:
            # Binary mode
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss: -α(1-p_t)^γ * log(p_t)

    Reduces the contribution of easy examples, focusing on hard ones.
    Useful when there's class imbalance (sky/water regions are often minority).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        logp = F.log_softmax(pred, dim=1)
        pt = torch.exp(logp)

        # Gather log probabilities for target classes
        logp = logp.transpose(1, 2).transpose(2, 3)  # (B, H, W, C)
        logp = logp.reshape(-1, logp.size(-1))
        target_flat = target.reshape(-1)

        # Ignore index
        mask = target_flat != self.ignore_index
        logp = logp[mask]
        target_flat = target_flat[mask]

        ce_loss = F.nll_loss(logp, target_flat, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


class JaccardLoss(nn.Module):
    """Jaccard / IoU Loss = 1 - IoU."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]

        # One-hot target
        mask = target != self.ignore_index
        target_oh = (
            F.one_hot(torch.clamp(target, 0, num_classes - 1), num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        mask = mask.unsqueeze(1).float()

        pred = pred * mask
        target_oh = target_oh * mask

        intersection = (pred * target_oh).sum(dim=(0, 2, 3))
        union = (pred + target_oh - pred * target_oh).sum(dim=(0, 2, 3))

        iou_per_class = (intersection + self.smooth) / (union + self.smooth)
        iou = iou_per_class[1:].mean()  # Exclude background

        return 1.0 - iou


class CombinedLoss(nn.Module):
    """Weighted combination of CrossEntropy and Dice loss.

    L = λ_ce * CE + λ_dice * Dice

    This is the recommended loss for sky/water segmentation:
    - CE provides per-pixel classification signal
    - Dice handles class imbalance and optimizes for region overlap
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[list] = None,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is not None:
            class_weights = torch.tensor(class_weights).float()

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(mode="multiclass", smooth=smooth, ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss(config) -> nn.Module:
    """Factory function to create loss from config."""
    loss_name = config.train.loss

    class_weights = config.train.class_weights
    if class_weights is not None:
        class_weights = torch.tensor(class_weights).float()

    if loss_name == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=config.data.ignore_index,
        )
    elif loss_name == "dice":
        return DiceLoss(mode="multiclass", ignore_index=config.data.ignore_index)
    elif loss_name == "focal":
        return FocalLoss(ignore_index=config.data.ignore_index)
    elif loss_name == "jaccard":
        return JaccardLoss(ignore_index=config.data.ignore_index)
    elif loss_name == "dice_ce":
        return CombinedLoss(
            ce_weight=config.train.ce_weight,
            dice_weight=config.train.dice_weight,
            class_weights=config.train.class_weights,
            ignore_index=config.data.ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
