"""
Training loop for sky/water segmentation.

Logging:
  - loguru: console + file (training.log) with rich formatting
  - TensorBoard: metrics, gradients, images, weights

Usage: python train.py --config configs/ade20k.yaml
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Suppress spurious LR scheduler warning (we call scheduler after optimizer)
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")

from skywater_seg.config import Config
from skywater_seg.losses import get_loss
from skywater_seg.model import create_model, get_model_info
from skywater_seg.utils import (
    CLASS_COLORS_RGB,
    compute_dice,
    compute_iou,
    compute_pixel_accuracy,
    configure_backend,
    create_scheduler,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    tensor_to_image,
    to_device,
)
from skywater_seg.visualization import colorize_mask, tensor_to_image

# TensorBoard tag hierarchy
TB_LOSS = "Loss"
TB_METRICS = "Metrics"
TB_IOU = "IoU"
TB_GRAD = "Gradients"
TB_WEIGHT = "Weights"
TB_IMAGE = "Images"
TB_LR = "LR"

CLASS_NAMES = {0: "BG", 1: "Sky", 2: "Water", 3: "Person"}


def _setup_logger(output_dir: Path) -> str:
    """Configure loguru: console (color) + file (machine-readable).

    Returns path to the log file.
    """
    logger.remove()  # Remove default handler
    log_path = output_dir / "training.log"

    # Console: colored, compact format
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # File: detailed with timestamps
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention=5,  # Keep 5 rotated log files
        encoding="utf-8",
    )

    return str(log_path)


class Trainer:
    """Training pipeline with loguru console logging + TensorBoard visualization."""

    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = get_device(config.device)
        configure_backend(self.device)
        if self.device.type == "mps":
            config.train.pin_memory = False
        set_seed(config.seed)

        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Logging ----
        self._log_path = _setup_logger(self.output_dir)
        logger.info(f"Log file: {self._log_path}")

        # ---- Model ----
        logger.info("Creating model...")
        self.model = create_model(config)
        self.model.to(self.device)
        info = get_model_info(self.model)
        logger.info(
            f"Model: {config.model.name} + {config.model.encoder_name} | "
            f"Params: {info['total_params']:,} ({info['size_mb_float32']} MB) | "
            f"Classes: {config.model.classes}"
        )

        # ---- Loss, Optimizer, Scheduler ----
        self.criterion = get_loss(config)
        self.criterion.to(self.device)
        self.optimizer = self._create_optimizer()
        self.scaler = GradScaler(
            enabled=(config.train.mixed_precision and self.device.type == "cuda")
        )
        self.scheduler = create_scheduler(self.optimizer, config, len(train_loader))

        # ---- TensorBoard ----
        self.writer = None
        self._log_dir = None
        if config.train.tensorboard:
            self._log_dir = self.output_dir / "logs"
            self._log_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(str(self._log_dir))
            logger.info(f"TensorBoard: {self._log_dir}")

        # ---- State ----
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_iou = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self._sample_batch = None

        if config.train.resume_from:
            self._resume(config.train.resume_from)

    # ================================================================
    # Main Loop
    # ================================================================

    def train(self):
        config = self.config
        cfg = config.train

        logger.info("=" * 60)
        logger.info(f"Training: {config.experiment_name}")
        logger.info(f"  Epochs: {cfg.epochs} | Batch: {cfg.batch_size} | LR: {cfg.learning_rate}")
        logger.info(f"  Loss: {cfg.loss} | Optimizer: {cfg.optimizer} | Scheduler: {cfg.scheduler}")
        logger.info(f"  AMP: {cfg.mixed_precision} | Device: {self.device}")
        logger.info(f"  Log: {self._log_path}")
        logger.info("=" * 60)

        logger.info(
            f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}"
        )

        for epoch in range(self.current_epoch, cfg.epochs):
            self.current_epoch = epoch

            # ---- Train ----
            train_loss, train_metrics = self._train_epoch(epoch)

            # TensorBoard: epoch-level metrics
            if self.writer:
                self.writer.add_scalar("Loss/epoch/train", train_loss, epoch)
                self.writer.add_scalar(TB_LR, self.optimizer.param_groups[0]["lr"], epoch)
                # Per-class IoU (training)
                for c in range(config.model.classes):
                    key = f"iou_class_{c}"
                    if key in train_metrics:
                        name = CLASS_NAMES.get(c, f"class_{c}")
                        self.writer.add_scalar(f"IoU/train/{name}", train_metrics[key], epoch)
                # Validation summary metrics
                self.writer.add_scalar(f"{TB_METRICS}/train/mIoU", train_metrics["mIoU"], epoch)
                self.writer.add_scalar(f"{TB_METRICS}/train/mDice", train_metrics["mDice"], epoch)

            # ---- Validate ----
            if (epoch + 1) % cfg.val_every == 0:
                # Clear CUDA cache to reduce fragmentation (critical on Windows/limited RAM)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                val_metrics = self._validate(epoch, log_images=True)

                current_iou = val_metrics["miou"]
                is_best = current_iou > self.best_val_iou
                if is_best:
                    self.best_val_iou = current_iou
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    best_path = self.output_dir / "best_model.pth"
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch + 1,
                        {"miou": self.best_val_iou},
                        str(best_path),
                        is_best=True,
                        model_meta=self._model_meta(),
                    )
                    logger.info(f"Best model saved: {best_path} (mIoU={self.best_val_iou:.4f})")
                else:
                    self.patience_counter += 1

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar("Loss/epoch/val", val_metrics["loss"], epoch)
                    self.writer.add_scalar(f"{TB_METRICS}/val/mIoU", current_iou, epoch)
                    self.writer.add_scalar(f"{TB_METRICS}/val/mDice", val_metrics["mdice"], epoch)
                    self.writer.add_scalar(
                        f"{TB_METRICS}/val/PixelAcc", val_metrics["pixel_acc"], epoch
                    )
                    for c in range(config.model.classes):
                        key = f"iou_class_{c}"
                        if key in val_metrics:
                            name = CLASS_NAMES.get(c, f"class_{c}")
                            self.writer.add_scalar(f"{TB_IOU}/{name}", val_metrics[key], epoch)

                self._log_val_progress(epoch, train_loss, val_metrics, is_best)

                # Early stopping
                if self.patience_counter >= cfg.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} (patience={cfg.early_stopping_patience})"
                    )
                    break

            # ---- Periodic: weight histograms ----
            if self.writer and (epoch + 1) % 10 == 0:
                self._log_weight_histograms(epoch)

            # ---- Checkpoint ----
            if (epoch + 1) % cfg.save_every == 0:
                ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    {"miou": self.best_val_iou},
                    str(ckpt_path),
                    model_meta=self._model_meta(),
                )
                logger.info(f"Checkpoint saved: {ckpt_path}")

        # ---- Done ----
        logger.info("=" * 60)
        logger.info(
            f"Training Complete! Best mIoU: {self.best_val_iou:.4f} (epoch {self.best_epoch + 1})"
        )
        logger.info(f"  TensorBoard: {self._log_dir}")
        logger.info(f"  Log file: {self._log_path}")
        logger.info("=" * 60)

        torch.save(self.model.state_dict(), self.output_dir / "final_model.pth")
        logger.info(f"Final model saved: {self.output_dir / 'final_model.pth'}")
        if self.writer:
            self.writer.close()

    # ================================================================
    # Train Epoch
    # ================================================================

    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        cfg = self.config.train

        total_loss = 0.0
        num_batches = len(self.train_loader)

        train_preds = []
        train_targets = []
        train_loss_samples = []

        # ---- Progress bar ----
        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Train {epoch + 1:3d}",
            unit="batch",
            ncols=120,
            leave=False,
        )

        for batch_idx, batch in pbar:
            batch = to_device(batch, self.device)
            images, masks = batch["image"], batch["mask"]

            # Forward
            with autocast(device_type=self.device.type, enabled=cfg.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, masks) / cfg.gradient_accumulation

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient step
            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
                if self.writer and (self.global_step % 50 == 0):
                    total_norm = self._compute_grad_norm()
                    self.writer.add_scalar(f"{TB_GRAD}/L2_norm", total_norm, self.global_step)

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.writer and (self.global_step % 200 == 0):
                    self._log_gradient_histograms(self.global_step)

                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * cfg.gradient_accumulation

            # Sample metrics (every 50 batches for representative estimate)
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    train_preds.append(preds.cpu())
                    train_targets.append(masks.cpu())
                    train_loss_samples.append(loss.item() * cfg.gradient_accumulation)

            # Update progress bar
            lr = self.optimizer.param_groups[0]["lr"]
            loss_scalar = loss.item() * cfg.gradient_accumulation
            pbar.set_postfix_str(f"loss={loss_scalar:.4f} lr={lr:.2e}")

            # TensorBoard: per optimizer step (log every step for fine-grained curve)
            if self.writer:
                self.writer.add_scalar("Loss/step", loss_scalar, self.global_step)

        avg_loss = total_loss / num_batches

        # Compute train IoU
        if train_preds:
            all_p = torch.cat(train_preds, dim=0)
            all_t = torch.cat(train_targets, dim=0)
            iou = compute_iou(
                all_p, all_t, self.config.model.classes, self.config.data.ignore_index
            )
            dice = compute_dice(
                all_p, all_t, self.config.model.classes, self.config.data.ignore_index
            )
        else:
            iou, dice = {"miou": 0.0}, {"mdice": 0.0}

        train_metrics = {
            "mIoU": iou.get("miou", 0.0),
            "mDice": dice.get("mdice", 0.0),
            **{k: v for k, v in iou.items() if k.startswith("iou_class_")},
        }

        logger.info(
            f"Epoch {epoch + 1:3d} train | loss={avg_loss:.4f} | "
            f"mIoU={train_metrics['mIoU']:.4f} | mDice={train_metrics['mDice']:.4f} | "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )

        return avg_loss, train_metrics

    # ================================================================
    # Validation
    # ================================================================

    @torch.no_grad()
    def _validate(self, epoch: int, log_images: bool = True) -> Dict[str, float]:
        self.model.eval()
        cfg = self.config

        val_loss = 0.0
        all_preds, all_targets = [], []

        pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc=f"Val   {epoch + 1:3d}",
            unit="batch",
            ncols=100,
            leave=False,
        )

        for batch_idx, batch in pbar:
            batch = to_device(batch, self.device)
            images, masks = batch["image"], batch["mask"]

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            pbar.set_postfix_str(f"loss={loss.item():.4f}")

            if log_images and self.writer and batch_idx == 0:
                self._log_predictions(epoch, images, masks, preds)

        val_loss /= len(self.val_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        iou_metrics = compute_iou(all_preds, all_targets, cfg.model.classes, cfg.data.ignore_index)
        dice_metrics = compute_dice(
            all_preds, all_targets, cfg.model.classes, cfg.data.ignore_index
        )
        pixel_acc = compute_pixel_accuracy(all_preds, all_targets, cfg.data.ignore_index)

        return {
            "loss": round(val_loss, 4),
            "miou": iou_metrics["miou"],
            "mdice": dice_metrics["mdice"],
            "pixel_acc": round(pixel_acc, 4),
            **iou_metrics,
            **dice_metrics,
        }

    # ================================================================
    # TensorBoard helpers
    # ================================================================

    def _compute_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def _log_gradient_histograms(self, step: int):
        for name, param in self.model.named_parameters():
            if param.grad is not None and ("encoder" in name or "decoder" in name):
                if any(k in name for k in ["conv", "bn", "aspp", "classifier"]):
                    self.writer.add_histogram(f"{TB_GRAD}/{name}", param.grad.data.cpu(), step)

    def _log_weight_histograms(self, epoch: int):
        for name, param in self.model.named_parameters():
            if "encoder" in name or "decoder" in name:
                if any(k in name for k in ["conv", "bn", "aspp", "classifier"]):
                    self.writer.add_histogram(f"{TB_WEIGHT}/{name}", param.data.cpu(), epoch)

    def _log_predictions(
        self,
        epoch: int,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        max_samples: int = 8,
    ):
        """Log GT vs Prediction overlays to TensorBoard."""
        import cv2

        n = min(images.size(0), max_samples)
        SKY = np.array(CLASS_COLORS_RGB[1], dtype=np.uint8)
        WTR = np.array(CLASS_COLORS_RGB[2], dtype=np.uint8)
        PRS = np.array(CLASS_COLORS_RGB[3], dtype=np.uint8)
        RED = np.array([255, 60, 60], dtype=np.uint8)
        YLW = np.array([255, 220, 50], dtype=np.uint8)
        DIM = np.array([30, 30, 30], dtype=np.uint8)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        def _blend(bg, mask, alpha=0.45):
            c = colorize_mask(mask).astype(np.float32)
            return (bg.astype(np.float32) * (1 - alpha) + c * alpha).clip(0, 255).astype(np.uint8)

        def _highlight(mask, cls_id, color):
            out = np.full((*mask.shape, 3), DIM, dtype=np.uint8)
            out[mask == cls_id] = color
            return out

        def _err(gt, pd, cls_id):
            err = np.zeros((*gt.shape, 3), dtype=np.uint8)
            err[(gt == cls_id) & (pd != cls_id)] = RED
            err[(gt != cls_id) & (pd == cls_id)] = YLW
            return err

        def _label(im, text):
            cv2.putText(im, text, (4, 14), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            return im

        for i in range(n):
            img = cv2.resize(tensor_to_image(images[i].cpu()), masks.shape[2:][::-1])
            gt = masks[i].cpu().numpy().astype(np.uint8)
            pd = preds[i].cpu().numpy().astype(np.uint8)

            row1 = np.hstack(
                [
                    _label(img.copy(), "Input"),
                    _label(_blend(img, gt), "GT"),
                    _label(_blend(img, pd), "Pred"),
                ]
            )
            row2 = np.hstack(
                [
                    _label(_highlight(gt, 1, SKY), "Sky GT"),
                    _label(_highlight(pd, 1, SKY), "Sky Pred"),
                    _label(_err(gt, pd, 1), "Sky Err (red=miss, yellow=false)"),
                ]
            )
            row3 = np.hstack(
                [
                    _label(_highlight(gt, 2, WTR), "Water GT"),
                    _label(_highlight(pd, 2, WTR), "Water Pred"),
                    _label(_err(gt, pd, 2), "Water Err (red=miss, yellow=false)"),
                ]
            )
            rows = [row1, row2, row3]
            if self.config.model.classes >= 4:
                rows.append(
                    np.hstack(
                        [
                            _label(_highlight(gt, 3, PRS), "Person GT"),
                            _label(_highlight(pd, 3, PRS), "Person Pred"),
                            _label(_err(gt, pd, 3), "Person Err (red=miss, yellow=false)"),
                        ]
                    )
                )
            self.writer.add_image(
                f"{TB_IMAGE}/sample_{i}", np.vstack(rows), epoch, dataformats="HWC"
            )

    # ---- Internal helpers ----

    def _create_optimizer(self):
        cfg = self.config.train
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=tuple(cfg.betas),
            )
        elif cfg.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=tuple(cfg.betas),
            )
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _resume(self, checkpoint_path: str):
        logger.info(f"Resuming from: {checkpoint_path}")
        ckpt = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, device=self.device
        )
        self.current_epoch = ckpt.get("epoch", 0)
        self.best_val_iou = ckpt.get("metrics", {}).get("miou", 0.0)
        logger.info(f"  Resumed at epoch {self.current_epoch}, best mIoU={self.best_val_iou:.4f}")

    def _model_meta(self) -> dict:
        """Build metadata for self-contained checkpoint inference."""
        cfg = self.config
        return {
            "model_name": cfg.model.name,
            "encoder_name": cfg.model.encoder_name,
            "encoder_weights": cfg.model.encoder_weights,
            "classes": cfg.model.classes,
            "in_channels": cfg.model.in_channels,
            "image_size": list(cfg.data.image_size),
            "num_classes": cfg.data.num_classes,
            "mean": cfg.data.mean,
            "std": cfg.data.std,
            "class_mapping": cfg.data.class_mapping,
        }

    def _log_val_progress(
        self, epoch: int, train_loss: float, metrics: Dict[str, float], is_best: bool
    ):
        marker = " [BEST]" if is_best else ""
        logger.info(
            f"Epoch {epoch + 1:3d} val{marker} | "
            f"loss={metrics['loss']:.4f} | "
            f"mIoU={metrics['miou']:.4f} | "
            f"mDice={metrics['mdice']:.4f} | "
            f"Acc={metrics['pixel_acc']:.4f}"
        )
        parts = []
        for c in range(self.config.model.classes):
            key = f"iou_class_{c}"
            if key in metrics:
                parts.append(f"{CLASS_NAMES.get(c, f'c{c}')}:{metrics[key]:.3f}")
        logger.info(f"  IoU: {' | '.join(parts)}")


def train(config: Config):
    """Entry point: create dataloaders and run training."""
    from skywater_seg.dataset import create_dataloaders

    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset):,} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_loader.dataset):,} samples, {len(val_loader)} batches")

    trainer = Trainer(config, train_loader, val_loader)
    trainer.train()
    return trainer
