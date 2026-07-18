# Phase 2 — Training

Fine-tune a segmentation model on annotated masks.

## Quick Start

```bash
# 1. Download dataset from HuggingFace
hf download Realcat/skywater --local-dir ./data
unzip data/ADEChallengeData2016.zip -d path/to/

# 2. Train (pick your config)
uv run python train.py --config configs/models/segformer_b2.yaml           # SegFormer B2 (best)
uv run python train.py --config configs/models/convnext_dinov3.yaml        # ConvNeXt + DINOv3
uv run python train.py --config configs/datasets/ade20k_person.yaml        # MobileNetV3, filtered
uv run python train.py --config configs/datasets/ade20k_full.yaml          # ADE20K full
uv run python train.py --config configs/models/mobilenetv3_flatdir.yaml    # Custom data

# CLI overrides (dot notation, auto-typed)
uv run python train.py --config configs/models/mobilenetv3_flatdir.yaml \
    --train.epochs 100 --train.batch_size 8 --train.learning_rate 0.0001

# Resume from checkpoint
uv run python train.py --config configs/models/mobilenetv3_flatdir.yaml \
    --train.resume_from checkpoints/skywater-seg/best_model.pth

# Monitor
uv run tensorboard --logdir checkpoints/skywater-seg/logs
```

## Datasets

The pipeline supports multiple dataset sources:

| Dataset | Format | Classes | Notes |
|---------|--------|---------|-------|
| Custom (flat dir) | `images/*.jpg` + `masks/*_mask.png` | Any | Default mode |
| ADE20K (ADEChallengeData2016) | `images/` + `annotations/` with class remapping | sky, water, person | 150→4 class mapping |
| Cityscapes | `leftImg8bit/` + `gtFine/` subdirectories | sky, water, person | Auto city-split detection |
| Multi-dataset | Mix any of the above with sampling weights | 4 | Combined `MultiDataset` |

### Preparing ADE20K

See [datasets.md](datasets.md) for detailed ADE20K setup instructions and
class mapping.

## Config Presets

| Config | Dataset | Model | Params | Use Case |
|--------|---------|-------|--------|----------|
| `models/mobilenetv3_flatdir.yaml` | Custom flat dir | MobileNetV3-Large | ~5M | Quick start |
| `datasets/ade20k_full.yaml` | ADE20K full | MobileNetV3-Large | ~5M | Cost-effective |
| `datasets/ade20k.yaml` | ADE20K filtered | MobileNetV3-Large | ~5M | Filtered split |
| `models/convnext_dinov3.yaml` | ADE20K filtered | ConvNeXt-Tiny + DINOv3 | ~29M | High quality |
| `datasets/multi_dataset.yaml` | ADE20K + Cityscapes | ConvNeXt-Tiny + DINOv3 | ~29M | Best generalization |
| `models/segformer_b2.yaml` | ADE20K filtered | SegFormer MiT-B2 | ~25M | **Best quality** |

## Model Architecture Options

| Encoder | Params | Weights | Notes |
|---------|--------|---------|-------|
| `timm-mobilenetv3_large_100` | ~5M | ImageNet | Lightweight |
| `timm-mobilenetv3_small_050` | ~2M | ImageNet | Ultra-lightweight |
| `timm-efficientnet-b0` | ~5M | ImageNet | Balanced |
| `timm-efficientnet-b3` | ~12M | ImageNet | Accurate |
| `convnext-tiny` | ~29M | DINOv3 / ImageNet-22K | High quality |
| `convnext-small` | ~50M | DINOv3 / ImageNet-22K | — |
| `convnext-base` | ~89M | DINOv3 / ImageNet-22K | — |
| `mit_b2` (SegFormer) | ~25M | ImageNet | Transformer-based |

All SMP-native encoders (ResNet, EfficientNet, MiT, etc.) are also supported
via the `model.encoder_name` config field.

## Loss Functions

| Loss | Config Key | Description |
|------|-----------|-------------|
| CrossEntropy + Dice | `dice_ce` | **Default.** Balanced per-pixel + region overlap |
| CrossEntropy | `ce` | Per-pixel only |
| Dice | `dice` | Region overlap only |
| Focal | `focal` | Focuses on hard examples, handles class imbalance |
| Jaccard (IoU) | `jaccard` | Direct IoU optimization |

Set via `train.loss` in config.
