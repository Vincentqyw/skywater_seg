# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sky-Water-Person Segmentation Pipeline — a three-phase system that automatically generates segmentation masks (Grounding DINO + SAM), trains a lightweight model on them, and deploys for fast inference. Built for NVIDIA GPUs and Apple Silicon (MPS/CoreML), managed with `uv`.

**Target:** Mask out sky, water, and person regions in images to eliminate their interference with SfM (Structure from Motion) and image matching pipelines.

**Version:** 0.3.0

## Environment & Package Management

This project uses **uv** (not pip). All commands run through `uv run`.

```bash
uv sync                          # Install all dependencies
uv sync --group annotate         # Only annotation deps (Grounding DINO + SAM)
uv sync --group train            # Only training deps
uv sync --group deploy           # Only ONNX/CoreML export deps
uv add <package>                 # Add a dependency
```

## Essential Commands

### Full Pipeline

```bash
uv run python run_pipeline.py --image-dir data/images                          # Full pipeline
uv run python run_pipeline.py --image-dir data/images --annotate-only          # Phase 1 only
uv run python run_pipeline.py --image-dir data/images --train-only             # Phase 2 only
uv run python run_pipeline.py --export-only --checkpoint checkpoints/skywater-seg/best_model.pth  # Phase 3 only
```

### Auto-Annotation (Phase 1)

```bash
uv run python scripts/auto_annotate.py -i data/images -o data/masks            # Directory of images
uv run python scripts/auto_annotate.py -i test.jpg -o ./output                 # Single image
uv run python scripts/auto_annotate.py -i data/images -o data/masks --gdino-model tiny --sam-model vit_b --fast  # Low-memory / MacBook fast mode
```

### Training (Phase 2)

```bash
uv run python train.py --config configs/default.yaml                           # Custom flat-dir dataset
uv run python train.py --config configs/ade_challenge.yaml                     # ADE20K full (4-class, small footprint)
uv run python train.py --config configs/ade20k_person.yaml                     # ADE20K filtered split (sky/water/person)
uv run python train.py --config configs/convnext_dinov3.yaml                   # ConvNeXt-Tiny + DINOv3 (high quality)
uv run python train.py --config configs/multi_dataset.yaml                     # ADE20K + Cityscapes mixed
uv run python train.py --config configs/default.yaml --train.batch_size=8 --train.epochs=50  # CLI overrides (dot notation)
uv run python train.py --config configs/default.yaml --train.resume_from checkpoints/skywater-seg/best_model.pth  # Resume
uv run tensorboard --logdir checkpoints/skywater-seg/logs                      # Monitor training
```

CLI overrides use dot-notation (`--train.batch_size 8`) and are auto-typed to match the config dataclass field type.

### Inference & Export (Phase 3)

```bash
# PyTorch inference
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth -i test.jpg

# ONNX Runtime inference (no PyTorch needed)
uv run python inference.py --onnx checkpoints/skywater-seg/skywater_seg.onnx -i test.jpg

# Export: PyTorch → ONNX
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth --export-onnx skywater_seg.onnx

# Export: ONNX → CoreML (macOS only)
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth --export-coreml skywater_seg.mlpackage

# Export: PyTorch → TorchScript
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth --export-torchscript skywater_seg.pt
```

### Data Preparation Scripts

```bash
# ADE20K → sky/water masks (legacy, 3-class)
uv run python scripts/extract_ade20k.py --ade-root /path/to/ADE20K_2021_17_01 --out-dir data/ade20k_skywater --splits training validation

# ADE20K → filter for sky/water/person images, generate train.txt / val.txt
uv run python scripts/prepare_ade20k_person.py
```

### Package CLI Entry Points

Registered in `pyproject.toml` via `[project.scripts]`:
```bash
skywater-annotate -i data/images -o data/masks
skywater-train --config configs/default.yaml
skywater-infer --checkpoint model.pth --input test.jpg
```

## Architecture

### Three-Phase Pipeline (`run_pipeline.py`)

```
Phase 1: Auto-Annotation     Phase 2: Training                Phase 3: Deployment
Grounding DINO + SAM    →    DeepLabV3+ / ConvNeXt       →    ONNX / CoreML (ANE)
(text→boxes→masks)           (~5M–30M params, CUDA/MPS)       (<5ms inference)
```

`run_pipeline.py` orchestrates all three via subprocess calls. It auto-detects Apple Silicon and picks optimal model defaults (tiny GDINO, mobile SAM, FP16, batch_size=8).

### Package Structure (`skywater_seg/`)

| Module | Role |
|---|---|
| `config.py` | Typed configuration via dataclasses (`DataConfig`, `ModelConfig`, `TrainConfig`, `DatasetConfig`, `Config`). Loads from YAML, supports dot-notation CLI overrides, multi-dataset sources. |
| `dataset.py` | `SkyWaterDataset` — loads image+mask pairs with class remapping, Cityscapes auto-detection, subdirectory mode. `MultiDataset` wrapper for mixed-dataset training with weighted sampling. `create_dataloaders()` handles train/val split across single and multi-dataset modes. MPS-aware (disables `pin_memory`). |
| `model.py` | Factory wrapping `segmentation-models-pytorch`. Supports DeepLabV3+, U-Net, FPN, PSPNet, PAN, Linknet with all SMP encoders. Extended encoders: ConvNeXt (tiny/small/base) via timm with DINOv3-distilled or ImageNet-22K weights. Includes named presets (`lightweight`, `ultra-lightweight`, `balanced`, `accurate`, `convnext_dinov3`). |
| `losses.py` | `DiceLoss`, `FocalLoss`, `JaccardLoss`, `CombinedLoss` (CE + Dice). Default: `dice_ce` with equal weights. Ignore index 255 support throughout. |
| `trainer.py` | `Trainer` class with AMP, gradient accumulation, gradient clipping, early stopping, checkpoint management. Logging via loguru (console + rotating file) and TensorBoard (loss, per-class IoU, Dice, pixel accuracy, gradient/weight histograms, prediction overlays with error maps). |
| `inference.py` | `SegmentationInference` (PyTorch), `ONNXRuntimeInference` (no PyTorch dep). Both return `{mask, sky_mask, water_mask}`. Supports batch inference, CRF post-processing, ONNX/TorchScript export. |
| `coreml_export.py` | ONNX→CoreML conversion via `coremltools`, direct PyTorch→CoreML tracing, and `CoreMLInference` class. macOS-only. |
| `utils.py` | Metrics (IoU, Dice, pixel accuracy), device management (`get_device` falls back cuda→mps→cpu), visualization (`tensor_to_image`, `mask_to_color`), checkpoint save/load, LR scheduler factory. |
| `cli.py` | Thin wrappers delegating to scripts/train/inference for the package console_scripts entry points. |

### Scripts (`scripts/`)

- **`auto_annotate.py`** (41KB): Full Grounding DINO + SAM annotation pipeline. Text prompts → bounding boxes (GDINO) → pixel masks (SAM). Supports custom class definitions via JSON (`custom_classes_example.json`). Outputs multi-class mask PNGs (0=bg, 1=sky, 2=water, 3=person) plus visualization overlays and summary JSON.
- **`extract_ade20k.py`**: Converts ADE20K 2021 dataset annotations to sky/water masks compatible with the training pipeline. Matches sky and water objects by WordNet name against predefined name sets.
- **`prepare_ade20k_person.py`**: Filters ADEChallengeData2016 annotations for images containing sky/water/person classes, generates `train.txt`/`val.txt` split files. Used as a preprocessing step before training with `ade20k_person.yaml` or `convnext_dinov3.yaml`.
- **`gen_readme_figures.py`**: Generates paper-style 2×2 comparison figures (Input/GT/Overlay+Contours/Prediction) from ADE20K val + SkySeg test images. Outputs fixed-size 1120×700 panels to `results/<name>/figure.jpg`.
- **`eval_segformer_b2.py`**: Computes per-class IoU, Dice, Precision, Recall on the ADE20K validation set.

### Configuration System

- Hierarchical dataclasses: `Config` → `DataConfig`, `ModelConfig`, `TrainConfig`, `DatasetConfig`
- **Single dataset mode**: set `data.image_dir` and `data.mask_dir` directly
- **Multi-dataset mode**: populate `datasets` list with `DatasetConfig` entries, optional `mix_weights` for weighted sampling
- YAML files in `configs/`:
  - `default.yaml` — custom flat-dir dataset, 3-class (bg/sky/water)
  - `ade_challenge.yaml` — ADE20K full, 4-class, 256px, minimal memory
  - `ade20k_person.yaml` — ADE20K filtered via `prepare_ade20k_person.py`, 4-class, MobileNetV3
  - `convnext_dinov3.yaml` — ADE20K filtered, ConvNeXt-Tiny + DINOv3, 4-class, high quality
  - `multi_dataset.yaml` — ADE20K + Cityscapes mixed, ConvNeXt-Tiny + DINOv3, weighted sampling
  - `segformer_b2.yaml` — ADE20K filtered, SegFormer MiT-B2, 4-class, transformer-based, best quality
- CLI overrides use dot-notation and auto-type-cast to match the dataclass field types (see `train.py:apply_dot_updates`)
- Saved config is written to `{output_dir}/{experiment_name}/config.yaml` on each training run

### Data Format

- **Input images**: Standard formats (jpg, png, tif, bmp)
- **Masks**: Single-channel PNG with pixel values 0=background, 1=sky, 2=water, 3=person
- **Mask naming** (flat-dir mode): `{image_stem}_mask.png` (matched by the dataset loader, falls back to `{stem}.png`)
- **Class remapping**: `class_mapping` dict in config maps arbitrary source class IDs → target indices (e.g. ADE20K 150 classes → 4 classes)
- **Split files**: Optional `train.txt` / `val.txt` with one image filename per line
- **Subdirectory mode**: Auto-detects `training/` and `validation/` subdirectories under `image_dir` and `mask_dir`
- **Cityscapes mode**: Set `cityscapes: true` for auto city-split directory layout detection

### Model Export Chain

```
PyTorch (.pth) → ONNX (.onnx) → CoreML (.mlpackage) → Apple Neural Engine
                → TorchScript (.pt)
                → TensorRT (.trt) via trtexec CLI
```

All exports now go through `inference.py` flags: `--export-onnx`, `--export-coreml`, `--export-torchscript`.

### Apple Silicon Optimizations

The codebase has explicit MPS/ANE awareness throughout:
- `get_device()` falls back cuda → mps → cpu
- `pin_memory` is disabled on MPS (not supported)
- `run_pipeline.py` auto-detects Apple Silicon and selects tiny/fast models, FP16, batch_size=8, 768px resolution
- CoreML export targets Apple Neural Engine for ~3ms inference (vs ~12ms MPS, ~50ms CPU)

## Key Technical Details

- **Default model**: DeepLabV3+ with `timm-mobilenetv3_large_100` encoder (~5M params, ImageNet pretrained)
- **High-quality model**: DeepLabV3+ with `convnext-tiny` encoder (~29M params, DINOv3-distilled weights from Meta LVD-1689M)
- **Best model**: SegFormer MiT-B2 (~24.7M params, ImageNet pretrained, transformer-based). Fine-tuned on ADE20K sky/water/person. **mIoU 94.7%, Sky IoU 92.1%, Pixel Accuracy 97.3%** on 1,111 val images.
- **Default loss**: `CombinedLoss` (0.5×CrossEntropy + 0.5×Dice), ignores index 255
- **Normalization**: ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Input size**: 512×512 default (configurable; 256×256 for memory-constrained setups)
- **4 classes**: 0=background, 1=sky, 2=water, 3=person
- **Dataset**: `SkyWaterDataset` supports flat directories, subdirectory splits, Cityscapes auto-detection, and class remapping. `_load_mask()` tries multiple naming conventions before defaulting to all-zeros.
- **Multi-dataset**: `MultiDataset` concatenates multiple `SkyWaterDataset` instances with optional per-dataset sampling weights (`mix_weights`).
- **Validation split**: If no explicit `.txt` files or subdirectories given, random 85/15 split seeded by `config.seed`
- **Checkpoint format**: Dict with `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `metrics`, `timestamp`
- **Inference result dict**: `{mask: (H,W) uint8, sky_mask: bool, water_mask: bool, probs?: (C,H,W) float32}`
- **Logging**: loguru for console + rotating file logs; TensorBoard for metrics, per-class IoU, gradients, weights, and prediction overlays with error maps
- **Class weights**: Optional per-class weights in config (e.g. `[0.3, 2.0, 2.0, 2.0]` to down-weight background, up-weight foreground)
- **Extended encoders**: ConvNeXt (tiny/small/base) built via timm with a custom stem layer so the 4-stage ConvNeXt output matches SMP's expected 5-stage feature pyramid
