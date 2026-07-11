# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sky-Water Segmentation Pipeline — a three-phase system that automatically generates sky/water segmentation masks, trains a lightweight model on them, and deploys for fast inference. Built for MacBook (Apple Silicon) with MPS acceleration, managed with `uv`.

**Target:** Mask out sky and water regions in images to eliminate their interference with SfM (Structure from Motion) and image matching pipelines.

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
uv run python scripts/auto_annotate.py -i data/images -o data/masks --gdino-model tiny --sam-model vit_b --fast  # MacBook fast mode
```

### Training (Phase 2)

```bash
uv run python train.py --config configs/default.yaml                           # With default config
uv run python train.py --config configs/ade20k.yaml                            # ADE20K dataset config
uv run python train.py --config configs/default.yaml --train.batch_size=8 --train.epochs=50  # CLI overrides (dot notation)
uv run python train.py --config configs/default.yaml --train.resume_from checkpoints/skywater-seg/best_model.pth  # Resume
uv run tensorboard --logdir checkpoints/skywater-seg/logs                      # Monitor training
```

CLI overrides use dot-notation (`--train.batch_size 8`) and are auto-typed to match the config dataclass field type.

### Inference (Phase 3)

```bash
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth -i test.jpg                    # PyTorch inference
uv run python inference.py --onnx checkpoints/skywater-seg/skywater_seg.onnx -i test.jpg                       # ONNX Runtime (no PyTorch)
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth --export-onnx skywater_seg.onnx  # Export ONNX
```

### ADE20K Data Extraction

```bash
uv run python scripts/extract_ade20k.py --ade-root /path/to/ADE20K_2021_17_01 --out-dir data/ade20k_skywater --splits training validation
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
Phase 1: Auto-Annotation     Phase 2: Training              Phase 3: Deployment
Grounding DINO + SAM    →    DeepLabV3+ MobileNetV3    →    ONNX / CoreML (ANE)
(text→boxes→masks)           (~5M params, MPS)              (<5ms inference)
```

`run_pipeline.py` orchestrates all three via subprocess calls. It auto-detects Apple Silicon and picks optimal model defaults (tiny GDINO, mobile SAM, FP16, batch_size=8).

### Package Structure (`skywater_seg/`)

| Module | Role |
|---|---|
| `config.py` | Typed configuration via dataclasses (`DataConfig`, `ModelConfig`, `TrainConfig`, `Config`). Loads from YAML, supports dot-notation CLI overrides. |
| `dataset.py` | `SkyWaterDataset` — loads image+mask pairs, mask matching by `{stem}_mask.png`. Augmentation via albumentations. `create_dataloaders()` handles train/val split (explicit .txt files or random split). MPS-aware (disables `pin_memory`). |
| `model.py` | Factory wrapping `segmentation-models-pytorch`. Supports DeepLabV3+, U-Net, FPN, PSPNet, PAN, Linknet. Default: DeepLabV3+ with MobileNetV3-Large encoder. Includes named presets (`lightweight`, `ultra-lightweight`, `balanced`, `accurate`). |
| `losses.py` | `DiceLoss`, `FocalLoss`, `JaccardLoss`, `CombinedLoss` (CE + Dice). Default: `dice_ce` with equal weights. Ignore index 255 support throughout. |
| `trainer.py` | `Trainer` class with AMP, gradient accumulation, gradient clipping, early stopping, checkpoint management, and rich TensorBoard logging (loss, metrics, per-class IoU, gradient/weight histograms, prediction overlays). |
| `inference.py` | `SegmentationInference` (PyTorch), `ONNXRuntimeInference` (no PyTorch dep). Both return `{mask, sky_mask, water_mask}`. Supports batch inference, CRF post-processing, ONNX/TorchScript export. |
| `coreml_export.py` | ONNX→CoreML conversion via `coremltools`, direct PyTorch→CoreML tracing, and `CoreMLInference` class. macOS-only. |
| `utils.py` | Metrics (IoU, Dice, pixel accuracy), device management (`get_device` falls back cuda→mps→cpu), visualization (`tensor_to_image`, `mask_to_color`), checkpoint save/load, LR scheduler factory. |
| `cli.py` | Thin wrappers delegating to scripts/train/inference for the package console_scripts entry points. |

### Scripts (`scripts/`)

- **`auto_annotate.py`** (41KB): Full Grounding DINO + SAM annotation pipeline. Text prompts → bounding boxes (GDINO) → pixel masks (SAM). Supports custom class definitions via JSON (`custom_classes_example.json`). Outputs multi-class mask PNGs (0=bg, 1=sky, 2=water) plus visualization overlays and summary JSON.
- **`extract_ade20k.py`**: Converts ADE20K 2021 dataset annotations to sky/water masks compatible with the training pipeline. Matches sky and water objects by WordNet name against predefined name sets.

### Configuration System

- Hierarchical dataclasses: `Config` → `DataConfig`, `ModelConfig`, `TrainConfig`
- YAML files in `configs/`: `default.yaml` (generic), `ade20k.yaml` (ADE20K dataset paths)
- CLI overrides use dot-notation and auto-type-cast to match the dataclass field types (see `train.py:apply_dot_updates`)
- Saved config is written to `{output_dir}/{experiment_name}/config.yaml` on each training run

### Data Format

- **Input images**: Standard formats (jpg, png, tif, bmp)
- **Masks**: Single-channel PNG with pixel values 0=background, 1=sky, 2=water
- **Mask naming**: `{image_stem}_mask.png` (matched by the dataset loader)
- **Split files**: Optional `train.txt` / `val.txt` with one image filename per line

### Model Export Chain

```
PyTorch (.pth) → ONNX (.onnx) → CoreML (.mlpackage) → Apple Neural Engine
                 ↘ TorchScript (.pt)
                 ↘ TensorRT (.trt) via trtexec CLI
```

### Apple Silicon Optimizations

The codebase has explicit MPS/ANE awareness throughout:
- `get_device()` falls back cuda → mps → cpu
- `pin_memory` is disabled on MPS (not supported)
- `run_pipeline.py` auto-detects Apple Silicon and selects tiny/fast models, FP16, batch_size=8, 768px resolution
- CoreML export targets Apple Neural Engine for ~3ms inference (vs ~12ms MPS, ~50ms CPU)

## Key Technical Details

- **Default model**: DeepLabV3+ with `timm-mobilenetv3_large_100` encoder (~5M params, ImageNet pretrained)
- **Default loss**: `CombinedLoss` (0.5×CrossEntropy + 0.5×Dice), ignores index 255
- **Normalization**: ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Input size**: 512×512 default (configurable)
- **3 classes**: 0=background, 1=sky, 2=water
- **Dataset**: `SkyWaterDataset` masks via `cv2.IMREAD_GRAYSCALE`, with `_load_mask()` trying multiple naming conventions before defaulting to all-zeros
- **Validation split**: If no explicit `.txt` files given, random 85/15 split seeded by `config.seed`
- **Checkpoint format**: Dict with `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `metrics`, `timestamp`
- **Inference result dict**: `{mask: (H,W) uint8, sky_mask: bool, water_mask: bool, probs?: (C,H,W) float32}`
