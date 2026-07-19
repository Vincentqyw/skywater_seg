# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sky-Water-Person Segmentation — fine-tuned SegFormer B2 for masking sky, water, and person regions in images. Built for NVIDIA GPUs and Apple Silicon (MPS/CoreML), managed with `uv`.

**Target:** Mask out sky, water, and person regions in images to eliminate their interference with SfM (Structure from Motion) and image matching pipelines.

**Version:** 0.3.0 | **HF Hub:** `Realcat/skywater_seg`

## Environment & Package Management

This project uses **uv** (not pip). All commands run through `uv run`.

```bash
uv sync                          # Install all dependencies
uv sync --group train            # Training deps only
uv sync --group dev              # Dev tools (pytest, jupyter, matplotlib)
uv add <package>                 # Add a dependency
```

## Essential Commands

### Auto-Annotation (Phase 1)

```bash
uv run python scripts/auto_annotate.py -i data/images -o data/masks
uv run python scripts/auto_annotate.py -i data/images -o data/masks --gdino-model tiny --sam-model vit_b --fast
```

### Training (Phase 2)

```bash
uv run python train.py --config configs/models/segformer_b2.yaml                 # SegFormer B2 (best)
uv run python train.py --config configs/models/convnext_dinov3.yaml              # ConvNeXt + DINOv3
uv run python train.py --config configs/models/mobilenetv3_flatdir.yaml          # Quick start, custom data
uv run python train.py --config configs/datasets/ade20k.yaml              # ADE20K filtered
uv run python train.py --config configs/datasets/ade20k_full.yaml                # ADE20K full
uv run python train.py --config configs/datasets/multi_dataset.yaml              # ADE20K + Cityscapes
uv run python train.py --config configs/models/segformer_b2.yaml --train.batch_size=8 --train.epochs=50
uv run python train.py --config configs/models/segformer_b2.yaml --train.resume_from checkpoints/xxx/checkpoint.pth
uv run tensorboard --logdir checkpoints/skywater-segformer-b2/logs
```

CLI overrides use dot-notation (`--train.batch_size=8`), auto-typed by OmegaConf.

### Inference & Export

```bash
# PyTorch — from HuggingFace
python -c "from skywater_seg import SkyWaterSegModel; m = SkyWaterSegModel.from_pretrained('Realcat/skywater_seg')"

# PyTorch inference
uv run python inference.py --checkpoint skywater_segformer_b2.pth -i test.jpg

# ONNX Runtime inference (no PyTorch needed)
uv run python inference.py --onnx skywater_segformer_b2_fp16.onnx -i test.jpg

# Export: PyTorch → ONNX
uv run python -c "from skywater_seg import export_onnx, convert_onnx_fp16; ..."
```

### Data Preparation

```bash
uv run python scripts/prepare_ade20k.py   # ADE20K → filtered splits
```

### Tests & CI

```bash
uv run pytest tests/ -v                          # 40 tests
uv run ruff check skywater_seg/                  # lint
```

### Demo

```bash
uv run jupyter notebook demo/demo.ipynb           # Full walkthrough
```

## Architecture

### Three-Phase Pipeline

```
Phase 1: Auto-Annotation     Phase 2: Training                Phase 3: Deployment
Grounding DINO + SAM    →    DeepLabV3+ / ConvNeXt       →    ONNX / CoreML (ANE)
(text→boxes→masks)           (~5M–30M params, CUDA/MPS)       (<5ms inference)
```

### Package Structure (`skywater_seg/`)

| Module | Role |
|---|---|
| `config.py` | OmegaConf-backed typed config (`DataConfig`, `ModelConfig`, `TrainConfig`, `DatasetConfig`, `Config`). YAML ↔ structured dataclass, `cli_to_dotlist()` for CLI overrides. |
| `model.py` | Model factory + `SkyWaterSegModel` (HF Hub mixin). `from_pretrained()` downloads safetensors. Supports SegFormer, DeepLabV3+, U-Net, FPN, PSPNet, PAN, Linknet + ConvNeXt (timm). |
| `dataset.py` | `SkyWaterDataset` + `MultiDataset` + `create_dataloaders()`. MPS-aware. |
| `losses.py` | `DiceLoss`, `FocalLoss`, `JaccardLoss`, `CombinedLoss`. |
| `trainer.py` | Full training loop with AMP, grad accumulation, early stopping, TensorBoard. |
| `inference.py` | `ONNXRuntimeInference` (CPU/CUDA/CoreML/TensorRT/ROCm/OpenVINO/DirectML/ACL), `export_onnx()`, `convert_onnx_fp16()`. |
| `visualization.py` | `colorize_mask()`, `overlay_mask()`, `plot_speed_comparison()`, `plot_iou_comparison()`, `make_comparison_grid()`, etc. |
| `utils.py` | Metrics, device management, checkpoint save/load, scheduler factory. |
| `cli.py` | CLI entry points for `skywater-*` commands. |

### Scripts (`scripts/`)

- **`auto_annotate.py`**: Grounding DINO + SAM pipeline.
- **`prepare_ade20k.py`**: Filter ADE20K for sky/water/person splits.
- **`eval_segformer_b2.py`**: Per-class IoU/Dice/Precision/Recall on ADE20K val.
- **`benchmark_full.py`**: ONNX export + latency + accuracy benchmark + figures.
- **`gen_readme_figures.py`**: Paper-style 2×2 comparison figures.

### Configuration System

- OmegaConf replaces manual PyYAML parsing. Dataclasses are the schema.
- Configs organized: `configs/models/` (architecture), `configs/datasets/` (data sources).
- CLI: `--train.batch_size=8` → `cli_to_dotlist()` → `OmegaConf.from_dotlist()` → merged.

### Key Technical Details

- **Best model**: SegFormer MiT-B2 (24.7M params). mIoU(fg) 88.1%, Sky IoU 92.1%, PA 97.2%.
- **Input**: 384×384, ImageNet normalization. 4 classes: bg/sky/water/person.
- **ONNX export**: FP32 95MB / FP16 48MB. ONNX FP16 GPU 13.6ms (1.7× faster than PyTorch 23.0ms).
- **HF Hub**: `SkyWaterSegModel.from_pretrained("Realcat/skywater_seg")` one-liner.
- **Apple Silicon**: CoreML provider (`provider="coreml"`) for ~3ms inference.
