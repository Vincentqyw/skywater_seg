# 🌊 Sky-Water-Person Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Model_on_HF-orange.svg)](https://huggingface.co/Realcat/skywater_seg)
[![Hugging Face Space](https://img.shields.io/badge/🤗-Live_Demo-blue.svg)](https://huggingface.co/spaces/Realcat/skywater_seg)

**Fine-tuned SegFormer B2** for sky, water, and person segmentation.  
Pre-filter images for robust Structure-from-Motion and image matching.

<p align="center">
  <img src="results/onnx_benchmark/sample_grid.png" width="100%" alt="SegFormer B2 predictions">
</p>

## 📊 Performance

**SegFormer MiT-B2** (24.7M params) · 384×384 · ADE20K filtered val (1,111 images)

### Accuracy

| Class | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| Background | 96.6% | 98.3% | 98.9% | 97.6% |
| Sky | 92.1% | 95.9% | 93.3% | 98.6% |
| Water | 79.4% | 88.5% | 89.9% | 87.2% |
| Person | 77.8% | 87.5% | 85.0% | 90.1% |
| **Foreground mIoU** | **88.1%** | — | — | — |
| **Overall mIoU** | **94.6%** | — | — | — |
| **Pixel Accuracy** | **97.2%** | — | — | — |


### Accuracy vs Speed

> All backends produce **pixel-identical** results (only 0.003% difference from FP16 weight quantization).

| Backend | mIoU (fg) | mIoU (all) | PA | Latency |
|---------|-----------|------------|-----|---------|
| PyTorch FP32 | 88.1% | 94.6% | 97.2% | 23.0 ms |
| ONNX FP32 GPU | 88.1% | 94.6% | 97.2% | 15.2 ms |
| ONNX FP16 GPU | 88.1% | 94.6% | 97.2% | **13.6 ms** |
| ONNX FP32 CPU | 88.1% | 94.6% | 97.2% | 169.5 ms |



## 🚀 Quick Start

<details>
<summary>📦 1. Install</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Vincentqyw/skywater_seg.git && cd skywater_seg
uv sync
```

</details>

<details>
<summary>🐍 2. Python API</summary>

```python
from skywater_seg import load_model, segment_skywater, overlay_mask

model = load_model()
mask  = segment_skywater("assets/ade/ade_ADE_val_00001674.jpg", model)   # 0=bg, 1=sky, 2=water, 3=person
vis   = overlay_mask("assets/ade/ade_ADE_val_00001674.jpg", mask)        # visualize
```

</details>

<details>
<summary>🖥️ 3. CLI</summary>

```bash
# From HuggingFace
uv run python inference.py --hf -i assets/ade/ade_ADE_val_00001674.jpg -o outputs/

# ONNX GPU (faster, no PyTorch needed)
uv run python inference.py --onnx skywater_segformer_b2_fp16.onnx -i assets/ade/ade_ADE_val_00001674.jpg -o outputs/
```

</details>

### 🎨 Interactive Demo (Gradio)

```bash
uv run python app.py                           # HF Hub model (default)
```

Launch a Gradio web UI to explore segmentation interactively. **Try it online at [🤗 HuggingFace Space](https://huggingface.co/spaces/Realcat/skywater_seg).**

### Reproduce the Best Model

<details>

<summary>📥 Download & Train</summary>

```bash
# 1. Download dataset from HuggingFace
hf download Realcat/skywater --local-dir ./data
unzip data/ADEChallengeData2016.zip -d path/to/your_dir

# 2. Train SegFormer B2 (75 epochs, RTX 3060 6GB)
uv run python train.py --config configs/models/segformer_b2.yaml
```

</details>

### Pre-trained Models

| Model | Format | Size | Link |
|-------|--------|------|------|
| SegFormer B2 | PyTorch (safetensors) | 95 MB | [HF Hub](https://huggingface.co/Realcat/skywater_seg) |
| SegFormer B2 | PyTorch (full ckpt) | 284 MB | [.pth](https://huggingface.co/Realcat/skywater_seg/resolve/main/skywater_segformer_b2.pth) |
| SegFormer B2 | ONNX FP32 | 95 MB | [.onnx](https://huggingface.co/Realcat/skywater_seg/resolve/main/skywater_segformer_b2_fp32.onnx) |
| SegFormer B2 | ONNX FP16 | 48 MB | [.onnx](https://huggingface.co/Realcat/skywater_seg/resolve/main/skywater_segformer_b2_fp16.onnx) |

## 📸 Sample Results

### ADE20K Validation

<table>
<tr>
<td><img src="results/ade_ADE_val_00000261/figure.jpg" width="100%"></td>
<td><img src="results/ade_ADE_val_00000260/figure.jpg" width="100%"></td>
</tr>
<tr>
<td><img src="results/ade_ADE_val_00000590/figure.jpg" width="100%"></td>
<td><img src="results/ade_ADE_val_00001354/figure.jpg" width="100%"></td>
</tr>
</table>

### Real-World (SkySeg test set)

<table>
<tr>
<td><img src="results/0015_096/figure.jpg" width="100%"></td>
<td><img src="results/264489593_6de914a0ab_o.jpg/figure.jpg" width="100%"></td>
</tr>
<tr>
<td><img src="results/3134760025_0aaa4fdc8b_o/figure.jpg" width="100%"></td>
<td><img src="results/331810308_2fe422b1ec_o.jpg/figure.jpg" width="100%"></td>
</tr>
</table>

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [Datasets](docs/datasets.md) | ADE20K setup, custom data format, multi-dataset training |
| [Training](docs/training.md) | Model configs, presets, loss functions, architecture options |
| [Evaluation & Benchmark](docs/evaluation.md) | Metrics, ONNX speed comparison, pixel identity validation |
| [Deployment](docs/deployment.md) | ONNX export, CoreML, TorchScript, Python API |

## 🏗️ Project Structure

```
skywater/
├── skywater_seg/          # Python package
│   ├── inference.py       # PyTorch + ONNX inference, export helpers
│   ├── model.py           # Model factory (SMP + ConvNeXt + SegFormer)
│   ├── config.py          # Typed config (dataclass + YAML)
│   ├── dataset.py         # Dataset + MultiDataset + dataloaders
│   ├── trainer.py         # Training loop (AMP, loguru, TensorBoard)
│   ├── losses.py          # Dice, Focal, Jaccard, Combined losses
│   ├── visualization.py   # Colorize, overlay, plots, comparison grids
│   └── utils.py           # Metrics, device, checkpoint, schedulers
├── scripts/
│   ├── auto_annotate.py   # Grounding DINO + SAM annotation pipeline
│   └── prepare_ade20k.py  # ADE20K → sky/water/person splits
├── configs/               # Training configs (YAML)
├── docs/                  # Documentation
├── tests/                 # Pytest suite
├── assets/                # Test images
├── results/               # Output: masks, figures, benchmarks
├── train.py               # Training entry point
├── inference.py           # Inference entry point
├── app.py                 # Gradio interactive demo
└── pyproject.toml
```

## 📝 Citation

```bibtex
@misc{qin2026skywater,
  author       = {Vincent Qin},
  title        = {{SkyWater-Seg}: Segmenting Sky, Water, and Person Regions for Robust Structure-from-Motion Pre-processing},
  year         = {2026},
  howpublished = {\url{https://github.com/Vincentqyw/skywater_seg}},
}
```

## 📄 License

MIT
