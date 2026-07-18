# Phase 3 — Deployment & Export

Convert trained PyTorch models to ONNX for cross-platform deployment.

## Model Export

```bash
# PyTorch → ONNX FP32
uv run python inference.py --checkpoint checkpoints/skywater-segformer-b2/best_model.pth \
    --export-onnx skywater_seg.onnx

# PyTorch → ONNX via Python API
```python
from skywater_seg import SegmentationInference, export_onnx, convert_onnx_fp16

infer = SegmentationInference("best_model.pth")
export_onnx(infer.model, infer.image_size, "model_fp32.onnx")
convert_onnx_fp16("model_fp32.onnx", "model_fp16.onnx")
```

# ONNX → CoreML (macOS only, Apple Neural Engine)
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth \
    --export-coreml skywater_seg.mlpackage

# PyTorch → TorchScript
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth \
    --export-torchscript skywater_seg.pt
```

## Export Chain

```
PyTorch (.pth)  →  ONNX (.onnx)  →  CoreML (.mlpackage)  →  Apple Neural Engine
                →  TorchScript (.pt)
                →  TensorRT (.trt) via trtexec CLI
```

## Inference

### PyTorch

```bash
# Single image
uv run python inference.py --checkpoint checkpoints/skywater-segformer-b2/best_model.pth \
    -i test.jpg -o output/

# Batch directory
uv run python inference.py --checkpoint best_model.pth -i images/ -o results/
```

### ONNX Runtime

```bash
# GPU (auto-detect)
uv run python inference.py --onnx skywater_segformer_b2_fp16.onnx -i test.jpg

# CPU only
uv run python inference.py --onnx model.onnx -i test.jpg --device cpu
```

### Python API

```python
from skywater_seg import ONNXRuntimeInference

# NVIDIA GPU (CUDA)
infer = ONNXRuntimeInference("skywater_segformer_b2_fp16.onnx", provider="cuda")

# Apple Silicon (macOS — CoreML / Neural Engine)
infer = ONNXRuntimeInference("skywater_segformer_b2_fp16.onnx", provider="coreml")

# CPU fallback
infer = ONNXRuntimeInference("model.onnx", provider="cpu")

result = infer.predict("test.jpg")
mask = result["mask"]  # (H, W) uint8: 0=bg, 1=sky, 2=water, 3=person
```

## Pre-exported Models

| Model | Format | Size | Speed (RTX 3060) |
|-------|--------|------|-------------------|
| `skywater_segformer_b2_fp32.onnx` | ONNX FP32 | 95 MB | 15.4 ms |
| `skywater_segformer_b2_fp16.onnx` | ONNX FP16 | 48 MB | **13.3 ms** |
| `best_model.pth` | PyTorch | 284 MB | 24.1 ms |

Available on Hugging Face: `Realcat/skywater_seg`
