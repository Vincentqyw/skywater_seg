# Evaluation & Benchmark

Model evaluation on ADE20K and ONNX inference benchmark.

## PyTorch Evaluation

```bash
# Evaluate SegFormer B2 on ADE20K filtered val set (1,111 images)
uv run python scripts/eval_segformer_b2.py
```

### SegFormer B2 Results

| Class | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| Background | 96.6% | 98.3% | 98.9% | 97.6% |
| Sky | 92.1% | 95.9% | 93.3% | 98.6% |
| Water | 79.4% | 88.5% | 89.9% | 87.2% |
| Person | 77.8% | 87.5% | 85.0% | 90.1% |
| **Foreground mIoU** | **88.1%** | — | — | — |
| **Overall mIoU** | **94.6%** | — | — | — |
| **Pixel Accuracy** | **97.2%** | — | — | — |

*ADE20K filtered val (1,111 images), 384×384 input, SegFormer MiT-B2 (24.7M params).*

## ONNX Export & Benchmark

Full PyTorch vs ONNX comparison on all 1,111 validation images:

```bash
# Full benchmark: export + latency + accuracy + figures
uv run python scripts/benchmark_full.py
```

### Speed (RTX 3060 Laptop, 384×384, batch=1)

| Backend | Latency | vs PyTorch FP32 |
|---------|---------|-----------------|
| **ONNX FP16 GPU** | **13–15 ms** | **~1.8× faster** |
| ONNX FP32 GPU | 15–17 ms | ~1.6× faster |
| PyTorch FP32 | 24–27 ms | baseline |
| PyTorch FP16 | 31–34 ms | slower |
| ONNX FP32 CPU | 188–198 ms | — |

### Pixel Identity

All ONNX variants produce **pixel-identical** results to PyTorch (only 0.003%
pixel difference from FP16 weight quantization).

### Visualization Output

```
results/onnx_benchmark/
├── speed_full.png          # Speed bar chart
├── iou_full.png            # Per-class IoU comparison
├── summary_table.png       # Metrics summary table
├── sample_grid.png         # Prediction comparison grid
└── benchmark_full.json     # All metrics as JSON
```
