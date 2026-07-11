# 🌊 Sky-Water Segmentation Pipeline

自动标注 + 轻量模型训练 + 部署的完整 pipeline。**uv 管理环境，针对 MacBook (Apple Silicon) 深度优化。**

目标：mask 掉图像中的天空和水面区域，消除它们对 SfM 与图像匹配的干扰。

---

## 整体架构

```
┌─────────────────────────────────┐
│  Phase 1: 自动标注               │
│  Grounding DINO + SAM (MPS)     │
│  → 生成 sky/water masks         │
└─────────────┬───────────────────┘
              │ masks (PNG)
              ▼
┌─────────────────────────────────┐
│  Phase 2: 小模型训练             │
│  DeepLabV3+ MobileNetV3 (MPS)   │
│  → ~5M params                     │
└─────────────┬───────────────────┘
              │ checkpoint (.pth)
              ▼
┌─────────────────────────────────┐
│  Phase 3: 部署                  │
│  ONNX Runtime / CoreML (ANE)   │
│  → <5ms MacBook, ~10MB          │
└─────────────────────────────────┘
```

---

## 🚀 快速开始 (MacBook)

### 环境安装 (uv)

```bash
# 安装 uv (如已安装跳过)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 一键安装所有依赖
cd skywater
uv sync

# 验证
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 一键运行完整 Pipeline

```bash
# MacBook: 自动检测 Apple Silicon，使用最优配置
uv run python run_pipeline.py --image-dir data/images

# 分步运行
uv run python run_pipeline.py --image-dir data/images --annotate-only   # 只标注
uv run python run_pipeline.py --image-dir data/images --train-only      # 只训练
uv run python run_pipeline.py --image-dir data/images --export-only \   # 只导出
    --checkpoint checkpoints/skywater-seg/best_model.pth
```

### 推理

```bash
# PyTorch (MPS 加速)
uv run python inference.py --checkpoint checkpoints/skywater-seg/best_model.pth -i test.jpg

# ONNX Runtime (跨平台)
uv run python inference.py --onnx checkpoints/skywater-seg/skywater_seg.onnx -i test.jpg
```

---

## 📋 详细用法

### Phase 1: 自动标注

```bash
# MacBook 优化（自动使用 tiny + vit_b，更快）
uv run python scripts/auto_annotate.py \
    -i data/images \
    -o data/masks

# 如果追求标注精度（需要更多内存和时间）
uv run python scripts/auto_annotate.py \
    -i data/images -o data/masks \
    --gdino-model base --sam-model vit_l

# 单张图像
uv run python scripts/auto_annotate.py -i test.jpg -o ./output
```

**输出**：
```
data/masks/
├── image001_mask.png      # 0=背景, 1=天空, 2=水面
├── image001_vis.jpg       # 可视化叠加
├── annotation_summary.json
└── ...
```

### Phase 2: 训练

```bash
# MacBook 优化训练
uv run python train.py --config configs/default.yaml \
    --data.image_dir data/images \
    --data.mask_dir data/masks \
    --train.epochs 100 \
    --train.batch_size 8     # MacBook 内存有限，batch 调小

# 监控
uv run tensorboard --logdir checkpoints/skywater-seg/logs
```

### Phase 3: 导出 & 部署

```bash
# → ONNX (跨平台)
uv run python inference.py \
    --checkpoint checkpoints/skywater-seg/best_model.pth \
    --export-onnx skywater_seg.onnx

# → CoreML (MacBook 专属，Apple Neural Engine 加速，最快！)
uv run python -c "
from skywater_seg.coreml_export import export_coreml
export_coreml('checkpoints/skywater-seg/skywater_seg.onnx',
              'skywater_seg.mlpackage')
"
```

---

## 🍎 MacBook 性能

| 推理方式 | 速度 (M3 Max) | 说明 |
|----------|--------------|------|
| **CoreML (ANE)** | **~3ms** | Apple Neural Engine，最快 |
| ONNX Runtime (CPU) | ~15ms | 跨平台，无需 GPU |
| PyTorch MPS | ~12ms | 开发调试用 |
| PyTorch CPU | ~50ms | 最慢 |

| 标注模型 | MacBook 适用 | 单张耗时 |
|----------|-------------|---------|
| GDINO-tiny + SAM vit_b | ✅ 推荐 | ~3-5s |
| GDINO-base + SAM vit_l | ⚠️ 16GB+ | ~8-12s |
| GDINO-base + SAM vit_h | ❌ 需要 32GB+ | ~15-20s |

---

## 📦 项目结构

```
skywater/
├── scripts/
│   └── auto_annotate.py          # Grounding DINO + SAM 自动标注
├── skywater_seg/
│   ├── cli.py                     # uv 命令行入口
│   ├── config.py                  # 配置管理
│   ├── dataset.py                 # PyTorch Dataset + 增强
│   ├── model.py                   # 模型工厂 (DeepLabV3+, U-Net, etc.)
│   ├── losses.py                  # 损失函数 (Dice, Focal, Combined)
│   ├── trainer.py                 # 训练循环 (AMP, MPS, 断点续训)
│   ├── inference.py               # PyTorch/ONNX 推理 + 导出
│   ├── coreml_export.py           # CoreML 导出 + ANE 推理 (MacBook 专属)
│   └── utils.py                   # 指标, 可视化, checkpoint 管理
├── configs/
│   ├── default.yaml               # 训练配置
│   └── custom_classes_example.json
├── pyproject.toml                  # uv 项目配置
├── run_pipeline.py                 # 端到端 pipeline (MacBook 优化)
├── train.py                       # 训练入口
├── inference.py                   # 推理入口
└── README.md
```

---

## 🔧 环境管理 (uv)

```bash
uv sync                          # 安装所有依赖
uv sync --group annotate         # 只装标注依赖
uv sync --group train            # 只装训练依赖
uv sync --group deploy           # 只装部署依赖
uv sync --no-dev                 # 生产环境（不含 dev 依赖）

uv add <package>                 # 添加依赖
uv remove <package>              # 移除依赖
uv tree                          # 查看依赖树
```

---

## 🔗 关键参考

- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- SAM: https://github.com/facebookresearch/segment-anything
- segmentation-models-pytorch: https://github.com/qubvel/segmentation_models.pytorch
- coremltools: https://github.com/apple/coremltools
- uv: https://github.com/astral-sh/uv
- SfM + 天空掩码: UFERN (2025) — YOLOv8 sky masking 减少 21.7% outliers

---

## License

MIT
