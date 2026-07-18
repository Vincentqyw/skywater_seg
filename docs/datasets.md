# Datasets

## Pre-packaged Dataset (Recommended)

The ADEChallengeData2016 dataset with pre-generated sky/water/person splits is
available on HuggingFace:

```bash
# 1. Download dataset (~1GB)
hf download Realcat/skywater --local-dir ./data

# 2. Extract
unzip data/ADEChallengeData2016.zip -d path/to/
```

After extraction you should have:

```
path/to/ADEChallengeData2016/
├── images/
│   ├── training/   (20,210 images)
│   └── validation/ (2,000 images)
└── annotations/
    ├── training/
    └── validation/
```

The `data/ade20k/` directory (also downloaded from HF) contains the filtered
train/val split files:

```
data/ade20k/
├── train.txt   (11,086 lines)
└── val.txt     (1,111 lines)
```

## Generate Splits Manually

If you already have ADE20K downloaded:

```bash
uv run python scripts/prepare_ade20k.py
```

## Class Mapping

ADE20K uses 150 semantic classes. We remap to 4:

| ADE20K ID | ADE20K Name | Skywater Class |
|-----------|-------------|----------------|
| 3 | sky | 1 (Sky) |
| 13 | person | 3 (Person) |
| 22, 27, 61, 105, 110, 114, 129 | water bodies | 2 (Water) |
| (all others) | — | 0 (Background) |

## Custom Data (Flat Directory)

```
data/
├── images/
│   ├── IMG_0001.jpg
│   └── ...
└── masks/
    ├── IMG_0001_mask.png   # 0=bg, 1=sky, 2=water, 3=person
    └── ...
```

Use `configs/models/mobilenetv3_flatdir.yaml` with custom paths.

## Cityscapes

Set `data.cityscapes: true` and point `image_dir`/`mask_dir` to the Cityscapes
root. Auto-detects `leftImg8bit/` and `gtFine/` subdirectories.

## Multi-Dataset

Mix datasets with weighted sampling — see `configs/datasets/multi_dataset.yaml`.
