# Datasets

Dataset preparation and supported formats for the skywater-seg pipeline.

## ADE20K Setup

### Download

Get `ADEChallengeData2016` from the [ADE20K website](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

### Expected Layout

```
E:/datasets/ADEChallengeData2016/
├── images/
│   ├── training/
│   │   ├── ADE_train_00000001.jpg
│   │   └── ...  (20,210 images)
│   └── validation/
│       ├── ADE_val_00000001.jpg
│       └── ...  (2,000 images)
└── annotations/
    ├── training/
    │   ├── ADE_train_00000001.png
    │   └── ...
    └── validation/
        ├── ADE_val_00000001.png
        └── ...
```

### Filtered Split (sky/water/person)

Only ~12,000 of the 22,000 ADE20K images contain sky, water, or person.
We provide pre-generated split files:

```bash
# Generate the filtered splits (requires ADE20K path)
uv run python scripts/prepare_ade20k.py

# Output:
#   data/ade20k/train.txt  (11,086 images)
#   data/ade20k/val.txt    (1,111 images)
```

These splits are also available on Hugging Face in
`data/train.txt` and `data/val.txt`.

### Class Mapping

ADE20K uses 150 semantic classes. We remap to 4:

| ADE20K ID | ADE20K Name | Skywater Class |
|-----------|-------------|----------------|
| 3 | sky | 1 (Sky) |
| 13 | person | 3 (Person) |
| 22 | water | 2 (Water) |
| 27 | sea | 2 (Water) |
| 61 | lake | 2 (Water) |
| 105 | pond | 2 (Water) |
| 110 | fountain | 2 (Water) |
| 114 | waterfall | 2 (Water) |
| 129 | river | 2 (Water) |
| (all others) | — | 0 (Background) |

Configured via `data.class_mapping` in the YAML config.

## Custom Data (Flat Directory)

Simplest format for your own images:

```
data/
├── images/
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
└── masks/
    ├── IMG_0001_mask.png   # 0=bg, 1=sky, 2=water, 3=person
    ├── IMG_0002_mask.png
    └── ...
```

Use `configs/models/mobilenetv3_flatdir.yaml` with `data.image_dir` and `data.mask_dir`
pointing to these directories.

## Cityscapes

The Cityscapes dataset is auto-detected via `data.cityscapes: true`:

```
Cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Multi-Dataset Training

Mix datasets with weighted sampling:

```yaml
datasets:
  - image_dir: path/to/ade20k/images
    mask_dir: path/to/ade20k/annotations
    ...
  - image_dir: path/to/cityscapes/images
    mask_dir: path/to/cityscapes/gtFine
    cityscapes: true
    ...
mix_weights: [0.7, 0.3]  # 70% ADE20K, 30% Cityscapes
```

See `configs/datasets/multi_dataset.yaml` for a complete example.
