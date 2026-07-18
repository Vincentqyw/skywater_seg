# Phase 1 — Auto-Annotation

Grounding DINO + SAM pipeline that generates pixel-level segmentation masks
from text prompts. No manual labeling required.

## Quick Start

```bash
# Directory of images → masks
uv run python scripts/auto_annotate.py -i data/images -o data/masks

# Single image
uv run python scripts/auto_annotate.py -i test.jpg -o ./output

# MacBook / low-memory: tiny + vit_b (fast, ~3–5s per image)
uv run python scripts/auto_annotate.py -i data/images -o data/masks \
    --gdino-model tiny --sam-model vit_b --fast

# GPU / high-quality: base + vit_l (~8–12s per image)
uv run python scripts/auto_annotate.py -i data/images -o data/masks \
    --gdino-model base --sam-model vit_l
```

## How It Works

```
Image → Grounding DINO (text→boxes) → SAM (boxes→masks) → Multi-class Mask
         "sky"   → [bbox₁, bbox₂, …]
         "water" → [bbox₃, …]
         "person"→ [bbox₄, …]
```

1. **Grounding DINO** detects regions matching each text prompt ("sky", "water",
   "person") and outputs bounding boxes with confidence scores.
2. **SAM** takes each bounding box and produces a binary mask at the pixel level.
3. Masks are merged into a single-channel PNG: 0=bg, 1=sky, 2=water, 3=person.

## Model Size Options

| GDINO Model | SAM Model | VRAM | Speed | Quality |
|-------------|-----------|------|-------|---------|
| tiny | vit_b | 6 GB | ~3 s/img | Good |
| tiny | vit_l | 10 GB | ~6 s/img | Better |
| base | vit_l | 16 GB | ~10 s/img | Best |
| base | vit_h | 32 GB | ~18 s/img | Excellent |

## Output

```
data/masks/
├── IMG_0001_mask.png          # 0=bg, 1=sky, 2=water, 3=person
├── IMG_0001_vis.jpg           # Visualization overlay
├── annotation_summary.json    # Per-image stats
└── ...
```

## Custom Classes

Define your own classes via a JSON file:

```json
{
  "classes": [
    {"name": "car", "prompt": "vehicle. car. automobile"},
    {"name": "tree", "prompt": "tree. vegetation. plant"}
  ]
}
```

```bash
uv run python scripts/auto_annotate.py -i images/ -o masks/ \
    --custom-classes my_classes.json
```
