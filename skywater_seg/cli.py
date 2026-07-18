from __future__ import annotations

"""
CLI entry points for uv-managed package scripts.

Usage:
  skywater-annotate -i data/images -o data/masks
  skywater-train --config configs/models/mobilenetv3_flatdir.yaml
  skywater-infer --checkpoint model.pth --input test.jpg
"""

import sys
from pathlib import Path


def annotate():
    """`skywater-annotate` CLI — auto-annotation with Grounding DINO + SAM."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.auto_annotate import main

    main()


def train_cmd():
    """`skywater-train` CLI — train lightweight segmentation model."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train import main

    main()


def infer():
    """`skywater-infer` CLI — run inference with trained model."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference import main

    main()
