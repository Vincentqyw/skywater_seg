#!/usr/bin/env python3
"""
Training entry point for Sky-Water Segmentation.

Usage:
  # Train with ADE20K config
  python train.py --config configs/datasets/ade20k.yaml

  # Override config values (use = syntax OR space)
  python train.py --config configs/datasets/ade20k.yaml --train.batch_size=16 --train.epochs=100

  # Or with spaces (auto-detected)
  python train.py --config configs/datasets/ade20k.yaml --train.batch_size 16 --train.epochs 100

  # Resume from checkpoint
  python train.py --config configs/datasets/ade20k.yaml --train.resume_from checkpoints/xxx/best_model.pth
"""

import sys
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from skywater_seg.config import Config, cli_to_dotlist
from skywater_seg.trainer import train


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train sky/water segmentation model")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/models/mobilenetv3_flatdir.yaml",
        help="Path to YAML config file",
    )
    args, unknown = parser.parse_known_args()

    # Build OmegaConf structured schema from defaults
    schema = OmegaConf.structured(Config())

    # Layer 1: YAML config file (optional — falls back to defaults)
    config_path = args.config
    if Path(config_path).exists():
        schema = OmegaConf.merge(schema, OmegaConf.load(config_path))
        logger.info(f"Loaded config from: {config_path}")
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")

    # Layer 2: CLI overrides (--key=val or --key val → OmegaConf dotlist)
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(cli_to_dotlist(unknown))
        schema = OmegaConf.merge(schema, cli_cfg)
        for k, v in OmegaConf.to_container(cli_cfg, resolve=True).items():
            logger.info(f"  CLI: {k} = {v}")

    # Convert to Config object
    config: Config = OmegaConf.to_object(schema)

    # Save resolved config
    out_path = Path(config.output_dir) / config.experiment_name / "config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(out_path))
    logger.info(f"Config saved to: {out_path}")

    trainer_ = train(config)
    return trainer_


if __name__ == "__main__":
    main()
