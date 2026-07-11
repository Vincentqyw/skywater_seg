#!/usr/bin/env python3
"""
Training entry point for Sky-Water Segmentation.

Usage:
  # Train with ADE20K config
  python train.py --config configs/ade20k.yaml

  # Override config values (use = syntax OR space)
  python train.py --config configs/ade20k.yaml --train.batch_size=16 --train.epochs=100

  # Or with spaces (auto-detected)
  python train.py --config configs/ade20k.yaml --train.batch_size 16 --train.epochs 100

  # Resume from checkpoint
  python train.py --config configs/ade20k.yaml --train.resume_from checkpoints/xxx/best_model.pth
"""

import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from skywater_seg.config import Config
from skywater_seg.trainer import train


def parse_cli_overrides(argv: list) -> dict:
    """Parse --key=value or --key value pairs from CLI, supporting dot-notation.

    Returns dict like {'train.batch_size': '16', 'train.epochs': '100'}
    """
    updates = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            arg = arg[2:]  # strip --
            if "=" in arg:
                key, value = arg.split("=", 1)
                updates[key] = value
            else:
                key = arg
                # Check if next arg is a value (not a --flag)
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    updates[key] = argv[i + 1]
                    i += 1  # skip next arg
                else:
                    updates[key] = True  # boolean flag
        i += 1
    return updates


def apply_dot_updates(config: Config, updates: dict):
    """Apply dot-notation updates, e.g. 'train.batch_size' → config.train.batch_size."""
    for key, value in updates.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field_name = parts[-1]

        current_val = getattr(obj, field_name, None)
        if current_val is not None and not isinstance(value, type(current_val)):
            try:
                if isinstance(current_val, bool):
                    value = str(value).lower() in ("true", "1", "yes")
                elif isinstance(current_val, int):
                    value = int(value)
                elif isinstance(current_val, float):
                    value = float(value)
                elif isinstance(current_val, (list, tuple)):
                    value = eval(str(value))
            except (ValueError, TypeError):
                pass
        setattr(obj, field_name, value)


def main():
    # Separate known args from overrides manually
    import argparse
    parser = argparse.ArgumentParser(description="Train sky/water segmentation model")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    args, unknown = parser.parse_known_args()

    # Load config
    config_path = args.config
    if Path(config_path).exists():
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded config from: {config_path}")
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = Config()

    # Parse and apply CLI overrides
    if unknown:
        updates = parse_cli_overrides(unknown)
        apply_dot_updates(config, updates)
        logger.info("Applied CLI overrides:")
        for k, v in updates.items():
            logger.info(f"  {k} = {v}")

    # Save config
    out_config_path = Path(config.output_dir) / config.experiment_name / "config.yaml"
    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(out_config_path))
    logger.info(f"Config saved to: {out_config_path}")

    # Train
    trainer = train(config)
    return trainer


if __name__ == "__main__":
    main()
