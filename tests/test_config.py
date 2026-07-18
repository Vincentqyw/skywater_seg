"""Tests for the typed configuration system (OmegaConf-backed)."""

import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from skywater_seg.config import Config, get_default_config


class TestConfigDefaults:
    def test_default_config_is_created(self):
        cfg = get_default_config()
        assert cfg.data.image_size == (512, 512)
        assert cfg.model.name == "deeplabv3plus"
        assert cfg.model.classes == 3
        assert cfg.seed == 42
        assert cfg.device == "cuda"

    def test_config_nesting(self):
        cfg = Config()
        assert cfg.data is not None
        assert cfg.model is not None
        assert cfg.train is not None


class TestConfigDictRoundTrip:
    def test_to_dict_and_back(self):
        cfg = Config()
        cfg.data.image_size = (384, 384)
        cfg.model.name = "segformer"
        cfg.model.encoder_name = "mit_b2"
        cfg.model.classes = 4

        d = cfg.to_dict()
        assert d["data"]["image_size"] == [384, 384]  # OmegaConf converts tuples→lists
        assert d["model"]["name"] == "segformer"

        cfg2 = Config.from_dict(d)
        assert cfg2.data.image_size == [384, 384]  # OmegaConf returns ListConfig→list
        assert cfg2.model.name == "segformer"
        assert cfg2.model.classes == 4

    def test_class_mapping_round_trip(self):
        cfg = Config()
        cfg.data.class_mapping = {3: 1, 22: 2, 13: 3}
        d = cfg.to_dict()
        cfg2 = Config.from_dict(d)
        assert cfg2.data.class_mapping == {3: 1, 22: 2, 13: 3}

    def test_tuple_becomes_list_in_yaml(self):
        """OmegaConf serialises tuples as lists for clean YAML."""
        cfg = Config()
        cfg.model.decoder_atrous_rates = (6, 12, 18)
        d = cfg.to_dict()
        assert isinstance(d["model"]["decoder_atrous_rates"], list)
        assert d["model"]["decoder_atrous_rates"] == [6, 12, 18]


class TestConfigYaml:
    def test_save_and_load(self):
        """Full YAML round-trip via OmegaConf — no !!python/tuple tags."""
        cfg = Config()
        cfg.experiment_name = "test-exp"
        cfg.data.image_size = (256, 256)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.yaml"
            cfg.save(str(path))
            content = path.read_text()
            assert "!!python/tuple" not in content

            loaded = Config.from_yaml(str(path))
            assert loaded.experiment_name == "test-exp"
            assert loaded.data.image_size == [256, 256]
            # Classes default preserved from schema
            assert loaded.model.classes == 3

    def test_save_and_load_via_dict(self):
        cfg = Config()
        cfg.experiment_name = "test-exp"
        cfg.data.image_size = (256, 256)
        d = cfg.to_dict()
        loaded = Config.from_dict(d)
        assert loaded.experiment_name == "test-exp"
        assert loaded.data.image_size == [256, 256]

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path/config.yaml")

    def test_load_partial_config_merges_defaults(self):
        """Missing keys in YAML are filled from schema defaults."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.yaml"
            OmegaConf.save({"model": {"name": "unet", "classes": 5}}, path)
            loaded = Config.from_yaml(str(path))
            assert loaded.model.name == "unet"
            assert loaded.model.classes == 5
            # Filled from defaults:
            assert loaded.data.image_size == [512, 512]
            assert loaded.train.epochs == 100


class TestOmegaConfStructured:
    def test_structured_config(self):
        oc = OmegaConf.structured(Config())
        assert oc.model.name == "deeplabv3plus"
        assert oc.train.batch_size == 16

    def test_cli_override_merge(self):
        """Simulate --train.batch_size=8 --train.epochs=50."""
        schema = OmegaConf.structured(Config())
        cli = OmegaConf.from_dotlist(["train.batch_size=8", "train.epochs=50"])
        merged = OmegaConf.merge(schema, cli)
        cfg = OmegaConf.to_object(merged)
        assert cfg.train.batch_size == 8
        assert cfg.train.epochs == 50
        # Unchanged:
        assert cfg.train.learning_rate == 1e-4

    def test_cli_type_coercion(self):
        """String values from CLI are auto-cast to field types."""
        schema = OmegaConf.structured(Config())
        cli = OmegaConf.from_dotlist(
            [
                "train.batch_size=32",
                "train.learning_rate=0.001",
                "train.mixed_precision=false",
                "data.image_size=[384, 384]",
            ]
        )
        merged = OmegaConf.merge(schema, cli)
        cfg = OmegaConf.to_object(merged)
        assert isinstance(cfg.train.batch_size, int)
        assert cfg.train.batch_size == 32
        assert isinstance(cfg.train.learning_rate, float)
        assert cfg.train.mixed_precision is False
        assert cfg.data.image_size == [384, 384]


class TestRealConfigs:
    """Load and validate every config file in the repo."""

    CONFIGS = [
        "configs/models/segformer_b2.yaml",
        "configs/models/convnext_dinov3.yaml",
        "configs/models/mobilenetv3_flatdir.yaml",
        "configs/datasets/ade20k_full.yaml",
        "configs/datasets/ade20k_person.yaml",
        "configs/datasets/multi_dataset.yaml",
    ]

    @pytest.mark.parametrize("path", CONFIGS)
    def test_config_loads_and_is_clean(self, path):
        cfg = Config.from_yaml(path)
        assert cfg.model.classes >= 3
        # All configs should have clean YAML
        content = Path(path).read_text()
        assert "!!python/tuple" not in content, f"{path} has !!python/tuple"
