"""Tests for utility functions."""

import numpy as np
import torch

from skywater_seg.utils import (
    CLASS_COLORS_RGB,
    class_colors_bgr,
    compute_dice,
    compute_iou,
    compute_pixel_accuracy,
    get_device,
    mask_to_color,
    set_seed,
    tensor_to_image,
)


class TestDevice:
    def test_get_device_returns_torch_device(self):
        dev = get_device("cpu")
        assert isinstance(dev, torch.device)

    def test_get_device_fallback(self):
        dev = get_device("nonexistent_device_string_12345")
        assert dev.type in ("cpu", "cuda", "mps")


class TestColors:
    def test_class_colors_rgb_has_four_classes(self):
        assert len(CLASS_COLORS_RGB) == 4
        assert 0 in CLASS_COLORS_RGB
        assert 3 in CLASS_COLORS_RGB

    def test_class_colors_bgr_matches_rgb_reversed(self):
        bgr = class_colors_bgr()
        for cid in CLASS_COLORS_RGB:
            rgb = CLASS_COLORS_RGB[cid]
            assert bgr[cid] == (rgb[2], rgb[1], rgb[0])


class TestMaskToColor:
    def test_output_shape(self):
        mask = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        rgb = mask_to_color(mask)
        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8

    def test_numpy_input(self):
        mask_np = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        rgb = mask_to_color(torch.from_numpy(mask_np))
        assert rgb[0, 0].tolist() == [0, 0, 0]  # bg = black
        assert rgb[0, 1].tolist() == list(CLASS_COLORS_RGB[1])


class TestTensorToImage:
    def test_denormalize(self):
        tensor = torch.randn(3, 64, 64)
        img = tensor_to_image(tensor)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255


class TestMetrics:
    def test_perfect_iou(self):
        pred = torch.tensor([[[0, 1], [2, 3]]])
        target = torch.tensor([[[0, 1], [2, 3]]])
        result = compute_iou(pred, target, num_classes=4, ignore_index=255)
        for k, v in result.items():
            assert abs(v - 1.0) < 0.01, f"{k} = {v}, expected ~1.0"

    def test_perfect_dice(self):
        pred = torch.tensor([[[0, 1], [2, 3]]])
        target = torch.tensor([[[0, 1], [2, 3]]])
        result = compute_dice(pred, target, num_classes=4, ignore_index=255)
        for k, v in result.items():
            assert abs(v - 1.0) < 0.01, f"{k} = {v}, expected ~1.0"

    def test_perfect_pixel_accuracy(self):
        pred = torch.tensor([[[0, 1], [2, 3]]])
        target = torch.tensor([[[0, 1], [2, 3]]])
        pa = compute_pixel_accuracy(pred, target, ignore_index=255)
        assert abs(pa - 1.0) < 0.01

    def test_ignore_index(self):
        pred = torch.tensor([[[0, 255], [255, 3]]])
        target = torch.tensor([[[0, 255], [255, 3]]])
        pa = compute_pixel_accuracy(pred, target, ignore_index=255)
        assert abs(pa - 1.0) < 0.01


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(10).tolist()
        set_seed(42)
        b = torch.randn(10).tolist()
        assert a == b
