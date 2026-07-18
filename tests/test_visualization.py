"""Tests for visualization functions."""

import numpy as np
import pytest

from skywater_seg.visualization import (
    CLASS_NAMES,
    NUM_CLASSES,
    colorize_mask,
    overlay_mask,
)


class TestClassNames:
    def test_num_classes_matches(self):
        assert NUM_CLASSES == 4
        assert len(CLASS_NAMES) == 4
        assert CLASS_NAMES[0] == "Background"
        assert CLASS_NAMES[1] == "Sky"
        assert CLASS_NAMES[2] == "Water"
        assert CLASS_NAMES[3] == "Person"


class TestColorizeMask:
    def test_output_shape_and_type(self):
        mask = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        rgb = colorize_mask(mask)
        assert rgb.shape == (1, 4, 3)
        assert rgb.dtype == np.uint8

    def test_background_is_dark(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        rgb = colorize_mask(mask)
        assert (rgb[0, 0] == [0, 0, 0]).all()

    def test_sky_is_colored(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        rgb = colorize_mask(mask)
        assert not (rgb[0, 0] == [0, 0, 0]).all()

    def test_all_classes_colorized_differently(self):
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        rgb = colorize_mask(mask)
        colors = [tuple(rgb[0, 0]), tuple(rgb[0, 1]),
                   tuple(rgb[1, 0]), tuple(rgb[1, 1])]
        assert len(set(colors)) == 4


class TestOverlayMask:
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_mask(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:60, 30:60] = 1  # sky patch
        mask[70:100, 70:100] = 2  # water patch
        return mask

    def test_overlay_bgr_output(self, sample_image, sample_mask):
        vis = overlay_mask(sample_image, sample_mask)
        assert vis.shape[2] == 3
        assert vis.dtype == np.uint8
        # BGR channel order — Blue channel non-zero for sky
        assert vis[40, 40, 0] > 0  # sky column = blue in BGR

    def test_overlay_no_contours(self, sample_image, sample_mask):
        vis = overlay_mask(sample_image, sample_mask, draw_contours=False)
        assert vis.shape == (128, 128, 3)

    def test_overlay_alpha(self, sample_image, sample_mask):
        vis_full = overlay_mask(sample_image, sample_mask, alpha=1.0)
        vis_none = overlay_mask(sample_image, sample_mask, alpha=0.0)
        assert not np.allclose(vis_full, vis_none)


class TestPlotFunctionsNoMpl:
    """These tests ensure the plot functions raise ImportError without
    matplotlib, not some cryptic error."""

    def test_plot_speed_needs_matplotlib(self, monkeypatch):
        import skywater_seg.visualization as viz
        # Temp remove matplotlib
        import sys
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ImportError, match="matplotlib"):
            viz.plot_speed_comparison({}, "/tmp/test.png")

    def test_plot_iou_needs_matplotlib(self, monkeypatch):
        import skywater_seg.visualization as viz
        import sys
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ImportError, match="matplotlib"):
            viz.plot_iou_comparison({}, "/tmp/test.png")
