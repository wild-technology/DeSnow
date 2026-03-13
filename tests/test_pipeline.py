"""Tests for the DeSnow pipeline components."""

import numpy as np
import pytest
import torch

from src.traditional.temporal_median import TemporalMedianFilter, AdaptiveTemporalFilter
from src.traditional.morphological import MorphologicalSnowRemover, FrequencyDomainFilter
from src.models.unet import UNet, LightweightUNet


# --- Temporal Median Filter ---

class TestTemporalMedianFilter:
    def test_init_odd_window(self):
        f = TemporalMedianFilter(window_size=7)
        assert f.window_size == 7

    def test_init_even_window_becomes_odd(self):
        f = TemporalMedianFilter(window_size=6)
        assert f.window_size == 7

    def test_returns_none_until_buffer_full(self):
        f = TemporalMedianFilter(window_size=3)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        assert f.add_frame(frame) is None
        assert f.add_frame(frame) is None
        result = f.add_frame(frame)
        assert result is not None
        assert result.shape == frame.shape

    def test_removes_transient_bright_spots(self):
        """Temporal median should remove a bright spot present in only one frame."""
        f = TemporalMedianFilter(window_size=3)
        base = np.full((64, 64, 3), 100, dtype=np.uint8)

        # Frame with a bright spot
        noisy = base.copy()
        noisy[30:35, 30:35] = 255

        f.add_frame(base)
        f.add_frame(noisy)  # bright spot only in this frame
        result = f.add_frame(base)

        # The bright spot should be removed (median of 100, 255, 100 = 100)
        assert result[32, 32, 0] == 100

    def test_reset_clears_buffer(self):
        f = TemporalMedianFilter(window_size=3)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        f.add_frame(frame)
        f.add_frame(frame)
        f.reset()
        assert f.add_frame(frame) is None  # Buffer cleared

    def test_process_frame_sequence(self):
        f = TemporalMedianFilter(window_size=3)
        frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(10)]
        results = f.process_frame_sequence(frames)
        assert len(results) == 10 - 2  # window_size - 1 frames lost


# --- Morphological Snow Remover ---

class TestMorphologicalSnowRemover:
    def test_detect_snow_mask(self):
        remover = MorphologicalSnowRemover(brightness_threshold=0.8)
        # Create dark image with bright spots
        image = np.full((128, 128, 3), 50, dtype=np.uint8)
        cv2_circle = __import__("cv2").circle
        cv2_circle(image, (64, 64), 5, (230, 230, 230), -1)

        mask = remover.detect_snow_mask(image)
        assert mask.shape == (128, 128)
        assert mask.dtype == np.uint8

    def test_remove_snow_returns_same_shape(self):
        remover = MorphologicalSnowRemover()
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = remover.remove_snow(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_process_batch(self):
        remover = MorphologicalSnowRemover()
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        results = remover.process_batch(images)
        assert len(results) == 3


# --- Frequency Domain Filter ---

class TestFrequencyDomainFilter:
    def test_output_shape_matches_input(self):
        filt = FrequencyDomainFilter()
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = filt.remove_snow(image)
        assert result.shape == image.shape

    def test_selective_remove(self):
        filt = FrequencyDomainFilter()
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = filt.selective_remove(image)
        assert result.shape == image.shape


# --- U-Net Model ---

class TestUNet:
    def test_forward_pass(self):
        model = UNet(in_channels=3, out_channels=3, base_filters=16, depth=2)
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_output_range(self):
        model = UNet(in_channels=3, out_channels=3, base_filters=16, depth=2)
        x = torch.rand(1, 3, 64, 64)
        y = model(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_different_depths(self):
        for depth in [2, 3, 4]:
            model = UNet(base_filters=8, depth=depth)
            x = torch.randn(1, 3, 64, 64)
            y = model(x)
            assert y.shape == x.shape

    def test_non_residual_mode(self):
        model = UNet(base_filters=16, depth=2, residual=False)
        x = torch.rand(1, 3, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_odd_input_dimensions(self):
        """U-Net should handle non-power-of-2 dimensions."""
        model = UNet(base_filters=16, depth=2)
        x = torch.randn(1, 3, 65, 63)
        y = model(x)
        assert y.shape == x.shape


class TestLightweightUNet:
    def test_forward_pass(self):
        model = LightweightUNet(base_filters=16, depth=2)
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_fewer_params_than_standard(self):
        standard = UNet(base_filters=32, depth=3)
        lightweight = LightweightUNet(base_filters=32, depth=3)
        standard_params = sum(p.numel() for p in standard.parameters())
        lightweight_params = sum(p.numel() for p in lightweight.parameters())
        assert lightweight_params < standard_params
