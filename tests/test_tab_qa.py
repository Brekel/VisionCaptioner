"""Tests for tab_qa.py - Quality analysis static methods."""

import os
import numpy as np
import cv2
import pytest
from PIL import Image as PILImage

from tab_qa import QAAnalysisWorker


# ===========================================================================
# TestAnalyzeBlur
# ===========================================================================
class TestAnalyzeBlur:

    def test_uniform_image_zero_variance(self):
        """A solid gray image has Laplacian variance of 0 (perfectly blurry)."""
        gray = np.ones((50, 50), dtype=np.uint8) * 128
        score = QAAnalysisWorker.analyze_blur(gray)
        assert score == 0.0

    def test_noisy_image_high_variance(self):
        """Random noise should have a high Laplacian variance (sharp)."""
        rng = np.random.RandomState(42)
        gray = rng.randint(0, 256, (50, 50), dtype=np.uint8)
        score = QAAnalysisWorker.analyze_blur(gray)
        assert score > 100  # random noise has very high variance

    def test_gradient_image_moderate(self):
        """A linear gradient should have moderate variance."""
        gray = np.tile(np.arange(0, 256, dtype=np.uint8), (256, 1))
        score = QAAnalysisWorker.analyze_blur(gray)
        assert score > 0
        # Gradient is less sharp than noise
        rng = np.random.RandomState(42)
        noise = rng.randint(0, 256, (256, 256), dtype=np.uint8)
        noise_score = QAAnalysisWorker.analyze_blur(noise)
        assert score < noise_score


# ===========================================================================
# TestAnalyzeResolution
# ===========================================================================
class TestAnalyzeResolution:

    def test_above_threshold(self):
        assert QAAnalysisWorker.analyze_resolution(1024, 768, 512) == 1.0

    def test_at_threshold(self):
        assert QAAnalysisWorker.analyze_resolution(512, 512, 512) == 1.0

    def test_below_half_threshold(self):
        assert QAAnalysisWorker.analyze_resolution(100, 100, 512) == 0.0

    def test_at_half_threshold(self):
        assert QAAnalysisWorker.analyze_resolution(256, 256, 512) == 0.0

    def test_between_half_and_threshold(self):
        # min_dim=384, half=256, score = (384-256)/(512-256) = 128/256 = 0.5
        result = QAAnalysisWorker.analyze_resolution(384, 1000, 512)
        assert abs(result - 0.5) < 1e-6

    def test_asymmetric_dimensions(self):
        # min_dim = min(2000, 300) = 300, half = 256, score = (300-256)/(512-256) = 44/256
        result = QAAnalysisWorker.analyze_resolution(2000, 300, 512)
        assert abs(result - 44 / 256) < 1e-6

    def test_min_dim_equals_width(self):
        # min_dim = 400, threshold = 800, half = 400 -> score = 0.0
        assert QAAnalysisWorker.analyze_resolution(400, 1200, 800) == 0.0


# ===========================================================================
# TestCheckMaskExists
# ===========================================================================
class TestCheckMaskExists:

    def test_masklabel_dash_png(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        (tmp_path / "photo-masklabel.png").write_bytes(b"fake")

        assert QAAnalysisWorker.check_mask_exists(str(img), str(tmp_path)) is True

    def test_masklabel_underscore(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        (tmp_path / "photo_masklabel.png").write_bytes(b"fake")

        assert QAAnalysisWorker.check_mask_exists(str(img), str(tmp_path)) is True

    def test_masks_subfolder(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "photo.png").write_bytes(b"fake")

        assert QAAnalysisWorker.check_mask_exists(str(img), str(tmp_path)) is True

    def test_alpha_channel_detected(self, tmp_path):
        # Create RGBA image with non-full alpha (some transparency)
        img = PILImage.new("RGBA", (4, 4), color=(255, 0, 0, 128))
        img_path = str(tmp_path / "transparent.png")
        img.save(img_path)

        assert QAAnalysisWorker.check_mask_exists(img_path, str(tmp_path)) is True

    def test_full_alpha_not_detected(self, tmp_path):
        # Create RGBA image with fully opaque alpha (mean==255)
        img = PILImage.new("RGBA", (4, 4), color=(255, 0, 0, 255))
        img_path = str(tmp_path / "opaque.png")
        img.save(img_path)

        assert QAAnalysisWorker.check_mask_exists(img_path, str(tmp_path)) is False

    def test_no_mask_at_all(self, tmp_path):
        # RGB image, no mask files
        img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
        img_path = str(tmp_path / "plain.jpg")
        img.save(img_path)

        assert QAAnalysisWorker.check_mask_exists(img_path, str(tmp_path)) is False

    def test_masklabel_jpg_variant(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        (tmp_path / "photo-masklabel.jpg").write_bytes(b"fake")

        assert QAAnalysisWorker.check_mask_exists(str(img), str(tmp_path)) is True

    def test_la_mode_alpha_detected(self, tmp_path):
        """Grayscale+alpha (LA mode) with partial transparency should be detected."""
        img = PILImage.new("LA", (4, 4), color=(128, 100))
        img_path = str(tmp_path / "gray_alpha.png")
        img.save(img_path)

        assert QAAnalysisWorker.check_mask_exists(img_path, str(tmp_path)) is True

    def test_la_mode_full_alpha_not_detected(self, tmp_path):
        """Grayscale+alpha (LA mode) with full opacity should NOT be detected."""
        img = PILImage.new("LA", (4, 4), color=(128, 255))
        img_path = str(tmp_path / "gray_opaque.png")
        img.save(img_path)

        assert QAAnalysisWorker.check_mask_exists(img_path, str(tmp_path)) is False

    def test_priority_masklabel_over_alpha(self, tmp_path):
        """When both a masklabel file and alpha channel exist, returns True (early exit)."""
        img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
        img_path = str(tmp_path / "photo.png")
        img.save(img_path)
        (tmp_path / "photo-masklabel.png").write_bytes(b"fake")

        assert QAAnalysisWorker.check_mask_exists(img_path, str(tmp_path)) is True
