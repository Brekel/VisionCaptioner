"""Tests for backend.py - File discovery, image processing, device detection."""

import os
import base64
import io
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage
import numpy as np
import cv2

from backend import QwenEngine


# ===========================================================================
# TestFindFiles
# ===========================================================================
class TestFindFiles:
    """Tests for QwenEngine.find_files()"""

    def _make_engine(self):
        engine = QwenEngine.__new__(QwenEngine)
        engine.model = None
        engine.processor = None
        engine.device = "cpu"
        engine.is_gguf = False
        return engine

    def test_finds_images(self, tmp_path):
        for name in ["a.jpg", "b.png", "c.webp"]:
            (tmp_path / name).write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        filenames = [os.path.basename(f) for f, _ in result]
        assert "a.jpg" in filenames
        assert "b.png" in filenames
        assert "c.webp" in filenames

    def test_excludes_masklabel(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "photo-masklabel.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        filenames = [os.path.basename(f) for f, _ in result]
        assert "photo.jpg" in filenames
        assert "photo-masklabel.png" not in filenames

    def test_pairs_mask_dash(self, tmp_path):
        (tmp_path / "img.jpg").write_bytes(b"fake")
        (tmp_path / "img-masklabel.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        assert len(result) == 1
        f, m = result[0]
        assert os.path.basename(f) == "img.jpg"
        assert m is not None
        assert "masklabel" in os.path.basename(m)

    def test_pairs_mask_underscore(self, tmp_path):
        (tmp_path / "img.jpg").write_bytes(b"fake")
        (tmp_path / "img_masklabel.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        assert len(result) == 1
        _, m = result[0]
        assert m is not None
        assert "_masklabel" in os.path.basename(m)

    def test_no_mask_returns_none(self, tmp_path):
        (tmp_path / "lonely.jpg").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        assert len(result) == 1
        _, m = result[0]
        assert m is None

    def test_skip_existing_with_txt(self, tmp_path):
        (tmp_path / "done.jpg").write_bytes(b"fake")
        (tmp_path / "done.txt").write_text("caption exists")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), skip_existing=True)
        assert len(result) == 0

    def test_skip_existing_without_txt(self, tmp_path):
        (tmp_path / "new.jpg").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), skip_existing=True)
        assert len(result) == 1

    def test_finds_videos(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        filenames = [os.path.basename(f) for f, _ in result]
        assert "clip.mp4" in filenames

    def test_empty_folder(self, tmp_path):
        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        assert result == []

    def test_sorted_and_deduplicated(self, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        filenames = [os.path.basename(f) for f, _ in result]
        assert filenames == sorted(filenames)

    def test_multiple_extensions_deduplicated(self, tmp_path):
        """Same file shouldn't appear twice even if matched by lower+upper globs."""
        (tmp_path / "photo.jpg").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        assert len(result) == 1

    def test_pairs_mask_jpg_variant(self, tmp_path):
        """Mask files with .jpg extension should also be paired."""
        (tmp_path / "img.jpg").write_bytes(b"fake")
        (tmp_path / "img-masklabel.jpg").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        assert len(result) == 1
        _, m = result[0]
        assert m is not None
        assert m.endswith("-masklabel.jpg")

    def test_mask_priority_dash_before_underscore(self, tmp_path):
        """When both dash and underscore masks exist, dash variant wins (checked first)."""
        (tmp_path / "img.jpg").write_bytes(b"fake")
        (tmp_path / "img-masklabel.png").write_bytes(b"fake")
        (tmp_path / "img_masklabel.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        assert len(result) == 1
        _, m = result[0]
        assert "-masklabel.png" in m

    def test_mixed_images_and_videos(self, tmp_path):
        """Both images and videos are found together."""
        (tmp_path / "pic.jpg").write_bytes(b"fake")
        (tmp_path / "vid.mp4").write_bytes(b"fake")
        (tmp_path / "pic.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path))
        filenames = [os.path.basename(f) for f, _ in result]
        assert "pic.jpg" in filenames
        assert "vid.mp4" in filenames
        assert "pic.png" in filenames

    def test_skip_existing_does_not_skip_without_txt(self, tmp_path):
        """skip_existing=True only skips when .txt exists, not when mask exists."""
        (tmp_path / "img.jpg").write_bytes(b"fake")
        (tmp_path / "img-masklabel.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), skip_existing=True)
        assert len(result) == 1  # not skipped since img.txt doesn't exist

    def test_finds_images_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.jpg").write_bytes(b"fake")
        (sub / "nested.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), recursive=True)
        filenames = [os.path.basename(f) for f, _ in result]
        assert "root.jpg" in filenames
        assert "nested.png" in filenames

    def test_recursive_excludes_unused_dir(self, tmp_path):
        unused = tmp_path / "unused"
        unused.mkdir()
        (unused / "old.jpg").write_bytes(b"fake")
        (tmp_path / "good.jpg").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), recursive=True)
        filenames = [os.path.basename(f) for f, _ in result]
        assert "good.jpg" in filenames
        assert "old.jpg" not in filenames

    def test_recursive_pairs_masks(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "img.jpg").write_bytes(b"fake")
        (sub / "img-masklabel.png").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), recursive=True)
        assert len(result) == 1
        f, m = result[0]
        assert os.path.basename(f) == "img.jpg"
        assert m is not None
        assert "masklabel" in os.path.basename(m)

    def test_recursive_skip_existing(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "done.jpg").write_bytes(b"fake")
        (sub / "done.txt").write_text("caption exists")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), skip_existing=True, recursive=True)
        assert len(result) == 0

    def test_recursive_false_is_flat(self, tmp_path):
        """Ensure recursive=False doesn't find subdir images (regression check)."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.jpg").write_bytes(b"fake")

        engine = self._make_engine()
        result = engine.find_files(str(tmp_path), recursive=False)
        assert len(result) == 0


# ===========================================================================
# TestImageToBase64
# ===========================================================================
class TestImageToBase64:

    def _make_engine(self):
        engine = QwenEngine.__new__(QwenEngine)
        engine.model = None
        engine.processor = None
        engine.device = "cpu"
        engine.is_gguf = False
        return engine

    def test_rgb_image(self):
        engine = self._make_engine()
        img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
        result = engine._image_to_base64(img)

        # Should be valid base64
        decoded = base64.b64decode(result)
        # Should be valid JPEG
        recovered = PILImage.open(io.BytesIO(decoded))
        assert recovered.format == "JPEG"

    def test_rgba_converts_to_rgb(self):
        engine = self._make_engine()
        img = PILImage.new("RGBA", (4, 4), color=(255, 0, 0, 128))
        result = engine._image_to_base64(img)

        decoded = base64.b64decode(result)
        recovered = PILImage.open(io.BytesIO(decoded))
        assert recovered.mode == "RGB"

    def test_grayscale(self):
        engine = self._make_engine()
        img = PILImage.new("L", (4, 4), color=128)
        result = engine._image_to_base64(img)

        decoded = base64.b64decode(result)
        recovered = PILImage.open(io.BytesIO(decoded))
        assert recovered.mode == "RGB"


# ===========================================================================
# TestApplyMask
# ===========================================================================
class TestApplyMask:

    def _make_engine(self):
        engine = QwenEngine.__new__(QwenEngine)
        engine.model = None
        engine.processor = None
        engine.device = "cpu"
        engine.is_gguf = False
        return engine

    def test_apply_mask_same_size(self, tmp_path):
        engine = self._make_engine()

        # Create a 8x8 red image
        img = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
        img_path = str(tmp_path / "img.png")
        img.save(img_path)

        # Create a mask: top half white (keep), bottom half black (mask out)
        mask_arr = np.zeros((8, 8), dtype=np.uint8)
        mask_arr[:4, :] = 255
        mask_path = str(tmp_path / "mask.png")
        cv2.imwrite(mask_path, mask_arr)

        result = engine.apply_mask(img_path, mask_path)
        assert result is not None
        assert isinstance(result, PILImage.Image)

        result_arr = np.array(result)
        # Top half should have color, bottom half should be black
        assert result_arr[:4, :, 0].mean() > 200  # red channel preserved
        assert result_arr[4:, :, :].mean() == 0  # masked out

    def test_apply_mask_resize(self, tmp_path):
        engine = self._make_engine()

        img = PILImage.new("RGB", (16, 16), color=(0, 255, 0))
        img_path = str(tmp_path / "img.png")
        img.save(img_path)

        # Mask is a different size - should be resized
        mask_arr = np.ones((8, 8), dtype=np.uint8) * 255
        mask_path = str(tmp_path / "mask.png")
        cv2.imwrite(mask_path, mask_arr)

        result = engine.apply_mask(img_path, mask_path)
        assert result is not None
        assert result.size == (16, 16)

    def test_apply_mask_invalid_mask_path(self, tmp_path):
        engine = self._make_engine()

        img = PILImage.new("RGB", (4, 4), color=(100, 100, 100))
        img_path = str(tmp_path / "img.png")
        img.save(img_path)

        result = engine.apply_mask(img_path, str(tmp_path / "nonexistent_mask.png"))
        assert result is None

    def test_apply_mask_invalid_image_path(self, tmp_path):
        engine = self._make_engine()

        mask_arr = np.ones((4, 4), dtype=np.uint8) * 255
        mask_path = str(tmp_path / "mask.png")
        cv2.imwrite(mask_path, mask_arr)

        result = engine.apply_mask(str(tmp_path / "nonexistent_img.png"), mask_path)
        assert result is None

    def test_apply_mask_returns_rgb(self, tmp_path):
        """apply_mask should return an RGB PIL Image (not BGR)."""
        engine = self._make_engine()

        # Pure red image
        img = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
        img_path = str(tmp_path / "red.png")
        img.save(img_path)

        # Full white mask (keep everything)
        mask_arr = np.ones((8, 8), dtype=np.uint8) * 255
        mask_path = str(tmp_path / "mask.png")
        cv2.imwrite(mask_path, mask_arr)

        result = engine.apply_mask(img_path, mask_path)
        assert result is not None
        assert result.mode == "RGB"
        # Verify it's actually red (R=255) not blue (B=255, which would mean BGR leak)
        r, g, b = result.getpixel((0, 0))
        assert r > 200
        assert b < 50

    def test_apply_mask_all_black_mask(self, tmp_path):
        """An all-black mask should zero out the entire image."""
        engine = self._make_engine()

        img = PILImage.new("RGB", (8, 8), color=(255, 255, 255))
        img_path = str(tmp_path / "white.png")
        img.save(img_path)

        mask_arr = np.zeros((8, 8), dtype=np.uint8)
        mask_path = str(tmp_path / "mask.png")
        cv2.imwrite(mask_path, mask_arr)

        result = engine.apply_mask(img_path, mask_path)
        assert result is not None
        assert np.array(result).max() == 0  # everything masked out


# ===========================================================================
# TestGetDeviceType
# ===========================================================================
class TestGetDeviceType:

    def _make_engine_raw(self):
        """Create engine without __init__ to test get_device_type independently."""
        engine = QwenEngine.__new__(QwenEngine)
        return engine

    def test_cuda_available(self):
        import torch
        torch.cuda.is_available.return_value = True
        torch.backends.mps.is_available.return_value = False

        engine = self._make_engine_raw()
        assert engine.get_device_type() == "cuda"

        # Reset
        torch.cuda.is_available.return_value = False

    def test_mps_available(self):
        import torch
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = True

        engine = self._make_engine_raw()
        assert engine.get_device_type() == "mps"

        # Reset
        torch.backends.mps.is_available.return_value = False

    def test_cpu_fallback(self):
        import torch
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False

        engine = self._make_engine_raw()
        assert engine.get_device_type() == "cpu"
