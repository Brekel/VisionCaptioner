"""Tests for cli.py - Configuration parsing and file discovery."""

import os
import json
import pytest
from unittest.mock import patch

from cli import parse_quant_string, RES_MAP, load_defaults_from_settings, find_images_in_folder, find_videos_in_folder, get_model_path_from_index


# ===========================================================================
# TestParseQuantString
# ===========================================================================
class TestParseQuantString:

    def test_fp16(self):
        assert parse_quant_string("FP16 (Half Precision)") == "FP16"

    def test_fp16_bare(self):
        assert parse_quant_string("FP16") == "FP16"

    def test_int8(self):
        assert parse_quant_string("Int8 (8-bit)") == "Int8"

    def test_nf4(self):
        assert parse_quant_string("NF4 (4-bit)") == "NF4"

    def test_none_string(self):
        assert parse_quant_string("None") == "None"

    def test_empty_string(self):
        assert parse_quant_string("") == "None"

    def test_unrecognized_string(self):
        assert parse_quant_string("something else") == "None"

    def test_non_string_int(self):
        assert parse_quant_string(42) == "None"

    def test_non_string_none(self):
        assert parse_quant_string(None) == "None"


# ===========================================================================
# TestResMap
# ===========================================================================
class TestResMap:

    def test_all_keys_present(self):
        assert set(RES_MAP.keys()) == {0, 1, 2, 3, 4}

    def test_values(self):
        expected = {0: 336, 1: 512, 2: 768, 3: 1024, 4: 1280}
        assert RES_MAP == expected


# ===========================================================================
# TestLoadDefaultsFromSettings
# ===========================================================================
class TestLoadDefaultsFromSettings:

    def test_full_settings(self, sample_settings, monkeypatch):
        import cli
        monkeypatch.setattr(cli, "SETTINGS_FILE", sample_settings)

        result = load_defaults_from_settings()
        assert result["model_index"] == 2
        assert result["quant"] == "FP16"
        assert result["res"] == 1024  # resolution_idx=3 -> 1024
        assert result["batch_size"] == 8
        assert result["frame_count"] == 4
        assert result["max_tokens"] == 512
        assert result["trigger"] == "photo"
        assert result["prompt"] == "Describe this image in detail."
        assert result["prompt_suffix"] == "Be concise."
        assert result["skip_existing"] is True
        # Mask defaults
        assert result["mask_prompt"] == "person"
        assert result["mask_res"] == 768
        assert result["mask_expand"] == 5.0
        assert result["mask_skip"] is False
        assert result["mask_crop"] is True
        # Video defaults
        assert result["video_step"] == 15
        assert result["video_start"] == 10
        assert result["video_end"] == 500
        assert result["video_conf"] == 0.3

    def test_missing_file(self, tmp_path, monkeypatch):
        import cli
        monkeypatch.setattr(cli, "SETTINGS_FILE", str(tmp_path / "nonexistent.json"))

        result = load_defaults_from_settings()
        assert result == {}

    def test_corrupt_json(self, tmp_path, monkeypatch):
        import cli
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{broken!")
        monkeypatch.setattr(cli, "SETTINGS_FILE", str(bad_file))

        result = load_defaults_from_settings()
        assert result == {}

    def test_partial_settings_no_generate_tab(self, tmp_path, monkeypatch):
        import cli
        settings_file = tmp_path / "partial.json"
        settings_file.write_text(json.dumps({"folder": "/some/path"}))
        monkeypatch.setattr(cli, "SETTINGS_FILE", str(settings_file))

        result = load_defaults_from_settings()
        # Should still return defaults from the get() calls with fallback values
        assert result["model_index"] == 0
        assert result["quant"] == "None"
        assert result["res"] == 512  # default resolution_idx=1 -> 512
        assert result["folder"] == "/some/path"

    def test_resolution_idx_mapping(self, tmp_path, monkeypatch):
        import cli
        settings_file = tmp_path / "res_test.json"
        settings_file.write_text(json.dumps({
            "generate_tab": {"resolution_idx": 0}
        }))
        monkeypatch.setattr(cli, "SETTINGS_FILE", str(settings_file))

        result = load_defaults_from_settings()
        assert result["res"] == 336  # idx 0 -> 336


# ===========================================================================
# TestFindImagesInFolder
# ===========================================================================
class TestFindImagesInFolder:

    def test_finds_all_extensions(self, tmp_path):
        for name in ["a.jpg", "b.jpeg", "c.png", "d.webp", "e.bmp"]:
            (tmp_path / name).write_bytes(b"fake")

        result = find_images_in_folder(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "a.jpg" in basenames
        assert "b.jpeg" in basenames
        assert "c.png" in basenames
        assert "d.webp" in basenames
        assert "e.bmp" in basenames

    def test_excludes_masklabel(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "photo-masklabel.png").write_bytes(b"fake")
        (tmp_path / "photo_masklabel.png").write_bytes(b"fake")

        result = find_images_in_folder(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "photo.jpg" in basenames
        assert "photo-masklabel.png" not in basenames
        assert "photo_masklabel.png" not in basenames

    def test_sorted_output(self, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"fake")

        result = find_images_in_folder(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert basenames == sorted(basenames)

    def test_empty_folder(self, tmp_path):
        result = find_images_in_folder(str(tmp_path))
        assert result == []

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "readme.txt").write_bytes(b"text")
        (tmp_path / "data.csv").write_bytes(b"csv")
        (tmp_path / "real.jpg").write_bytes(b"img")

        result = find_images_in_folder(str(tmp_path))
        assert len(result) == 1
        assert os.path.basename(result[0]) == "real.jpg"

    def test_uppercase_extensions(self, tmp_path):
        """The source globs both *.jpg and *.JPG so uppercase should be found."""
        (tmp_path / "PHOTO.JPG").write_bytes(b"fake")
        (tmp_path / "SCAN.PNG").write_bytes(b"fake")

        result = find_images_in_folder(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "PHOTO.JPG" in basenames
        assert "SCAN.PNG" in basenames

    def test_finds_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.jpg").write_bytes(b"fake")
        (sub / "nested.png").write_bytes(b"fake")

        result = find_images_in_folder(str(tmp_path), recursive=True)
        basenames = [os.path.basename(f) for f in result]
        assert "root.jpg" in basenames
        assert "nested.png" in basenames

    def test_flat_default_unchanged(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.jpg").write_bytes(b"fake")

        result = find_images_in_folder(str(tmp_path))
        assert len(result) == 0


# ===========================================================================
# TestFindVideosInFolder
# ===========================================================================
class TestFindVideosInFolder:

    def test_finds_video_extensions(self, tmp_path):
        for name in ["a.mp4", "b.mkv", "c.avi", "d.mov", "e.webm", "f.flv"]:
            (tmp_path / name).write_bytes(b"fake")

        result = find_videos_in_folder(str(tmp_path))
        assert len(result) == 6

    def test_single_file_path(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        result = find_videos_in_folder(str(video))
        assert result == [str(video)]

    def test_empty_folder(self, tmp_path):
        result = find_videos_in_folder(str(tmp_path))
        assert result == []

    def test_sorted_output(self, tmp_path):
        for name in ["z.mp4", "a.mp4", "m.mp4"]:
            (tmp_path / name).write_bytes(b"fake")

        result = find_videos_in_folder(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert basenames == sorted(basenames)

    def test_uppercase_extensions(self, tmp_path):
        (tmp_path / "clip.MP4").write_bytes(b"fake")
        (tmp_path / "vid.AVI").write_bytes(b"fake")

        result = find_videos_in_folder(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "clip.MP4" in basenames
        assert "vid.AVI" in basenames

    def test_finds_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.mp4").write_bytes(b"fake")
        (sub / "nested.avi").write_bytes(b"fake")

        result = find_videos_in_folder(str(tmp_path), recursive=True)
        basenames = [os.path.basename(f) for f in result]
        assert "root.mp4" in basenames
        assert "nested.avi" in basenames

    def test_flat_default_unchanged(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.mp4").write_bytes(b"fake")

        result = find_videos_in_folder(str(tmp_path))
        assert len(result) == 0


# ===========================================================================
# TestGetModelPathFromIndex
# ===========================================================================
class TestGetModelPathFromIndex:

    def test_valid_index(self, tmp_path, monkeypatch):
        """Returns model dir path when index is in range."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "alpha-model").mkdir()
        (models_dir / "beta-model").mkdir()
        (models_dir / "gamma-model").mkdir()

        monkeypatch.chdir(tmp_path)
        # Sorted: alpha, beta, gamma -> index 1 = beta
        result = get_model_path_from_index(1)
        assert result is not None
        assert "beta-model" in result

    def test_index_zero(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "first-model").mkdir()
        (models_dir / "second-model").mkdir()

        monkeypatch.chdir(tmp_path)
        result = get_model_path_from_index(0)
        assert result is not None
        assert "first-model" in result

    def test_index_out_of_range(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "only-model").mkdir()

        monkeypatch.chdir(tmp_path)
        result = get_model_path_from_index(5)
        assert result is None

    def test_negative_index(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model-a").mkdir()

        monkeypatch.chdir(tmp_path)
        result = get_model_path_from_index(-1)
        assert result is None

    def test_no_models_dir(self, tmp_path, monkeypatch):
        """Returns None when the models/ directory doesn't exist."""
        monkeypatch.chdir(tmp_path)
        result = get_model_path_from_index(0)
        assert result is None

    def test_skips_files_in_models_dir(self, tmp_path, monkeypatch):
        """Only directories inside models/ count, not loose files."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "real-model").mkdir()
        (models_dir / "README.md").write_text("docs")

        monkeypatch.chdir(tmp_path)
        result = get_model_path_from_index(0)
        assert result is not None
        assert "real-model" in result
        # Index 1 should be out of range (only 1 dir)
        assert get_model_path_from_index(1) is None
