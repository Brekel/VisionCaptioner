"""Tests for model_probe.py - Model introspection and caching."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from model_probe import ModelProbe


# ===========================================================================
# TestProbeFolder
# ===========================================================================
class TestProbeFolder:
    """Tests for ModelProbe._probe_folder()"""

    def test_qwen25_vl(self, tmp_path):
        model_dir = tmp_path / "qwen-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "qwen_hf"
        assert result["unified_vision"] is True
        assert result["format"] == "hf_folder"
        assert result["architecture"] == "qwen2_5_vl"

    def test_qwen3_vl(self, tmp_path):
        model_dir = tmp_path / "qwen3-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "qwen_hf"
        assert result["unified_vision"] is True

    def test_mllama(self, tmp_path):
        model_dir = tmp_path / "llama-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "mllama"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "llama_hf"
        assert result["unified_vision"] is False

    def test_llava(self, tmp_path):
        model_dir = tmp_path / "llava-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llava_next"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "llava_hf"

    def test_sam3_by_config(self, tmp_path):
        model_dir = tmp_path / "sam-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "sam3_video"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "sam3"

    def test_sam3_by_name_and_pt_file(self, tmp_path):
        model_dir = tmp_path / "sam3-weights"
        model_dir.mkdir()
        (model_dir / "sam3.pt").write_bytes(b"fake")

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["architecture"] == "sam3"
        assert result["backend"] == "sam3"

    def test_path_based_fallback_qwen_vl(self, tmp_path):
        model_dir = tmp_path / "Qwen2.5-VL-7B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "custom_unknown"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "qwen_hf"
        assert result["unified_vision"] is True

    def test_path_based_fallback_sam3(self, tmp_path):
        model_dir = tmp_path / "my-sam3-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "custom_unknown"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "sam3"

    def test_no_config_json(self, tmp_path):
        model_dir = tmp_path / "empty-model"
        model_dir.mkdir()

        result = ModelProbe._probe_folder(str(model_dir))
        assert "error" in result

    def test_invalid_json(self, tmp_path):
        model_dir = tmp_path / "broken-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{not valid json!!!")

        result = ModelProbe._probe_folder(str(model_dir))
        assert "error" in result

    def test_unknown_model_type_no_fallback(self, tmp_path):
        model_dir = tmp_path / "mystery-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "totally_custom"}))

        result = ModelProbe._probe_folder(str(model_dir))
        assert result["backend"] == "unknown"


# ===========================================================================
# TestProbeGguf
# ===========================================================================
class TestProbeGguf:
    """Tests for ModelProbe._probe_gguf()"""

    def test_no_gguf_library(self, monkeypatch):
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", False)

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert "error" in result
        assert "gguf" in result["error"].lower()

    def test_vision_tensors_detected(self, monkeypatch):
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "v.patch_embed.weight"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["unified_vision"] is True
        assert result["format"] == "gguf"

    def test_text_only_with_mmproj(self, tmp_path, monkeypatch):
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        # Create main and mmproj files
        main_gguf = tmp_path / "model-qwen-7b.gguf"
        main_gguf.write_bytes(b"fake")
        mmproj = tmp_path / "model-qwen-7b-mmproj-f16.gguf"
        mmproj.write_bytes(b"fake")

        mock_tensor = MagicMock()
        mock_tensor.name = "output.weight"  # No vision tensors

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf(str(main_gguf))
        assert result["unified_vision"] is False
        assert result["mmproj_detected"] is not None

    def test_architecture_from_metadata(self, monkeypatch):
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "output.weight"

        mock_field = MagicMock()
        mock_field.name = "general.architecture"
        mock_field.data = [0]
        mock_field.parts = [b"qwen2vl"]

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {"general.architecture": mock_field}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["architecture"] == "qwen2vl"

    def test_vision_tensors_clip(self, monkeypatch):
        """Vision detected via 'clip' in tensor name."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "clip.encoder.layer.0.weight"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["unified_vision"] is True

    def test_vision_tensors_visual(self, monkeypatch):
        """Vision detected via 'visual' in tensor name."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "visual.proj.weight"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["unified_vision"] is True

    def test_vision_tensors_mm_prefix(self, monkeypatch):
        """Vision detected via 'mm.' in tensor name."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "mm.proj.0.weight"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["unified_vision"] is True

    def test_no_vision_tensors(self, monkeypatch):
        """No vision patterns in tensor names -> unified_vision=False."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "model.layers.0.self_attn.weight"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["unified_vision"] is False

    def test_architecture_from_numpy_array(self, monkeypatch):
        """Architecture decoded via numpy array -> tolist() -> list of ints (ASCII)."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "output.weight"

        # Simulate a numpy-like array that has tolist() returning ASCII code points
        mock_val = MagicMock()
        mock_val.tolist.return_value = [108, 108, 97, 109, 97, 0]  # "llama" + null padding
        mock_val.__class__ = type("ndarray", (), {})  # has tolist

        mock_field = MagicMock()
        mock_field.name = "general.architecture"
        mock_field.data = [0]
        mock_field.parts = [mock_val]

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {"general.architecture": mock_field}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["architecture"] == "llama"

    def test_architecture_from_string(self, monkeypatch):
        """Architecture decoded from a plain string (no decode, no tolist)."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "output.weight"

        mock_field = MagicMock()
        mock_field.name = "general.architecture"
        mock_field.data = [0]
        mock_field.parts = ["gemma2"]  # plain string

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {"general.architecture": mock_field}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert result["architecture"] == "gemma2"

    def test_architecture_fallback_from_filename(self, monkeypatch):
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "output.weight"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/qwen-vl-7b-q4.gguf")
        assert result["architecture"] == "qwen2"

    def test_architecture_no_fallback_non_qwen(self, monkeypatch):
        """Filename without 'qwen' leaves architecture as 'unknown'."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        mock_tensor = MagicMock()
        mock_tensor.name = "output.weight"

        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe._probe_gguf("/fake/llama-7b.gguf")
        assert result["architecture"] == "unknown"

    def test_reader_exception(self, monkeypatch):
        """GGUFReader raising an exception returns error dict."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.side_effect = RuntimeError("corrupt file")

        result = ModelProbe._probe_gguf("/fake/model.gguf")
        assert "error" in result


# ===========================================================================
# TestFindMatchingMmproj
# ===========================================================================
class TestFindMatchingMmproj:
    """Tests for ModelProbe.find_matching_mmproj()"""

    def test_exact_token_overlap(self, tmp_path):
        main = tmp_path / "model-qwen3-vl-8b.gguf"
        main.write_bytes(b"x")
        mmproj = tmp_path / "model-qwen3-vl-8b-mmproj-f16.gguf"
        mmproj.write_bytes(b"x")

        result = ModelProbe.find_matching_mmproj(str(main))
        assert result is not None
        assert "mmproj" in os.path.basename(result).lower()

    def test_best_match_by_score(self, tmp_path):
        main = tmp_path / "qwen3-vl-8b-instruct.gguf"
        main.write_bytes(b"x")
        # Good match (shares qwen3, vl, 8b, instruct)
        good = tmp_path / "qwen3-vl-8b-instruct-mmproj.gguf"
        good.write_bytes(b"x")
        # Weak match (shares only qwen3)
        weak = tmp_path / "qwen3-mmproj.gguf"
        weak.write_bytes(b"x")

        result = ModelProbe.find_matching_mmproj(str(main))
        assert result == str(good)

    def test_no_candidates(self, tmp_path):
        main = tmp_path / "model.gguf"
        main.write_bytes(b"x")

        result = ModelProbe.find_matching_mmproj(str(main))
        assert result is None

    def test_skip_tokens_excluded(self, tmp_path):
        """Tokens like 'gguf', 'mmproj', 'q4', 'f16', 'model' are excluded from scoring.
        When all shared tokens are in the skip set, overlap score is 0 and no match is returned."""
        main = tmp_path / "model-f16.gguf"
        main.write_bytes(b"x")
        # The only shared tokens are 'model' and 'f16' which are in the skip set
        mmproj = tmp_path / "model-f16-mmproj.gguf"
        mmproj.write_bytes(b"x")

        result = ModelProbe.find_matching_mmproj(str(main))
        # Score stays 0 (never exceeds best_score=0) so no match returned
        assert result is None

    def test_skip_tokens_dont_affect_real_overlap(self, tmp_path):
        """Non-skip tokens still contribute to matching even when skip tokens are present."""
        main = tmp_path / "model-qwen3-vl-f16.gguf"
        main.write_bytes(b"x")
        mmproj = tmp_path / "model-qwen3-vl-mmproj-f16.gguf"
        mmproj.write_bytes(b"x")

        result = ModelProbe.find_matching_mmproj(str(main))
        assert result is not None  # 'qwen3' and 'vl' provide real overlap


# ===========================================================================
# TestCache
# ===========================================================================
class TestCache:
    """Tests for cache load/save/prune and cache integration with probe()."""

    def test_load_cache_exists(self, tmp_path, monkeypatch):
        cache_data = {"path/to/model": {"_mtime": 123, "backend": "qwen_hf"}}
        cache_file = tmp_path / "model_cache.json"
        cache_file.write_text(json.dumps(cache_data))
        monkeypatch.setattr(ModelProbe, "CACHE_FILE", str(cache_file))

        result = ModelProbe.load_cache()
        assert result == cache_data

    def test_load_cache_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ModelProbe, "CACHE_FILE", str(tmp_path / "nonexistent.json"))
        result = ModelProbe.load_cache()
        assert result == {}

    def test_load_cache_corrupt(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "bad_cache.json"
        cache_file.write_text("{broken json!")
        monkeypatch.setattr(ModelProbe, "CACHE_FILE", str(cache_file))

        result = ModelProbe.load_cache()
        assert result == {}

    def test_save_cache_roundtrip(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ModelProbe, "CACHE_FILE", str(cache_file))

        data = {"model/path": {"_mtime": 999, "backend": "llama_hf"}}
        ModelProbe.save_cache(data)

        loaded = ModelProbe.load_cache()
        assert loaded == data

    def test_prune_cache_removes_missing(self, tmp_path):
        cache = {
            "/nonexistent/path/a": {"_mtime": 1},
            "/nonexistent/path/b": {"_mtime": 2},
        }
        removed = ModelProbe.prune_cache(cache)
        assert removed == 2
        assert len(cache) == 0

    def test_prune_cache_keeps_existing(self, tmp_path):
        real_file = tmp_path / "real_model"
        real_file.mkdir()
        cache = {str(real_file): {"_mtime": 1}}

        removed = ModelProbe.prune_cache(cache)
        assert removed == 0
        assert len(cache) == 1

    def test_probe_uses_cache_on_mtime_match(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "cached-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))

        abs_path = os.path.abspath(str(model_dir))
        mtime = os.path.getmtime(abs_path)

        cache = {
            abs_path: {
                "_mtime": mtime,
                "backend": "cached_backend",
                "format": "cached"
            }
        }

        result = ModelProbe.probe(str(model_dir), cache=cache)
        assert result["backend"] == "cached_backend"

    def test_probe_ignores_cache_on_mtime_mismatch(self, tmp_path):
        model_dir = tmp_path / "stale-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))

        abs_path = os.path.abspath(str(model_dir))
        cache = {
            abs_path: {
                "_mtime": 0.0,  # stale
                "backend": "stale_backend"
            }
        }

        result = ModelProbe.probe(str(model_dir), cache=cache)
        assert result["backend"] == "qwen_hf"  # freshly probed

    def test_probe_nonexistent_path(self):
        result = ModelProbe.probe("/totally/fake/path")
        assert "error" in result

    def test_probe_unknown_file_type(self, tmp_path):
        """A non-directory, non-GGUF file returns format=unknown."""
        weird_file = tmp_path / "model.bin"
        weird_file.write_bytes(b"data")

        result = ModelProbe.probe(str(weird_file))
        assert result["format"] == "unknown"

    def test_probe_gguf_file_delegates(self, tmp_path, monkeypatch):
        """probe() dispatches to _probe_gguf for .gguf files."""
        import model_probe
        monkeypatch.setattr(model_probe, "HAS_GGUF", True)

        gguf_file = tmp_path / "test.gguf"
        gguf_file.write_bytes(b"fake")

        mock_tensor = MagicMock()
        mock_tensor.name = "output.weight"
        mock_reader = MagicMock()
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}

        monkeypatch.setattr(model_probe, "gguf", MagicMock())
        model_probe.gguf.GGUFReader.return_value = mock_reader

        result = ModelProbe.probe(str(gguf_file))
        assert result["format"] == "gguf"
        assert result["backend"] == "llama_cpp"

    def test_probe_error_not_cached(self, tmp_path):
        """When probe returns an error, it should NOT be stored in cache."""
        cache = {}
        result = ModelProbe.probe("/totally/fake/path", cache=cache)
        assert "error" in result
        assert len(cache) == 0

    def test_probe_updates_cache(self, tmp_path):
        model_dir = tmp_path / "new-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "mllama"}))

        cache = {}
        result = ModelProbe.probe(str(model_dir), cache=cache)

        abs_path = os.path.abspath(str(model_dir))
        assert abs_path in cache
        assert cache[abs_path]["backend"] == "llama_hf"
        assert "_mtime" in cache[abs_path]
