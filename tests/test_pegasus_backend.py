"""Tests for pegasus_backend.py - TwelveLabs Pegasus captioning engine.

No-network unit tests mock the TwelveLabs SDK; the live test is gated on
TWELVELABS_API_KEY and skipped without it (it hits the real API).
"""

import os
import sys
import types
import pytest
from unittest.mock import MagicMock

from pegasus_backend import PegasusEngine, PEGASUS_MIN_TOKENS


def _install_fake_twelvelabs(monkeypatch):
    """Inject a fake `twelvelabs` package so load_model/generate_batch run
    without the real SDK or network access. Returns the fake TwelveLabs class."""
    tl = types.ModuleType("twelvelabs")
    tl.TwelveLabs = MagicMock(name="TwelveLabs")

    ctx_mod = types.ModuleType("twelvelabs.types.video_context")
    ctx_mod.VideoContext_Url = lambda url=None: ("url", url)
    ctx_mod.VideoContext_AssetId = lambda asset_id=None: ("asset", asset_id)
    types_pkg = types.ModuleType("twelvelabs.types")

    monkeypatch.setitem(sys.modules, "twelvelabs", tl)
    monkeypatch.setitem(sys.modules, "twelvelabs.types", types_pkg)
    monkeypatch.setitem(sys.modules, "twelvelabs.types.video_context", ctx_mod)
    return tl.TwelveLabs


# --- find_files: video/image discovery + skip/mask handling ----------------
class TestFindFiles:
    def test_pairs_have_no_mask(self, tmp_path):
        (tmp_path / "a.mp4").write_text("x")
        (tmp_path / "b.jpg").write_text("x")
        eng = PegasusEngine(api_key="k")
        results = eng.find_files(str(tmp_path))
        assert {os.path.basename(f) for f, _ in results} == {"a.mp4", "b.jpg"}
        assert all(m is None for _, m in results)

    def test_excludes_masklabel(self, tmp_path):
        (tmp_path / "a.mp4").write_text("x")
        (tmp_path / "a-masklabel.png").write_text("x")
        eng = PegasusEngine(api_key="k")
        names = {os.path.basename(f) for f, _ in eng.find_files(str(tmp_path))}
        assert "a-masklabel.png" not in names

    def test_skip_existing(self, tmp_path):
        (tmp_path / "a.mp4").write_text("x")
        (tmp_path / "a.txt").write_text("done")
        eng = PegasusEngine(api_key="k")
        assert eng.find_files(str(tmp_path), skip_existing=True) == []


# --- load_model: pure client construction, no network ----------------------
class TestLoadModel:
    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
        ok, msg = PegasusEngine(api_key=None).load_model()
        assert ok is False and "TWELVELABS_API_KEY" in msg

    def test_success_sets_sentinel(self, monkeypatch):
        _install_fake_twelvelabs(monkeypatch)
        eng = PegasusEngine(api_key="k", model_name="pegasus1.5")
        ok, msg = eng.load_model()
        assert ok is True
        assert eng.model == "pegasus1.5"  # sentinel the CLI checks
        assert eng.client is not None

    def test_unload_clears(self, monkeypatch):
        _install_fake_twelvelabs(monkeypatch)
        eng = PegasusEngine(api_key="k")
        eng.load_model()
        eng.unload_model()
        assert eng.model is None and eng.client is None


# --- generate_batch: wiring, token clamp, image rejection, errors ----------
class TestGenerateBatch:
    def _ready_engine(self, monkeypatch, analyze_data="a dog runs"):
        TLClass = _install_fake_twelvelabs(monkeypatch)
        client = MagicMock()
        TLClass.return_value = client
        # Asset upload returns a ready asset immediately.
        client.assets.create.return_value = MagicMock(id="asset123", status="ready")
        client.analyze.return_value = MagicMock(data=analyze_data)
        eng = PegasusEngine(api_key="k", model_name="pegasus1.5")
        eng.load_model()
        return eng, client

    def test_video_calls_analyze(self, tmp_path, monkeypatch):
        vid = tmp_path / "clip.mp4"
        vid.write_text("x")
        eng, client = self._ready_engine(monkeypatch)
        out = eng.generate_batch([str(vid)], prompt_text="Describe.", max_tokens=384)
        assert out == ["a dog runs"]
        client.assets.create.assert_called_once()
        # max_tokens below Pegasus min must be clamped up.
        assert client.analyze.call_args.kwargs["max_tokens"] == PEGASUS_MIN_TOKENS

    def test_trigger_word_prepended(self, tmp_path, monkeypatch):
        vid = tmp_path / "clip.mp4"
        vid.write_text("x")
        eng, _ = self._ready_engine(monkeypatch)
        out = eng.generate_batch([str(vid)], trigger_word="MYLORA", max_tokens=512)
        assert out[0] == "MYLORA, a dog runs"

    def test_image_rejected(self, tmp_path, monkeypatch):
        img = tmp_path / "pic.jpg"
        img.write_text("x")
        eng, client = self._ready_engine(monkeypatch)
        out = eng.generate_batch([str(img)], max_tokens=512)
        assert out[0].startswith("Error:")
        client.analyze.assert_not_called()

    def test_api_error_per_item(self, tmp_path, monkeypatch):
        vid = tmp_path / "clip.mp4"
        vid.write_text("x")
        eng, client = self._ready_engine(monkeypatch)
        client.analyze.side_effect = RuntimeError("boom")
        out = eng.generate_batch([str(vid)], max_tokens=512)
        assert out[0].startswith("Error:") and "boom" in out[0]

    def test_not_loaded(self):
        eng = PegasusEngine(api_key="k")
        out = eng.generate_batch(["clip.mp4"])
        assert out[0].startswith("Error:")


# --- Live test: real API, requires a key (skipped in CI without it) --------
@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"),
                    reason="TWELVELABS_API_KEY not set; skipping live TwelveLabs test")
def test_live_embed_dimension():
    """Sanity-check connectivity/auth via a Marengo text embedding (512-dim)."""
    from twelvelabs import TwelveLabs
    c = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])
    r = c.embed.create(model_name="marengo3.0", text="a cat playing piano")
    assert len(r.text_embedding.segments[0].float_) == 512
