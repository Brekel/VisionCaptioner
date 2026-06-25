"""
pegasus_backend.py - Optional cloud captioning backend powered by TwelveLabs Pegasus.

VisionCaptioner normally captions videos by extracting a handful of frames and
feeding them to a local Vision-Language Model. That works well for images, but
throws away most of the motion and temporal context in a video.

TwelveLabs Pegasus is a video-native understanding model: it watches the whole
clip (motion, sequence of events, on-screen action) instead of a few stills, so
it is well suited to captioning video datasets. This module wraps it in an engine
that duck-types `QwenEngine` (find_files / load_model / generate_batch /
unload_model / is_gguf / model), so the CLI can swap it in with `--backend pegasus`
without touching any of the existing local-model code paths.

This backend is fully OPT-IN:
  * the `twelvelabs` SDK is imported lazily inside load_model, so users who never
    use it don't need the dependency installed;
  * nothing here runs unless the user explicitly selects the pegasus backend;
  * images are still routed to the local engine (Pegasus is video-only) so mixed
    datasets behave sensibly.

Set the API key via the TWELVELABS_API_KEY environment variable.
Grab a free key at https://twelvelabs.io
"""

import os

from file_utils import find_media_files, IMAGE_EXTS, VIDEO_EXTS

# Pegasus enforces a minimum on max_tokens server-side; clamp to avoid a 400.
PEGASUS_MIN_TOKENS = 512
DEFAULT_MODEL = "pegasus1.5"
# How long (seconds) to wait for a local file to finish uploading/indexing.
ASSET_READY_TIMEOUT = 600
ASSET_POLL_INTERVAL = 5

_VIDEO_EXT_SET = tuple(e.lstrip("*").lower() for e in VIDEO_EXTS)


class PegasusEngine:
    """Cloud video-captioning engine backed by TwelveLabs Pegasus.

    Mirrors the subset of the QwenEngine interface the CLI relies on so it can be
    dropped in as an alternative backend.
    """

    def __init__(self, api_key=None, model_name=DEFAULT_MODEL):
        self.client = None
        # `model` is checked by the CLI/UI as a "loaded?" sentinel; reuse it.
        self.model = None
        self.api_key = api_key or os.environ.get("TWELVELABS_API_KEY")
        self.model_name = model_name
        self.is_gguf = False  # keeps batch-sizing logic in the CLI happy

    # --- File discovery: identical contract to QwenEngine.find_files ---------
    def find_files(self, folder_path, skip_existing=False, recursive=False):
        files = find_media_files(folder_path, exts=IMAGE_EXTS + VIDEO_EXTS,
                                 recursive=recursive, exclude_masks=False)
        results = []
        for f in files:
            if "masklabel" in os.path.basename(f):
                continue
            if skip_existing:
                if os.path.exists(os.path.splitext(f)[0] + ".txt"):
                    continue
            # Pegasus is video-native; masks don't apply, so always pair with None.
            results.append((f, None))
        return results

    def load_model(self, model_path=None, quantization_type="None", max_resolution=512,
                   attn_impl="sdpa", use_compile=False, vision_token_budget=None,
                   media_mode="image"):
        """'Loading' a cloud model just means constructing an authenticated client.

        Returns (success: bool, message: str) to match QwenEngine.load_model.
        """
        if not self.api_key:
            return False, ("TWELVELABS_API_KEY not set. Export your key "
                           "(get one free at https://twelvelabs.io).")
        try:
            from twelvelabs import TwelveLabs
        except ImportError:
            return False, ("twelvelabs SDK not installed. Run: "
                           "pip install 'twelvelabs>=1.2.8'")
        try:
            self.client = TwelveLabs(api_key=self.api_key)
            self.model = self.model_name  # sentinel: "loaded"
            return True, f"TwelveLabs Pegasus ready ({self.model_name})"
        except Exception as e:
            return False, str(e)

    def unload_model(self):
        self.client = None
        self.model = None
        return "TwelveLabs client released."

    def _is_video(self, path):
        return path.lower().endswith(_VIDEO_EXT_SET)

    def _video_context(self, video_path, log_callback=None):
        """Upload a local file as a TwelveLabs asset and return its VideoContext.

        Pegasus also accepts public URLs, but VisionCaptioner works on local
        dataset files, so we upload each one as a direct asset and reference it
        by id.
        """
        import time
        from twelvelabs.types.video_context import VideoContext_AssetId

        with open(video_path, "rb") as fh:
            asset = self.client.assets.create(method="direct", file=fh,
                                              filename=os.path.basename(video_path))
        if log_callback:
            log_callback(f"☁️  Uploaded {os.path.basename(video_path)} (asset {asset.id})")

        # Wait until the asset is processed before analysing it.
        deadline = time.time() + ASSET_READY_TIMEOUT
        status = asset.status
        while status not in ("ready", "failed") and time.time() < deadline:
            time.sleep(ASSET_POLL_INTERVAL)
            status = self.client.assets.retrieve(asset_id=asset.id).status
        if status != "ready":
            raise RuntimeError(f"asset {asset.id} not ready (status={status})")

        return VideoContext_AssetId(asset_id=asset.id)

    def generate_batch(self, file_paths, prompt_text="Describe this.", trigger_word="",
                       frame_count=8, mask_paths=None, max_tokens=1024,
                       log_callback=None, stop_event=None):
        """Caption each video with Pegasus. Images are skipped with a clear note
        (Pegasus is video-only); pipe those through the local engine instead.

        Returns one string per input path, matching QwenEngine.generate_batch.
        Errors are returned per-item as "Error: ..." so the CLI skips writing
        them, exactly like the local engine.
        """
        if not self.client:
            return ["Error: TwelveLabs client not loaded"] * len(file_paths)

        from twelvelabs.types.video_context import VideoContext_Url  # noqa: F401 (parity import)

        # Pegasus rejects max_tokens below its server-side minimum.
        tokens = max(int(max_tokens), PEGASUS_MIN_TOKENS)

        results = []
        for f_path in file_paths:
            if stop_event and stop_event():
                return []
            if not self._is_video(f_path):
                results.append("Error: Pegasus backend supports video only "
                               "(use the local engine for images).")
                continue
            try:
                video = self._video_context(f_path, log_callback=log_callback)
                resp = self.client.analyze(
                    model_name=self.model_name,
                    video=video,
                    prompt=prompt_text,
                    max_tokens=tokens,
                )
                clean = (resp.data or "").strip()
                if trigger_word and trigger_word.strip():
                    clean = f"{trigger_word.strip()}, {clean}"
                results.append(clean)
                if log_callback:
                    log_callback(f"✅ Captioned {os.path.basename(f_path)}")
            except Exception as e:
                msg = str(e)
                print(f"Pegasus Error on {f_path}: {msg}")
                results.append(f"Error: {msg}")
        return results
