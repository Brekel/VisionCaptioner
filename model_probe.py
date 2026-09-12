import os
import json
import glob
import re
import time

# Try to import gguf library (usually installed with llama-cpp-python)
try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

class ModelProbe:
    """
    Utility to inspect model files/folders and determine their capabilities and backend requirements.
    Supports caching to speed up GGUF probing.
    """
    
    CACHE_FILE = "model_cache.json"
    # Bumped whenever probing changes what it records, so entries written by an
    # older build are re-probed instead of trusted. v2: metadata-based mmproj pairing.
    CACHE_VERSION = 2

    @staticmethod
    def load_cache():
        if os.path.exists(ModelProbe.CACHE_FILE):
            try:
                with open(ModelProbe.CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    @staticmethod
    def save_cache(cache):
        try:
            with open(ModelProbe.CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    @staticmethod
    def probe(path, cache=None):
        """
        Main entry point. Probes a path and returns a dictionary of metadata.
        Uses cache if provided and mtime matches.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return {"error": "Path not found"}

        # Caching Logic
        mtime = os.path.getmtime(path)
        if cache is not None:
            if path in cache:
                cached_data = cache[path]
                # Check if file has been modified since cache, and whether the
                # projectors beside it still look the same (their mtimes are not
                # reflected in the model's own).
                if cached_data.get("_mtime") == mtime and \
                   cached_data.get("_v") == ModelProbe.CACHE_VERSION:
                    sig = cached_data.get("_mmproj_sig")
                    if sig is None or sig == ModelProbe._mmproj_signature(path):
                        return cached_data

        result = {}
        if os.path.isdir(path):
            result = ModelProbe._probe_folder(path)
        elif path.lower().endswith(".gguf"):
            result = ModelProbe._probe_gguf(path)
        else:
            result = {"format": "unknown", "type": "unknown"}

        # Update Cache
        if cache is not None and "error" not in result:
            result["_mtime"] = mtime
            result["_v"] = ModelProbe.CACHE_VERSION
            cache[path] = result
            
        return result

    @staticmethod
    def _probe_folder(folder_path):
        """
        Inspects a HuggingFace directory.
        """
        config_path = os.path.join(folder_path, "config.json")
        res = {
            "format": "hf_folder",
            "path": folder_path,
            "architecture": "unknown",
            "backend": "unknown",
            "unified_vision": False
        }

        if not os.path.exists(config_path):
            if "sam" in os.path.basename(folder_path).lower():
                if any(f.endswith(".pt") for f in os.listdir(folder_path)):
                     res.update({"architecture": "sam3", "backend": "sam3"})
                     return res
            return {"error": "No config.json found"}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 1. Architecture
            model_type = config.get("model_type", "").lower()
            res["architecture"] = model_type
            
            # 2. Backend Detection
            if model_type in ["qwen2_5_vl", "qwen2", "qwen2.5", "qwen3_vl"]:
                res["backend"] = "qwen_hf"
                res["unified_vision"] = True
            elif model_type.startswith("gemma4") or model_type == "gemma4":
                res["backend"] = "gemma_hf"
                res["unified_vision"] = bool(config.get("vision_config")) or True
            elif model_type == "mllama":
                res["backend"] = "llama_hf"
            elif "llava" in model_type:
                res["backend"] = "llava_hf"
            elif "sam" in model_type:
                 res["backend"] = "sam3"

            if res["backend"] == "unknown":
                 lower_path = folder_path.lower()
                 if "qwen" in lower_path and "vl" in lower_path:
                      res["backend"] = "qwen_hf"
                      res["unified_vision"] = True
                 elif "gemma-4" in lower_path or "gemma4" in lower_path:
                      res["backend"] = "gemma_hf"
                      res["unified_vision"] = True
                 elif "sam3" in lower_path:
                      res["backend"] = "sam3"

            return res
            
        except Exception as e:
            return {"error": f"Failed to parse config.json: {e}"}

    @staticmethod
    def _gguf_field(reader, name):
        """Raw first value of a GGUF metadata field, or None if absent/unreadable."""
        for field in reader.fields.values():
            if field.name != name:
                continue
            try:
                val = field.parts[field.data[0]]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                return val
            except Exception:
                return None
        return None

    @staticmethod
    def _gguf_str(reader, name):
        """String metadata value. Strings arrive as a list of byte values."""
        val = ModelProbe._gguf_field(reader, name)
        if val is None:
            return None
        try:
            if isinstance(val, list):
                val = bytes([b for b in val if b != 0]).decode("utf-8", errors="ignore")
            elif hasattr(val, "decode"):
                val = val.decode("utf-8", errors="ignore")
        except Exception:
            return None
        return str(val).strip("\0")

    @staticmethod
    def _gguf_int(reader, name):
        """Integer metadata value (GGUF wraps scalars in a 1-element array)."""
        val = ModelProbe._gguf_field(reader, name)
        if isinstance(val, list):
            val = val[0] if val else None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _probe_gguf(file_path):
        """
        Inspects a GGUF file header.
        """
        res = {
            "format": "gguf",
            "path": file_path,
            "architecture": "unknown",
            "backend": "llama_cpp",
            "unified_vision": False,
            "mmproj_detected": None
        }

        if not HAS_GGUF:
            return {"error": "python 'gguf' library not found. Please `pip install gguf`."}

        try:
            reader = gguf.GGUFReader(file_path, mode='r')
            
            # TENSOR CHECK
            tensor_names = {t.name for t in reader.tensors}
            
            has_vision_tensors = any("clip" in t for t in tensor_names) or \
                                 any("v.patch_embed" in t for t in tensor_names) or \
                                 any("visual" in t for t in tensor_names) or \
                                 any("mm." in t for t in tensor_names)
            
            res["unified_vision"] = has_vision_tensors
            
            # Architecture Metadata (silent on decoding errors, fallback logic handles it)
            arch = ModelProbe._gguf_str(reader, "general.architecture")
            if arch:
                res["architecture"] = arch

            if res["architecture"] == "unknown":
                fname_lower = os.path.basename(file_path).lower()
                if "qwen" in fname_lower:
                    res["architecture"] = "qwen2"
                elif "gemma-4" in fname_lower or "gemma4" in fname_lower:
                    res["architecture"] = "gemma4"

            if res["architecture"].startswith("gemma4"):
                res["backend"] = "gemma_gguf"
            
            if not has_vision_tensors:
                # The width the text model expects from a projector; the pairing key.
                embd = ModelProbe._gguf_int(reader, f"{res['architecture']}.embedding_length")
                res["mmproj_detected"] = ModelProbe.find_matching_mmproj(file_path, main_embd=embd)
                # Projectors live beside the model, and dropping one in does not touch
                # the model's own mtime, so the cache needs its own fingerprint for them.
                res["_mmproj_sig"] = ModelProbe._mmproj_signature(file_path)

            return res

        except Exception as e:
            return {"error": f"GGUF Probe failed: {e}"}

    @staticmethod
    def _mmproj_signature(main_gguf_path):
        """Fingerprint of the projector files sitting beside a model."""
        folder = os.path.dirname(main_gguf_path)
        sig = []
        for p in sorted(glob.glob(os.path.join(folder, "*mmproj*.gguf"))):
            try:
                sig.append(f"{os.path.basename(p)}:{os.path.getmtime(p)}")
            except OSError:
                sig.append(os.path.basename(p))
        return sig

    @staticmethod
    def _projector_dim(mmproj_path):
        """Width this projector outputs; must equal the text model's embedding_length."""
        if not HAS_GGUF:
            return None
        try:
            reader = gguf.GGUFReader(mmproj_path, mode='r')
            return ModelProbe._gguf_int(reader, "clip.vision.projection_dim")
        except Exception:
            return None

    @staticmethod
    def _model_embd(main_gguf_path):
        """Embedding width of a text GGUF, read through its own architecture prefix."""
        if not HAS_GGUF:
            return None
        try:
            reader = gguf.GGUFReader(main_gguf_path, mode='r')
            arch = ModelProbe._gguf_str(reader, "general.architecture")
            if not arch:
                return None
            return ModelProbe._gguf_int(reader, f"{arch}.embedding_length")
        except Exception:
            return None

    @staticmethod
    def find_matching_mmproj(main_gguf_path, main_embd=None):
        folder = os.path.dirname(main_gguf_path)
        filename = os.path.basename(main_gguf_path)
        base_name = os.path.splitext(filename)[0]

        candidates = glob.glob(os.path.join(folder, "*mmproj*.gguf"))
        if not candidates:
            print(f"ℹ️ No *mmproj*.gguf files found in {folder}")
            print(f"  Tip: Download the matching mmproj file and place it next to your model.")
            print(f"  Name it to share the model name, e.g.: {base_name}-mmproj-BF16.gguf")
            return None

        # Pair on metadata before filenames: a projector fits only if the width it
        # outputs (clip.vision.projection_dim) equals the model's embedding_length.
        # That survives generic names like "mmproj-BF16.gguf" and rejects a projector
        # belonging to another model that merely shares a quant suffix.
        if main_embd is None:
            main_embd = ModelProbe._model_embd(main_gguf_path)

        if main_embd:
            typed = [(c, ModelProbe._projector_dim(c)) for c in candidates]
            mismatched = [(c, d) for c, d in typed if d is not None and d != main_embd]
            exact = [c for c, d in typed if d == main_embd]
            unknown = [c for c, d in typed if d is None]

            if not exact and not unknown:
                print(f"⚠️ No *mmproj*.gguf in {folder} projects to {main_embd} dims — vision unavailable.")
                for c, d in mismatched:
                    print(f"  ✗ {os.path.basename(c)}: projects {d}-d")
                print(f"  Download the projector built for {filename}.")
                return None

            candidates = exact or unknown
            if exact and len(exact) == 1:
                print(f"✅ Projector matched on projection_dim={main_embd}: {os.path.basename(exact[0])}")
                if mismatched:
                    print(f"  ({len(mismatched)} other projector(s) here project a different width)")
                return exact[0]
            if not exact:
                print(f"ℹ️ Could not read clip.vision.projection_dim from any projector here;"
                      f" falling back to filename matching.")

        # Several (or untyped) candidates left - fall back to filename overlap.
        best_match = None
        best_score = -1   # a lone candidate scoring 0 still wins; 0 would reject it

        main_tokens = set(re.split(r'[._-]', base_name.lower()))
        # Quant/format tokens carry no identity - "q8"/"0" alone once paired a Qwen
        # projector with a Gemma model.
        skip = {'gguf', 'mmproj', 'model', 'lora',
                'q2', 'q3', 'q4', 'q5', 'q6', 'q8', 'f16', 'f32', 'bf16',
                'k', 's', 'm', 'l', '0', '1'}

        scored = []
        for cand in candidates:
            c_name = os.path.basename(cand)
            c_base = os.path.splitext(c_name)[0]
            c_tokens = set(re.split(r'[._-]', c_base.lower()))

            overlap = main_tokens.intersection(c_tokens) - skip
            score = len(overlap)
            scored.append((score, c_name, cand))

            if score > best_score:
                best_score = score
                best_match = cand

        # Log matching details so user can verify or fix naming
        if len(scored) > 1 or best_score < 2:
            scored.sort(key=lambda x: x[0], reverse=True)
            print(f"ℹ️ mmproj candidates for {filename}:")
            for sc, name, _ in scored:
                marker = " ← selected" if name == os.path.basename(best_match) else ""
                print(f"  score {sc}: {name}{marker}")
            if best_score < 2:
                print(f"  ⚠️ Low match score ({best_score}). If vision fails, rename the mmproj to share")
                print(f"  the model name, e.g.: {base_name}-mmproj-BF16.gguf")

        return best_match

    @staticmethod
    def prune_cache(cache):
        """
        Removes entries from cache that no longer exist on disk.
        """
        keys_to_remove = []
        for path in cache.keys():
            if not os.path.exists(path):
                keys_to_remove.append(path)
        
        if keys_to_remove:
            for k in keys_to_remove:
                del cache[k]
        
        return len(keys_to_remove)

    @staticmethod
    def print_report(root_folder):
        print(f"--- Scanning Models in {root_folder} ---")
        if not HAS_GGUF:
            print("⚠️ WARNING: 'gguf' python library not found. GGUF probing will fail.")
        
        # Load Cache
        cache = ModelProbe.load_cache()
        
        # Prune Cache
        removed_count = ModelProbe.prune_cache(cache)
        
        print(f"(Loaded {len(cache)} entries from cache. Pruned {removed_count} missing paths)")

        # 1. Folders
        subdirs = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
        for d in subdirs:
            # Skip folders starting with _
            if os.path.basename(d).startswith("_"):
                continue

            info = ModelProbe.probe(d, cache=cache)
            print(f"\n[Folder] {os.path.basename(d)}")
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Type: {info.get('architecture')}")
                print(f"  Backend: {info.get('backend')}")

        # 2. Files (GGUF)
        files = glob.glob(os.path.join(root_folder, "*.gguf"))
        files = [f for f in files if "mmproj" not in os.path.basename(f).lower()]
        
        for f in files:
            # Skip files starting with _ (if any)
            if os.path.basename(f).startswith("_"):
                continue

            info = ModelProbe.probe(f, cache=cache)
            print(f"\n[GGUF] {os.path.basename(f)}")
            if "error" in info:
                 print(f"  Error: {info['error']}")
            else:
                print(f"  Unified Vision: {info.get('unified_vision')}")
                if info.get('architecture') != 'unknown':
                     print(f"  Arch: {info.get('architecture')}")
                
                if info.get('mmproj_detected'):
                    print(f"  + Projector: {os.path.basename(info['mmproj_detected'])}")
                elif not info.get('unified_vision'):
                    print(f"  ⚠️ Text-Only (No Projector Found)")
                    
        # Save Cache
        ModelProbe.save_cache(cache)
        print("\n\n(Cache updated)")

if __name__ == "__main__":
    test_path = r"E:\_python_tools\VisionCaptioner\models"
    if len(os.sys.argv) > 1:
        test_path = os.sys.argv[1]
        
    if os.path.exists(test_path):
        ModelProbe.print_report(test_path)
    else:
        print("Model path not found for testing.")
