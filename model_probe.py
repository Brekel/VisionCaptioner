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
                # Check if file has been modified since cache
                if cached_data.get("_mtime") == mtime:
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
                 elif "sam3" in lower_path:
                      res["backend"] = "sam3"

            return res
            
        except Exception as e:
            return {"error": f"Failed to parse config.json: {e}"}

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
            
            # Architecture Metadata
            for field in reader.fields.values():
                if field.name == "general.architecture":
                    try:
                        val = field.parts[field.data[0]]
                        
                        # Handle numpy arrays (common in gguf lib)
                        if hasattr(val, "tolist"):
                             val = val.tolist()
                             
                        # Fix for raw list of ints (ASCII)
                        if isinstance(val, list):
                             # Clean up 0 padding if present
                             val = bytes([b for b in val if b != 0]).decode('utf-8', errors='ignore')
                        elif hasattr(val, "decode"):
                            val = val.decode('utf-8', errors='ignore')
                            
                        res["architecture"] = str(val).strip('\0')
                    except Exception as e:
                        # Keep silent on decoding errors, fallback logic handles it
                        pass
            
            if res["architecture"] == "unknown":
                if "qwen" in os.path.basename(file_path).lower():
                    res["architecture"] = "qwen2"
            
            if not has_vision_tensors:
                res["mmproj_detected"] = ModelProbe.find_matching_mmproj(file_path)

            return res

        except Exception as e:
            return {"error": f"GGUF Probe failed: {e}"}

    @staticmethod
    def find_matching_mmproj(main_gguf_path):
        folder = os.path.dirname(main_gguf_path)
        filename = os.path.basename(main_gguf_path)
        base_name = os.path.splitext(filename)[0]
        
        candidates = glob.glob(os.path.join(folder, "*mmproj*.gguf"))
        if not candidates:
            return None
            
        best_match = None
        best_score = 0
        
        main_tokens = set(re.split(r'[._-]', base_name.lower()))
        
        for cand in candidates:
            c_name = os.path.basename(cand)
            c_base = os.path.splitext(c_name)[0]
            c_tokens = set(re.split(r'[._-]', c_base.lower()))
            
            skip = {'gguf', 'mmproj', 'q4', 'k', 'm', 'f16', 'model', 'lora'}
            overlap = main_tokens.intersection(c_tokens) - skip
            
            score = len(overlap)
            
            if score > best_score:
                best_score = score
                best_match = cand
                
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
