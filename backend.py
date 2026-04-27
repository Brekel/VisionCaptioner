import os
import base64
import io
import torch
from file_utils import find_media_files, IMAGE_EXTS, VIDEO_EXTS
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from model_family import get_family, QwenFamily
import gc
import cv2
from PIL import Image, ImageOps
import logging
import numpy as np
import warnings
import platform

# --- OPTIONAL IMPORTS (Hardware Specific) ---
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# --- CHECK FOR LLAMA-CPP ---
HAS_LLAMA = False
Llava15ChatHandler = None
Gemma4ChatHandler = None

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError as e:
    HAS_LLAMA = False

if HAS_LLAMA:
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler, Qwen25VLChatHandler
    except ImportError:
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError:
            pass
        
    # Attempt to import Qwen3 if available (newer llama-cpp-python)
    try:
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    except ImportError:
        pass

    # Attempt to import Gemma4 if available (llama-cpp-python v0.3.35+)
    try:
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
    except ImportError:
        pass


# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*fast processor by default.*")
warnings.filterwarnings("ignore", message=".*max_length is ignored.*")
warnings.filterwarnings("ignore", message=".*The following generation flags are not valid.*")
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class StopTrigger(StoppingCriteria):
    def __init__(self, check_fn):
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        if self.check_fn and self.check_fn():
            return True 
        return False


class QwenEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = self.get_device_type()
        self.is_gguf = False
        self.family = QwenFamily()

    def get_device_type(self):
        """Detects the available hardware acceleration."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def find_files(self, folder_path, skip_existing=False, recursive=False):
        files = find_media_files(folder_path, exts=IMAGE_EXTS + VIDEO_EXTS,
                                 recursive=recursive, exclude_masks=False)
        
        mask_suffix = "masklabel"
        results = []

        for f in files:
            if mask_suffix in os.path.basename(f):
                continue
            
            if skip_existing:
                txt_path = os.path.splitext(f)[0] + ".txt"
                if os.path.exists(txt_path):
                    continue

            mask_found = None
            base_no_ext = os.path.splitext(f)[0]
            candidates = [ 
                f"{base_no_ext}-{mask_suffix}.png", 
                f"{base_no_ext}-{mask_suffix}.jpg", 
                f"{base_no_ext}_{mask_suffix}.png", 
                f"{base_no_ext}_{mask_suffix}.jpg", 
            ]
            for c in candidates:
                if os.path.exists(c):
                    mask_found = c
                    break
            
            results.append((f, mask_found))
            
        return results

    def unload_model(self):
        # 1. Clear compiled graphs
        try:
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
        except Exception:
            pass

        # 2. Delete Processor & Model
        if self.processor is not None:
            del self.processor
            self.processor = None

        if self.model is not None:
            # Try to move to CPU to detach hardware hooks
            try:
                self.model.to("cpu")
            except Exception:
                pass
            del self.model
            self.model = None

        self.is_gguf = False
        self.family = QwenFamily()

        # 3. GC
        for _ in range(3):
            gc.collect()

        # 4. Clear Hardware Cache (Cross-Platform)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        
        return "Model unloaded. Memory cleared."

    def load_model(self, model_path, quantization_type="None", max_resolution=512, attn_impl="sdpa", use_compile=False, vision_token_budget=None, media_mode="image"):
        try:
            from model_probe import ModelProbe
            
            self.device = self.get_device_type()
            print(f"Loading model from: {model_path}...")
            print(f"Platform: {platform.system()} | Device: {self.device}")
            
            self.unload_model()

            # --- PROBE THE MODEL ---
            probe_info = ModelProbe.probe(model_path)
            if "error" in probe_info:
                return False, f"Probe Failed: {probe_info['error']}"
                
            print(f"Probe Result: {probe_info}")
            
            # --- GGUF HANDLING ---
            if probe_info.get("format") == "gguf":
                if not HAS_LLAMA:
                    return False, "LLAMA_CPP_NOT_INSTALLED"

                # Verify Llama version
                try:
                    import llama_cpp
                    curr_ver = getattr(llama_cpp, "__version__", "0.0.0")
                    # Simple version check (parsing logic omitted for brevity, assuming user has recent version if they installed requirements)
                    print(f"llama-cpp-python version: {curr_ver}")
                except:
                    pass

                print(f"Loading GGUF: {os.path.basename(model_path)}")
                n_gpu_layers = -1 if self.device in ["cuda", "mps"] else 0
                
                # Vision Handler Logic
                chat_handler = None
                mmproj_path = None
                
                backend_type = probe_info.get("backend", "")

                if probe_info.get("unified_vision"):
                    print("✅ Probe detected Unified Vision Model.")
                    if backend_type == "gemma_gguf":
                        if Gemma4ChatHandler is None:
                            return False, (
                                "Gemma 4 GGUF requires llama-cpp-python v0.3.35+ with Gemma4ChatHandler. "
                                "Please update your llama-cpp-python installation."
                            )
                        chat_handler = Gemma4ChatHandler(clip_model_path=model_path, verbose=False)
                        mmproj_path = model_path
                    else:
                        try:
                            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                            chat_handler = Qwen25VLChatHandler(clip_model_path=model_path, verbose=False)
                            mmproj_path = model_path
                        except ImportError:
                            print("⚠️ Qwen25VLChatHandler not found in llama_cpp.")

                elif probe_info.get("mmproj_detected"):
                    mmproj_path = probe_info["mmproj_detected"]
                    print(f"✅ Probe detected compatible projector: {os.path.basename(mmproj_path)}")

                    if backend_type == "gemma_gguf":
                        if Gemma4ChatHandler is None:
                            return False, (
                                "Gemma 4 GGUF requires llama-cpp-python v0.3.35+ with Gemma4ChatHandler. "
                                "Please update your llama-cpp-python installation."
                            )
                        chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False)
                    else:
                        try:
                            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)
                        except:
                            try:
                                from llama_cpp.llama_chat_format import Llava15ChatHandler
                                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
                            except:
                                pass
                else:
                    model_base = os.path.splitext(os.path.basename(model_path))[0]
                    print("⚠️ Text-Only GGUF — no built-in vision and no matching mmproj found.")
                    print(f"   To enable vision, place an mmproj file next to the model and name it")
                    print(f"   to share the model name, e.g.: {model_base}-mmproj-BF16.gguf")

                # Pick context window + prompt-ingest chunk based on the active media mode.
                # n_ctx drives the KV-cache size and n_batch the prefill scratch buffer; both are
                # the dominant VRAM consumers for GGUF on top of the weights themselves.
                if media_mode == "video":
                    n_ctx, n_batch = 32768, 2048
                else:
                    n_ctx, n_batch = 8192, 512
                print(f"GGUF mode: {media_mode} -> n_ctx={n_ctx}, n_batch={n_batch}")

                # Load Llama
                try:
                    llm_kwargs = {
                        "model_path": model_path,
                        "n_ctx": n_ctx,
                        "n_batch": n_batch,
                        "n_gpu_layers": n_gpu_layers,
                        "verbose": False,
                        "chat_handler": chat_handler,
                    }
                    self.model = Llama(**llm_kwargs)
                    self.is_gguf = True
                    self.family = get_family(probe_info, model_path)
                    print(f"Model family: {self.family.name}")

                    msg = "GGUF Loaded ✅"
                    if chat_handler:
                        msg += " (Vision Enabled)"
                    else:
                        msg += " (Text Only)"
                    msg += f" — {media_mode} mode (n_ctx={n_ctx}, n_batch={n_batch})"
                    return True, msg
                    
                except Exception as e:
                    return False, f"Llama Load Failed: {e}"

            # --- HF TRANSFORMERS HANDLING ---
            self.is_gguf = False

            # Check backend recommendation
            backend_type = probe_info.get("backend", "unknown")
            print(f"Detected Backend: {backend_type}")

            # Select family strategy (Qwen, Gemma4, ...)
            self.family = get_family(probe_info, model_path)
            print(f"Model family: {self.family.name}")

            # Family-specific processor kwargs
            proc_kwargs = self.family.processor_kwargs(max_resolution=max_resolution, vision_token_budget=vision_token_budget)
            # use_fast=True is deprecated in favor of backend="torchvision", but the
            # latter is forwarded to video sub-processors (e.g. Gemma4VideoProcessor)
            # where `backend` is a read-only property and setattr raises.
            proc_kwargs.update({"trust_remote_code": True, "use_fast": True})

            try:
                self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
            except TypeError as e:
                # Some processors reject family-specific kwargs (e.g. visual_token_budget on older versions).
                # Retry without them.
                print(f"⚠️ AutoProcessor rejected family kwargs ({e}); retrying with defaults.")
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
            
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"

            # Attention Implementation
            if attn_impl == "flash_attention_2" and (not HAS_FLASH_ATTN or self.device != "cuda"):
                 print("⚠️ Flash Attn 2 unavailable, using SDPA.")
                 attn_impl = "sdpa"

            # DataType
            torch_dtype = torch.float16
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            
            # Quantization (BitsAndBytes)
            quant_config = None
            if quantization_type in ["Int8", "NF4"]:
                if self.device == "cuda" and HAS_BNB:
                     if quantization_type == "Int8":
                         quant_config = BitsAndBytesConfig(load_in_8bit=True)
                     elif quantization_type == "NF4":
                         quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True)
                else:
                     print(f"⚠️ Quantization {quantization_type} not supported on {self.device}")

            # Load Arguments
            load_args = {
                "dtype": torch_dtype,
                "trust_remote_code": True,
                "attn_implementation": attn_impl,
                "device_map": "auto" if self.device == "cuda" else "cpu"
            }
            if self.device == "mps": load_args["device_map"] = "cpu" # Move later

            if quant_config:
                load_args["quantization_config"] = quant_config
                use_compile = False

            # Load Model
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_args)
            
            if self.device == "mps":
                self.model.to("mps")
            
            if use_compile and self.device == "cuda" and not quant_config:
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except:
                    pass

            self.model.eval()
            return True, f"HF Model Loaded ({backend_type})"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _image_to_base64(self, image_obj):
        """Helper to convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        # Convert to RGB to ensure JPEG compatibility
        if image_obj.mode != 'RGB':
            image_obj = image_obj.convert('RGB')
        image_obj.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_video_frames(self, video_path, num_frames=8, log_callback=None, stop_event=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if log_callback: log_callback(f"⚠️ Error: Could not open video {os.path.basename(video_path)}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            if log_callback: log_callback(f"⚠️ Error: Video {os.path.basename(video_path)} has 0 frames.")
            return []
            
        if log_callback:
            log_callback(f"🎬 Video: {os.path.basename(video_path)} - Extracting {num_frames} frames.")

        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        indices = sorted(list(set(indices))) 
        
        frames = []
        for i in indices:
            if stop_event and stop_event():
                cap.release()
                return [] 
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames

    def apply_mask(self, image_path, mask_path):
        try:
            # Load with PIL to handle EXIF rotation automatically
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            
            # Convert to RGB (ensure consistent channels)
            pil_img = pil_img.convert("RGB")
            
            # Convert PIL to OpenCV format (numpy array)
            # PIL is RGB, OpenCV expects BGR
            img = np.array(pil_img)
            img = img[:, :, ::-1].copy() 

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: return None

            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            # Convert back to RGB for the AI
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(masked_img)
        except Exception as e:
            print(f"Error applying mask: {e}")
            return None

    def generate_batch(self, file_paths, prompt_text="Describe this.", trigger_word="", frame_count=8, mask_paths=None, max_tokens=1024, log_callback=None, stop_event=None):
        if not self.model: return ["Error: Model not loaded"] * len(file_paths)

        # --- GGUF GENERATION BRANCH ---
        if self.is_gguf:
            results = []
            for i, f_path in enumerate(file_paths):
                if stop_event and stop_event(): return []
                
                try:
                    # 1. Determine if video or image
                    ext = os.path.splitext(f_path)[1].lower()
                    is_video = ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']
                    mask_path = mask_paths[i] if mask_paths and i < len(mask_paths) else None
                    
                    # 2. Load image(s)
                    pil_images = []  # List of PIL images to send
                    
                    if is_video:
                        # Video: Extract multiple frames (matches reference implementation)
                        frames = self.extract_video_frames(f_path, num_frames=frame_count, stop_event=stop_event)
                        if frames:
                            pil_images = frames
                            if log_callback:
                                log_callback(f"🎬 Video: Extracted {len(frames)} frames for GGUF")
                        else:
                            results.append("[Video Error]")
                            continue
                    else:
                        # Single image
                        pil_img = None
                        if mask_path and os.path.exists(mask_path):
                            pil_img = self.apply_mask(f_path, mask_path)
                        else:
                            try:
                                pil_img = Image.open(f_path)
                                pil_img = ImageOps.exif_transpose(pil_img)
                            except Exception as e:
                                print(f"Error loading image {f_path}: {e}")
                                results.append(f"Error: {e}")
                                continue
                        pil_images = [pil_img]
                    
                    # 3. Construct Message
                    # Check if we have a valid chat_handler for vision
                    has_vision = hasattr(self.model, 'chat_handler') and self.model.chat_handler is not None
                    
                    if has_vision:
                        # Vision Mode: Send all images + Text
                        # Build content list with text first, then all images
                        
                        # Apply prompt modification for masks
                        local_prompt = prompt_text
                        if mask_path and os.path.exists(mask_path):
                            local_prompt += " The background is masked and transparent. Describe the foreground subject ONLY. Do not mention the background or transparency."
                            
                        content = [{"type": "text", "text": local_prompt}]
                        b64_images = []
                        for pil_img in pil_images:
                            b64 = self._image_to_base64(pil_img)
                            b64_images.append(b64)
                            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": content}
                        ]
                    else:
                        # Text-Only Fallback (No mmproj/ChatHandler)
                        print("⚠️ No vision handler - sending text only (caption will be hallucinated!)")
                        b64_images = []
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt_text}
                        ]

                    # 4. Generate
                    # Temperature 0.2 is usually good for factual descriptions, repeat_penalty=1.1 helps prevent repeating text
                    output = self.model.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.2,
                        repeat_penalty=1.1, # this prevents repeating text, best to keep this > 1 but not too high
                    )
                    
                    text = output["choices"][0]["message"]["content"]
                    
                    # Format Output (family.clean_output strips thinking tokens / turn markers for Gemma)
                    clean = self.family.clean_output(self.processor, text).strip()
                    if trigger_word and trigger_word.strip():
                        clean = f"{trigger_word.strip()}, {clean}"
                    results.append(clean)
                    
                    # Garbage Collection for GGUF
                    if b64_images:
                        del b64_images
                    del messages, output, pil_images
                    gc.collect()

                except Exception as e:
                    err_msg = str(e)
                    print(f"GGUF Error on {f_path}: {err_msg}")
                    
                    if "mtmd" in err_msg or "multimodal" in err_msg.lower():
                        model_base = os.path.splitext(os.path.basename(model_path))[0]
                        results.append(
                            f"Error: Vision failed — the mmproj file may be missing or mismatched. "
                            f"Download the matching *mmproj*.gguf for this model and name it to share "
                            f"the model name, e.g.: {model_base}-mmproj-BF16.gguf"
                        )
                    else:
                        results.append(f"Error: {err_msg}")
            
            return results

        # --- STANDARD TRANSFORMERS GENERATION ---
        try:
            texts = []
            all_image_inputs = []
            all_video_inputs = []

            for i, f_path in enumerate(file_paths):
                if stop_event and stop_event(): return []

                ext = os.path.splitext(f_path)[1].lower()
                is_video = ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']
                mask_path = mask_paths[i] if mask_paths and i < len(mask_paths) else None

                pil_frames = None
                final_image_obj = None
                load_error = False

                if is_video:
                    pil_frames = self.extract_video_frames(f_path, num_frames=frame_count, log_callback=log_callback, stop_event=stop_event)
                    if not pil_frames:
                        if stop_event and stop_event(): return []
                        load_error = True
                else:
                    if mask_path and os.path.exists(mask_path):
                        final_image_obj = self.apply_mask(f_path, mask_path)
                    else:
                        try:
                            pil_img = Image.open(f_path)
                            pil_img = ImageOps.exif_transpose(pil_img)
                            final_image_obj = pil_img
                        except Exception as e:
                            print(f"Error loading image {f_path}: {e}")
                            final_image_obj = f_path

                # Apply prompt modification for masks
                local_prompt = prompt_text
                if mask_path and os.path.exists(mask_path) and not is_video:
                    local_prompt += " The background is masked and transparent. Describe the foreground subject ONLY. Do not mention the background or transparency."

                if load_error:
                    content = [{"type": "text", "text": "[Video Load Error] " + local_prompt}]
                else:
                    content = self.family.build_content_block(is_video, final_image_obj, pil_frames, local_prompt)

                messages = [{"role": "user", "content": content}]

                text = self.family.apply_template(self.processor, messages)
                texts.append(text)

                if stop_event and stop_event(): return []
                img_in, vid_in = self.family.extract_vision_inputs(messages)
                all_image_inputs.append(img_in)
                all_video_inputs.append(vid_in)

            if stop_event and stop_event(): return []

            if self.family.flatten_vision_inputs:
                # Qwen-style: flat list of all images/videos across the batch.
                final_image_inputs = [item for sublist in all_image_inputs if sublist for item in sublist]
                final_video_inputs = [item for sublist in all_video_inputs if sublist for item in sublist]
                if not final_image_inputs: final_image_inputs = None
                if not final_video_inputs: final_video_inputs = None
            else:
                # Gemma 4-style: nested per-sample lists so the processor can pair
                # each image list with the corresponding text prompt.
                final_image_inputs = [sublist if sublist else [] for sublist in all_image_inputs]
                final_video_inputs = [sublist if sublist else [] for sublist in all_video_inputs]
                if not any(final_image_inputs): final_image_inputs = None
                if not any(final_video_inputs): final_video_inputs = None

            inputs = self.processor(text=texts, images=final_image_inputs, videos=final_video_inputs, padding="longest", return_tensors="pt").to(self.model.device)

            del all_image_inputs, all_video_inputs, final_image_inputs, final_video_inputs
            gc.collect()

            stopping_criteria = None
            if stop_event:
                stopping_criteria = StoppingCriteriaList([StopTrigger(stop_event)])

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    do_sample=False, 
                    use_cache=True, 
                    stopping_criteria=stopping_criteria,
                    repetition_penalty=1.1, # Penalty for repeated tokens
                    no_repeat_ngram_size=3  # Hard block on repeating 3-word phrases
                )

            if stop_event and stop_event():
                del inputs, generated_ids
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    try: torch.mps.empty_cache()
                    except: pass
                return [] 

            generated_ids_trimmed = [ out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids) ]
            skip_special = self.family.decode_skip_special_tokens()
            output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=skip_special, clean_up_tokenization_spaces=False)

            del inputs, generated_ids, generated_ids_trimmed
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                try: torch.mps.empty_cache()
                except: pass

            final_results = []
            for txt in output_texts:
                clean = self.family.clean_output(self.processor, txt).strip()
                if trigger_word and trigger_word.strip():
                    clean = f"{trigger_word.strip()}, {clean}"
                final_results.append(clean)

            return final_results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return [f"Error: {str(e)}"] * len(file_paths)


class SAM3Engine:
    def __init__(self):
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def is_available(self):
        try:
            import sam3
            return True
        except Exception as e:
            print(f"⚠️ SAM3 Import Failed: {e}")
            return False

    def load_model(self, model_folder):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self.unload()

            ckpt_path = os.path.join(model_folder, "sam3.pt")
            if not os.path.exists(ckpt_path):
                candidates = glob.glob(os.path.join(model_folder, "*.pt"))
                if candidates:
                    ckpt_path = candidates[0]
                else:
                    return False, f"sam3.pt not found in {model_folder}"

            print(f"Loading SAM3 from {ckpt_path}...")
            
            self.model = build_sam3_image_model(
                checkpoint_path=ckpt_path, 
                device=self.device
            )
            
            self.processor = Sam3Processor(self.model)
            
            return True, "SAM3 Model Loaded"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed to load SAM3: {str(e)}"

    def generate_mask(self, image_input, prompt, max_dimension=1024, conf_threshold=0.25, expand_ratio=0.0):
        """
        image_input: Can be a file path (str) OR a PIL.Image object.
        expand_ratio: Float (e.g., 0.05 for 5% expansion). 
        """
        if not self.model or not self.processor:
            return None, "Model not loaded"

        try:
            # 1. Load Image (Handle Path vs Object)
            if isinstance(image_input, str):
                pil_img = Image.open(image_input)
                pil_img = ImageOps.exif_transpose(pil_img)
            else:
                # Assume it is already a PIL object
                pil_img = image_input

            pil_img = pil_img.convert("RGB")
            
            orig_w, orig_h = pil_img.size
            
            # 2. Resize if too big
            # (Create a copy to resize so we don't affect the original object passed in)
            proc_img = pil_img.copy()
            if max(orig_w, orig_h) > max_dimension:
                proc_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            # 3. Inference
            # SAM3 loads with mixed bf16/fp32 weights and expects autocast at call time;
            # without it some Torch/CUDA combos raise "mat1 and mat2 must have the same dtype".
            if self.device == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                import contextlib
                autocast_ctx = contextlib.nullcontext()
            with autocast_ctx:
                inference_state = self.processor.set_image(proc_img)
                output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            masks = output.get("masks")
            scores = output.get("scores")
            
            if masks is None: return None, "No masks returned"
            
            # .float() before .numpy() — autocast can leave these as bfloat16, which numpy doesn't support
            if hasattr(masks, "cpu"): masks = masks.float().cpu().numpy()
            if hasattr(scores, "cpu"): scores = scores.float().cpu().numpy()
            
            if masks.size == 0: return None, "No masks found"

            # 4. Filter & Flatten
            if scores is not None and scores.size > 0:
                scores = scores.flatten()
                valid_indices = scores > conf_threshold
                if not np.any(valid_indices):
                    return None, f"No detections above threshold {conf_threshold:.2f}"
                masks = masks[valid_indices]

            while masks.ndim > 2:
                masks = np.any(masks, axis=0)
            
            final_mask_uint8 = (masks * 255).astype(np.uint8)
            
            # 5. Cleanup (Morphological Opening)
            kernel_clean = np.ones((5,5), np.uint8)
            final_mask_uint8 = cv2.morphologyEx(final_mask_uint8, cv2.MORPH_OPEN, kernel_clean)
            
            # 6. Expansion (Dilation by Percentage)
            if expand_ratio > 0.0:
                h, w = final_mask_uint8.shape
                expansion_pixels = int(max(h, w) * expand_ratio)
                
                if expansion_pixels > 0:
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion_pixels*2+1, expansion_pixels*2+1))
                    final_mask_uint8 = cv2.dilate(final_mask_uint8, kernel_dilate)

            mask_img = Image.fromarray(final_mask_uint8)
            
            # 7. Resize back to original
            if mask_img.size != (orig_w, orig_h):
                mask_img = mask_img.resize((orig_w, orig_h), resample=Image.NEAREST)
            
            return mask_img, "Success"

        except Exception as e:
            if "memory" in str(e).lower():
                return None, "OOM: Image too large for VRAM"
            return None, str(e)

    def unload(self):
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            try:
                torch.mps.empty_cache()
            except:
                pass