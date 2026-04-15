import os
import time
import gc
import random
import platform
import datetime
import shutil
from PySide6.QtCore import QThread, Signal, QObject

# --- SIGNALS ---
class WorkerSignals(QObject):
    progress = Signal(int, float)
    total = Signal(int)
    log = Signal(str)
    image_processed = Signal(str, str)
    finished = Signal(float, float)
    error = Signal(str)

# --- WORKERS ---
class GPUMonitorWorker(QThread):
    stats_update = Signal(float, float, float, int, bool)

    def run(self):
        is_supported = platform.system() in ["Windows", "Linux"]
        if not is_supported:
            self.stats_update.emit(0, 0, 0, 0, False)
            return

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            alpha = 0.2 
            smoothed_core = 0.0
            smoothed_mem = 0.0
            
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            smoothed_mem = (mem.used / mem.total) * 100
            smoothed_core = float(util.gpu)

            while not self.isInterruptionRequested():
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    used_gb = mem.used / (1024**3)
                    total_gb = mem.total / (1024**3)
                    
                    cur_vram = (mem.used / mem.total) * 100
                    cur_core = float(util.gpu)
                    
                    smoothed_core = (cur_core * alpha) + (smoothed_core * (1 - alpha))
                    smoothed_mem = (cur_vram * alpha) + (smoothed_mem * (1 - alpha))

                    self.stats_update.emit(used_gb, total_gb, smoothed_mem, int(smoothed_core), True)
                except:
                    pass
                self.msleep(250)
                
            pynvml.nvmlShutdown()
        except:
            self.stats_update.emit(0, 0, 0, 0, False)

class TestWorker(QThread):
    result_ready = Signal(str, str, float) 
    error = Signal(str)
    log_signal = Signal(str) 
    
    def __init__(self, engine, image_folder, settings, mode="random"):
        super().__init__()
        self.engine = engine
        self.image_folder = image_folder
        self.settings = settings
        self.mode = mode

    def run(self):
        recursive = self.settings.get('recursive', False)
        pairs = self.engine.find_files(self.image_folder, skip_existing=False, recursive=recursive)
        if not pairs:
            self.error.emit("No images found.")
            return

        target_pair = pairs[0] if self.mode == "first" else random.choice(pairs)
        target_file, mask_file = target_pair

        self.log_signal.emit(f"Test Target: {os.path.basename(target_file)}")

        use_masks = self.settings.get('use_masks', True)

        try:
            start_t = time.time()
            captions = self.engine.generate_batch(
                [target_file],
                self.settings['prompt'],
                self.settings['trigger'],
                frame_count=self.settings['frame_count'],
                mask_paths=[mask_file] if use_masks else [None],
                max_tokens=self.settings['max_tokens'],
                log_callback=lambda m: self.log_signal.emit(m),
                stop_event=self.isInterruptionRequested
            )
            
            if self.isInterruptionRequested() or not captions:
                return

            self.result_ready.emit(target_file, captions[0], time.time() - start_t)
            gc.collect()
        except Exception as e:
            self.error.emit(str(e))

class CaptionWorker(QThread):
    def __init__(self, engine, image_folder, settings):
        super().__init__()
        self.engine = engine
        self.image_folder = image_folder
        self.settings = settings
        self.signals = WorkerSignals()
        self.is_running = True

    def run(self):
        recursive = self.settings.get('recursive', False)
        all_pairs = self.engine.find_files(self.image_folder, skip_existing=False, recursive=recursive)
        total = len(all_pairs)
        self.signals.total.emit(total)
        self.signals.log.emit(f"Found {total} files.")

        if total == 0:
            self.signals.finished.emit(0, 0)
            return

        batch_size = self.settings['batch_size']
        start_time_global = time.time() # For final summary
        processed_count = 0
        
        # Exponential Moving Average for smoother ETA
        avg_speed = 0.0 
        alpha = 0.3 

        # Calculate Total Batches for display
        total_batches = (total + batch_size - 1) // batch_size

        def stop_check():
            return self.isInterruptionRequested() or not self.is_running

        for i in range(0, total, batch_size):
            if not self.is_running: break

            batch = all_pairs[i : i + batch_size]
            current_batch_num = (i // batch_size) + 1
            
            files = []
            masks = []
            skipped = 0
            
            # Filter skip/process
            use_masks = self.settings.get('use_masks', True)
            for fpath, maskpath in batch:
                txt_path = os.path.splitext(fpath)[0] + ".txt"
                if self.settings['skip_existing'] and os.path.exists(txt_path):
                    skipped += 1
                else:
                    files.append(fpath)
                    masks.append(maskpath if use_masks else None)

            # 1. Handle SKIPPED files
            if skipped > 0:
                processed_count += skipped
                # We emit -1.0 for speed to tell UI "Don't update speed, just update progress bar"
                self.signals.progress.emit(processed_count, -1.0) 

            # 2. Handle GENERATED files
            if files:
                range_end = min(i + batch_size, total)
                msg = (f"⚙️ Batch {current_batch_num}/{total_batches} | "
                       f"Processing {len(files)} items (Files {i+1}-{range_end} of {total})...")
                self.signals.log.emit(msg)
                
                # Measure ONLY the generation time
                t0 = time.time()
                try:
                    captions = self.engine.generate_batch(
                        files, 
                        self.settings['prompt'], 
                        self.settings['trigger'], 
                        frame_count=self.settings['frame_count'], 
                        mask_paths=masks, 
                        max_tokens=self.settings['max_tokens'],
                        log_callback=lambda m: self.signals.log.emit(m),
                        stop_event=stop_check
                    )
                    
                    if not self.is_running: break
                    if not captions: break # Likely stopped

                    # Save files (Error logic handled in previous fix)
                    for idx, f_path in enumerate(files):
                        caption = captions[idx]
                        if caption.startswith("Error") or caption.startswith("[Video Load Error]"):
                            self.signals.log.emit(f"⚠️ Skipping save for {os.path.basename(f_path)}: {caption}")
                            continue

                        txt_path = os.path.splitext(f_path)[0] + ".txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(caption)
                    
                    # Calculate Speed
                    t1 = time.time()
                    batch_duration = t1 - t0
                    current_speed = batch_duration / len(files) # Seconds per item

                    # Smooth the speed using EMA
                    if avg_speed == 0.0:
                        avg_speed = current_speed
                    else:
                        avg_speed = (alpha * current_speed) + ((1 - alpha) * avg_speed)

                    processed_count += len(files)
                    
                    # Emit new count AND valid speed
                    self.signals.progress.emit(processed_count, avg_speed)
                    
                    self.signals.image_processed.emit(files[-1], captions[-1])
                    gc.collect()

                except Exception as e:
                    self.signals.error.emit(str(e))
                    break
            
            if processed_count >= total:
                break

        elapsed_total = time.time() - start_time_global
        self.signals.finished.emit(elapsed_total, avg_speed)

    def stop(self):
        self.is_running = False
        self.requestInterruption()

class ScanWorker(QThread):
    finished = Signal(dict)
    log = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.models_dir = os.path.join(os.getcwd(), 'models')
        
    def run(self):
        # Import locally to avoid circular imports
        from model_probe import ModelProbe
        import glob
        
        # Run the probe on the directory
        if not os.path.exists(self.models_dir):
            self.finished.emit({})
            return
            
        # We manually walk and probe
        results = {}
        
        # Load Cache ONCE
        cache = ModelProbe.load_cache()
        
        # Prune Cache First
        try:
             removed = ModelProbe.prune_cache(cache)
             if removed > 0:
                 self.log.emit(f'Pruned {removed} stale entries from model cache.')
        except:
             pass
        
        # 1. Folders
        subdirs = [os.path.join(self.models_dir, d) for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
        for d in subdirs:
            if os.path.basename(d).startswith('_'): continue
            
            # Smart Logging: Only log if NOT in cache or mtime changed
            path = os.path.abspath(d)
            should_log = True
            if path in cache:
                if cache[path].get("_mtime") == os.path.getmtime(path):
                    should_log = False
            
            if should_log:
                self.log.emit(f"Scanning model folder: {os.path.basename(d)}...")
                
            info = ModelProbe.probe(d, cache=cache)
            if 'error' not in info:
                results[os.path.basename(d)] = info

        # 2. Files
        files = glob.glob(os.path.join(self.models_dir, '*.gguf'))
        for f in files:
            if os.path.basename(f).startswith('_') or 'mmproj' in os.path.basename(f).lower(): continue
            
            path = os.path.abspath(f)
            should_log = True
            if path in cache:
                if cache[path].get("_mtime") == os.path.getmtime(path):
                    should_log = False
            
            if should_log:
                self.log.emit(f"Scanning GGUF file: {os.path.basename(f)}...")
            
            info = ModelProbe.probe(f, cache=cache)
            if 'error' not in info:
                results[os.path.basename(f)] = info
                # Report Projector Status if we just scanned (or always? User said "when probing... log also")
                # Even if cached, user might want to know? 
                # User said "It can take a while... make it log what it's doing".
                # If cached, it's instant, so maybe no log needed.
                # But "when probing a gguf file it would be nice that when it's done the log also shows..."
                # I'll stick to logging ONLY if we did a fresh scan (should_log=True) to keep it clean as requested.
                if should_log and not info.get('unified_vision', False):
                    proj = info.get('mmproj_detected')
                    if proj:
                        self.log.emit(f"  -> Found external projector: {os.path.basename(proj)}")
                    else:
                        self.log.emit(f"  -> No matching projector found.")
                
        # Save cache after scan
        ModelProbe.save_cache(cache)
        self.finished.emit(results)
        self.requestInterruption()

class ModelLoaderWorker(QThread):
    finished = Signal(bool, str)
    def __init__(self, engine, path, quant, res, attn_impl="sdpa", use_compile=False, vision_token_budget=None):
        super().__init__()
        self.engine, self.path, self.quant, self.res = engine, path, quant, res
        self.attn_impl = attn_impl
        self.use_compile = use_compile
        self.vision_token_budget = vision_token_budget
    def run(self):
        try:
            success, msg = self.engine.load_model(self.path, self.quant, self.res, self.attn_impl, self.use_compile, vision_token_budget=self.vision_token_budget)
            self.finished.emit(success, msg)
        except Exception as e:
            self.finished.emit(False, f"Critical Worker Error: {e}")

class CropWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, file_pairs, output_dir, unused_dir=None, ignore_no_mask=True):
        super().__init__()
        self.file_pairs = file_pairs # List of tuples: (image_path, mask_path_or_None, has_alpha)
        self.output_dir = output_dir
        # Where to move images whose mask is empty (would crop to nothing).
        # Defaults to a sibling 'unused' folder next to 'uncropped'.
        self.unused_dir = unused_dir or os.path.join(os.path.dirname(output_dir), "unused")
        self.ignore_no_mask = ignore_no_mask
        self.is_running = True
        
    def run(self):
        import concurrent.futures
        from PIL import Image, ImageOps 
        # Import internally to ensure we don't have circular dep issues or pollution
        
        total_items = len(self.file_pairs)
        completed_count = 0
        
        # Determine optimal worker count
        # IO bound mixed with CPU bound (image processing)
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_crop, item): item for item in self.file_pairs}
            
            for future in concurrent.futures.as_completed(future_to_file):
                if not self.is_running:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                try:
                    result_msg = future.result()
                    if result_msg:
                        self.log.emit(result_msg)
                except Exception as exc:
                    self.log.emit(f"❌ Error processing: {exc}")
                
                completed_count += 1
                self.progress.emit(completed_count)
        
        self.finished.emit()

    def process_single_crop(self, item):
        # Unpack
        f_path, mask_source, source_type = item 
        # source_type: 'subfolder', 'file', 'alpha'
        
        if not self.is_running: return None
        
        try:
            from PIL import Image, ImageOps
            import numpy as np

            uncropped_image_path = os.path.join(self.output_dir, os.path.basename(f_path))
            filename = os.path.basename(f_path)
            base_name = os.path.splitext(filename)[0]

            # 1. Load Image
            pil_image = Image.open(f_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            
            bbox = None
            pil_mask = None

            # 2. Extract BBox based on source logic
            if source_type == 'subfolder' or source_type == 'file':
                # Load Mask from file
                pil_mask = Image.open(mask_source)
                bbox = pil_mask.getbbox()
                
            elif source_type == 'alpha':
                if pil_image.mode in ('RGBA', 'LA') or (pil_image.mode == 'P' and 'transparency' in pil_image.info):
                    img_rgba = pil_image.convert("RGBA")
                    # Invert alpha to get mask (Alpha 0 = Transparent/Background, 255 = Opaque/Foreground)
                    # wait, getbbox works on non-zero regions.
                    # Usually Alpha channel: 255 is Variable, 0 is Transparent.
                    # We want to crop to the VISIBLE area (Alpha > 0).
                    alpha = img_rgba.split()[-1]
                    bbox = alpha.getbbox()
                    # We don't have a separate mask file to save/crop for alpha mode specifically 
                    # unless we want to extract it, but usually alpha crop just modifies the image.

            if not bbox:
                # Nothing to crop to — move the image (and its mask / caption)
                # to the 'unused' folder since the whole image is unusable.
                return self._move_to_unused(f_path, mask_source, source_type, filename)

            # 3. Handle Backup & Crop
            
            # Subfolder/File Mask Logic
            if pil_mask:
                uncropped_masks_dir = os.path.join(self.output_dir, "masks")
                if not os.path.exists(uncropped_masks_dir):
                    os.makedirs(uncropped_masks_dir, exist_ok=True)
                    
                # Determining backup mask name
                if source_type == 'subfolder':
                    uncropped_mask_path = os.path.join(uncropped_masks_dir, f"{base_name}.png")
                else:
                    # Same dir mask (suffix)
                    # We should probably just move it to uncropped/masks to be clean
                    uncropped_mask_path = os.path.join(uncropped_masks_dir, os.path.basename(mask_source))

                cropped_image = pil_image.crop(bbox)
                cropped_mask = pil_mask.crop(bbox)

                # Save Originals
                pil_image.save(uncropped_image_path, compress_level=1)
                pil_mask.save(uncropped_mask_path, compress_level=1)

                # Overwrite with Cropped
                cropped_image.save(f_path, compress_level=1)
                
                # Careful not to overwrite if source was different
                # IF source was subfolder, overwrite subfolder
                # IF source was file, overwrite file
                cropped_mask.save(mask_source, compress_level=1)

            else:
                # Alpha Logic
                # Save Original
                pil_image.save(uncropped_image_path, compress_level=1)
                
                # Crop
                cropped_image = pil_image.crop(bbox)
                
                # Overwrite
                cropped_image.save(f_path, compress_level=1)

            return None # Success (silent to avoid log spam? or return msg?)
            # return f"✅ Cropped: {filename}" 

        except Exception as e:
            return f"❌ Error {filename}: {str(e)}"

    def stop(self):
        self.is_running = False

    def _move_to_unused(self, f_path, mask_source, source_type, filename):
        """Move image + sibling caption + separate mask file to 'unused/'.

        Used when a mask is empty and cropping would produce nothing — the whole
        image is treated as unusable, mirroring the 'delete' behaviour in Review.
        """
        try:
            os.makedirs(self.unused_dir, exist_ok=True)

            def _pick_dest(src_path):
                dst = os.path.join(self.unused_dir, os.path.basename(src_path))
                if os.path.exists(dst):
                    stem, ext = os.path.splitext(dst)
                    dst = f"{stem}_dup{ext}"
                return dst

            shutil.move(f_path, _pick_dest(f_path))

            # Move sibling caption if present.
            base = os.path.splitext(f_path)[0]
            txt_path = base + ".txt"
            if os.path.exists(txt_path):
                shutil.move(txt_path, _pick_dest(txt_path))

            # Move separate mask file (file / subfolder modes) if present.
            if source_type in ('file', 'subfolder') and mask_source and os.path.exists(mask_source):
                shutil.move(mask_source, _pick_dest(mask_source))

            return f"🗑️ Moved to 'unused/' (empty mask): {filename}"
        except Exception as e:
            return f"❌ Could not move {filename} to 'unused/': {e}"


class LlamaCppInstallWorker(QThread):
    """Background worker that detects the system and installs llama-cpp-python."""
    log = Signal(str)
    finished = Signal(bool, str)  # success, message

    def run(self):
        from llama_cpp_installer import run_install
        success, message = run_install(log=lambda msg: self.log.emit(msg))
        self.finished.emit(success, message)