import os
import time
import gc
import random
import platform
import datetime
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
        pairs = self.engine.find_files(self.image_folder, skip_existing=False)
        if not pairs:
            self.error.emit("No images found.")
            return

        target_pair = pairs[0] if self.mode == "first" else random.choice(pairs)
        target_file, mask_file = target_pair
        
        self.log_signal.emit(f"Test Target: {os.path.basename(target_file)}")

        try:
            start_t = time.time()
            captions = self.engine.generate_batch(
                [target_file], 
                self.settings['prompt'], 
                self.settings['trigger'], 
                frame_count=self.settings['frame_count'], 
                mask_paths=[mask_file], 
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
        all_pairs = self.engine.find_files(self.image_folder, skip_existing=False)
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
            for fpath, maskpath in batch:
                txt_path = os.path.splitext(fpath)[0] + ".txt"
                if self.settings['skip_existing'] and os.path.exists(txt_path):
                    skipped += 1
                else:
                    files.append(fpath)
                    masks.append(maskpath)

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

class ModelLoaderWorker(QThread):
    finished = Signal(bool, str)
    def __init__(self, engine, path, quant, res, attn_impl="sdpa", use_compile=False):
        super().__init__()
        self.engine, self.path, self.quant, self.res = engine, path, quant, res
        self.attn_impl = attn_impl
        self.use_compile = use_compile
    def run(self):
        try:
            success, msg = self.engine.load_model(self.path, self.quant, self.res, self.attn_impl, self.use_compile)
            self.finished.emit(success, msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, f"Critical Worker Error: {e}")