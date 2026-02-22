import os
import glob
import time
import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QProgressBar, QComboBox, QCheckBox, QTextEdit, QSpinBox, 
                               QGroupBox, QSplitter, QSizePolicy, QApplication, QLineEdit,
                               QMessageBox, QGridLayout)
from PySide6.QtCore import Qt, Signal
from gui_widgets import ResizableImageLabel
from gui_workers import ModelLoaderWorker, TestWorker, CaptionWorker
from gui_model_manager import ModelManagerDialog


class CaptionsTab(QWidget):
    # Signals to communicate with Main Window
    log_msg = Signal(str)
    request_lock = Signal(bool) # Lock shared UI (like tabs) during processing
    batch_finished = Signal()
    IGNORED_MODELS = ["sam3"]

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.worker = None
        self.test_worker = None
        self.loader_worker = None
        self.current_folder = ""
        self.PROMPTS_DIR = "prompts"
        self.start_time = 0.0

        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- LEFT PANEL (Settings) ---
        left_panel = QGroupBox("Generation Settings")
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(380)

        # Model
        left_layout.addWidget(QLabel("Model Path:"))
        
        model_select_layout = QHBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Select the local folder containing the Qwen-VL model files.\nPlace models in the 'models' subdirectory.")
        
        # Connect signal immediately
        self.model_combo.currentIndexChanged.connect(self.on_model_combo_changed)
        
        self.btn_manager = QPushButton("📥💾")
        self.btn_manager.setFixedWidth(70)
        self.btn_manager.setToolTip("Download new models or delete existing ones.")
        self.btn_manager.clicked.connect(self.open_model_manager)

        model_select_layout.addWidget(self.model_combo)
        model_select_layout.addWidget(self.btn_manager)
        
        left_layout.addLayout(model_select_layout)
        # ----------------------------------------
        
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Model")
        self.btn_load.clicked.connect(self.load_model)
        self.btn_load.setStyleSheet(self.get_btn_style("green"))
        self.btn_load.setToolTip("Load the selected model into memory (VRAM).")
        
        self.btn_unload = QPushButton("Unload")
        self.btn_unload.clicked.connect(self.unload_model)
        self.btn_unload.setEnabled(False)
        self.btn_unload.setStyleSheet(self.get_btn_style("red"))
        self.btn_unload.setToolTip("Free up VRAM by removing the model from memory.")

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_unload)
        left_layout.addLayout(btn_layout)

        left_layout.addSpacing(10)

        # Configs
        # Configs - Compact Grid Layout
        grid_model = QGridLayout()
        grid_model.setContentsMargins(0, 0, 0, 0)
        
        # Row 1: Quantization & Resolution
        grid_model.addWidget(QLabel("Quantization:"), 0, 0)
        self.combo_quant = QComboBox()
        self.combo_quant.addItems(["None (BF16 - Default)", "FP16 (Half Precision)", "Int8 (BitsAndBytes)", "NF4 (4-bit)"])
        self.combo_quant.setToolTip(
            "None: Native quality (BF16).\n"
            "FP16: Slightly less VRAM on older cards.\n"
            "Int8: ~50% VRAM usage, excellent quality. (CUDA Only)\n"
            "NF4: ~25% VRAM usage, good quality. (CUDA Only)"
        )
        grid_model.addWidget(self.combo_quant, 1, 0)

        grid_model.addWidget(QLabel("Max Resolution:"), 0, 1)
        self.combo_res = QComboBox()
        self.combo_res.addItem("336px (Super Fast)", 336)
        self.combo_res.addItem("512px (Fast)", 512)
        self.combo_res.addItem("768px (Balanced)", 768)
        self.combo_res.addItem("1024px (Detailed)", 1024)
        self.combo_res.addItem("1280px (Max/Native)", 1280)
        self.combo_res.setCurrentIndex(2) # Default 768px
        self.combo_res.setToolTip("Maximum side length for resizing images.\nHigher = Better detail but slower and more VRAM.")
        grid_model.addWidget(self.combo_res, 1, 1)

        
        left_layout.addLayout(grid_model)

        left_layout.addSpacing(10)

        # Grid for Numeric Params
        grid_params = QGridLayout()
        grid_params.setContentsMargins(0, 0, 0, 0)
        
        grid_params.addWidget(QLabel("Batch Size:"), 0, 0)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(8)
        self.spin_batch.setToolTip("Number of images to process simultaneously.\nIncrease for speed, decrease if running out of VRAM.")
        grid_params.addWidget(self.spin_batch, 1, 0)

        grid_params.addWidget(QLabel("Video Frames:"), 0, 1)
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(1, 64)
        self.spin_frames.setValue(16)
        self.spin_frames.setToolTip("Number of frames to extract from video files for analysis.\nMore frames = better understanding of motion, but significantly more VRAM.")
        grid_params.addWidget(self.spin_frames, 1, 1)

        grid_params.addWidget(QLabel("Max Tokens:"), 0, 2)
        self.spin_tokens = QSpinBox()
        self.spin_tokens.setRange(50, 4096)
        self.spin_tokens.setValue(1024)
        self.spin_tokens.setToolTip("Maximum length of the generated text description.")
        grid_params.addWidget(self.spin_tokens, 1, 2)
        
        left_layout.addLayout(grid_params)

        left_layout.addSpacing(5)
        self.chk_skip = QCheckBox("Skip existing .txt files")
        self.chk_skip.setToolTip("If checked, files that already have a corresponding .txt caption file will be ignored.")
        left_layout.addWidget(self.chk_skip)

        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Trigger Word (Optional):"))
        self.txt_trigger = QLineEdit()
        self.txt_trigger.setPlaceholderText("e.g. style of xyz, ")
        self.txt_trigger.setToolTip("Text to automatically prepend to every caption (e.g., an activation keyword).")
        left_layout.addWidget(self.txt_trigger)

        # Prompts
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("System Prompt:"))
        self.combo_prompts = QComboBox()
        self.combo_prompts.addItem("Custom (Type below...)", "")
        self.populate_prompts()
        self.combo_prompts.currentIndexChanged.connect(self.on_prompt_preset_changed)
        self.combo_prompts.setToolTip("Choose a pre-defined instruction set for the AI.")
        left_layout.addWidget(self.combo_prompts)

        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlainText("Analyze the image and write a single concise sentence that describes the main subject and setting. Keep it grounded in visible details only.")
        self.txt_prompt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.txt_prompt.setToolTip("The main instruction given to the AI (e.g., 'Describe the image').")
        left_layout.addWidget(self.txt_prompt)

        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Append to System Prompt (Always active):"))
        self.txt_suffix = QTextEdit()
        self.txt_suffix.setPlainText("Output only the caption text.\nDo NOT use any ambiguous language.")
        self.txt_suffix.setFixedHeight(60)
        self.txt_suffix.setToolTip("Text added to the end of the system prompt.\nUseful for enforcing rules like 'No moralizing' or 'Be concise'.")
        left_layout.addWidget(self.txt_suffix)

        main_layout.addWidget(left_panel)

        # --- RIGHT PANEL (Preview & Controls) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(2)
        
        self.lbl_preview = ResizableImageLabel()
        self.lbl_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_preview.setScaledContents(False)
        self.lbl_preview.setToolTip("Preview of the most recently processed image/video frame.")
        
        # Group file label and text edit into one widget to avoid extra splitter handle
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_filename = QLabel("")
        self.lbl_filename.setAlignment(Qt.AlignCenter)
        self.lbl_filename.setStyleSheet("font-weight: bold; color: #ccc; margin-top: 1px; margin-bottom: 1px;")
        
        self.txt_preview = QTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setPlaceholderText("Generated caption will appear here...")
        self.txt_preview.setMinimumHeight(50)
        self.txt_preview.setToolTip("The generated caption output for the previewed image.")
        
        bottom_layout.addWidget(self.lbl_filename)
        bottom_layout.addWidget(self.txt_preview)
        
        self.splitter.addWidget(self.lbl_preview)
        self.splitter.addWidget(bottom_container)
        self.splitter.setSizes([500, 150])
        right_layout.addWidget(self.splitter, stretch=1)

        # Buttons
        test_layout = QHBoxLayout()
        self.btn_test_first = QPushButton("TEST FIRST IMAGE/VIDEO")
        self.btn_test_first.clicked.connect(lambda: self.run_test("first"))
        self.btn_test_first.setMinimumHeight(40)
        self.btn_test_first.setToolTip("Process the first image in the folder alphabetically. Useful for comparing settings on the same file.")

        self.btn_test_rand = QPushButton("TEST RANDOM IMAGE/VIDEO")
        self.btn_test_rand.clicked.connect(lambda: self.run_test("random"))
        self.btn_test_rand.setMinimumHeight(40)
        self.btn_test_rand.setToolTip("Pick one random file from the folder and caption it to test settings.")

        for b in [self.btn_test_first, self.btn_test_rand]:
            b.setEnabled(False)
            b.setStyleSheet(self.get_btn_style("blue"))
            test_layout.addWidget(b)
        right_layout.addLayout(test_layout)

        proc_layout = QHBoxLayout()
        self.btn_start = QPushButton("START PROCESSING")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet(self.get_btn_style("green"))
        self.btn_start.setEnabled(False)
        self.btn_start.setToolTip("Begin processing all images in the selected folder.")
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet(self.get_btn_style("red"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop processing after the current batch is finished.")

        proc_layout.addWidget(self.btn_start)
        proc_layout.addWidget(self.btn_stop)
        right_layout.addLayout(proc_layout)

        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #333;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #a15f13;
                width: 20px;
            }
        """)
        right_layout.addWidget(self.progress)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #ccc;")
        right_layout.addWidget(self.lbl_status)

        main_layout.addWidget(right_panel)

        self.populate_models()

    # --- METHODS ---
    def on_model_combo_changed(self):
        # Defensive check: Ensure UI elements exist before accessing them
        if not hasattr(self, 'combo_quant') or not hasattr(self, 'combo_res'):
            return

        # Use currentData() to get the actual model name/filename, not the display text
        txt = self.model_combo.currentData()
        if not txt: return

        is_gguf = txt.lower().endswith(".gguf")
        
        # Disable settings that don't apply to GGUF models
        self.combo_quant.setEnabled(not is_gguf)
        self.combo_res.setEnabled(True) 
        
        if is_gguf:
            self.combo_quant.setToolTip("GGUF models are pre-quantized.")
        else:
            self.combo_quant.setToolTip("Select quantization for Transformers model.")

    def open_model_manager(self):
        dlg = ModelManagerDialog(self)
        dlg.exec() # Modal blocking
        self.populate_models() # Refresh list after close

    def update_folder(self, folder):
        self.current_folder = folder

    def get_btn_style(self, color_type):
        base_css = """
            QPushButton { border-radius: 5px; font-weight: bold; padding: 5px; }
            QPushButton:disabled { background-color: #444444; color: #888888; border: 1px solid #333; }
        """
        if color_type == "green":
            return base_css + """
                QPushButton:enabled { background-color: #2da44e; color: white; }
                QPushButton:enabled:hover { background-color: #2c974b; }
            """
        elif color_type == "red":
            return base_css + """
                QPushButton:enabled { background-color: #d73a49; color: white; }
                QPushButton:enabled:hover { background-color: #cb2431; }
            """
        elif color_type == "blue":
            return base_css + """
                QPushButton:enabled { background-color: #0366d6; color: white; }
                QPushButton:enabled:hover { background-color: #005cc5; }
            """
        return ""

    def _get_path_size_gb(self, path):
        total_size = 0
        try:
            if os.path.isfile(path):
                total_size = os.path.getsize(path)
            else:
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        # skip if it is symbolic link
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
        except Exception:
            pass # Ignore permissions errors etc
        
        return total_size / (1024**3) # Convert bytes to GB

    def populate_models(self):
        self.model_combo.clear()
        base = os.path.join(os.getcwd(), "models")
        if os.path.exists(base):
            # 1. Get Directories (Transformers models)
            # Filter folders starting with _, -, +
            dirs = [d for d in os.listdir(base) 
                    if os.path.isdir(os.path.join(base, d)) 
                    and not d.startswith(('_', '-', '+'))]
            
            items = sorted(dirs)
            
            # 2. Get GGUF Files (excluding mmproj files)
            ggufs = glob.glob(os.path.join(base, "*.gguf"))
            gguf_names = sorted([os.path.basename(g) for g in ggufs 
                                 if "mmproj" not in os.path.basename(g).lower()
                                 and not os.path.basename(g).startswith(('_', '-', '+'))])
            
            items.extend(gguf_names)
            
            # Add items with validatiion and size
            for name in items:
                if name in self.IGNORED_MODELS: continue
                
                full_path = os.path.join(base, name)
                size_gb = self._get_path_size_gb(full_path)
                
                display_text = f"{name} ({size_gb:.1f} GB)"
                self.model_combo.addItem(display_text, userData=name)

        else:
            self.model_combo.addItem("Folder /models not found", userData=None)
            
        # Trigger UI update
        self.on_model_combo_changed()

    def populate_prompts(self):
        path = os.path.join(os.getcwd(), self.PROMPTS_DIR)
        if os.path.exists(path):
            for f in sorted(glob.glob(os.path.join(path, "*.txt"))):
                try: 
                    with open(f, "r", encoding="utf-8") as file: 
                        self.combo_prompts.addItem(os.path.basename(f)[:-4], file.read().strip())
                except: pass

    def on_prompt_preset_changed(self, index):
        prompt_text = self.combo_prompts.itemData(index)
        if prompt_text is not None and prompt_text != "":
            self.txt_prompt.setPlainText(prompt_text)

    def get_settings(self):
        """Returns a dict of currently selected settings."""
        return {
            "model": self.model_combo.currentData(), # Save the real name/path, not display text
            "quantization": self.combo_quant.currentText(),
            "resolution_idx": self.combo_res.currentIndex(),
            "batch_size": self.spin_batch.value(),
            "frames": self.spin_frames.value(),
            "tokens": self.spin_tokens.value(),
            "skip_existing": self.chk_skip.isChecked(),
            "trigger": self.txt_trigger.text(),
            "system_prompt_idx": self.combo_prompts.currentIndex(),
            "prompt_text": self.txt_prompt.toPlainText(),
            "suffix": self.txt_suffix.toPlainText()
        }

    def set_settings(self, settings):
        """Loads settings from a dict."""
        if not settings: return

        def set_combo_by_text(combo, text):
            idx = combo.findText(text)
            if idx >= 0: combo.setCurrentIndex(idx)

        if "model" in settings:
            # Must find by Data (the real name), not display text
            frame_idx = self.model_combo.findData(settings["model"])
            if frame_idx >= 0:
                self.model_combo.setCurrentIndex(frame_idx)
        
        if "quantization" in settings:
            set_combo_by_text(self.combo_quant, settings["quantization"])

        if "resolution_idx" in settings:
            self.combo_res.setCurrentIndex(min(max(0, settings["resolution_idx"]), self.combo_res.count()-1))

        if "batch_size" in settings: self.spin_batch.setValue(settings["batch_size"])
        if "frames" in settings: self.spin_frames.setValue(settings["frames"])
        if "tokens" in settings: self.spin_tokens.setValue(settings["tokens"])
        if "skip_existing" in settings: self.chk_skip.setChecked(settings["skip_existing"])
        if "trigger" in settings: self.txt_trigger.setText(settings["trigger"])
        if "suffix" in settings: self.txt_suffix.setPlainText(settings["suffix"])
        
        if "system_prompt_idx" in settings:
            idx = settings["system_prompt_idx"]
            if 0 <= idx < self.combo_prompts.count():
                self.combo_prompts.setCurrentIndex(idx)
        
        # Explicitly set prompt text last
        if "prompt_text" in settings:
            self.txt_prompt.setPlainText(settings["prompt_text"])

    def load_model(self):
        self.toggle_ui(False)
        self.lbl_status.setText("Loading Model... Please Wait")
        self.progress.setRange(0, 0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        model_name = self.model_combo.currentData() # Get real name
        if not model_name: 
            self.on_model_loaded(False, "No model selected")
            return

        path = os.path.join(os.getcwd(), "models", model_name)
        quant = self.combo_quant.currentText().split()[0]
        res = self.combo_res.currentData()
        attn_impl = "sdpa"
        use_compile = False
        
        self.log_msg.emit(f"⏳ Loading model: {model_name}...")

        self.loader_worker = ModelLoaderWorker(self.engine, path, quant, res, attn_impl, use_compile)
        self.loader_worker.finished.connect(self.on_model_loaded)
        self.loader_worker.start()

    def on_model_loaded(self, success, msg):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        QApplication.restoreOverrideCursor()
        
        if success:
            self.log_msg.emit(f"✅ {msg}")
            self.toggle_ui(True)
            self.btn_load.setEnabled(False)
            self.btn_unload.setEnabled(True)
            self.lbl_status.setText("Ready")
        else:
            self.log_msg.emit(f"❌ Error: {msg}")
            self.toggle_ui(True, loaded=False)
            self.lbl_status.setText("Load Failed")

    def unload_model(self):
        if self.worker: self.worker.stop(); self.worker.wait()
        
        self.log_msg.emit("⏳ Unloading model...")
        QApplication.processEvents() 

        msg = self.engine.unload_model()
        
        self.log_msg.emit(f"✅ {msg}")

        self.toggle_ui(True, loaded=False)
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(False)

    def toggle_ui(self, enabled, loaded=True):
        self.request_lock.emit(not enabled)
        
        settings_enabled = enabled and not loaded
        
        self.model_combo.setEnabled(settings_enabled)
        self.combo_quant.setEnabled(settings_enabled)
        self.combo_res.setEnabled(settings_enabled)
        
        if loaded:
            self.btn_start.setEnabled(enabled)
            self.btn_test_first.setEnabled(enabled)
            self.btn_test_rand.setEnabled(enabled)
        else:
            # Ensure start buttons are disabled if model is not loaded
            self.btn_start.setEnabled(False)
            self.btn_test_first.setEnabled(False)
            self.btn_test_rand.setEnabled(False)

    def run_test(self, mode):
        if not self.current_folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select an image folder first.")
            return
        
        self.toggle_ui(False)
        self.btn_stop.setEnabled(False)
        self.btn_unload.setEnabled(False) 
        
        p = self.txt_prompt.toPlainText() + "\n" + self.txt_suffix.toPlainText()
        settings = {"frame_count": self.spin_frames.value(), "max_tokens": self.spin_tokens.value(), "prompt": p, "trigger": self.txt_trigger.text()}

        self.lbl_status.setText("Testing..." if mode == "random" else "Testing First Image...")

        self.test_worker = TestWorker(self.engine, self.current_folder, settings, mode)
        self.test_worker.result_ready.connect(lambda f, c, t: self.on_test_done(f, c, t))
        self.test_worker.error.connect(lambda e: self.log_msg.emit(f"Error: {e}"))

        self.test_worker.log_signal.connect(self.log_msg.emit)
        
        self.test_worker.finished.connect(lambda: self.toggle_ui(True))
        self.test_worker.finished.connect(lambda: self.btn_unload.setEnabled(True))
        
        self.test_worker.start()

    def on_test_done(self, f, c, t):
        self.lbl_preview.set_image(f)
        self.txt_preview.setText(c)
        self.lbl_filename.setText(os.path.basename(f))
        self.lbl_status.setText(f"Test finished in {t:.2f}s")
        self.log_msg.emit(f"🧪 Test Sample: {os.path.basename(f)} (Time: {t:.2f}s)")

    def start_processing(self):
        if not self.current_folder: 
            QMessageBox.warning(self, "No Folder Selected", "Please select an image folder first.")
            return

        self.toggle_ui(False)
        self.btn_stop.setEnabled(True)
        self.btn_unload.setEnabled(False) # Disable Unload
        
        self.start_time = time.time()
        self.progress.setValue(0)
        
        self.lbl_status.setText("Initializing...")
        self.log_msg.emit("🚀 Starting processing...")

        p = self.txt_prompt.toPlainText() + "\n" + self.txt_suffix.toPlainText()
        settings = {
            "batch_size": self.spin_batch.value(), "skip_existing": self.chk_skip.isChecked(),
            "frame_count": self.spin_frames.value(), "max_tokens": self.spin_tokens.value(),
            "prompt": p, "trigger": self.txt_trigger.text()
        }

        self.worker = CaptionWorker(self.engine, self.current_folder, settings)
        
        # Connect to custom progress handler for stats
        self.worker.signals.progress.connect(self.on_progress_update)
        
        self.worker.signals.total.connect(self.progress.setMaximum)
        self.worker.signals.log.connect(self.log_msg.emit)
        self.worker.signals.image_processed.connect(lambda f, c: (self.lbl_preview.set_image(f), self.txt_preview.setText(c)))
        self.worker.signals.finished.connect(self.on_process_done)
        self.worker.start()

    def on_progress_update(self, val, speed_per_item):
        """
        val: Total items processed (including skipped)
        speed_per_item: Rolling average seconds per item (ONLY generation time). 
                        -1.0 if the update was purely skipped files.
        """
        self.progress.setValue(val)
        
        # Only update text stats if we have a valid speed from actual generation
        if speed_per_item > 0:
            remaining_items = self.progress.maximum() - val
            eta_seconds = remaining_items * speed_per_item
            
            # Calculate Batch Speed
            current_batch_size = self.spin_batch.value()
            
            # Base string
            speed_str = f"{speed_per_item:.2f} s/item"
            
            # Append Batch speed if batch size > 1
            if current_batch_size > 1:
                speed_per_batch = speed_per_item * current_batch_size
                speed_str += f" ({speed_per_batch:.2f} s/batch)"

            # Format time strings
            rem_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            # Calculate Finish Time
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            finish_str = finish_time.strftime("%H:%M:%S")
            
            # Updated Label Text
            self.lbl_status.setText(f"Speed: {speed_str}    |    Remaining: {rem_str}    |    ETA: {finish_str}")
            
        elif val == 0:
            self.lbl_status.setText("Initializing...")

    def stop_processing(self):
        self.log_msg.emit("🛑 Stopping... Please wait for the current batch to finish.")
        self.lbl_status.setText("Stopping...")
        self.btn_stop.setEnabled(False) # Prevent double-clicking
        if self.worker: 
            self.worker.stop()

    def on_process_done(self, elapsed, speed):
        self.toggle_ui(True)
        self.btn_stop.setEnabled(False)
        self.btn_unload.setEnabled(True) # Re-enable Unload
        self.lbl_status.setText(f"Complete. Total: {elapsed:.1f}s | Avg: {speed:.2f} s/item")
        self.log_msg.emit(f"Done. Total time: {elapsed:.2f}s")
        self.batch_finished.emit()