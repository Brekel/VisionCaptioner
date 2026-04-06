import os
import time
import datetime
import shutil
from file_utils import find_media_files, IMAGE_EXTS
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QLineEdit, QProgressBar, QGroupBox, QMessageBox,
                               QSplitter, QApplication, QCheckBox, QSpinBox, QDoubleSpinBox,
                               QSizePolicy, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, QByteArray
from PySide6.QtGui import QPixmap, QImage
from PIL import Image, ImageOps 

# Imports from your existing project
from gui_widgets import ResizableImageLabel
from backend import SAM3Engine
from gui_model_manager import ModelManagerDialog


class SAM3LoaderWorker(QThread):
    finished = Signal(bool, str)

    def __init__(self, engine, model_path):
        super().__init__()
        self.engine = engine
        self.model_path = model_path

    def run(self):
        # This now runs in the background
        success, msg = self.engine.load_model(self.model_path)
        self.finished.emit(success, msg)


class SAM3Worker(QThread):
    progress = Signal(int, float)
    log_msg = Signal(str)
    preview_ready = Signal(str, str) # original_path, mask_file_path
    finished = Signal(float, float) # total_time, avg_speed
    
    def __init__(self, engine, folder, prompt, files, max_dim, conf_thresh, expand_ratio, skip_existing, crop_to_mask, mask_format): 
        super().__init__()
        self.engine = engine
        self.folder = folder
        self.prompt = prompt
        self.files = files
        
        # Settings
        self.max_dim = max_dim
        self.conf_thresh = conf_thresh
        self.expand_ratio = expand_ratio 
        self.skip_existing = skip_existing
        self.crop_to_mask = crop_to_mask
        self.mask_format = mask_format # "Separate File" or "Alpha Channel"
        
        self.is_running = True

    def run(self):
        total = len(self.files)
        processed_count = 0
        
        # Speed calc variables
        start_time_global = time.time()
        avg_speed = 0.0 
        alpha = 0.3 # Smoothing factor for EMA
        
        for i, f_path in enumerate(self.files):
            if not self.is_running: break
            
            # Start timer for this item
            t0 = time.time()
            
            base_name = os.path.splitext(f_path)[0]
            # Determine behavior based on mask_format
            use_alpha_channel = "Alpha Channel" in self.mask_format
            use_subfolder = "Masks Subfolder" in self.mask_format
            
            if use_alpha_channel:
                 save_path = f"{base_name}.png"
            elif use_subfolder:
                 masks_dir = os.path.join(self.folder, "masks")
                 if not os.path.exists(masks_dir):
                     os.makedirs(masks_dir, exist_ok=True)
                 save_path = os.path.join(masks_dir, f"{os.path.basename(base_name)}.png")
            else:
                 save_path = f"{base_name}-masklabel.png"

            # Skip Existing Logic
            if self.skip_existing and os.path.exists(save_path):
                processed_count += 1
                self.progress.emit(processed_count, -1.0)
                continue

            filename = os.path.basename(f_path)
            self.log_msg.emit(f"[{i+1}/{total}] 🖌️ Processing: {filename}...")

            # Generate
            mask_img, msg = self.engine.generate_mask(
                f_path, 
                self.prompt, 
                max_dimension=self.max_dim, 
                conf_threshold=self.conf_thresh,
                expand_ratio=self.expand_ratio
            )
            
            if mask_img:
                try:
                    # 1. Prepare Images
                    original_pil_img = Image.open(f_path)
                    original_pil_img = ImageOps.exif_transpose(original_pil_img)
                    original_pil_img = original_pil_img.convert("RGBA") # Use RGBA for all operations
                    
                    bbox = mask_img.getbbox()
                    has_content = bbox is not None
                    
                    if not has_content:
                        self.log_msg.emit("   -> ⚠️ Mask is empty.")

                    # 2. Cropping Logic (Optional)
                    if self.crop_to_mask and has_content:
                         cropped_img = original_pil_img.crop(bbox)
                         cropped_mask = mask_img.crop(bbox)
                         
                         final_img = cropped_img
                         final_mask = cropped_mask
                         was_cropped = True
                    else:
                         final_img = original_pil_img
                         final_mask = mask_img
                         was_cropped = False

                    # 3. Apply Alpha Channel or Save Split
                    if use_alpha_channel:
                        # Put mask into Alpha Channel
                        # Ensure mask is same size as image
                        if final_mask.size != final_img.size:
                             final_mask = final_mask.resize(final_img.size, Image.NEAREST)
                             
                        r, g, b, _ = final_img.split()
                        final_img = Image.merge("RGBA", (r, g, b, final_mask))
                        
                        # Save Alpha PNG
                        final_img.save(save_path, compress_level=1)
                        
                        # Handle Original File (Move to unused if it wasn't a PNG or if we cropped)
                        # Case A: We cropped -> Original is now partial invalid, move it.
                        # Case B: We didn't crop, but changed format (jpg->png) -> Move jpg to avoid dupe.
                        # Case C: We didn't crop, and input was png -> Overwritten naturally?
                        # Let's simple rule: If source != dest, move source.
                        
                        abs_src = os.path.abspath(f_path)
                        abs_dst = os.path.abspath(save_path)
                        
                        if was_cropped or (abs_src.lower() != abs_dst.lower()):
                            uncropped_dir = os.path.join(self.folder, "unused") # Use 'unused' folder
                            os.makedirs(uncropped_dir, exist_ok=True)
                            move_dest_path = os.path.join(uncropped_dir, os.path.basename(f_path))
                            
                             # If we haven't overwritten it yet (i.e. src != dst), move it
                            if abs_src.lower() != abs_dst.lower():
                                shutil.move(f_path, move_dest_path)
                                self.log_msg.emit(f"   -> Moved original to 'unused/'")
                            else:
                                # Start and end are same file (input was png, no crop, just alpha added)
                                # We already saved over it with final_img.save(save_path)
                                pass

                    else:
                        # Standard Separate Mask File
                        # If we cropped, we need to save the cropped ORIGINAL image too
                        if was_cropped:
                             # Move original full image to 'uncropped' backup
                             uncropped_dir = os.path.join(self.folder, "uncropped")
                             os.makedirs(uncropped_dir, exist_ok=True)
                             move_dest_path = os.path.join(uncropped_dir, os.path.basename(f_path))
                             
                             # Save current state (pre-crop but loaded) to backup? No, move physical file better
                             # But we have 'original_pil_img' in memory.
                             # Actually standard logic: Move old file, save new file.
                             
                             # We can't move f_path if we are about to save to f_path (Windows lock?)
                             # Actually Pillow save is fine.
                             
                             # Save backup
                             # Re-open or use in-memory? In-memory 'original_pil_img' is full size
                             original_pil_img.convert("RGB").save(move_dest_path, compress_level=1)
                             
                             # Save cropped original (overwrite f_path)
                             final_img.convert("RGB").save(f_path, compress_level=1)
                             self.log_msg.emit(f"   -> Cropped and moved original to 'uncropped/'")
                             
                        # Save Mask
                        final_mask.save(save_path, compress_level=1)


                    # Determine what to show in preview
                    # If Alpha Channel mode, the "original" is now the PNG we saved (save_path)
                    # and the "mask" is also embedded in that PNG.
                    if use_alpha_channel:
                        self.preview_ready.emit(save_path, save_path)
                    else:
                        self.preview_ready.emit(f_path, save_path)
                except Exception as e:
                    self.log_msg.emit(f"❌ Error saving {os.path.basename(save_path)}: {e}")
            else:
                if "No detections" in msg:
                    self.log_msg.emit(f"⚠️ Skipped {filename}: {msg}")
                else:
                    self.log_msg.emit(f"❌ Failed {filename}: {msg}")
            
            # Speed Calculation
            t1 = time.time()
            item_duration = t1 - t0
            
            if avg_speed == 0.0:
                avg_speed = item_duration
            else:
                avg_speed = (alpha * item_duration) + ((1 - alpha) * avg_speed)

            processed_count += 1
            self.progress.emit(processed_count, avg_speed)
            
        elapsed_total = time.time() - start_time_global
        self.finished.emit(elapsed_total, avg_speed)

    def stop(self):
        self.is_running = False


class MaskTab(QWidget):
    log_msg = Signal(str)
    batch_finished = Signal()

    def __init__(self, sam_engine=None):
        super().__init__()
        if sam_engine:
            self.sam_engine = sam_engine
        else:
            self.sam_engine = SAM3Engine()
        self.current_folder = ""
        self.recursive = False
        self.worker = None
        self.loader_worker = None
        
        self.setup_ui()
        # Initialize button state based on model presence
        self.check_model_status()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # --- LEFT PANEL (Settings) ---
        left_panel = QGroupBox("Mask Settings")
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(280) 

        # Prompt
        left_layout.addWidget(QLabel("Prompt:"))
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setText("person")
        self.txt_prompt.setPlaceholderText("e.g. person, car")
        self.txt_prompt.setToolTip("Enter the object(s) you want to mask.")
        left_layout.addWidget(self.txt_prompt)
        
        left_layout.addSpacing(15)

        # Max Resolution
        left_layout.addWidget(QLabel("Max Res (Processing):"))
        self.spin_res = QSpinBox()
        self.spin_res.setRange(256, 4096)
        self.spin_res.setValue(1024)
        self.spin_res.setSingleStep(128)
        self.spin_res.setToolTip("Downscale image to this size for processing (saves VRAM).")
        left_layout.addWidget(self.spin_res)
        
        left_layout.addSpacing(15)
        
        # Expand Mask
        left_layout.addWidget(QLabel("Expand Mask (%):"))
        self.spin_expand = QDoubleSpinBox()
        self.spin_expand.setRange(0.0, 50.0)
        self.spin_expand.setValue(0.0)
        self.spin_expand.setSingleStep(0.5)
        self.spin_expand.setSuffix("%")
        self.spin_expand.setToolTip("Expands the mask by a percentage of the image size.\nUseful to ensure the full object is covered.")
        left_layout.addWidget(self.spin_expand)

        left_layout.addSpacing(15)

        # Skip Existing
        self.chk_skip = QCheckBox("Skip Existing Masks")
        self.chk_skip.setChecked(True)
        self.chk_skip.setToolTip("If a *-masklabel.png already exists, skip processing this image.")
        left_layout.addWidget(self.chk_skip)
        
        # Crop Checkbox
        self.chk_crop = QCheckBox("Crop image to mask")
        self.chk_crop.setChecked(False)
        self.chk_crop.setToolTip("Crops the image and mask to the bounding box of the mask.\nMoves the original, uncropped image to an 'uncropped' subfolder.")
        left_layout.addWidget(self.chk_crop)

        left_layout.addSpacing(15)

        # Mask Format Selector
        left_layout.addWidget(QLabel("Mask Format:"))
        self.combo_format = QComboBox()
        self.combo_format.addItem("Separate File (*-masklabel.png)")
        self.combo_format.addItem("Masks Subfolder (masks/*.png)")
        self.combo_format.addItem("Alpha Channel (PNG)")
        self.combo_format.setToolTip("Choose how to save the mask.\n'Separate File': Saves a black/white mask alongside the image.\n'Masks Subfolder': Saves masks in a 'masks' subfolder.\n'Alpha Channel': Saves the image as a PNG with a transparent background.")
        left_layout.addWidget(self.combo_format)

        left_layout.addStretch() 
        main_layout.addWidget(left_panel)

        # --- RIGHT PANEL (Preview & Controls) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Preview Area
        self.splitter = QSplitter(Qt.Horizontal)
        splitter = self.splitter
        
        self.lbl_orig = ResizableImageLabel()
        self.lbl_orig.setText("Original Image")
        self.lbl_orig.setToolTip("Original Image")
        self.lbl_orig.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.lbl_mask = ResizableImageLabel()
        self.lbl_mask.setText("Generated Mask")
        self.lbl_mask.setToolTip("Generated Binary Mask")
        self.lbl_mask.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        splitter.addWidget(self.lbl_orig)
        splitter.addWidget(self.lbl_mask)
        splitter.setSizes([400, 400])
        right_layout.addWidget(splitter, stretch=1)

        # Progress & Buttons
        ctrl_layout = QHBoxLayout()
        
        # Button setup
        self.btn_start = QPushButton("START PROCESSING")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.clicked.connect(self.on_click_start) 
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.clicked.connect(self.stop_generation)
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self.get_btn_style("red"))

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        
        right_layout.addLayout(ctrl_layout)
        
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

    def check_model_status(self):
        """Updates UI appearance only. Logic is handled in on_click_start."""
        model_path = os.path.join(os.getcwd(), "models", "sam3")
        is_installed = os.path.exists(model_path) and os.path.isdir(model_path)
        
        if is_installed:
            self.btn_start.setText("START PROCESSING")
            self.btn_start.setStyleSheet(self.get_btn_style("green"))
            self.lbl_status.setText("Ready")
            
            # Check engine availability
            if not self.sam_engine.is_available():
                self.log_msg.emit("⚠️ SAM3 library not found.")
                self.btn_start.setEnabled(False)
            else:
                self.btn_start.setEnabled(True)
        else:
            self.btn_start.setText("Install SAM3 Model")
            self.btn_start.setStyleSheet(self.get_btn_style("blue"))
            self.lbl_status.setText("Model Missing")
            self.btn_start.setEnabled(True)

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

    # --- Persistence Methods ---
    def get_settings(self):
        return {
            "prompt": self.txt_prompt.text(),
            "max_res": self.spin_res.value(),
            "expand_percent": self.spin_expand.value(),
            "skip_existing": self.chk_skip.isChecked(),
            "crop_to_mask": self.chk_crop.isChecked(),
            "mask_format_index": self.combo_format.currentIndex(),
            "splitter_state": self.splitter.saveState().toHex().data().decode()
        }

    def set_settings(self, settings):
        if not settings: return
        if "prompt" in settings: self.txt_prompt.setText(settings["prompt"])
        if "max_res" in settings: self.spin_res.setValue(int(settings["max_res"]))
        if "expand_percent" in settings: self.spin_expand.setValue(float(settings["expand_percent"]))
        if "skip_existing" in settings: self.chk_skip.setChecked(bool(settings["skip_existing"]))
        if "crop_to_mask" in settings: self.chk_crop.setChecked(bool(settings["crop_to_mask"]))
        if "mask_format_index" in settings: self.combo_format.setCurrentIndex(int(settings["mask_format_index"]))
        if "splitter_state" in settings:
            try: self.splitter.restoreState(QByteArray.fromHex(settings["splitter_state"].encode()))
            except: pass

    # --- Logic ---
    def open_manager_for_sam(self):
        dlg = ModelManagerDialog(self)
        dlg.exec()
        # Refresh status after dialog closes
        self.check_model_status()

    def on_click_start(self):
        """Handler that decides action based on model status."""
        model_path = os.path.join(os.getcwd(), "models", "sam3")
        if os.path.exists(model_path) and os.path.isdir(model_path):
            self.start_generation()
        else:
            self.open_manager_for_sam()
        
    def on_model_loaded(self, success, msg):
        if not success:
            self.log_msg.emit(f"❌ Load Failed: {msg}")
            QMessageBox.critical(self, "Model Load Error", f"Could not load SAM3.\n\n{msg}")
            self.cleanup_ui_state()
            return

        self.log_msg.emit("✅ Model loaded. Starting generation...")
        self.lbl_status.setText("Processing...")

        # Convert percent to ratio
        exp_ratio = self.spin_expand.value() / 100.0

        # Start the actual processing worker
        self.worker = SAM3Worker(
            self.sam_engine, 
            self.current_folder, 
            self.pending_prompt, 
            self.pending_files,
            self.spin_res.value(),
            0.25, 
            exp_ratio,
            self.chk_skip.isChecked(),
            self.chk_crop.isChecked(),
            self.combo_format.currentText()
        )
        
        self.worker.progress.connect(self.on_progress_update)
        self.worker.log_msg.connect(self.log_msg.emit)
        self.worker.preview_ready.connect(self.update_preview)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def set_recursive(self, recursive):
        self.recursive = recursive

    def update_folder(self, folder, recursive=None):
        self.current_folder = folder
        if recursive is not None:
            self.recursive = recursive

    def start_generation(self):
        # 1. Validation Logic
        if not self.current_folder or not os.path.exists(self.current_folder):
            QMessageBox.warning(self, "Error", "Please select a valid image folder first.")
            return

        prompt = self.txt_prompt.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Error", "Please enter a text prompt.")
            return

        # 2. Prepare Files List
        files = find_media_files(self.current_folder, exts=IMAGE_EXTS,
                                 recursive=self.recursive)
        
        if not files:
            self.log_msg.emit("No images found to process.")
            return

        # 3. Confirmation for Alpha Channel Mode
        if "Alpha Channel" in self.combo_format.currentText():
            ret = QMessageBox.warning(
                self, 
                "Confirm Alpha Channel Output",
                "You have selected <b>Alpha Channel (PNG)</b> output.<br><br>"
                "Original non-PNG files (e.g., .jpg) will be <b>MOVED</b> to an 'unused' subfolder to prevent duplicates.<br><br>"
                "Are you sure you want to proceed?",
                QMessageBox.Yes | QMessageBox.No
            )
            if ret == QMessageBox.No:
                return

        # 4. Lock UI
        self.toggle_ui(False)
        self.progress.setValue(0)
        self.progress.setMaximum(len(files))
        self.lbl_status.setText("Loading Model...")
        self.lbl_mask.setText("Loading Model...")
        self.log_msg.emit("⏳ Auto-loading SAM3 model (Background)...")
        
        # 4. START LOADER WORKER
        model_path = os.path.join(os.getcwd(), "models", "sam3")
        
        # Store these for use after loading is done
        self.pending_files = files
        self.pending_prompt = prompt
        
        self.loader_worker = SAM3LoaderWorker(self.sam_engine, model_path)
        self.loader_worker.finished.connect(self.on_model_loaded)
        self.loader_worker.start()

    def on_progress_update(self, count, speed):
        self.progress.setValue(count)
        
        if speed > 0:
            remaining = self.progress.maximum() - count
            eta_seconds = remaining * speed
            
            speed_str = f"{speed:.2f} s/img"
            rem_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            finish_str = finish_time.strftime("%H:%M:%S")
            
            self.lbl_status.setText(f"Speed: {speed_str}    |    Remaining: {rem_str}    |    ETA: {finish_str}")
        else:
            if count == 0:
                self.lbl_status.setText("Initializing...")
            pass

    def stop_generation(self):
        if self.worker:
            self.log_msg.emit("🛑 Stopping...")
            self.lbl_status.setText("Stopping...")
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def on_worker_finished(self, total_time=0.0, avg_speed=0.0):
        self.log_msg.emit("✅ Generation sequence ended.")
        self.lbl_status.setText(f"Complete. Total: {total_time:.1f}s")
        self.cleanup_ui_state()
        self.log_msg.emit("Ready.")
        self.batch_finished.emit()

    def cleanup_ui_state(self):
        self.toggle_ui(True)
        if self.lbl_mask.text() == "Loading Model...":
            self.lbl_mask.setText("Generated Mask")

    def toggle_ui(self, enabled):
        self.btn_start.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)
        self.txt_prompt.setEnabled(enabled)
        self.spin_res.setEnabled(enabled)
        self.spin_expand.setEnabled(enabled)
        self.chk_skip.setEnabled(enabled)
        self.chk_crop.setEnabled(enabled)
        self.combo_format.setEnabled(enabled)

    def update_preview(self, orig_path, mask_path):
        # 1. Check if we are in Alpha Channel mode (orig_path == mask_path or png+alpha detection)
        is_alpha_mode = (orig_path == mask_path)
        
        if is_alpha_mode:
            # Load the RGBA image
            # Set Left Image (Color + Alpha)
            self.lbl_orig.set_image(orig_path)
            
            # Extract Alpha for Right Image
            try:
                reader = Image.open(orig_path)
                if reader.mode == 'RGBA':
                    alpha = reader.split()[-1]
                    # Convert to QPixmap
                    data = alpha.convert("L").tobytes()
                    q_img = QImage(data, alpha.width, alpha.height, alpha.width, QImage.Format_Grayscale8)
                    self.lbl_mask.current_pixmap = QPixmap.fromImage(q_img)
                    self.lbl_mask.update_view()
                else:
                     # Fallback if no alpha found?
                     self.lbl_mask.set_image(orig_path) 
            except Exception as e:
                print(f"Error extracting alpha for preview: {e}")
                self.lbl_mask.set_image(orig_path) # Fallback
        else:
            # Standard Mode
            self.lbl_orig.set_image(orig_path)
            self.lbl_mask.set_image(mask_path)