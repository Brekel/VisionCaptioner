import os
import cv2
import time
import datetime
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QLineEdit, QProgressBar, QGroupBox, QMessageBox,
                               QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
                               QSizePolicy, QFileDialog, QSlider, QStyle)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMutex, QWaitCondition
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QIcon
from gui_widgets import ResizableImageLabel
from gui_model_manager import ModelManagerDialog

# --- HELPER: THREADED FRAME LOADING ---
class FrameLoader(threading.Thread):
    def __init__(self, video_path, start_frame, end_frame, step, queue_obj):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.queue = queue_obj
        self.is_running = True
        self.daemon = True 

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        current_idx = self.start_frame

        while current_idx <= self.end_frame and self.is_running:
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.is_running:
                while self.is_running:
                    try:
                        self.queue.put((current_idx, frame_rgb), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            
            next_target = current_idx + self.step
            frames_to_skip = self.step - 1
            
            if frames_to_skip > 0:
                if frames_to_skip < 100:
                    for _ in range(frames_to_skip):
                        if not self.is_running: break
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_target)

            current_idx = next_target

        cap.release()
        if self.is_running:
            self.queue.put(None)

    def stop(self):
        self.is_running = False

# Preview Worker With Playback
class PreviewWorker(QThread):
    # Emits (Image, Frame Index, IsPlaying)
    frame_ready = Signal(QImage, int)
    playback_stopped = Signal() # Signal when video ends naturally

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.fps = 30.0
        self.is_running = True
        
        # State
        self.pending_seek = -1
        self.is_playing = False
        self.current_frame_idx = 0
        
        # Synchronization
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def request_seek(self, frame_idx):
        """Request a jump to a specific frame."""
        self.mutex.lock()
        self.pending_seek = frame_idx
        # Usually, if user scrubs manually, we pause playback
        self.is_playing = False 
        self.condition.wakeOne()
        self.mutex.unlock()

    def play(self):
        """Start continuous playback."""
        self.mutex.lock()
        self.is_playing = True
        self.condition.wakeOne()
        self.mutex.unlock()

    def pause(self):
        """Pause playback."""
        self.mutex.lock()
        self.is_playing = False
        self.mutex.unlock()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / self.fps
        
        while self.is_running:
            self.mutex.lock()
            
            # If not playing and no seek request, sleep and wait
            if not self.is_playing and self.pending_seek == -1:
                self.condition.wait(self.mutex)
            
            if not self.is_running:
                self.mutex.unlock()
                break

            # Check flags
            do_seek = self.pending_seek
            do_play = self.is_playing
            
            # Clear pending request
            self.pending_seek = -1
            self.mutex.unlock()

            # --- EXECUTION (Outside Mutex) ---
            
            # 1. Handle Seek (Priority)
            if do_seek != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, do_seek)
                self.current_frame_idx = do_seek
                
            # 2. Read Frame
            start_time = time.time()
            ret, frame = cap.read()
            
            if ret:
                # Process Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                # Copy is essential here
                q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                
                # Update index tracking
                if not do_seek != -1: # If we didn't just seek, we advanced by 1
                    self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                self.frame_ready.emit(q_img, self.current_frame_idx)
            else:
                # End of video or error
                if do_play:
                    self.playback_stopped.emit()
                    self.mutex.lock()
                    self.is_playing = False
                    self.mutex.unlock()

            # 3. Handle Playback Timing
            if do_play and not do_seek != -1:
                # Sleep to match FPS
                elapsed = time.time() - start_time
                wait_time = max(0.001, frame_delay - elapsed)
                time.sleep(wait_time)

        cap.release()

    def stop(self):
        self.mutex.lock()
        self.is_running = False
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()

# Helper: Async Saver
def save_image_task(img, path, quality=95):
    try:
        img.save(path, quality=quality)
    except Exception as e:
        print(f"Save error: {e}")

# Main Worker
class VideoExtractWorker(QThread):
    progress = Signal(int, int, float, int, int, int, int, str) 
    log_msg = Signal(str)
    status_update = Signal(str)
    preview_ready = Signal(object, object)
    finished = Signal(int, int, float)

    def __init__(self, engine, video_tasks, output_folder, prompt, step, sam_settings, output_flags):
        super().__init__()
        self.engine = engine
        self.video_tasks = video_tasks 
        self.output_folder = output_folder
        self.prompt = prompt
        self.step = step
        self.sam_settings = sam_settings 
        self.output_flags = output_flags
        self.is_running = True
        
        self.frame_queue = queue.Queue(maxsize=5)
        self.loader = None
        self.save_executor = None

    def run(self):
        saved_count = 0
        scanned_count = 0
        start_time_global = time.time()
        
        try:
            model_path = os.path.join(os.getcwd(), "models", "sam3")
            if not self.engine.model:
                self.log_msg.emit("⏳ Auto-loading SAM3 model...")
                self.status_update.emit("⏳ Auto-loading SAM3 model...")
                success, msg = self.engine.load_model(model_path)
                if not success:
                    self.log_msg.emit(f"❌ Model Load Failed: {msg}")
                    return

            if not os.path.exists(self.output_folder):
                try:
                    os.makedirs(self.output_folder)
                except Exception as e:
                    self.log_msg.emit(f"❌ Error creating output folder: {e}")
                    return
            
            self.save_executor = ThreadPoolExecutor(max_workers=4)
            num_videos = len(self.video_tasks)

            for v_idx, task in enumerate(self.video_tasks):
                if not self.is_running: break

                video_path = task['path']
                base_filename = os.path.splitext(os.path.basename(video_path))[0]
                self.log_msg.emit(f"🎬 Starting video {v_idx+1}/{num_videos}: {base_filename}")
                self.status_update.emit(f"Starting {base_filename}...")

                cap_temp = cv2.VideoCapture(video_path)
                total_frames_in_video = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_temp.release()

                start_f = task['start']
                end_f = task['end']
                
                if end_f == -1 or end_f >= total_frames_in_video:
                    end_f = total_frames_in_video - 1

                with self.frame_queue.mutex:
                    self.frame_queue.queue.clear()
                
                self.loader = FrameLoader(video_path, start_f, end_f, self.step, self.frame_queue)
                self.loader.start()

                avg_speed = 0.0
                alpha = 0.3
                
                while self.is_running:
                    try:
                        item = self.frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        if not self.loader.is_alive() and self.frame_queue.empty():
                            break 
                        continue
                    
                    if item is None: break
                        
                    frame_idx, frame_rgb_np = item
                    t0 = time.time()
                    
                    pil_img = Image.fromarray(frame_rgb_np)
                    scanned_count += 1

                    mask_img, msg = self.engine.generate_mask(
                        pil_img, 
                        self.prompt,
                        max_dimension=self.sam_settings['max_res'],
                        conf_threshold=self.sam_settings['conf'],
                        expand_ratio=self.sam_settings['expand']
                    )

                    if mask_img:
                        save_img = pil_img
                        save_mask = mask_img
                        suffix = ""
                        
                        if self.sam_settings['crop']:
                            bbox = mask_img.getbbox()
                            if bbox:
                                save_img = pil_img.crop(bbox)
                                save_mask = mask_img.crop(bbox)
                            else:
                                suffix = "_empty"
                        
                        frame_name = f"{base_filename}_frame_{frame_idx:06d}{suffix}.jpg"
                        mask_name = f"{base_filename}_frame_{frame_idx:06d}{suffix}-masklabel.png"
                        
                        out_path_img = os.path.join(self.output_folder, frame_name)
                        out_path_mask = os.path.join(self.output_folder, mask_name)
                        
                        did_save = False
                        if self.output_flags['save_color']:
                            self.save_executor.submit(save_image_task, save_img.copy(), out_path_img)
                            did_save = True
                        if self.output_flags['save_mask']:
                            self.save_executor.submit(save_image_task, save_mask.copy(), out_path_mask)
                            did_save = True
                        
                        if did_save:
                            saved_count += 1
                    
                    self.preview_ready.emit(pil_img, mask_img)

                    t1 = time.time()
                    dt = t1 - t0
                    if avg_speed == 0: avg_speed = dt
                    else: avg_speed = (alpha * dt) + ((1 - alpha) * avg_speed)
                    
                    self.progress.emit(frame_idx, total_frames_in_video, avg_speed, saved_count, scanned_count, v_idx + 1, num_videos, base_filename)

            # Drain queue to ensure loader can exit if blocked
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            if self.loader:
                self.loader.stop()
                self.loader.join()

        except Exception as e:
            self.log_msg.emit(f"❌ Critical Error in Video Worker: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.save_executor:
                self.save_executor.shutdown(wait=True)

            total_time = time.time() - start_time_global
            self.log_msg.emit("♻️ Unloading SAM3 model...")
            self.engine.unload()
            self.finished.emit(saved_count, scanned_count, total_time)

    def stop(self):
        self.is_running = False
        if self.loader:
            self.loader.stop()

class VideoTab(QWidget):
    log_msg = Signal(str)

    def __init__(self, sam_engine):
        super().__init__()
        self.sam_engine = sam_engine
        self.video_path = ""
        self.current_folder = "" 
        
        self.total_frames = 0
        self.fps = 30.0
        
        self.worker = None 
        self.preview_worker = None 
        
        self.video_loaded = False 
        self.is_batch_mode = False 
        self.is_playing_preview = False

        self.setAcceptDrops(True)
        self.setup_ui()
        
        # Check status immediately
        self.check_model_status()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
               
        self.content_layout = QHBoxLayout()
        main_layout.addLayout(self.content_layout)

        # Left Panel (Settings)
        left_widget = QWidget()
        left_widget.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        
        settings_group = QGroupBox("Extraction Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Video File:"))
        file_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open Video")
        self.btn_open.clicked.connect(self.open_video)
        
        self.btn_batch = QPushButton("Batch Process")
        self.btn_batch.setStyleSheet("background-color: #5a3fa3; color: white;")
        self.btn_batch.clicked.connect(self.batch_process_videos)
        
        file_layout.addWidget(self.btn_open)
        file_layout.addWidget(self.btn_batch)
        settings_layout.addLayout(file_layout)
        
        self.lbl_file = QLabel("No file loaded (Drag & Drop supported)")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.lbl_file.setStyleSheet("color: #888; font-size: 10px;")
        settings_layout.addWidget(self.lbl_file)
        
        settings_layout.addSpacing(10)
        
        settings_layout.addWidget(QLabel("Prompt:"))
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setText("person")
        self.txt_prompt.setPlaceholderText("e.g. person, car")
        settings_layout.addWidget(self.txt_prompt)
        
        sam_grid = QHBoxLayout()
        v1 = QVBoxLayout(); v1.addWidget(QLabel("Resolution:"))
        self.spin_res = QSpinBox(); self.spin_res.setRange(256, 4096); self.spin_res.setValue(1024); self.spin_res.setSingleStep(128)
        v1.addWidget(self.spin_res)
        
        v2 = QVBoxLayout(); v2.addWidget(QLabel("Confidence:"))
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.01, 1.0); self.spin_conf.setValue(0.25); self.spin_conf.setSingleStep(0.05)
        v2.addWidget(self.spin_conf)
        
        v3 = QVBoxLayout(); v3.addWidget(QLabel("Expand Mask:"))
        self.spin_expand = QDoubleSpinBox()
        self.spin_expand.setRange(0.0, 50.0)
        self.spin_expand.setSingleStep(0.5)
        self.spin_expand.setValue(0.0)
        self.spin_expand.setSuffix("%")
        v3.addWidget(self.spin_expand)
        
        sam_grid.addLayout(v1)
        sam_grid.addLayout(v2)
        sam_grid.addLayout(v3)
        settings_layout.addLayout(sam_grid)
        
        self.chk_crop = QCheckBox("Crop output to mask")
        settings_layout.addWidget(self.chk_crop)

        settings_layout.addSpacing(10)

        range_group = QGroupBox("Frame Range (Single Video Only)")
        range_layout = QVBoxLayout(range_group)
        
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Start:"))
        self.spin_start = QSpinBox(); self.spin_start.setRange(0, 999999)
        self.spin_start.valueChanged.connect(self.update_scan_estimate)
        
        self.btn_set_start = QPushButton("Set Start")
        self.btn_set_start.clicked.connect(lambda: self.spin_start.setValue(self.slider.value()))
        
        r1.addWidget(self.spin_start); r1.addWidget(self.btn_set_start)
        range_layout.addLayout(r1)
        
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("End:"))
        self.spin_end = QSpinBox(); self.spin_end.setRange(0, 999999)
        self.spin_end.valueChanged.connect(self.update_scan_estimate)
        
        self.btn_set_end = QPushButton("Set End")
        self.btn_set_end.clicked.connect(lambda: self.spin_end.setValue(self.slider.value()))
        
        r2.addWidget(self.spin_end); r2.addWidget(self.btn_set_end)
        range_layout.addLayout(r2)
        
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Step:"))
        self.spin_step = QSpinBox(); self.spin_step.setRange(1, 10000); self.spin_step.setValue(100)
        self.spin_step.valueChanged.connect(self.update_scan_estimate)
        r3.addWidget(self.spin_step)
        range_layout.addLayout(r3)
        
        self.lbl_scan_est = QLabel("Scan Count: 0 frames")
        self.lbl_scan_est.setStyleSheet("color: #aaa; font-style: italic; margin-top: 5px;")
        self.lbl_scan_est.setAlignment(Qt.AlignCenter)
        range_layout.addWidget(self.lbl_scan_est)
        
        settings_layout.addWidget(range_group)
        
        out_group = QGroupBox("Output Files")
        out_layout = QHBoxLayout(out_group)
        self.chk_save_img = QCheckBox("Color")
        self.chk_save_img.setChecked(True)
        self.chk_save_mask = QCheckBox("Mask")
        self.chk_save_mask.setChecked(True)
        out_layout.addWidget(self.chk_save_img)
        out_layout.addWidget(self.chk_save_mask)
        settings_layout.addWidget(out_group)
        
        settings_layout.addStretch()
        left_layout.addWidget(settings_group)
        
        # Right Panel (Preview)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)

        self.lbl_preview = ResizableImageLabel()
        self.lbl_preview.setText("No Video Loaded")
        self.lbl_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_layout.addWidget(self.lbl_preview, stretch=1)
        
        # Playback Controls
        slider_layout = QHBoxLayout()
        
        # Play Button
        self.btn_play_preview = QPushButton()
        self.btn_play_preview.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play_preview.setFixedSize(30, 30)
        self.btn_play_preview.setEnabled(False)
        self.btn_play_preview.clicked.connect(self.toggle_playback)
        slider_layout.addWidget(self.btn_play_preview)

        self.lbl_time = QLabel("00:00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.lbl_frame = QLabel("0 / 0")
        
        slider_layout.addWidget(self.lbl_time)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.lbl_frame)
        right_layout.addLayout(slider_layout)
        
        ctrl_layout = QHBoxLayout()
        
        self.btn_process = QPushButton("START EXTRACTION")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.on_click_process)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setStyleSheet("""
            QPushButton { background-color: #d73a49; color: white; font-weight: bold; padding: 10px; border-radius: 4px; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QPushButton:hover:!disabled { background-color: #cb2431; }
        """)
        self.btn_stop.setEnabled(False)
        
        ctrl_layout.addWidget(self.btn_process)
        ctrl_layout.addWidget(self.btn_stop)
        right_layout.addLayout(ctrl_layout)
        
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter) 
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        
        self.progress.setStyleSheet("""
            QProgressBar {
                text-align: center;
                color: white;
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #333;
            }
            QProgressBar::chunk {
                background-color: #2da44e;
                border-radius: 4px;
            }
        """)
        
        right_layout.addWidget(self.progress)
        
        self.content_layout.addWidget(left_widget, 0)
        self.content_layout.addWidget(right_widget, 1)

    def on_click_process(self):
        """Handler that decides action based on model status."""
        model_path = os.path.join(os.getcwd(), "models", "sam3")
        if os.path.exists(model_path) and os.path.isdir(model_path):
            self.start_processing()
        else:
            self.open_model_manager()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv']:
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_video(file_path)

    def get_btn_style(self, color):
        base = "QPushButton { color: white; font-weight: bold; padding: 10px; border-radius: 4px; }"
        disabled = "QPushButton:disabled { background-color: #555; color: #888; }"
        
        if color == "green":
            return base + "QPushButton { background-color: #2da44e; } QPushButton:hover:!disabled { background-color: #2c974b; } " + disabled
        elif color == "blue":
            return base + "QPushButton { background-color: #0366d6; } QPushButton:hover:!disabled { background-color: #005cc5; } " + disabled
        return ""

    def check_model_status(self):
        """Updates UI appearance only. Logic is handled in on_click_process."""
        model_path = os.path.join(os.getcwd(), "models", "sam3")
        is_installed = os.path.exists(model_path) and os.path.isdir(model_path)

        if is_installed:
            self.btn_process.setText("START EXTRACTION")
            self.btn_process.setStyleSheet(self.get_btn_style("green"))
            # Only enable if video is loaded
            self.btn_process.setEnabled(self.video_loaded)
        else:
            self.btn_process.setText("Install SAM3 Model")
            self.btn_process.setStyleSheet(self.get_btn_style("blue"))
            self.btn_process.setEnabled(True) # Always enable the install button

    def open_model_manager(self):
        dlg = ModelManagerDialog(self)
        dlg.exec()
        self.check_model_status()

    def get_settings(self):
        return {
            "video_path": self.video_path,
            "prompt": self.txt_prompt.text(),
            "res": self.spin_res.value(),
            "conf": self.spin_conf.value(),
            "expand": self.spin_expand.value(),
            "crop": self.chk_crop.isChecked(),
            "step": self.spin_step.value(),
            "save_color": self.chk_save_img.isChecked(),
            "save_mask": self.chk_save_mask.isChecked(),
            "start_frame": self.spin_start.value(),
            "end_frame": self.spin_end.value(),
        }

    def set_settings(self, s):
        if not s: return
        if "prompt" in s: self.txt_prompt.setText(s["prompt"])
        if "res" in s: self.spin_res.setValue(s["res"])
        if "conf" in s: self.spin_conf.setValue(s["conf"])
        if "expand" in s: self.spin_expand.setValue(s["expand"])
        if "crop" in s: self.chk_crop.setChecked(s["crop"])
        if "step" in s: self.spin_step.setValue(s["step"])
        if "save_color" in s: self.chk_save_img.setChecked(s["save_color"])
        if "save_mask" in s: self.chk_save_mask.setChecked(s["save_mask"])
        
        if "video_path" in s and s["video_path"] and os.path.exists(s["video_path"]):
            self.load_video(s["video_path"])
            if "start_frame" in s: self.spin_start.setValue(s["start_frame"])
            if "end_frame" in s: self.spin_end.setValue(s["end_frame"])  
            self.update_scan_estimate()

    def update_folder(self, folder):
        self.current_folder = folder

    def open_video(self):
        start_dir = os.path.dirname(self.video_path) if self.video_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", start_dir, "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)")
        if path:
            self.load_video(path)

    def load_video(self, path):
        if self.preview_worker:
            self.preview_worker.stop()
            self.preview_worker = None
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        
        self.video_path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        filename = os.path.basename(path)
        # Insert zero-width spaces to allow wrapping on underscores, dots, etc.
        display_name = filename.replace("_", "_\u200b").replace(".", ".\u200b").replace("-", "-\u200b")
        self.lbl_file.setText(display_name)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.btn_play_preview.setEnabled(True)
        self.set_playback_ui(False)
        
        self.spin_start.setRange(0, self.total_frames - 1)
        self.spin_start.setValue(0)
        
        self.spin_end.setRange(0, self.total_frames - 1)
        self.spin_end.setValue(self.total_frames - 1)
        
        self.update_scan_estimate()
        
        self.video_loaded = True
        # Re-check status to potentially enable the start button
        self.check_model_status()
        self.lbl_preview.setText("")
        
        self.preview_worker = PreviewWorker(path)
        self.preview_worker.frame_ready.connect(self.on_preview_frame_ready)
        self.preview_worker.playback_stopped.connect(lambda: self.set_playback_ui(False))
        self.preview_worker.start()
        
        # Load first frame
        self.preview_worker.request_seek(0)

    def update_scan_estimate(self):
        start = self.spin_start.value()
        end = self.spin_end.value()
        step = self.spin_step.value()
        if step <= 0 or end < start: count = 0
        else: count = (end - start) // step + 1
        self.lbl_scan_est.setText(f"Scan Count: ~{count} frames")

    # --- PLAYBACK LOGIC ---
    def toggle_playback(self):
        if not self.preview_worker: return
        
        if self.is_playing_preview:
            # Stop
            self.preview_worker.pause()
            self.set_playback_ui(False)
        else:
            # Start
            self.preview_worker.play()
            self.set_playback_ui(True)

    def set_playback_ui(self, playing):
        self.is_playing_preview = playing
        icon = QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
        self.btn_play_preview.setIcon(self.style().standardIcon(icon))

    def on_slider_changed(self):
        frame = self.slider.value()
        self.lbl_frame.setText(f"{frame} / {self.total_frames}")
        seconds = frame / self.fps
        self.lbl_time.setText(str(datetime.timedelta(seconds=int(seconds))))
        
        # If the user drags the slider, we request a seek.
        # However, if this change was triggered programmatically (by playback),
        # we skip the request (handled in on_preview_frame_ready).
        if self.preview_worker and not self.slider.signalsBlocked():
            # If user scrubs, we pause playback to avoid conflict
            if self.is_playing_preview:
                self.set_playback_ui(False)
            self.preview_worker.request_seek(frame)

    def on_preview_frame_ready(self, q_img, frame_idx):
        # Update Image
        self.lbl_preview.current_pixmap = QPixmap.fromImage(q_img)
        self.lbl_preview.update_view()
        
        # Update Slider (Block signals to prevent seek-loop)
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        
        # Update Time/Text manually since signal didn't fire
        self.lbl_frame.setText(f"{frame_idx} / {self.total_frames}")
        seconds = frame_idx / self.fps
        self.lbl_time.setText(str(datetime.timedelta(seconds=int(seconds))))

    def closeEvent(self, event):
        if self.preview_worker:
            self.preview_worker.stop()
        super().closeEvent(event)

    def batch_process_videos(self):
        start_dir = os.path.dirname(self.video_path) if self.video_path else ""
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Multiple Videos", start_dir, "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)")
        
        if not paths:
            return

        tasks = []
        for p in paths:
            tasks.append({
                'path': p,
                'start': 0,
                'end': -1 
            })
            
        out_folder = self.current_folder
        if not out_folder or not os.path.exists(out_folder):
            ret = QMessageBox.warning(self, "No Output Folder", 
                "The main output folder is not set.\nSave to the folder of the first video?", QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes: 
                out_folder = os.path.dirname(paths[0])
            else: 
                return

        self.is_batch_mode = True
        self.launch_worker(tasks, out_folder)

    def start_processing(self):
        if not self.video_path: return

        prompt = self.txt_prompt.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a prompt (e.g., 'person').")
            return
        
        out_folder = self.current_folder
        if not out_folder or not os.path.exists(out_folder):
            ret = QMessageBox.warning(self, "No Output Folder", 
                "The main output folder is not set.\nSave to video location instead?", QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes: out_folder = os.path.dirname(self.video_path)
            else: return

        tasks = [{
            'path': self.video_path,
            'start': self.spin_start.value(),
            'end': self.spin_end.value()
        }]

        self.is_batch_mode = False
        self.launch_worker(tasks, out_folder)

    def launch_worker(self, tasks, out_folder):
        # Pause preview before starting heavy work
        if self.preview_worker:
            self.set_playback_ui(False)
            self.preview_worker.pause()
            
        if not self.chk_save_img.isChecked() and not self.chk_save_mask.isChecked():
            QMessageBox.warning(self, "No Output", "Select at least one output type (Color or Mask).")
            return

        sam_settings = {
            'max_res': self.spin_res.value(),
            'conf': self.spin_conf.value(),
            'expand': self.spin_expand.value() / 100.0, 
            'crop': self.chk_crop.isChecked()
        }
        
        output_flags = {
            'save_color': self.chk_save_img.isChecked(),
            'save_mask': self.chk_save_mask.isChecked()
        }
        
        prompt = self.txt_prompt.text().strip()
        if not prompt: 
            QMessageBox.warning(self, "Error", "Prompt is required.")
            return

        self.toggle_ui(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Initializing / Loading Model...")
        
        self.worker = VideoExtractWorker(
            self.sam_engine,
            tasks,
            out_folder,
            prompt,
            self.spin_step.value(),
            sam_settings,
            output_flags
        )
        
        self.worker.progress.connect(self.on_progress)
        self.worker.log_msg.connect(self.log_msg.emit)
        self.worker.status_update.connect(lambda s: self.progress.setFormat(s))
        self.worker.preview_ready.connect(self.on_worker_preview)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, current_frame, total_frames, speed, saved, scanned, v_idx, total_videos, filename):
        # --- 1. Update Progress Bar Value ---
        if not self.is_batch_mode:
            self.slider.blockSignals(True)
            self.slider.setValue(current_frame)
            self.slider.blockSignals(False)
            self.lbl_frame.setText(f"{current_frame} / {self.total_frames}")

            start = self.spin_start.value()
            end = self.spin_end.value()
            total_range = end - start
            if total_range > 0:
                relative_pos = current_frame - start
                percent = int((relative_pos / total_range) * 100)
                if self.progress.maximum() == 0: self.progress.setRange(0, 100)
                self.progress.setValue(max(0, min(100, percent)))
        else:
            if self.progress.maximum() != total_frames:
                self.progress.setRange(0, total_frames)
            
            self.progress.setValue(current_frame)

        # --- 2. Calculate Time Remaining & ETA ---
        time_stats = ""
        if speed > 0:
            # Determine the target end frame
            # In batch mode, we assume processing goes to the end of the video
            target_end = total_frames
            if not self.is_batch_mode:
                # In single mode, we might stop early based on user settings
                # Clamp to total_frames in case user input is out of bounds
                target_end = min(self.spin_end.value(), total_frames)
            
            # Calculate how many frames are left to scan
            frames_remaining = max(0, target_end - current_frame)
            
            # Account for the 'Step' setting (we don't process every frame)
            step = max(1, self.spin_step.value())
            iterations_remaining = frames_remaining / step
            
            # Calculate seconds remaining
            seconds_left = iterations_remaining * speed
            
            # Format Duration (Time Remaining)
            m, s = divmod(int(seconds_left), 60)
            h, m = divmod(m, 60)
            if h > 0:
                rem_str = f"{h}h {m:02d}m"
            else:
                rem_str = f"{m:02d}m {s:02d}s"

            # Calculate ETA (Clock time)
            eta_dt = datetime.datetime.now() + datetime.timedelta(seconds=seconds_left)
            eta_str = eta_dt.strftime("%H:%M:%S")
            
            time_stats = f" | {rem_str} left | ETA: {eta_str}"

        # --- 3. Format Display Text ---
        # Calculate percentage for display
        display_percent = 0
        if self.progress.maximum() > 0:
            val = self.progress.value()
            mx = self.progress.maximum()
            display_percent = int((val / mx) * 100)

        stat_text = f"{display_percent}% | Video {v_idx}/{total_videos} | {filename} | Saved: {saved} | Scanned: {scanned}"
        
        if speed > 0:
            stat_text += f" | {speed:.2f}s/it"
        
        stat_text += time_stats
            
        self.progress.setFormat(stat_text)

    def on_worker_preview(self, pil_img, pil_mask):
        img_np = np.array(pil_img)
        h, w = img_np.shape[:2]
        
        if pil_mask is None:
            qim = QImage(img_np.data, w, h, 3 * w, QImage.Format_RGB888)
            self.lbl_preview.current_pixmap = QPixmap.fromImage(qim)
            self.lbl_preview.update_view()
            return

        mask_np = np.array(pil_mask)
        if len(mask_np.shape) == 3:
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            
        overlay_color = np.full((h, w, 3), (20, 20, 40), dtype=np.uint8) 
        opacity = 0.8 
        brightness = 1.0 - opacity
        dimmed_bg = cv2.addWeighted(img_np, brightness, overlay_color, opacity, 0)
        
        _, binary_mask = cv2.threshold(mask_np, 127, 1, cv2.THRESH_BINARY)
        binary_mask_3ch = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        
        final_img = (img_np * binary_mask_3ch) + (dimmed_bg * (1 - binary_mask_3ch))
        final_img = final_img.astype(np.uint8)
        
        qim = QImage(final_img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.lbl_preview.current_pixmap = QPixmap.fromImage(qim)
        self.lbl_preview.update_view()

    def on_finished(self, saved, scanned, total_time):
        self.toggle_ui(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat(f"Complete! Saved: {saved} | Time: {total_time:.1f}s")
        QMessageBox.information(self, "Extraction Complete", f"Finished.\nSaved {saved} items.")

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.progress.setFormat("Stopping...")

    def toggle_ui(self, enabled):
        start_enabled = enabled and self.video_loaded
        self.btn_open.setEnabled(enabled)
        self.btn_batch.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)
        self.btn_process.setEnabled(start_enabled)
        self.slider.setEnabled(start_enabled)
        self.btn_play_preview.setEnabled(start_enabled)
        self.btn_set_start.setEnabled(start_enabled)
        self.btn_set_end.setEnabled(start_enabled)
        
        self.spin_res.setEnabled(enabled)
        self.txt_prompt.setEnabled(enabled)
        self.spin_start.setEnabled(enabled)
        self.spin_end.setEnabled(enabled)
        self.spin_step.setEnabled(enabled)
        self.spin_conf.setEnabled(enabled)
        self.spin_expand.setEnabled(enabled)
        self.chk_crop.setEnabled(enabled)
        self.chk_save_img.setEnabled(enabled)
        self.chk_save_mask.setEnabled(enabled)