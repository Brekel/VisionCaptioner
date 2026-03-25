import sys
import os
import json

# === FAST IMPORTS (PySide6 core) - Load immediately for splash screen ===
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QCheckBox,
                               QHBoxLayout, QPushButton, QFileDialog, QTabWidget,
                               QTextEdit, QProgressBar, QLabel, QGroupBox, QSplashScreen, QSplitter)
from PySide6.QtGui import QIcon, QPixmap, QFont, QColor, QTextCursor, QDesktopServices
from PySide6.QtCore import Qt, QByteArray, QUrl, Signal, QObject
from PySide6.QtMultimedia import QSoundEffect

# === DEFERRED IMPORTS ===
# Heavy imports (torch, transformers, etc.) are done AFTER splash is shown
# These will be imported in the `if __name__ == "__main__"` block

# Placeholder globals - will be set after deferred import
QwenEngine = None
SAM3Engine = None
DropLineEdit = None
GPUMonitorWorker = None
CaptionsTab = None
ReviewTab = None
MaskTab = None
VideoTab = None
QATab = None

SETTINGS_FILE = "settings.json"

class EmittingStream(QObject):
    textWritten = Signal(str)
    
    def write(self, text):
        try:
            self.textWritten.emit(str(text))
        except:
            # Fallback to true stdio to avoid recursion if emitting fails
            try:
                # We can't easily access the 'original' streams here globally without keeping a ref
                # but we can try to write to fd 1/2 if needed, or just suppress to prevent crash
                pass 
            except: pass
        
    def flush(self):
        pass
        
    def isatty(self):
        return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Captioner")
        self.resize(1200, 950)
        
        # Instantiate Engines Centrally
        self.engine = QwenEngine()       # For Captioning
        self.sam_engine = SAM3Engine()   # Shared for Masking & Video Extraction

        self.setup_ui()
        
        # --- GLOBAL LOGGING REDIRECT ---
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        self.stdout_stream = EmittingStream()
        self.stdout_stream.textWritten.connect(self.write_to_log)
        self.stderr_stream = EmittingStream()
        self.stderr_stream.textWritten.connect(self.write_to_log)
        
        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream
        
        self.load_settings()
        
        self.gpu_monitor = GPUMonitorWorker()
        self.gpu_monitor.stats_update.connect(self.update_gpu_stats)
        self.gpu_monitor.start()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        top_bar = QHBoxLayout()
        self.txt_folder = DropLineEdit()
        self.txt_folder.setToolTip("Path to the folder containing images or videos to caption.")
        self.txt_folder.textChanged.connect(self.on_folder_changed)
        
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_browse.setToolTip("Open file explorer to select folder.")
        
        self.btn_open_folder = QPushButton("📂")
        self.btn_open_folder.setToolTip("Open current folder in file explorer")
        self.btn_open_folder.setFixedWidth(32)
        self.btn_open_folder.clicked.connect(self.open_folder_in_explorer)

        self.chk_sound = QCheckBox("🔊")
        self.chk_sound.setToolTip("Play sound when processing finishes")
        self.chk_sound.setChecked(True)

        # Initialize Sound Effect
        self.sound_player = QSoundEffect()
        # Ensure you have a 'notification.wav' in your folder
        wav_path = os.path.join(os.getcwd(), "notification.wav")
        self.sound_player.setSource(QUrl.fromLocalFile(wav_path))
        self.sound_player.setVolume(0.5) # 0.0 to 1.0

        top_bar.addWidget(self.txt_folder)
        top_bar.addWidget(self.btn_browse)
        top_bar.addWidget(self.btn_open_folder)
        top_bar.addWidget(self.chk_sound)
        layout.addLayout(top_bar)

        # Initialize Tabs
        self.tabs = QTabWidget()
        
        # 1. Video Extraction Tab (New)
        self.tab_video = VideoTab(self.sam_engine)
        self.tab_video.log_msg.connect(self.log)
        self.tab_video.work_finished.connect(self._try_unload_sam3)

        # 2. Mask Tab (Pass shared engine)
        self.tab_mask = MaskTab(self.sam_engine)
        self.tab_mask.log_msg.connect(self.log)
        self.tab_mask.batch_finished.connect(self.play_notification)
        self.tab_mask.batch_finished.connect(self._try_unload_sam3)
        
        # 3. Captions Tab
        self.tab_gen = CaptionsTab(self.engine)
        self.tab_gen.log_msg.connect(self.log)
        self.tab_gen.request_lock.connect(self.lock_interface)
        self.tab_gen.batch_finished.connect(self.play_notification) 
        
        # 4. Review Tab
        self.tab_rev = ReviewTab()
        self.tab_rev.log_msg.connect(self.log)

        # 5. QA Tab
        self.tab_qa = QATab()
        self.tab_qa.log_msg.connect(self.log)

        # Add Tabs in Order
        self.tabs.addTab(self.tab_video, "Video Extract")
        self.tabs.addTab(self.tab_mask, "Mask Segmentation")
        self.tabs.addTab(self.tab_gen, "Captions")
        self.tabs.addTab(self.tab_rev, "Review && Edit")
        self.tabs.addTab(self.tab_qa, "Quality Assurance")
        
        self.tabs.setCurrentWidget(self.tab_gen)
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # --- REFACTOR: SPLITTER FOR LOGS ---
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(self.tabs)
        
        # Bottom Container (Stats + Log + Footer)
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        stats_group = QGroupBox()
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        bar_style = """
            QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; background-color: #333; color: white; font-weight: bold; }
            QProgressBar::chunk { background-color: #a15f13; width: 20px; }
        """
        self.bar_vram = QProgressBar(); self.bar_vram.setStyleSheet(bar_style); self.bar_vram.setFormat("VRAM: %v GB")
        self.bar_gpu = QProgressBar(); self.bar_gpu.setStyleSheet(bar_style); self.bar_gpu.setFormat("Core: %p%")
        stats_layout.addWidget(QLabel("VRAM:")); stats_layout.addWidget(self.bar_vram)
        stats_layout.addWidget(QLabel("GPU Load:")); stats_layout.addWidget(self.bar_gpu)
        bottom_layout.addWidget(stats_group)

        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True) 
        # Height is now controlled by splitter, but set a sensible minimum
        self.txt_log.setMinimumHeight(50)
        bottom_layout.addWidget(self.txt_log)
        
        # --- FOOTER (Version + Link) ---
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(5, 0, 5, 5)
        
        import torch
        py_ver = sys.version.split(' ')[0]
        pt_ver = torch.__version__
        
        version_str = f"Python: {py_ver}    |    PyTorch: {pt_ver}"
        self.lbl_version = QLabel(version_str)
        self.lbl_version.setStyleSheet("color: #777; font-size: 11px;")
        footer_layout.addWidget(self.lbl_version)
        
        footer_layout.addStretch()
        
        self.lbl_link = QLabel('<a href="https://brekel.com" style="color: #999; text-decoration: none;">brekel.com</a>')
        self.lbl_link.setOpenExternalLinks(True); self.lbl_link.setAlignment(Qt.AlignRight)
        self.lbl_link.setStyleSheet("QLabel { font-size: 11px; margin-right: 5px; }")
        footer_layout.addWidget(self.lbl_link)
        
        bottom_layout.addLayout(footer_layout)
        
        self.main_splitter.addWidget(bottom_widget)
        # Set initial sizes (give most space to tabs)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        
        layout.addWidget(self.main_splitter)
        # -----------------------------------

    def write_to_log(self, text):
        # Move cursor to end to avoid inserting in middle if user clicked
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.txt_log.setTextCursor(cursor)
        self.txt_log.insertPlainText(text)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def log(self, msg):
        self.write_to_log(msg + "\n")

    def browse_folder(self):
        current_path = self.txt_folder.text()
        start_dir = current_path if os.path.exists(current_path) else ""
        f = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if f: self.txt_folder.setText(f)

    def open_folder_in_explorer(self):
        folder = self.txt_folder.text()
        if folder and os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def on_folder_changed(self, text):
        if os.path.exists(text):
            self.tab_gen.update_folder(text)
            self.tab_rev.update_folder(text)
            self.tab_mask.update_folder(text)
            self.tab_video.update_folder(text)
            self.tab_qa.update_folder(text)

    def on_tab_changed(self, index):
        current_tab = self.tabs.widget(index)
        if current_tab == self.tab_rev:
            self.tab_rev.refresh_file_list()
            self.tab_rev.set_focus_to_image() # <-- SET FOCUS
        elif current_tab == self.tab_qa:
            self.tab_qa.refresh_file_list()
            self.tab_qa.set_focus_to_image()
        self._try_unload_sam3()

    def _is_sam3_busy(self):
        """Check if any SAM3 worker is currently running."""
        if self.tab_video.worker and self.tab_video.worker.isRunning():
            return True
        if hasattr(self.tab_video, 'grab_worker') and self.tab_video.grab_worker and self.tab_video.grab_worker.isRunning():
            return True
        if self.tab_mask.worker and self.tab_mask.worker.isRunning():
            return True
        if self.tab_mask.loader_worker and self.tab_mask.loader_worker.isRunning():
            return True
        return False

    def _try_unload_sam3(self):
        """Unload SAM3 if not on a SAM3 tab and no SAM3 work is in progress."""
        current_tab = self.tabs.currentWidget()
        on_sam3_tab = current_tab in (self.tab_video, self.tab_mask)
        if not on_sam3_tab and not self._is_sam3_busy():
            if self.sam_engine.model is not None:
                self.log("♻️ Auto-unloading SAM3 model (left SAM3 tabs)")
                self.sam_engine.unload()

    def showEvent(self, event):
        """Called when the window is first shown."""
        super().showEvent(event)
        # Set initial focus when app starts
        self.on_tab_changed(self.tabs.currentIndex())
        
        # Trigger background model scan (prunes cache & updates)
        # We do this silently
        from gui_workers import ScanWorker
        self.scan_worker = ScanWorker()
        self.scan_worker.log.connect(self.log)
        self.scan_worker.finished.connect(lambda r: self.log(f"Background Model Scan: Cache updated ({len(r)} models found)."))
        self.scan_worker.start()

    def lock_interface(self, locked):
        self.txt_folder.setEnabled(not locked)
        if hasattr(self, 'btn_browse'):
            self.btn_browse.setEnabled(not locked)

    def update_gpu_stats(self, used_gb, total_gb, vram_pct, core_pct, available):
        if not available:
            self.bar_vram.setFormat("No NVIDIA GPU"); self.bar_vram.setValue(0); self.bar_gpu.setValue(0); return
        self.bar_vram.setMaximum(int(total_gb * 100)); self.bar_vram.setValue(int(used_gb * 100))
        self.bar_vram.setFormat(f"VRAM: {used_gb:.1f} GB / {total_gb:.1f} GB ({int(vram_pct)}%)")
        self.bar_gpu.setValue(int(core_pct)); self.bar_gpu.setFormat(f"GPU Load: {int(core_pct)}%")

    def play_notification(self):
        if self.chk_sound.isChecked():
            self.sound_player.play()

    def save_settings(self):
        data = { "folder": self.txt_folder.text(), "geometry": self.saveGeometry().toHex().data().decode(),
                 "main_splitter_state": self.main_splitter.saveState().toHex().data().decode(),
                 "generate_tab": self.tab_gen.get_settings(), "review_tab": self.tab_rev.get_settings(),
                 "mask_tab": self.tab_mask.get_settings(), "video_tab": self.tab_video.get_settings(),
                 "qa_tab": self.tab_qa.get_settings() }
        try:
            with open(SETTINGS_FILE, 'w') as f: json.dump(data, f, indent=4)
        except Exception as e: print(f"Failed to save settings: {e}")

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f: data = json.load(f)
                if "geometry" in data: self.restoreGeometry(QByteArray.fromHex(data["geometry"].encode()))
                if "main_splitter_state" in data: self.main_splitter.restoreState(QByteArray.fromHex(data["main_splitter_state"].encode()))
                if "folder" in data: self.txt_folder.setText(data["folder"])
                if "generate_tab" in data: self.tab_gen.set_settings(data["generate_tab"])
                if "review_tab" in data: self.tab_rev.set_settings(data["review_tab"])
                if "mask_tab" in data: self.tab_mask.set_settings(data["mask_tab"])
                if "video_tab" in data: self.tab_video.set_settings(data["video_tab"])
                if "qa_tab" in data: self.tab_qa.set_settings(data["qa_tab"])
            except Exception as e: print(f"Failed to load settings: {e}")

    def closeEvent(self, event):
        # Restore streams
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
        self.save_settings()
        if self.gpu_monitor and self.gpu_monitor.isRunning():
            self.gpu_monitor.requestInterruption(); self.gpu_monitor.wait(500)
        if self.tab_gen.worker and self.tab_gen.worker.isRunning():
            self.tab_gen.worker.terminate(); self.tab_gen.worker.wait(100)
        if self.tab_gen.test_worker and self.tab_gen.test_worker.isRunning():
            self.tab_gen.test_worker.terminate(); self.tab_gen.test_worker.wait(100)
        if self.tab_gen.loader_worker and self.tab_gen.loader_worker.isRunning():
            self.tab_gen.loader_worker.terminate(); self.tab_gen.loader_worker.wait(100)
        if hasattr(self, 'tab_mask'): self.tab_mask.sam_engine.unload()
        
        # Clean up Video Worker if running
        if hasattr(self.tab_video, 'worker') and self.tab_video.worker and self.tab_video.worker.isRunning():
            self.tab_video.worker.stop()
            self.tab_video.worker.wait(100)

        # Clean up QA Worker if running
        if hasattr(self.tab_qa, 'worker') and self.tab_qa.worker and self.tab_qa.worker.isRunning():
            self.tab_qa.worker.stop()
            self.tab_qa.worker.wait(100)
            
        # Unload engines
        if self.sam_engine: self.sam_engine.unload()
        
        event.accept(); os._exit(0)

def apply_dark_theme(app):
    app.setStyle("Fusion")
    dark_stylesheet = """
    QWidget { background-color: #2b2b2b; color: #dddddd; font-family: Segoe UI, Arial, sans-serif; }
    QWidget:disabled { color: #888888; }
    QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #aaa; }
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QListWidget { background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px; color: #eee; padding: 4px; }
    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled { background-color: #262626; color: #666666; border: 1px solid #333; }
    QLineEdit:focus, QTextEdit:focus, QComboBox:focus { border: 1px solid #3daee9; }
    QListWidget::item:selected { background-color: #3daee9; color: white; }
    QListWidget::item:selected:!active { background-color: #a15f13; color: white; }
    QComboBox { background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px; padding: 4px; color: #eee; }
    QComboBox:disabled { background-color: #262626; color: #666666; border: 1px solid #333; }
    QComboBox::drop-down { border: 0px; background: #2b2b2b; width: 20px; }
    QComboBox::drop-down:disabled { background: #262626; }
    QComboBox QAbstractItemView { background-color: #2b2b2b; color: #eee; border: 1px solid #444; selection-background-color: #3daee9; }
    QScrollBar:vertical { background: #2b2b2b; width: 12px; margin: 0px; }
    QScrollBar::handle:vertical { background: #555; min-height: 20px; border-radius: 6px; margin: 2px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }
    QTabBar::tab { background: #1e1e1e; color: #888; padding: 8px 12px; border: 1px solid #444; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
    QTabBar::tab:selected { background: #3daee9; color: white; font-weight: bold; border-bottom: 1px solid #3daee9; }
    QTabBar::tab:hover { background: #333; color: #ddd; }
    QPushButton { background-color: #444; border: 1px solid #333; color: white; padding: 5px; border-radius: 4px; }
    QPushButton:hover { background-color: #555; }
    QPushButton:pressed { background-color: #333; }
    QPushButton:disabled { background-color: #333; color: #666; border: 1px solid #2a2a2a; }
    QCheckBox:disabled { color: #666; }
    """
    app.setStyleSheet(dark_stylesheet)

def do_deferred_imports(splash, app):
    """Import heavy modules after splash is visible, with progress updates."""
    global QwenEngine, SAM3Engine, DropLineEdit, GPUMonitorWorker
    global CaptionsTab, ReviewTab, MaskTab, VideoTab, QATab
    
    def update_splash(msg):
        splash.showMessage(msg, Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()
    
    update_splash("Loading UI components...")
    from gui_widgets import DropLineEdit as _DropLineEdit
    DropLineEdit = _DropLineEdit
    
    update_splash("Loading GPU monitor...")
    from gui_workers import GPUMonitorWorker as _GPUMonitorWorker
    GPUMonitorWorker = _GPUMonitorWorker
    
    update_splash("Loading AI backend (this may take a moment)...")
    from backend import QwenEngine as _QwenEngine, SAM3Engine as _SAM3Engine
    update_splash("Loading AI backend (this may take a moment)... Qwen")
    QwenEngine = _QwenEngine
    update_splash("Loading AI backend (this may take a moment)...SAM3")
    SAM3Engine = _SAM3Engine
    
    update_splash("Loading Captions tab...")
    from tab_captions import CaptionsTab as _CaptionsTab
    CaptionsTab = _CaptionsTab
    
    update_splash("Loading Review tab...")
    from tab_review import ReviewTab as _ReviewTab
    ReviewTab = _ReviewTab
    
    update_splash("Loading Mask tab...")
    from tab_mask import MaskTab as _MaskTab
    MaskTab = _MaskTab
    
    update_splash("Loading Video tab...")
    from tab_video import VideoTab as _VideoTab
    VideoTab = _VideoTab

    update_splash("Loading QA tab...")
    from tab_qa import QATab as _QATab
    QATab = _QATab

    update_splash("Initializing main window...")

if __name__ == "__main__":
    # Fix Taskbar Icon on Windows
    import ctypes
    import platform
    if platform.system() == 'Windows':
        myappid = 'mycompany.myproduct.subproduct.version' # Arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    # === SHOW SPLASH IMMEDIATELY ===
    icon_path = "logo.png" 
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        splash_pix = QPixmap(icon_path)
    else:
        splash_pix = QPixmap(500, 300)
        splash_pix.fill(QColor(30, 30, 30))
    
    splash = QSplashScreen(splash_pix)
    font = QFont("Arial", 14, QFont.Bold)
    splash.setFont(font)
    splash.show()
    splash.showMessage("Starting...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    app.processEvents()
    
    # === NOW DO HEAVY IMPORTS (splash is visible) ===
    do_deferred_imports(splash, app)
    
    try:
        window = MainWindow()
        window.show()
        splash.finish(window)
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        traceback.print_exc()