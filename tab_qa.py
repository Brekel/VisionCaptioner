import os
import glob
import json
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QTextEdit, QSlider, QLabel, QSplitter, QGroupBox,
                               QLineEdit, QCheckBox, QPushButton, QMessageBox, QFormLayout, QFrame, QSizePolicy, QComboBox,
                               QProgressBar, QRadioButton, QButtonGroup, QSpinBox, QTreeWidget, QTreeWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QTimer, Signal, QByteArray, QThread
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QShortcut, QKeySequence
from gui_widgets import ResizableImageLabel


# ---------------------------------------------------------------------------
# Analysis Worker
# ---------------------------------------------------------------------------
class QAAnalysisWorker(QThread):
    progress = Signal(int, int)           # current, total
    result_ready = Signal(str, dict)      # filepath, result dict
    log_msg = Signal(str)
    finished_analysis = Signal()

    BLINK_THRESHOLD = 0.45  # blendshape score above this = eyes closed

    def __init__(self, image_files, settings, folder):
        super().__init__()
        self.image_files = image_files
        self.settings = settings
        self.folder = folder
        self._stop = False

        # Per-thread detectors (MediaPipe / YuNet / Haar are not thread-safe)
        self._tls = threading.local()
        self._landmarkers_lock = threading.Lock()
        self._all_landmarkers = []  # track for cleanup

    def stop(self):
        self._stop = True

    # -- blur ---------------------------------------------------------------
    @staticmethod
    def analyze_blur(cv_gray):
        return cv2.Laplacian(cv_gray, cv2.CV_64F).var()

    # -- resolution ---------------------------------------------------------
    @staticmethod
    def analyze_resolution(w, h, threshold):
        min_dim = min(w, h)
        if min_dim >= threshold:
            return 1.0
        half = threshold / 2.0
        if min_dim <= half:
            return 0.0
        return (min_dim - half) / (threshold - half)

    # -- mask existence -----------------------------------------------------
    @staticmethod
    def check_mask_exists(img_path, folder):
        base = os.path.splitext(os.path.basename(img_path))[0]
        # 1. subfolder
        if os.path.exists(os.path.join(folder, "masks", f"{base}.png")):
            return True
        # 2. separate file
        base_path = os.path.splitext(img_path)[0]
        for suffix in ["-masklabel.png", "-masklabel.jpg", "_masklabel.png", "_masklabel.jpg"]:
            if os.path.exists(base_path + suffix):
                return True
        # 3. alpha channel
        try:
            pil = Image.open(img_path)
            if pil.mode in ("RGBA", "LA"):
                alpha = np.array(pil.split()[-1])
                if alpha.mean() < 255:
                    return True
        except Exception:
            pass
        return False

    # -- face + eyes (MediaPipe) --------------------------------------------
    def _get_face_landmarker(self):
        """Lazy-init a per-thread MediaPipe FaceLandmarker with blendshapes."""
        landmarker = getattr(self._tls, "face_landmarker", None)
        if landmarker is not None:
            return landmarker
        if not _HAS_MEDIAPIPE:
            return None
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "_")
        model_path = os.path.join(model_dir, "face_landmarker_v2_with_blendshapes.task")
        if not os.path.exists(model_path):
            self._download_model(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                model_path)
        if not os.path.exists(model_path):
            return None
        try:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=10,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5,
            )
            landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._tls.face_landmarker = landmarker
            with self._landmarkers_lock:
                self._all_landmarkers.append(landmarker)
        except Exception as e:
            self.log_msg.emit(f"Failed to create MediaPipe FaceLandmarker: {e}")
            return None
        return landmarker

    def analyze_faces_and_eyes(self, cv_img):
        """Detect faces and eye-blink state in one pass.

        Returns (face_count, eyes_closed_count).
        Uses MediaPipe FaceLandmarker with blendshapes when available,
        falls back to YuNet + Haar cascade otherwise.
        """
        landmarker = self._get_face_landmarker()
        if landmarker is not None:
            return self._analyze_mediapipe(cv_img, landmarker)
        return self._analyze_fallback(cv_img)

    def _analyze_mediapipe(self, cv_img, landmarker):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_img)
        result = landmarker.detect(mp_image)
        face_count = len(result.face_landmarks)
        eyes_closed = 0

        for blendshapes in (result.face_blendshapes or []):
            blink_scores = {}
            for bs in blendshapes:
                if bs.category_name in ("eyeBlinkLeft", "eyeBlinkRight"):
                    blink_scores[bs.category_name] = bs.score
            left = blink_scores.get("eyeBlinkLeft", 0.0)
            right = blink_scores.get("eyeBlinkRight", 0.0)
            if left > self.BLINK_THRESHOLD or right > self.BLINK_THRESHOLD:
                eyes_closed += 1

        return face_count, eyes_closed

    # -- fallback: YuNet face + Haar eyes -----------------------------------
    def _analyze_fallback(self, cv_img):
        faces = self._detect_faces_yunet(cv_img)
        face_count = len(faces) if faces is not None else 0
        eyes_closed = 0
        if face_count > 0:
            cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            for face in faces:
                if self._eyes_closed_haar(cv_gray, face):
                    eyes_closed += 1
        return face_count, eyes_closed

    def _detect_faces_yunet(self, cv_img):
        h, w = cv_img.shape[:2]
        detector = getattr(self._tls, "yunet_detector", None)
        if detector is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "_", "face_detection_yunet_2023mar.onnx")
            if not os.path.exists(model_path):
                self._download_model(
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    model_path)
            if not os.path.exists(model_path):
                return []
            detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.7, 0.3, 5000)
            self._tls.yunet_detector = detector
        detector.setInputSize((w, h))
        _, faces = detector.detect(cv_img)
        return faces if faces is not None else []

    def _eyes_closed_haar(self, cv_gray, face_bbox):
        """Fallback: returns True if eyes appear closed using Haar cascades."""
        try:
            eye_cascade = getattr(self._tls, "eye_cascade", None)
            if eye_cascade is None:
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                self._tls.eye_cascade = eye_cascade
            img_h, img_w = cv_gray.shape[:2]
            if len(face_bbox) >= 8:
                fw = int(face_bbox[2])
                eye_radius = max(int(fw * 0.15), 15)
                eyes_found = 0
                for ei in range(2):
                    ex, ey = int(face_bbox[4 + ei * 2]), int(face_bbox[5 + ei * 2])
                    x1, y1 = max(0, ex - eye_radius), max(0, ey - eye_radius)
                    x2, y2 = min(img_w, ex + eye_radius), min(img_h, ey + eye_radius)
                    patch = cv_gray[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue
                    min_sz = max(int(eye_radius * 0.4), 5)
                    if len(eye_cascade.detectMultiScale(patch, 1.05, 3, minSize=(min_sz, min_sz))) > 0:
                        eyes_found += 1
                return eyes_found < 2
            elif len(face_bbox) >= 4:
                x, y, w, h = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
                region = cv_gray[y:y + int(h * 0.5), x:x + w]
                if region.size == 0:
                    return True
                min_sz = max(int(w * 0.06), 5)
                return len(eye_cascade.detectMultiScale(region, 1.05, 3, minSize=(min_sz, min_sz))) < 2
        except Exception:
            pass
        return False

    @staticmethod
    def _download_model(url, dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception:
            pass

    # -- per-image analysis (called from thread pool) ------------------------
    def _analyze_single(self, img_path):
        """Analyze a single image. Thread-safe via thread-local detectors."""
        blur_enabled = self.settings.get("blur_enabled", True)
        res_enabled = self.settings.get("resolution_enabled", True)
        mask_enabled = self.settings.get("mask_enabled", True)
        face_enabled = self.settings.get("face_enabled", True)
        eyes_enabled = self.settings.get("eyes_enabled", True)
        res_threshold = self.settings.get("resolution_threshold", 512)
        blur_low = self.settings.get("blur_threshold_low", 50)
        blur_high = self.settings.get("blur_threshold_high", 500)

        result = {
            "filepath": img_path,
            "blur_variance": 0.0, "blur_score": 1.0,
            "width": 0, "height": 0, "min_dimension": 0, "resolution_score": 1.0,
            "has_mask": False, "mask_score": 1.0,
            "face_count": 0, "face_score": 1.0,
            "eyes_closed_count": 0, "eyes_score": 1.0,
        }

        try:
            pil_img = Image.open(img_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            cv_img = np.array(pil_img.convert("RGB"))
            h, w = cv_img.shape[:2]
            result["width"] = w
            result["height"] = h
            result["min_dimension"] = min(w, h)

            cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

            if blur_enabled:
                variance = self.analyze_blur(cv_gray)
                result["blur_variance"] = round(variance, 2)
                if blur_high > blur_low:
                    result["blur_score"] = round(max(0.0, min(1.0, (variance - blur_low) / (blur_high - blur_low))), 4)
                else:
                    result["blur_score"] = 1.0 if variance >= blur_low else 0.0

            if res_enabled:
                result["resolution_score"] = round(self.analyze_resolution(w, h, res_threshold), 4)

            if mask_enabled:
                has_mask = self.check_mask_exists(img_path, self.folder)
                result["has_mask"] = has_mask
                result["mask_score"] = 1.0 if has_mask else 0.0

            if face_enabled:
                face_count, eyes_closed = self.analyze_faces_and_eyes(cv_img)
                result["face_count"] = face_count
                if eyes_enabled and face_count > 0:
                    result["eyes_closed_count"] = eyes_closed
                    result["eyes_score"] = 0.0 if eyes_closed > 0 else 1.0

        except Exception as e:
            self.log_msg.emit(f"Error analyzing {os.path.basename(img_path)}: {e}")

        return img_path, result

    # -- main run -----------------------------------------------------------
    def run(self):
        total = len(self.image_files)
        num_workers = min(os.cpu_count() or 4, 8)
        completed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._analyze_single, p): p for p in self.image_files}

            for future in as_completed(futures):
                if self._stop:
                    for f in futures:
                        f.cancel()
                    break
                completed += 1
                self.progress.emit(completed, total)
                try:
                    img_path, result = future.result()
                    self.result_ready.emit(img_path, result)
                except Exception as e:
                    img_path = futures[future]
                    self.log_msg.emit(f"Error analyzing {os.path.basename(img_path)}: {e}")

        # Cleanup all per-thread MediaPipe landmarkers
        for lm in self._all_landmarkers:
            try:
                lm.close()
            except Exception:
                pass
        self._all_landmarkers.clear()

        self.progress.emit(total, total)
        self.finished_analysis.emit()


# ---------------------------------------------------------------------------
# Tree item with numeric sort
# ---------------------------------------------------------------------------
class _ScoreTreeItem(QTreeWidgetItem):
    """QTreeWidgetItem that sorts score columns numerically."""
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        if col == 0:
            return self.text(0).lower() < other.text(0).lower()
        a = self.data(col, Qt.UserRole)
        b = other.data(col, Qt.UserRole)
        if a is None: a = -1.0
        if b is None: b = -1.0
        return a < b


# ---------------------------------------------------------------------------
# QA Tab
# ---------------------------------------------------------------------------
class QATab(QWidget):
    log_msg = Signal(str)

    SCORE_COLORS = {
        "bad": "#e06c75",      # red
        "medium": "#e5c07b",   # yellow
        "good": "#98c379",     # green
    }

    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.image_files = []       # all images (unsorted)
        self.sorted_files = []      # sorted by score after analysis
        self.current_index = -1
        self.analysis_results = {}  # filepath -> result dict
        self.worker = None
        self.cv_img_original = None

        self.setup_ui()
        self.setup_shortcuts()

    # -----------------------------------------------------------------------
    # UI Setup
    # -----------------------------------------------------------------------
    def setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter

        # === LEFT PANEL ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # -- Analysis Settings --
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setContentsMargins(5, 10, 5, 5)

        # Blur
        self.chk_blur = QCheckBox("Blur Detection")
        self.chk_blur.setChecked(True)
        self.chk_blur.setToolTip("Detect blurry images using Laplacian variance")
        settings_layout.addRow(self.chk_blur)

        blur_thresh_layout = QHBoxLayout()
        blur_thresh_layout.addWidget(QLabel("Low:"))
        self.spin_blur_low = QSpinBox()
        self.spin_blur_low.setRange(1, 9999)
        self.spin_blur_low.setValue(50)
        self.spin_blur_low.setToolTip("Laplacian variance below this = definitely blurry (score 0)")
        blur_thresh_layout.addWidget(self.spin_blur_low)
        blur_thresh_layout.addWidget(QLabel("High:"))
        self.spin_blur_high = QSpinBox()
        self.spin_blur_high.setRange(1, 9999)
        self.spin_blur_high.setValue(500)
        self.spin_blur_high.setToolTip("Laplacian variance above this = definitely sharp (score 1)")
        blur_thresh_layout.addWidget(self.spin_blur_high)
        settings_layout.addRow("  Thresholds:", blur_thresh_layout)

        # Resolution
        self.chk_resolution = QCheckBox("Low Resolution")
        self.chk_resolution.setChecked(True)
        self.chk_resolution.setToolTip("Flag images below a minimum resolution")
        settings_layout.addRow(self.chk_resolution)

        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(32, 8192)
        self.spin_resolution.setValue(512)
        self.spin_resolution.setSuffix(" px")
        self.spin_resolution.setToolTip("Minimum acceptable shortest dimension")
        settings_layout.addRow("  Min dimension:", self.spin_resolution)

        # Mask
        self.chk_mask = QCheckBox("Missing Mask")
        self.chk_mask.setChecked(True)
        self.chk_mask.setToolTip("Check if images have a corresponding mask. Auto-disabled if no images have masks.")
        settings_layout.addRow(self.chk_mask)

        # Face Detection
        self.chk_face = QCheckBox("Face Detection")
        self.chk_face.setChecked(True)
        self.chk_face.setToolTip("Detect faces using YuNet (OpenCV). Downloads a small model on first use.")
        settings_layout.addRow(self.chk_face)

        face_mode_layout = QHBoxLayout()
        self.radio_face_missing = QRadioButton("Flag no face")
        self.radio_face_present = QRadioButton("Flag face found")
        self.radio_face_missing.setChecked(True)
        self.radio_face_missing.setToolTip("Bad score if no face detected (portrait dataset)")
        self.radio_face_present.setToolTip("Bad score if face detected (landscape/object dataset)")
        self.face_mode_group = QButtonGroup(self)
        self.face_mode_group.addButton(self.radio_face_missing, 0)
        self.face_mode_group.addButton(self.radio_face_present, 1)
        face_mode_layout.addWidget(self.radio_face_missing)
        face_mode_layout.addWidget(self.radio_face_present)
        settings_layout.addRow("  Mode:", face_mode_layout)

        # Eyes Closed
        self.chk_eyes = QCheckBox("Eyes Closed")
        self.chk_eyes.setChecked(True)
        self.chk_eyes.setToolTip("Detect closed eyes in detected faces (approximate heuristic)")
        settings_layout.addRow(self.chk_eyes)

        left_layout.addWidget(settings_group)

        # -- Weights --
        weights_group = QGroupBox("Criterion Weights")
        weights_layout = QFormLayout(weights_group)
        weights_layout.setContentsMargins(5, 10, 5, 5)

        self.weight_sliders = {}
        for name, default in [("Blur", 10), ("Resolution", 5), ("Mask", 10), ("Face", 7), ("Eyes", 3)]:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 20)  # 0.0 to 2.0
            slider.setValue(default)
            lbl = QLabel(f"{default / 10:.1f}")
            lbl.setFixedWidth(28)
            slider.valueChanged.connect(lambda v, l=lbl: l.setText(f"{v / 10:.1f}"))
            row = QHBoxLayout()
            row.addWidget(slider)
            row.addWidget(lbl)
            weights_layout.addRow(f"{name}:", row)
            self.weight_sliders[name.lower()] = slider

        left_layout.addWidget(weights_group)

        # -- Analyze Button + Progress --
        analyze_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze All")
        self.btn_analyze.setStyleSheet("background-color: #5d99c4; color: white; font-weight: bold;")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        analyze_layout.addWidget(self.btn_analyze)
        analyze_layout.addWidget(self.btn_stop)
        left_layout.addLayout(analyze_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # -- Sorted Image Table --
        self.COLUMN_KEYS = ["overall_score", "blur_score", "resolution_score", "mask_score", "face_score", "eyes_score"]
        self.COLUMN_HEADERS = ["File", "Overall", "Blur", "Res", "Mask", "Face", "Eyes"]

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(self.COLUMN_HEADERS)
        self.tree_widget.setRootIsDecorated(False)
        self.tree_widget.setAllColumnsShowFocus(True)
        self.tree_widget.setSortingEnabled(True)
        self.tree_widget.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree_widget.currentItemChanged.connect(self._on_tree_item_changed)
        self.tree_widget.header().sectionClicked.connect(self._on_header_clicked)

        # Column sizing
        header = self.tree_widget.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(self.COLUMN_HEADERS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        # Default sort: Overall descending (best first)
        self.tree_widget.sortByColumn(1, Qt.DescendingOrder)
        self._current_sort_col = 1
        self._current_sort_order = Qt.DescendingOrder

        left_layout.addWidget(self.tree_widget)

        # -- Batch Actions --
        batch_group = QGroupBox("Batch Actions")
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setContentsMargins(5, 10, 5, 5)

        # Caption append text
        append_layout = QHBoxLayout()
        append_layout.addWidget(QLabel("Append text:"))
        self.txt_append = QLineEdit(", low quality, worst quality, blurry")
        self.txt_append.setToolTip("Text to append to caption files")
        append_layout.addWidget(self.txt_append)
        batch_layout.addLayout(append_layout)

        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        self.combo_append_preset = QComboBox()
        self.combo_append_preset.addItems([
            "Custom",
            ", low quality, worst quality, jpeg artifacts, blurry",
            ", blurry, out of focus",
            ", low resolution, pixelated",
            ", eyes closed",
        ])
        self.combo_append_preset.currentIndexChanged.connect(self.on_append_preset_changed)
        preset_layout.addWidget(self.combo_append_preset)
        batch_layout.addLayout(preset_layout)

        # Criterion selector for threshold
        self.BATCH_CRITERIA = [
            ("Overall", "overall_score"),
            ("Blur", "blur_score"),
            ("Resolution", "resolution_score"),
            ("Mask", "mask_score"),
            ("Face", "face_score"),
            ("Eyes", "eyes_score"),
        ]
        criterion_layout = QHBoxLayout()
        criterion_layout.addWidget(QLabel("Criterion:"))
        self.combo_batch_criterion = QComboBox()
        for label, _ in self.BATCH_CRITERIA:
            self.combo_batch_criterion.addItem(label)
        self.combo_batch_criterion.setToolTip("Which score to compare against the threshold")
        self.combo_batch_criterion.currentIndexChanged.connect(lambda: self.on_threshold_changed(self.slider_threshold.value()))
        criterion_layout.addWidget(self.combo_batch_criterion)
        batch_layout.addLayout(criterion_layout)

        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(50)
        self.slider_threshold.valueChanged.connect(self.on_threshold_changed)
        self.lbl_threshold = QLabel("0.50")
        self.lbl_threshold.setFixedWidth(32)
        thresh_layout.addWidget(self.slider_threshold)
        thresh_layout.addWidget(self.lbl_threshold)
        batch_layout.addLayout(thresh_layout)

        self.lbl_below_count = QLabel("0 of 0 images below threshold")
        self.lbl_below_count.setStyleSheet("color: #888;")
        batch_layout.addWidget(self.lbl_below_count)

        # Batch action buttons
        batch_btn_layout = QHBoxLayout()
        self.btn_batch_append = QPushButton("Append to All Below")
        self.btn_batch_append.setToolTip("Append text to captions of all images scoring below the threshold on the selected criterion")
        self.btn_batch_append.clicked.connect(lambda: self.apply_batch_action("append"))
        self.btn_batch_move = QPushButton("Move All Below to Unused")
        self.btn_batch_move.setToolTip("Move all images (+ caption + mask) scoring below the threshold to the unused folder")
        self.btn_batch_move.clicked.connect(lambda: self.apply_batch_action("move"))
        batch_btn_layout.addWidget(self.btn_batch_append)
        batch_btn_layout.addWidget(self.btn_batch_move)
        batch_layout.addLayout(batch_btn_layout)

        left_layout.addWidget(batch_group)

        # === RIGHT PANEL ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Per-image action buttons (above image for visibility)
        action_layout = QHBoxLayout()
        self.btn_append_single = QPushButton("Append to Caption [A]")
        self.btn_append_single.setToolTip("Append the configured text to this image's caption file (A)")
        self.btn_append_single.clicked.connect(self.append_to_current_caption)
        self.btn_move_single = QPushButton("Move to Unused [D]")
        self.btn_move_single.setToolTip("Move this image (and its caption/mask) to the unused folder (D)")
        self.btn_move_single.clicked.connect(self.move_current_to_unused)
        action_layout.addWidget(self.btn_append_single)
        action_layout.addWidget(self.btn_move_single)
        right_layout.addLayout(action_layout)

        # Image display
        self.lbl_image = ResizableImageLabel()
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(200, 200)
        # Wheel navigation on image
        self.lbl_image.wheelEvent = self.image_wheel_event
        right_layout.addWidget(self.lbl_image, stretch=1)

        # Info bar
        self.lbl_info = QLabel("0 / 0")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color: #aaa; padding: 2px;")
        right_layout.addWidget(self.lbl_info)

        # Navigation slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        right_layout.addWidget(self.slider)

        # Per-image quality details
        details_group = QGroupBox("Quality Details")
        details_layout = QFormLayout(details_group)
        details_layout.setContentsMargins(5, 10, 5, 5)

        self.lbl_detail_blur = QLabel("-")
        self.lbl_detail_resolution = QLabel("-")
        self.lbl_detail_mask = QLabel("-")
        self.lbl_detail_face = QLabel("-")
        self.lbl_detail_eyes = QLabel("-")
        self.lbl_detail_overall = QLabel("-")
        self.lbl_detail_overall.setStyleSheet("font-weight: bold; font-size: 14px;")

        details_layout.addRow("Blur:", self.lbl_detail_blur)
        details_layout.addRow("Resolution:", self.lbl_detail_resolution)
        details_layout.addRow("Mask:", self.lbl_detail_mask)
        details_layout.addRow("Face:", self.lbl_detail_face)
        details_layout.addRow("Eyes:", self.lbl_detail_eyes)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #555;")
        details_layout.addRow(sep)
        details_layout.addRow("Overall Score:", self.lbl_detail_overall)

        right_layout.addWidget(details_group)

        # Caption preview
        caption_group = QGroupBox("Caption")
        caption_layout = QVBoxLayout(caption_group)
        caption_layout.setContentsMargins(5, 10, 5, 5)
        self.txt_caption = QTextEdit()
        self.txt_caption.setReadOnly(True)
        self.txt_caption.setMaximumHeight(80)
        caption_layout.addWidget(self.txt_caption)
        right_layout.addWidget(caption_group)

        # Assemble splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 700])
        layout.addWidget(splitter)

    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self.navigate_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.navigate_next)
        QShortcut(QKeySequence(Qt.Key_A), self, self.append_to_current_caption)
        QShortcut(QKeySequence(Qt.Key_D), self, self.move_current_to_unused)

    # -----------------------------------------------------------------------
    # Folder / File Discovery
    # -----------------------------------------------------------------------
    def update_folder(self, path):
        if path and os.path.isdir(path):
            self.current_folder = path
            self.refresh_file_list()
            self.load_cache()

    def refresh_file_list(self):
        if not self.current_folder:
            self.tree_widget.clear()
            return

        exts = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.current_folder, ext)))
            files.extend(glob.glob(os.path.join(self.current_folder, ext.upper())))

        unique_files = sorted(set(files))
        self.image_files = [f for f in unique_files if not f.lower().endswith("-masklabel.png")]
        self.sorted_files = list(self.image_files)

        self._rebuild_list_widget()

    def _rebuild_list_widget(self):
        self.tree_widget.blockSignals(True)
        self.tree_widget.setSortingEnabled(False)
        self.tree_widget.clear()

        for f in self.sorted_files:
            basename = os.path.basename(f)
            item = _ScoreTreeItem(self.tree_widget)
            item.setText(0, basename)
            item.setData(0, Qt.UserRole, f)  # store full path

            if f in self.analysis_results:
                r = self.analysis_results[f]
                scores = [
                    r.get("overall_score", -1),
                    r.get("blur_score", -1),
                    r.get("resolution_score", -1),
                    r.get("mask_score", -1),
                    r.get("face_score", -1),
                    r.get("eyes_score", -1),
                ]
                for col, score in enumerate(scores, start=1):
                    if score < 0:
                        item.setText(col, "-")
                    else:
                        item.setText(col, f"{score:.2f}")
                        item.setForeground(col, QColor(self._score_color(score)))
                    item.setData(col, Qt.UserRole, score)  # for sorting
                # Color the filename by overall score
                item.setForeground(0, QColor(self._score_color(scores[0])))
            else:
                for col in range(1, len(self.COLUMN_HEADERS)):
                    item.setText(col, "-")
                    item.setData(col, Qt.UserRole, -1.0)

        self.tree_widget.setSortingEnabled(True)
        self.tree_widget.sortByColumn(self._current_sort_col, self._current_sort_order)
        self.tree_widget.blockSignals(False)

        # Update slider and select first
        self.slider.setRange(0, max(0, len(self.sorted_files) - 1))
        if self.tree_widget.topLevelItemCount() > 0:
            self.tree_widget.setCurrentItem(self.tree_widget.topLevelItem(0))

    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------
    def _current_visual_index(self):
        """Return the visual row index of the currently selected tree item."""
        item = self.tree_widget.currentItem()
        if item is None:
            return -1
        return self.tree_widget.indexOfTopLevelItem(item)

    def _filepath_from_item(self, item):
        if item is None:
            return None
        return item.data(0, Qt.UserRole)

    def _on_tree_item_changed(self, current, previous):
        f_path = self._filepath_from_item(current)
        if f_path is None:
            return
        row = self._current_visual_index()
        self.current_index = row

        self.slider.blockSignals(True)
        self.slider.setValue(row)
        self.slider.blockSignals(False)

        self.load_image(f_path)
        self.load_caption_preview(f_path)
        self.update_detail_panel(f_path)

        # Info label
        filename = os.path.basename(f_path)
        total = self.tree_widget.topLevelItemCount()
        res_info = ""
        if self.cv_img_original is not None:
            h, w = self.cv_img_original.shape[:2]
            res_info = f"    |    {w} x {h}"
        score_info = ""
        if f_path in self.analysis_results:
            score_info = f"    |    Score: {self.analysis_results[f_path].get('overall_score', 0):.2f}"
        self.lbl_info.setText(f"{row + 1} / {total}    |    {filename}{res_info}{score_info}")

    def _on_header_clicked(self, logical_index):
        """Track which column / order the user picked so rebuilds preserve it."""
        self._current_sort_col = logical_index
        self._current_sort_order = self.tree_widget.header().sortIndicatorOrder()

    def on_slider_changed(self, value):
        count = self.tree_widget.topLevelItemCount()
        if 0 <= value < count:
            self.tree_widget.setCurrentItem(self.tree_widget.topLevelItem(value))

    def image_wheel_event(self, event):
        self.navigate(1 if event.angleDelta().y() < 0 else -1)

    def navigate(self, steps):
        idx = self._current_visual_index()
        new_row = idx + steps
        count = self.tree_widget.topLevelItemCount()
        if 0 <= new_row < count:
            self.tree_widget.setCurrentItem(self.tree_widget.topLevelItem(new_row))

    def navigate_prev(self):
        self.navigate(-1)

    def navigate_next(self):
        self.navigate(1)

    def set_focus_to_image(self):
        self.lbl_image.setFocus()

    # -----------------------------------------------------------------------
    # Image Display
    # -----------------------------------------------------------------------
    def load_image(self, f_path):
        self.cv_img_original = None
        try:
            pil_img = Image.open(f_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            self.cv_img_original = np.array(pil_img.convert("RGB"))
        except Exception as e:
            self.log_msg.emit(f"Error loading {os.path.basename(f_path)}: {e}")
            self.lbl_image.set_image(None)
            return
        self.update_image_display()

    def update_image_display(self):
        if self.cv_img_original is None:
            self.lbl_image.set_image(None)
            return
        img = self.cv_img_original
        height, width, _ = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.lbl_image.current_pixmap = QPixmap.fromImage(q_img)
        self.lbl_image.update_view()

    def load_caption_preview(self, f_path):
        txt_path = os.path.splitext(f_path)[0] + ".txt"
        self.txt_caption.blockSignals(True)
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.txt_caption.setPlainText(f.read())
            except Exception:
                self.txt_caption.setPlainText("")
        else:
            self.txt_caption.setPlainText("")
        self.txt_caption.blockSignals(False)

    # -----------------------------------------------------------------------
    # Detail Panel
    # -----------------------------------------------------------------------
    def update_detail_panel(self, f_path):
        if f_path not in self.analysis_results:
            for lbl in [self.lbl_detail_blur, self.lbl_detail_resolution, self.lbl_detail_mask,
                        self.lbl_detail_face, self.lbl_detail_eyes, self.lbl_detail_overall]:
                lbl.setText("-")
            return

        r = self.analysis_results[f_path]

        # Blur
        bv = r.get("blur_variance", 0)
        bs = r.get("blur_score", 1.0)
        self.lbl_detail_blur.setText(f"Score: {bs:.2f}  (variance: {bv:.1f})")
        self.lbl_detail_blur.setStyleSheet(f"color: {self._score_color(bs)};")

        # Resolution
        rs = r.get("resolution_score", 1.0)
        md = r.get("min_dimension", 0)
        self.lbl_detail_resolution.setText(f"Score: {rs:.2f}  ({r.get('width', 0)}x{r.get('height', 0)}, min: {md})")
        self.lbl_detail_resolution.setStyleSheet(f"color: {self._score_color(rs)};")

        # Mask
        has_mask = r.get("has_mask", False)
        ms = r.get("mask_score", 1.0)
        self.lbl_detail_mask.setText(f"{'YES' if has_mask else 'NO'}  (score: {ms:.2f})")
        self.lbl_detail_mask.setStyleSheet(f"color: {self._score_color(ms)};")

        # Face
        fc = r.get("face_count", 0)
        fs = r.get("face_score", 1.0)
        self.lbl_detail_face.setText(f"{fc} face(s) detected  (score: {fs:.2f})")
        self.lbl_detail_face.setStyleSheet(f"color: {self._score_color(fs)};")

        # Eyes
        ec = r.get("eyes_closed_count", 0)
        es = r.get("eyes_score", 1.0)
        if fc > 0:
            self.lbl_detail_eyes.setText(f"{ec} with eyes closed  (score: {es:.2f})")
        else:
            self.lbl_detail_eyes.setText("N/A (no faces)")
        self.lbl_detail_eyes.setStyleSheet(f"color: {self._score_color(es)};")

        # Overall
        overall = r.get("overall_score", 0)
        self.lbl_detail_overall.setText(f"{overall:.2f}")
        self.lbl_detail_overall.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {self._score_color(overall)};")

    @staticmethod
    def _score_color(score):
        if score < 0.3:
            return "#e06c75"
        elif score < 0.6:
            return "#e5c07b"
        return "#98c379"

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    def _gather_settings(self):
        return {
            "blur_enabled": self.chk_blur.isChecked(),
            "blur_threshold_low": self.spin_blur_low.value(),
            "blur_threshold_high": self.spin_blur_high.value(),
            "resolution_enabled": self.chk_resolution.isChecked(),
            "resolution_threshold": self.spin_resolution.value(),
            "mask_enabled": self.chk_mask.isChecked(),
            "face_enabled": self.chk_face.isChecked(),
            "face_flag_mode": "not_detected" if self.radio_face_missing.isChecked() else "detected",
            "eyes_enabled": self.chk_eyes.isChecked(),
        }

    def start_analysis(self):
        if not self.image_files:
            self.log_msg.emit("No images to analyze.")
            return
        if self.worker and self.worker.isRunning():
            self.log_msg.emit("Analysis already running.")
            return

        self.analysis_results = {}
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.image_files))
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(True)

        settings = self._gather_settings()
        self.worker = QAAnalysisWorker(self.image_files, settings, self.current_folder)
        self.worker.progress.connect(self.on_analysis_progress)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.log_msg.connect(lambda msg: self.log_msg.emit(msg))
        self.worker.finished_analysis.connect(self.on_analysis_finished)
        self.worker.start()

        self.log_msg.emit(f"Analyzing {len(self.image_files)} images...")

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_msg.emit("Stopping analysis...")

    def on_analysis_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current} / {total}")

    def on_result_ready(self, filepath, result):
        # Compute face score based on mode
        settings = self._gather_settings()
        face_count = result.get("face_count", 0)
        if settings["face_flag_mode"] == "not_detected":
            result["face_score"] = 1.0 if face_count > 0 else 0.0
        else:
            result["face_score"] = 0.0 if face_count > 0 else 1.0

        # Compute overall score
        result["overall_score"] = self._compute_overall_score(result, settings)
        self.analysis_results[filepath] = result

    def on_analysis_finished(self):
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_bar.setFormat("Done")

        # Check if mask criterion should be auto-disabled
        settings = self._gather_settings()
        if settings["mask_enabled"]:
            any_masks = any(r.get("has_mask", False) for r in self.analysis_results.values())
            if not any_masks:
                self.log_msg.emit("No masks found for any image. Mask criterion auto-disabled.")
                # Recalculate scores without mask
                for fp, r in self.analysis_results.items():
                    r["mask_score"] = 1.0  # neutral
                    r["overall_score"] = self._compute_overall_score(r, settings, force_mask_disabled=True)

        # Build sorted_files list (tree widget handles visual sort order via headers)
        self.sorted_files = [f for f in self.image_files if f in self.analysis_results]
        # Add any unanalyzed files at the end
        analyzed_set = set(self.sorted_files)
        for f in self.image_files:
            if f not in analyzed_set:
                self.sorted_files.append(f)

        self._rebuild_list_widget()
        self.on_threshold_changed(self.slider_threshold.value())
        self.save_cache()

        count = len(self.analysis_results)
        self.log_msg.emit(f"Analysis complete. {count} images scored.")

    def _compute_overall_score(self, result, settings, force_mask_disabled=False):
        weights = {}
        scores = {}

        if settings.get("blur_enabled", True):
            weights["blur"] = self.weight_sliders["blur"].value() / 10.0
            scores["blur"] = result.get("blur_score", 1.0)

        if settings.get("resolution_enabled", True):
            weights["resolution"] = self.weight_sliders["resolution"].value() / 10.0
            scores["resolution"] = result.get("resolution_score", 1.0)

        if settings.get("mask_enabled", True) and not force_mask_disabled:
            weights["mask"] = self.weight_sliders["mask"].value() / 10.0
            scores["mask"] = result.get("mask_score", 1.0)

        if settings.get("face_enabled", True):
            weights["face"] = self.weight_sliders["face"].value() / 10.0
            scores["face"] = result.get("face_score", 1.0)

        if settings.get("eyes_enabled", True):
            weights["eyes"] = self.weight_sliders["eyes"].value() / 10.0
            scores["eyes"] = result.get("eyes_score", 1.0)

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 1.0

        weighted_sum = sum(weights[k] * scores[k] for k in weights)
        return round(weighted_sum / total_weight, 4)

    # -----------------------------------------------------------------------
    # Threshold
    # -----------------------------------------------------------------------
    def _batch_criterion_key(self):
        idx = self.combo_batch_criterion.currentIndex()
        if 0 <= idx < len(self.BATCH_CRITERIA):
            return self.BATCH_CRITERIA[idx][1]
        return "overall_score"

    def on_threshold_changed(self, value):
        threshold = value / 100.0
        self.lbl_threshold.setText(f"{threshold:.2f}")
        key = self._batch_criterion_key()
        label = self.combo_batch_criterion.currentText()
        below = sum(1 for r in self.analysis_results.values() if r.get(key, 1.0) < threshold)
        total = len(self.analysis_results)
        self.lbl_below_count.setText(f"{below} of {total} images below {label} threshold")

    def on_append_preset_changed(self, index):
        if index > 0:
            self.txt_append.setText(self.combo_append_preset.itemText(index))

    # -----------------------------------------------------------------------
    # Per-Image Actions
    # -----------------------------------------------------------------------
    def append_to_current_caption(self):
        f_path = self._filepath_from_item(self.tree_widget.currentItem())
        if not f_path:
            return
        self._append_to_caption(f_path)
        self.load_caption_preview(f_path)

    def _append_to_caption(self, img_path):
        append_text = self.txt_append.text()
        if not append_text:
            return
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        existing = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except Exception:
                pass
        # Don't append if it's already there
        if append_text in existing:
            self.log_msg.emit(f"Text already present in {os.path.basename(txt_path)}, skipping.")
            return
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(existing + append_text)
        self.log_msg.emit(f"Appended to {os.path.basename(txt_path)}")

    def move_current_to_unused(self):
        item = self.tree_widget.currentItem()
        f_path = self._filepath_from_item(item)
        if not f_path:
            return
        visual_idx = self._current_visual_index()
        self._move_to_unused(f_path)
        # Remove from data lists
        if f_path in self.sorted_files:
            self.sorted_files.remove(f_path)
        if f_path in self.image_files:
            self.image_files.remove(f_path)
        if f_path in self.analysis_results:
            del self.analysis_results[f_path]
        # Remove just the single tree item instead of rebuilding everything
        self.tree_widget.blockSignals(True)
        self.tree_widget.takeTopLevelItem(visual_idx)
        count = self.tree_widget.topLevelItemCount()
        self.slider.setRange(0, max(0, count - 1))
        # Advance to next image (or last if we were at the end) for triage flow
        if count > 0:
            new_idx = min(visual_idx, count - 1)
            new_item = self.tree_widget.topLevelItem(new_idx)
            self.tree_widget.setCurrentItem(new_item)
            self.tree_widget.blockSignals(False)
            # Explicitly refresh since setCurrentItem may not fire the signal
            self._on_tree_item_changed(new_item, None)
        else:
            self.tree_widget.blockSignals(False)
            self.lbl_image.set_image(None)
            self.txt_caption.setPlainText("")
            self.lbl_info.setText("0 / 0")

    def _move_to_unused(self, img_path):
        """Move image + caption + mask to unused folder."""
        folder = os.path.dirname(img_path)
        unused_dir = os.path.join(folder, "unused")
        os.makedirs(unused_dir, exist_ok=True)

        filename = os.path.basename(img_path)
        base_path = os.path.splitext(img_path)[0]
        base_name = os.path.splitext(filename)[0]

        files_to_move = [img_path]
        txt_path = base_path + ".txt"
        if os.path.exists(txt_path):
            files_to_move.append(txt_path)
        mask_path = base_path + "-masklabel.png"
        if os.path.exists(mask_path):
            files_to_move.append(mask_path)
        mask_sub = os.path.join(folder, "masks", f"{base_name}.png")
        if os.path.exists(mask_sub):
            files_to_move.append(mask_sub)

        for src in files_to_move:
            rel = os.path.relpath(src, folder)
            dst = os.path.join(unused_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                base, ex = os.path.splitext(dst)
                dst = f"{base}_dup_{np.random.randint(1000)}{ex}"
            try:
                shutil.move(src, dst)
            except Exception as e:
                self.log_msg.emit(f"Error moving {os.path.basename(src)}: {e}")

        self.log_msg.emit(f"Moved {filename} to unused/")

    # -----------------------------------------------------------------------
    # Batch Actions
    # -----------------------------------------------------------------------
    def apply_batch_action(self, action):
        threshold = self.slider_threshold.value() / 100.0
        key = self._batch_criterion_key()
        criterion_label = self.combo_batch_criterion.currentText()
        targets = [fp for fp, r in self.analysis_results.items() if r.get(key, 1.0) < threshold]

        if not targets:
            QMessageBox.information(self, "No Images", f"No images fall below the {criterion_label} threshold.")
            return

        action_label = "append quality tags to captions of" if action == "append" else "move to unused"
        reply = QMessageBox.question(
            self, "Confirm Batch Action",
            f"This will {action_label} {len(targets)} image(s) with {criterion_label} score below {threshold:.2f}.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        count = 0
        if action == "append":
            for fp in targets:
                self._append_to_caption(fp)
                count += 1
            self.log_msg.emit(f"Batch: appended text to {count} captions.")
        elif action == "move":
            for fp in list(targets):
                self._move_to_unused(fp)
                if fp in self.image_files:
                    self.image_files.remove(fp)
                if fp in self.analysis_results:
                    del self.analysis_results[fp]
                if fp in self.sorted_files:
                    self.sorted_files.remove(fp)
                count += 1
            self._rebuild_list_widget()
            self.log_msg.emit(f"Batch: moved {count} images to unused/.")

        self.on_threshold_changed(self.slider_threshold.value())

        # Refresh current display
        count = self.tree_widget.topLevelItemCount()
        if count > 0:
            new_idx = min(self.current_index, count - 1)
            self.tree_widget.setCurrentItem(self.tree_widget.topLevelItem(max(0, new_idx)))
        else:
            self.lbl_image.set_image(None)
            self.txt_caption.setPlainText("")
            self.lbl_info.setText("0 / 0")

    # -----------------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------------
    def _cache_path(self):
        if self.current_folder:
            return os.path.join(self.current_folder, "_qa_cache.json")
        return None

    def save_cache(self):
        path = self._cache_path()
        if not path:
            return
        cache = {}
        for fp, result in self.analysis_results.items():
            try:
                mtime = os.path.getmtime(fp)
            except OSError:
                mtime = 0
            cache[os.path.basename(fp)] = {
                "mtime": mtime,
                **{k: v for k, v in result.items() if k != "filepath"}
            }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            self.log_msg.emit(f"Failed to save QA cache: {e}")

    def load_cache(self):
        path = self._cache_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            return

        loaded = 0
        for fp in self.image_files:
            basename = os.path.basename(fp)
            if basename in cache:
                entry = cache[basename]
                try:
                    current_mtime = os.path.getmtime(fp)
                except OSError:
                    continue
                if abs(entry.get("mtime", 0) - current_mtime) < 1.0:
                    result = dict(entry)
                    result.pop("mtime", None)
                    result["filepath"] = fp
                    self.analysis_results[fp] = result
                    loaded += 1

        if loaded > 0:
            # Rebuild (tree widget handles visual sort via headers)
            self.sorted_files = [f for f in self.image_files if f in self.analysis_results]
            analyzed_set = set(self.sorted_files)
            for f in self.image_files:
                if f not in analyzed_set:
                    self.sorted_files.append(f)
            self._rebuild_list_widget()
            self.on_threshold_changed(self.slider_threshold.value())
            self.log_msg.emit(f"Loaded cached QA results for {loaded} images. Click Analyze to re-scan.")

    # -----------------------------------------------------------------------
    # Settings Persistence
    # -----------------------------------------------------------------------
    def get_settings(self):
        current_file = None
        f_path = self._filepath_from_item(self.tree_widget.currentItem())
        if f_path:
            current_file = os.path.basename(f_path)
        return {
            "blur_enabled": self.chk_blur.isChecked(),
            "blur_threshold_low": self.spin_blur_low.value(),
            "blur_threshold_high": self.spin_blur_high.value(),
            "resolution_enabled": self.chk_resolution.isChecked(),
            "resolution_threshold": self.spin_resolution.value(),
            "mask_enabled": self.chk_mask.isChecked(),
            "face_enabled": self.chk_face.isChecked(),
            "face_flag_mode": "not_detected" if self.radio_face_missing.isChecked() else "detected",
            "eyes_enabled": self.chk_eyes.isChecked(),
            "weights": {k: s.value() for k, s in self.weight_sliders.items()},
            "append_text": self.txt_append.text(),
            "batch_criterion": self.combo_batch_criterion.currentIndex(),
            "batch_threshold": self.slider_threshold.value(),
            "main_splitter_state": self.main_splitter.saveState().toHex().data().decode(),
            "last_selected_file": current_file,
        }

    def set_settings(self, settings):
        if not settings:
            return
        if "blur_enabled" in settings: self.chk_blur.setChecked(settings["blur_enabled"])
        if "blur_threshold_low" in settings: self.spin_blur_low.setValue(settings["blur_threshold_low"])
        if "blur_threshold_high" in settings: self.spin_blur_high.setValue(settings["blur_threshold_high"])
        if "resolution_enabled" in settings: self.chk_resolution.setChecked(settings["resolution_enabled"])
        if "resolution_threshold" in settings: self.spin_resolution.setValue(settings["resolution_threshold"])
        if "mask_enabled" in settings: self.chk_mask.setChecked(settings["mask_enabled"])
        if "face_enabled" in settings: self.chk_face.setChecked(settings["face_enabled"])
        if "face_flag_mode" in settings:
            if settings["face_flag_mode"] == "detected":
                self.radio_face_present.setChecked(True)
            else:
                self.radio_face_missing.setChecked(True)
        if "eyes_enabled" in settings: self.chk_eyes.setChecked(settings["eyes_enabled"])
        if "weights" in settings:
            for k, v in settings["weights"].items():
                if k in self.weight_sliders:
                    self.weight_sliders[k].setValue(v)
        if "append_text" in settings: self.txt_append.setText(settings["append_text"])
        if "batch_criterion" in settings: self.combo_batch_criterion.setCurrentIndex(settings["batch_criterion"])
        if "batch_threshold" in settings: self.slider_threshold.setValue(settings["batch_threshold"])
        if "main_splitter_state" in settings:
            try:
                self.main_splitter.restoreState(QByteArray.fromHex(settings["main_splitter_state"].encode()))
            except Exception:
                pass
