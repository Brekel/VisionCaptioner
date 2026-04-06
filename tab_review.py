import os
import glob
import re
import cv2
import json
import shutil
import numpy as np
from PIL import Image, ImageOps
from file_utils import find_media_files, get_display_name, ALL_MEDIA_EXTS
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QTextEdit, QSlider, QLabel, QSplitter, QGroupBox, 
                               QLineEdit, QCheckBox, QPushButton, QMessageBox, QFormLayout, QFrame, QSizePolicy, QColorDialog, QButtonGroup, QProgressDialog, QComboBox)
from PySide6.QtCore import Qt, QTimer, Signal, QByteArray, QCoreApplication
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QShortcut, QKeySequence
from gui_widgets import ResizableImageLabel

class RefreshableComboBox(QComboBox):
    def __init__(self, parent=None, preset_file="search_replace_presets.json"):
        super().__init__(parent)
        self.preset_file = preset_file
        self.refresh_items()

    def showPopup(self):
        self.refresh_items()
        super().showPopup()

    def refresh_items(self):
        current_text = self.currentText()
        self.blockSignals(True)
        self.clear()
        self.addItem("Select a Preset...", None)
        
        if os.path.exists(self.preset_file):
            try:
                with open(self.preset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, values in data.items():
                        self.addItem(name, values)
            except Exception as e:
                print(f"Error loading presets: {e}")
        
        # Restore selection if it still exists
        index = self.findText(current_text)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            self.setCurrentIndex(0)
            
        self.blockSignals(False)

class ReviewTab(QWidget):
    log_msg = Signal(str)

    SAVE_BUTTON_STYLE_NORMAL = "background-color: #5d99c4; color: white; font-weight: bold;"
    SAVE_BUTTON_STYLE_DIRTY = "background-color: #f1c40f; color: black; font-weight: bold;"
    TOOL_BUTTON_STYLE = """
        QPushButton { background-color: #3b3b3b; border: 1px solid #555; border-radius: 4px; }
        QPushButton:checked { background-color: #5d99c4; border: 1px solid #fff; }
        QPushButton:hover:!checked { background-color: #555; }
    """

    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.recursive = False
        self.image_files = []
        self.current_index = -1
        self.last_selected_file = None
        
        self.cv_img_original = None
        self.cv_mask = None
        self.mask_is_dirty = False
        self.last_paint_pos = None
        self.mask_overlay_color = QColor(0, 0, 50)
        self.undo_snapshot = {}
        self.mask_undo_stack = []
        self.mask_redo_stack = []
        self.cv_dimmed_cache = None 
        self.is_alpha_mask_mode = False # Track if we are editing an embedded alpha mask
        
        # Undo Delete Stack: Stores dicts of {original_index, items: [(src, dst), ...]}
        self.deleted_files_stack = []

        # Persistent state for "Show Mask" toggle (to survive video switching)
        self.show_mask_state = True 

        self.video_cap = None
        self.total_frames = 0
        
        self.save_timer = QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(1000)
        self.save_timer.timeout.connect(self.save_current_caption)

        self.setup_ui()
        self.setup_shortcuts()
        self.on_tool_changed()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter
        
        # --- LEFT SIDE (Stats & List) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Folder Stats
        stats_group = QGroupBox("Folder Stats")
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl_stat_img = QLabel("Images: 0")
        self.lbl_stat_img.setStyleSheet("color: #a3be8c; font-weight: bold;")
        
        self.lbl_stat_txt = QLabel("Captions: 0")
        self.lbl_stat_txt.setStyleSheet("color: #a3be8c; font-weight: bold;")
        
        self.lbl_stat_mask = QLabel("Masks: 0")
        self.lbl_stat_mask.setStyleSheet("color: #666666; font-weight: bold;")
        
        stats_layout.addWidget(self.lbl_stat_img)
        stats_layout.addWidget(self.lbl_stat_txt)
        stats_layout.addWidget(self.lbl_stat_mask)
        left_layout.addWidget(stats_group)
        
        # File List
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_row_changed)
        left_layout.addWidget(self.list_widget)
        
        # Delete Button
        self.btn_delete = QPushButton("🗑️ Move to 'unused' Folder [DEL]")
        self.btn_delete.setToolTip("Move current image, text, and mask to /unused subfolder (Del)")
        self.btn_delete.clicked.connect(self.delete_current_image)
        left_layout.addWidget(self.btn_delete)
        
        # Undo Delete Button
        self.btn_undo_delete = QPushButton("↩️ Undo Delete")
        self.btn_undo_delete.setToolTip("Restore the last deleted image.")
        self.btn_undo_delete.clicked.connect(self.undo_delete)
        self.btn_undo_delete.setEnabled(False) 
        left_layout.addWidget(self.btn_undo_delete)
        
        # Batch Crop Button
        self.btn_crop_all = QPushButton("✂️ Crop All Images to Masks")
        self.btn_crop_all.setToolTip("Crop ALL images in the list to their corresponding masks.")
        self.btn_crop_all.clicked.connect(self.crop_all_masks)
        left_layout.addWidget(self.btn_crop_all)
        
        # Batch Find & Replace
        fr_group = QGroupBox("Batch Find && Replace")
        fr_layout = QVBoxLayout(fr_group)
        input_layout = QFormLayout()
        
        self.txt_find = QLineEdit()
        self.txt_find.setPlaceholderText("Find...")
        self.txt_find.setToolTip("The text or pattern to search for in all caption files.")
        
        self.txt_replace = QLineEdit()
        self.txt_replace.setPlaceholderText("Replace with...")
        self.txt_replace.setToolTip("The text to replace the found matches with.")
        
        # Presets ComboBox
        self.combo_presets = RefreshableComboBox(preset_file="search_replace_presets.json")
        self.combo_presets.currentIndexChanged.connect(self.apply_preset)
        self.combo_presets.setToolTip("Select a preset to automatically fill the Find and Replace fields.")
        input_layout.addRow("Presets:", self.combo_presets)

        input_layout.addRow("Find:", self.txt_find)
        input_layout.addRow("Replace:", self.txt_replace)
        
        fr_layout.addLayout(input_layout)
        
        self.chk_case = QCheckBox("Match Case")
        self.chk_whole = QCheckBox("Match Whole Word Only")
        self.chk_whole.setChecked(True)
        
        opts_layout = QHBoxLayout()
        opts_layout.addWidget(self.chk_case)
        opts_layout.addWidget(self.chk_whole)
        fr_layout.addLayout(opts_layout)
        
        self.btn_apply = QPushButton("Replace All")
        self.btn_apply.clicked.connect(self.apply_replace)
        self.btn_apply.setStyleSheet("background-color: #d73a49; color: white; font-weight: bold;")
        
        self.btn_undo = QPushButton("Undo Last Replace")
        self.btn_undo.clicked.connect(self.undo_last_replace)
        self.btn_undo.setEnabled(False)
        
        fr_layout.addWidget(self.btn_apply)
        fr_layout.addWidget(self.btn_undo)
        left_layout.addWidget(fr_group)
        
        self.main_splitter.addWidget(left_widget)
        
        # --- RIGHT SIDE (Image & Tools) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        
        self.chk_show_mask = QCheckBox("Show Mask Overlay")
        self.chk_show_mask.setChecked(True)
        self.chk_show_mask.clicked.connect(self.on_show_mask_toggled)
        self.chk_show_mask.toggled.connect(self.update_image_display)
        self.chk_show_mask.setFixedWidth(190)
        
        self.btn_color_picker = QPushButton()
        self.btn_color_picker.setToolTip("Set overlay background color")
        self.btn_color_picker.setFixedSize(28, 28)
        self.btn_color_picker.clicked.connect(self.pick_overlay_color)
        self._update_color_button_style()
        
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(80)
        self.slider_opacity.setFixedWidth(120)
        self.slider_opacity.valueChanged.connect(self.on_opacity_changed)
        self.slider_opacity.setToolTip("Adjust background dimming intensity.")
        
        self.lbl_opacity_val = QLabel("80%")
        self.lbl_opacity_val.setFixedWidth(35)
        self.slider_opacity.valueChanged.connect(lambda v: self.lbl_opacity_val.setText(f"{v}%"))
        
        toolbar.addWidget(self.chk_show_mask)
        toolbar.addSpacing(5)
        toolbar.addWidget(self.btn_color_picker)
        toolbar.addWidget(self.slider_opacity)
        toolbar.addWidget(self.lbl_opacity_val)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        toolbar.addWidget(separator)
        
        emoji_font = QFont("Segoe UI Emoji", 12)
        
        # Tool Selection
        self.btn_tool_brush = QPushButton("🖌️")
        self.btn_tool_brush.setFont(emoji_font)
        self.btn_tool_brush.setCheckable(True)
        self.btn_tool_brush.setChecked(True)
        self.btn_tool_brush.setToolTip("Brush Tool (B)")
        self.btn_tool_brush.setFixedSize(32, 30)
        self.btn_tool_brush.setStyleSheet(self.TOOL_BUTTON_STYLE)
        
        self.btn_tool_bucket = QPushButton("🪣")
        self.btn_tool_bucket.setFont(emoji_font)
        self.btn_tool_bucket.setCheckable(True)
        self.btn_tool_bucket.setToolTip("Bucket Fill Tool (F)\nFills connected mask area.")
        self.btn_tool_bucket.setFixedSize(32, 30)
        self.btn_tool_bucket.setStyleSheet(self.TOOL_BUTTON_STYLE)
        
        # Group to ensure only one is active at a time
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_tool_brush)
        self.tool_group.addButton(self.btn_tool_bucket)
        
        # signal connection to a new dedicated handler
        self.tool_group.buttonClicked.connect(self.on_tool_changed) 

        toolbar.addWidget(self.btn_tool_bucket)
        toolbar.addWidget(self.btn_tool_brush)
        toolbar.addSpacing(10)
        
        # Brush Controls
        self.lbl_brush_size = QLabel("Brush:")
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(1, 30)
        self.slider_brush.setValue(10)
        self.slider_brush.setFixedWidth(100)
        self.slider_brush.setToolTip("Adjust brush size. Shortcuts: [ and ]")
        self.slider_brush.valueChanged.connect(self.on_brush_size_changed)
        
        self.lbl_brush_val = QLabel("10%")
        self.lbl_brush_val.setFixedWidth(35)
        
        toolbar.addWidget(self.lbl_brush_size)
        toolbar.addWidget(self.slider_brush)
        toolbar.addWidget(self.lbl_brush_val)
        
        # Mask Action Buttons
        self.btn_fill_black = QPushButton("⬛ Fill")
        self.btn_fill_black.setToolTip("Fill mask with black (background)")
        self.btn_fill_black.setFixedSize(60, 30)
        self.btn_fill_black.clicked.connect(lambda: self.fill_mask(0))
        
        self.btn_fill_white = QPushButton("⬜ Fill")
        self.btn_fill_white.setToolTip("Fill mask with white (foreground)")
        self.btn_fill_white.setFixedSize(60, 30)
        self.btn_fill_white.clicked.connect(lambda: self.fill_mask(255))
        
        self.btn_save_mask = QPushButton("💾 Save")
        self.btn_save_mask.setToolTip("Save current mask to disk (Ctrl+S)")
        self.btn_save_mask.setFixedSize(60, 30)
        self.btn_save_mask.clicked.connect(self.save_current_mask)
        
        self.btn_discard_mask = QPushButton("↩️ Discard")
        self.btn_discard_mask.setToolTip("Discard unsaved changes to mask (Reverts to last saved state)")
        self.btn_discard_mask.setFixedSize(90, 30)
        self.btn_discard_mask.clicked.connect(self.discard_mask_changes)

        self.btn_crop_mask = QPushButton("✂️ Crop")
        self.btn_crop_mask.setToolTip("Crop image and mask to the mask's bounding box (C).\nThis moves the original files to an 'uncropped' subfolder.")
        self.btn_crop_mask.setFixedSize(70, 30)
        self.btn_crop_mask.clicked.connect(self.crop_to_mask)
        
        # Undo
        self.btn_mask_undo = QPushButton("↩️ Undo")
        self.btn_mask_undo.setToolTip("Undo mask change (Ctrl+Z)")
        self.btn_mask_undo.setFixedSize(60, 30)
        self.btn_mask_undo.clicked.connect(self.undo_mask_action)
        self.btn_mask_undo.setEnabled(False)

           # Redo
        self.btn_mask_redo = QPushButton("↪️ Redo")
        self.btn_mask_redo.setToolTip("Redo mask change (Ctrl+Y)")
        self.btn_mask_redo.setFixedSize(60, 30)
        self.btn_mask_redo.clicked.connect(self.redo_mask_action)
        self.btn_mask_redo.setEnabled(False)
        
        # Mask Expansion / Contraction
        self.btn_mask_contract = QPushButton("➖ Contract")
        self.btn_mask_contract.setToolTip("Contract Mask (Minus Key)")
        self.btn_mask_contract.setFixedSize(80, 30)
        self.btn_mask_contract.clicked.connect(lambda: self.modify_mask(-1))
        
        self.btn_mask_expand = QPushButton("➕ Expand")
        self.btn_mask_expand.setToolTip("Expand Mask (Plus Key)")
        self.btn_mask_expand.setFixedSize(80, 30)
        self.btn_mask_expand.clicked.connect(lambda: self.modify_mask(1))

        # Add buttons to toolbar
        toolbar.addWidget(self.btn_fill_black)
        toolbar.addWidget(self.btn_fill_white)
        toolbar.addWidget(self.btn_mask_contract)
        toolbar.addWidget(self.btn_mask_expand)
        toolbar.addWidget(self.btn_crop_mask)
        toolbar.addWidget(self.btn_save_mask)
        toolbar.addWidget(self.btn_discard_mask)
        toolbar.addWidget(self.btn_mask_undo)
        toolbar.addWidget(self.btn_mask_redo)
        
        # Spacer
        toolbar.addStretch()
        
        # Confirm Toggle
        self.chk_confirm_actions = QCheckBox("Confirm Actions")
        self.chk_confirm_actions.setChecked(True)
        self.chk_confirm_actions.setToolTip("Uncheck to disable confirmation popups for Fill, Crop, and Discard operations.")
        toolbar.addWidget(self.chk_confirm_actions)
        
        right_layout.addLayout(toolbar)
        
        # Mouse Help Label
        self.lbl_mouse_help = QLabel("Left Mouse Button: Draw / Fill (mask foreground)       |       Right Mouse Button: Erase / Fill (mask background)       |       Middle Mouse Button: Switch Tool       |       Scroll Wheel: scroll through image list")
        self.lbl_mouse_help.setAlignment(Qt.AlignCenter)
        self.lbl_mouse_help.setStyleSheet("color: #888; font-size: 11px; margin-bottom: 2px;")
        right_layout.addWidget(self.lbl_mouse_help)
        
        # Image Display
        self.lbl_image = ResizableImageLabel()
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_image.setFocusPolicy(Qt.StrongFocus)
        self.lbl_image.wheelEvent = self.image_wheel_event
        self.lbl_image.paint_start.connect(self.on_paint_start)
        self.lbl_image.paint_move.connect(self.on_paint_move)
        self.lbl_image.paint_end.connect(self.on_paint_end)
        self.lbl_image.view_resized.connect(self._update_brush_cursor_size)
        self.lbl_image.middle_click.connect(self.toggle_tool)
        
        # Video Controls
        self.video_controls_layout = QHBoxLayout()
        
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setStyleSheet("color: #888;")
        
        self.slider_video = QSlider(Qt.Horizontal)
        self.slider_video.setToolTip("Scrub video frames")
        self.slider_video.valueChanged.connect(self.seek_video)
        self.slider_video.setEnabled(False)
        
        self.video_controls_layout.addWidget(QLabel("🎞️"))
        self.video_controls_layout.addWidget(self.slider_video)
        self.video_controls_layout.addWidget(self.lbl_frame_info)
        
        # Container widget to easily hide/show
        self.wid_video_controls = QWidget()
        self.wid_video_controls.setLayout(self.video_controls_layout)
        self.wid_video_controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.wid_video_controls.setVisible(False) # Hide by default
        
        # Caption Edit
        self.txt_caption = QTextEdit()
        self.txt_caption.setPlaceholderText("Select an image to edit caption...")
        self.txt_caption.textChanged.connect(self.on_text_changed)
        self.txt_caption.setMaximumHeight(16777215)
        
        # Info & Slider
        self.lbl_info = QLabel("0 / 0")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)
        
        # Vertical Splitter
        self.vertical_splitter = QSplitter(Qt.Vertical)
        self.vertical_splitter.setHandleWidth(2)
        
        # Top Pane: Image + Video
        top_pane = QWidget()
        top_layout = QVBoxLayout(top_pane)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.addWidget(self.lbl_image, stretch=1)
        top_layout.addWidget(self.wid_video_controls)
        
        self.vertical_splitter.addWidget(top_pane)
        self.vertical_splitter.addWidget(self.txt_caption)
        self.vertical_splitter.setSizes([600, 150]) 
        self.vertical_splitter.setCollapsible(0, False)
        self.vertical_splitter.setCollapsible(1, False)

        right_layout.addWidget(self.vertical_splitter, stretch=1)
        right_layout.addWidget(self.lbl_info)
        right_layout.addWidget(self.slider)

        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.main_splitter)
        
        self._update_mask_button_states()

    def set_focus_to_image(self):
        """Public method to allow the main window to set focus here."""
        self.lbl_image.setFocus()
    
    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current_mask)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_mask_action)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo_mask_action)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo_mask_action)
        QShortcut(QKeySequence("Del"), self, self.delete_current_image)
        QShortcut(QKeySequence("["), self, lambda: self.slider_brush.setValue(self.slider_brush.value() - 1))
        QShortcut(QKeySequence("]"), self, lambda: self.slider_brush.setValue(self.slider_brush.value() + 1))
        QShortcut(QKeySequence("Left"), self, lambda: self.navigate(-1))
        QShortcut(QKeySequence("Right"), self, lambda: self.navigate(1))
        QShortcut(QKeySequence("B"), self, self.btn_tool_brush.click)
        QShortcut(QKeySequence("F"), self, self.btn_tool_bucket.click)
        QShortcut(QKeySequence("C"), self, self.crop_to_mask)
        QShortcut(QKeySequence("+"), self, lambda: self.modify_mask(1))
        QShortcut(QKeySequence("="), self, lambda: self.modify_mask(1)) # Support = as +
        QShortcut(QKeySequence("-"), self, lambda: self.modify_mask(-1))

    def get_settings(self):
        # Get current filename safely
        current_file = None
        if 0 <= self.current_index < len(self.image_files):
            current_file = get_display_name(self.image_files[self.current_index], self.current_folder, self.recursive)
        
        return {
            "find_text": self.txt_find.text(), 
            "replace_text": self.txt_replace.text(), 
            "match_case": self.chk_case.isChecked(), 
            "match_whole": self.chk_whole.isChecked(), 
            "show_mask": self.show_mask_state, 
            "confirm_actions": self.chk_confirm_actions.isChecked(),
            "mask_opacity": self.slider_opacity.value(), 
            "brush_size": self.slider_brush.value(), 
            "mask_overlay_color": self.mask_overlay_color.name(), 
            "main_splitter_state": self.main_splitter.saveState().toHex().data().decode(),
            "vertical_splitter_state": self.vertical_splitter.saveState().toHex().data().decode(),
            "last_selected_file": current_file
        }

    def set_settings(self, settings):
        if not settings: return
        if "find_text" in settings: self.txt_find.setText(settings["find_text"])
        if "replace_text" in settings: self.txt_replace.setText(settings["replace_text"])
        if "match_case" in settings: self.chk_case.setChecked(settings["match_case"])
        if "match_whole" in settings: self.chk_whole.setChecked(settings["match_whole"])
        if "show_mask" in settings:
            self.show_mask_state = settings["show_mask"]
            self.chk_show_mask.setChecked(self.show_mask_state)
        if "confirm_actions" in settings: self.chk_confirm_actions.setChecked(settings["confirm_actions"])
        if "mask_opacity" in settings: self.slider_opacity.setValue(settings["mask_opacity"])
        if "brush_size" in settings: self.slider_brush.setValue(settings["brush_size"])
        if "mask_overlay_color" in settings:
            self.mask_overlay_color = QColor(settings["mask_overlay_color"])
            self._update_color_button_style()
        if "last_selected_file" in settings:
            self.last_selected_file = settings["last_selected_file"]
        if "main_splitter_state" in settings:
            try: self.main_splitter.restoreState(QByteArray.fromHex(settings["main_splitter_state"].encode()))
            except: pass
        elif "splitter_state" in settings:
            # Migration for old settings
            try: self.main_splitter.restoreState(QByteArray.fromHex(settings["splitter_state"].encode()))
            except: pass
            
        if "vertical_splitter_state" in settings:
            try: self.vertical_splitter.restoreState(QByteArray.fromHex(settings["vertical_splitter_state"].encode()))
            except: pass

    def _refresh_dimmed_cache(self):
        """
        Performance Optimization:
        Pre-calculates the 'dimmed' background (Image + Color Overlay) so we don't 
        have to run cv2.addWeighted() on every single mouse movement.
        """
        if self.cv_img_original is None:
            self.cv_dimmed_cache = None
            return

        opacity = self.slider_opacity.value() / 100.0
        brightness = 1.0 - opacity
        r, g, b, _ = self.mask_overlay_color.getRgb()

        # Create the solid color block (Allocating this 60fps is slow, so we do it here)
        color_overlay = np.full(self.cv_img_original.shape, (r, g, b), dtype=np.uint8)
        
        # Perform the heavy blending operation once
        self.cv_dimmed_cache = cv2.addWeighted(
            self.cv_img_original, brightness, 
            color_overlay, opacity, 
            0
        )

    def _update_mask_button_states(self):
        has_image = self.cv_img_original is not None
        has_mask = self.cv_mask is not None
        is_video = self.video_cap is not None  # Check if video
        
        # Disable mask-specific items if it is a video
        self.chk_show_mask.setEnabled(has_mask and not is_video)
        self.slider_opacity.setEnabled(has_mask and not is_video)
        self.btn_color_picker.setEnabled(has_mask and not is_video)
        
        self.btn_fill_black.setEnabled(has_image and not is_video)
        self.btn_fill_white.setEnabled(has_image and not is_video)
        self.btn_mask_contract.setEnabled(has_mask and not is_video)
        self.btn_mask_expand.setEnabled(has_mask and not is_video)
        self.slider_brush.setEnabled(has_mask and not is_video)
        
        # Crop logic
        can_crop = has_mask and not is_video
        self.btn_crop_mask.setEnabled(can_crop)
        
        self.btn_save_mask.setEnabled(self.mask_is_dirty and not is_video)
        self.btn_discard_mask.setEnabled(self.mask_is_dirty and not is_video)
        
        # Styles
        if self.mask_is_dirty:
            self.btn_save_mask.setStyleSheet(self.SAVE_BUTTON_STYLE_DIRTY)
            if has_mask: self.chk_show_mask.setText("Show Mask (* Unsaved)")
        else:
            self.btn_save_mask.setStyleSheet(self.SAVE_BUTTON_STYLE_NORMAL)
            if has_mask and not is_video: 
                self.chk_show_mask.setText("Show Mask (Saved)")
            elif is_video:
                self.chk_show_mask.setText("Masks Disabled (Video)")

    def _draw_on_mask(self, x, y, button):
        if self.cv_mask is None or self.cv_img_original is None: return
        
        brush_percentage = self.slider_brush.value() / 100.0
        h, w = self.cv_img_original.shape[:2]
        max_dimension = max(h, w)
        brush_size_px = max(1, int(max_dimension * brush_percentage))
        
        color = 255 if button == Qt.LeftButton else 0
        
        if self.last_paint_pos:
            cv2.line(self.cv_mask, self.last_paint_pos, (x, y), color, brush_size_px, cv2.FILLED)
        else:
            cv2.circle(self.cv_mask, (x, y), brush_size_px // 2, color, -1, cv2.FILLED)
            
        self.last_paint_pos = (x, y)
        if not self.mask_is_dirty:
            self.mask_is_dirty = True
            self._update_mask_button_states()
        self.update_image_display()

    def toggle_tool(self):
        """Swaps the active tool (Brush <-> Bucket) and triggers UI updates."""
        if self.btn_tool_brush.isChecked():
            self.btn_tool_bucket.click()
        else:
            self.btn_tool_brush.click()

    def on_tool_changed(self, button=None):
        """
        Handles switching between tools.
        Updates the cursor and the brush overlay visibility.
        """
        if self.btn_tool_bucket.isChecked():
            # Bucket Mode: 
            # 1. Set cursor to a specific icon (Pointing Hand or Arrow)
            self.lbl_image.setCursor(Qt.PointingHandCursor)
            # 2. Hide the brush circle overlay
            self.lbl_image.set_brush_outline_size(0)
        else:
            # Brush Mode:
            # 1. Hide system cursor (so we only see the custom circle)
            self.lbl_image.setCursor(Qt.BlankCursor)
            # 2. Update/Show the brush circle overlay
            self._update_brush_cursor_size()

    def on_brush_size_changed(self, value):
        self.lbl_brush_val.setText(f"{value}%")
        self._update_brush_cursor_size()

    def _update_brush_cursor_size(self, _=None):
        # Safety check: If bucket is selected, force brush size to 0 and exit.
        if self.btn_tool_bucket.isChecked():
            self.lbl_image.set_brush_outline_size(0)
            return

        if self.cv_img_original is None or self.lbl_image.pixmap() is None or self.lbl_image.pixmap().isNull():
            self.lbl_image.set_brush_outline_size(0)
            # Restore standard cursor if no image is loaded
            self.lbl_image.setCursor(Qt.ArrowCursor) 
            return
            
        # Ensure we are using the Blank cursor when brush is active 
        # (in case it was switched elsewhere)
        if self.lbl_image.cursor().shape() != Qt.BlankCursor:
            self.lbl_image.setCursor(Qt.BlankCursor)

        brush_percentage = self.slider_brush.value() / 100.0
        h, w = self.cv_img_original.shape[:2]
        max_dimension = max(h, w)
        brush_size_image_px = max(1, int(max_dimension * brush_percentage))
        
        scale_w = self.lbl_image.pixmap().width() / w
        scale_h = self.lbl_image.pixmap().height() / h
        scale = min(scale_w, scale_h)
        
        widget_brush_size = brush_size_image_px * scale
        self.lbl_image.set_brush_outline_size(widget_brush_size)

    def _flood_fill_mask(self, x, y, button):
        if self.cv_img_original is None: return

        # Ensure mask exists
        h, w = self.cv_img_original.shape[:2]
        if self.cv_mask is None:
            self.cv_mask = np.zeros((h, w), dtype=np.uint8)

        # Determine fill color: Left Click = White (255), Right Click = Black (0)
        fill_value = 255 if button == Qt.LeftButton else 0

        # cv2.floodFill modifies the image in-place.
        # It requires a mask slightly larger than the image (h+2, w+2) for processing
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Perform fill
        # loDiff and upDiff are 0 because we are filling a binary mask (0 or 255)
        # flags=4 means 4-connected pixels (up/down/left/right)
        try:
            cv2.floodFill(self.cv_mask, flood_mask, (x, y), fill_value, 0, 0, flags=4)
            
            self.mask_is_dirty = True
            self._update_mask_button_states()
            self.update_image_display()
        except Exception as e:
            # Click might be outside bounds or other CV error
            print(f"Fill error: {e}")

    def on_paint_start(self, x, y, button):
        # Ensure mask exists before painting starts
        if self.cv_img_original is not None:
            h, w = self.cv_img_original.shape[:2]
            if self.cv_mask is None:
                self.cv_mask = np.zeros((h, w), dtype=np.uint8)
                
        # Save state before drawing
        self._push_undo_state()
    
        if self.btn_tool_bucket.isChecked():
            self._flood_fill_mask(x, y, button)
        else:
            self.last_paint_pos = None
            self._draw_on_mask(x, y, button)

    def on_paint_move(self, x, y, button):
        if self.btn_tool_bucket.isChecked():
            return
        self._draw_on_mask(x, y, button)

    def on_paint_end(self, button):
        self.last_paint_pos = None

    def _get_current_mask_path(self):
        if self.current_index < 0 or self.current_index >= len(self.image_files): return None
        f_path = self.image_files[self.current_index]
        base, _ = os.path.splitext(f_path)
        return f"{base}-masklabel.png"

    def save_current_mask(self):
        if self.cv_mask is None or not self.mask_is_dirty: return
        
        try:
            if self.is_alpha_mask_mode:
                # Alpha Mode: Save back to the original image file
                f_path = self.image_files[self.current_index]
                
                # 1. Load original image (to get color channels)
                # We can use self.cv_img_original, but need to be sure it's valid RGB
                pil_img = Image.fromarray(self.cv_img_original)
                
                # 2. Resize mask if needed
                pil_mask = Image.fromarray(self.cv_mask)
                if pil_mask.size != pil_img.size:
                    pil_mask = pil_mask.resize(pil_img.size, Image.NEAREST)
                    
                # 3. Merge
                r, g, b = pil_img.split()
                pil_final = Image.merge("RGBA", (r, g, b, pil_mask))
                
                # 4. Save (Overwrite)
                pil_final.save(f_path, compress_level=1)
                self.log_msg.emit(f"✅ Mask saved to alpha channel: {os.path.basename(f_path)}")
                
            elif getattr(self, 'mask_subfolder_mode', False):
                # Subfolder Mode: masks/filename.png
                f_path = self.image_files[self.current_index]
                folder = os.path.dirname(f_path)
                filename = os.path.basename(f_path)
                base_name = os.path.splitext(filename)[0]
                
                masks_dir = os.path.join(folder, "masks")
                if not os.path.exists(masks_dir):
                    os.makedirs(masks_dir, exist_ok=True)
                    
                mask_path = os.path.join(masks_dir, f"{base_name}.png")
                cv2.imwrite(mask_path, self.cv_mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                self.log_msg.emit(f"✅ Mask saved to subfolder: {os.path.basename(mask_path)}")
                
            else:
                # Separate File Mode
                mask_path = self._get_current_mask_path()
                if not mask_path: return
                
                cv2.imwrite(mask_path, self.cv_mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                self.log_msg.emit(f"✅ Mask saved: {os.path.basename(mask_path)}")
                
            self.mask_is_dirty = False
            self._update_mask_button_states()
            self.update_stats()
            
        except Exception as e:
            self.log_msg.emit(f"❌ Error saving mask: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save mask.\n\nError: {e}")

    def crop_all_masks(self):
        """Iterates through all files, checks for masks, and crops them."""
        
        # 1. Check for unsaved changes on the current image first
        if not self.check_unsaved_changes():
            return

        # 2. Confirm Action
        msg = (
            "⚠️ <b>Are you sure you want to crop ALL images?</b><br><br>"
            "This will:<br>"
            "1. Scan every image in the folder.<br>"
            "2. If a matching mask exists, crop the image and mask to the content.<br>"
            "3. Move original files to an 'uncropped' subfolder.<br><br>"
            "This process might take a moment."
        )
        reply = QMessageBox.question(self, "Confirm Batch Crop", msg, QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.No:
            return

        # 3. Setup
        uncropped_dir = os.path.join(self.current_folder, "uncropped")
        count_processed = 0
        count_skipped = 0
        errors = []
        
        total_images = len(self.image_files)

        # --- PROGRESS BAR SETUP ---
        progress = QProgressDialog("Preparing to crop...", "Cancel", 0, total_images, self)
        progress.setWindowTitle("Batch Crop Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0) # Force it to show immediately
        
        # Apply stylesheet to enforce consistent dark theme and prevent "inactive" graying
        progress.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
                font-weight: bold;
            }
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
            QPushButton {
                background-color: #d73a49;
                color: white;
                border: 1px solid #ab2636;
                border-radius: 4px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cb2431;
            }
        """)

        progress.setValue(0)
        # --------------------------

        # Change cursor to wait
        self.setCursor(Qt.WaitCursor)
        
        try:
            # Ensure output directory exists if we are going to use it
            if not os.path.exists(uncropped_dir):
                os.makedirs(uncropped_dir, exist_ok=True)

            file_pairs = []
            for f_path in self.image_files:
                base, _ = os.path.splitext(f_path)
                filename = os.path.basename(f_path)
                base_name = os.path.splitext(filename)[0]
                
                mask_path_std = f"{base}-masklabel.png"
                mask_path_sub = os.path.join(os.path.dirname(f_path), "masks", f"{base_name}.png")

                # Check Priority: Subfolder > Separate File > Alpha
                has_sub_mask = os.path.exists(mask_path_sub)
                has_mask_file = False
                has_alpha = False
                
                final_source_type = None
                final_mask_source = None
                
                if has_sub_mask:
                    final_source_type = 'subfolder'
                    final_mask_source = mask_path_sub
                else:
                    has_mask_file = os.path.exists(mask_path_std)
                    if has_mask_file:
                        final_source_type = 'file'
                        final_mask_source = mask_path_std
                    else:
                        # Simple check for alpha candidacy without opening file
                        final_source_type = 'alpha' 

                if final_source_type:
                    file_pairs.append((f_path, final_mask_source, final_source_type))
            
            # -------------------------------
            
            if not file_pairs:
                self.log_msg.emit("⚠️ No masks (or candidates) found to crop.")
                self.setCursor(Qt.ArrowCursor)
                progress.close()
                return

            self.log_msg.emit(f"🚀 Starting batch crop on {len(file_pairs)} images...")

            # --- LAUNCH WORKER ---
            from gui_workers import CropWorker
            self.crop_worker = CropWorker(file_pairs, uncropped_dir)
            
            # Connect Signals
            self.crop_worker.progress.connect(progress.setValue)
            self.crop_worker.log.connect(lambda m: self.log_msg.emit(m))
            
            def on_finished():
                self.setCursor(Qt.ArrowCursor)
                progress.close()
                
                # Refresh file list / current view
                self.refresh_file_list()
                if self.current_index >= 0:
                     self.load_image_and_data(self.current_index) # Reload current
                
                QMessageBox.information(self, "Batch Crop", f"Batch crop finished.\nOriginals saved to 'uncropped' folder.")
                self.crop_worker = None # cleanup

            self.crop_worker.finished.connect(on_finished)
            
            # Handle Progress Cancel
            progress.canceled.connect(self.crop_worker.stop)
            
            # Adjust Progress Range
            progress.setMaximum(len(file_pairs))
            progress.setValue(0)
            
            self.crop_worker.start()
            # ---------------------

        except Exception as e:
            self.setCursor(Qt.ArrowCursor)
            progress.close()
            self.log_msg.emit(f"❌ Batch crop failed: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        
    def crop_to_mask(self):
        if self.current_index < 0 or self.cv_mask is None:
            return

        # If mask has changes, save them first so the backup file 
        # created later contains the most recent edits.
        if self.mask_is_dirty:
            self.save_current_mask()
            # Safety check: If save failed (still dirty), abort crop
            if self.mask_is_dirty: 
                return

        # 1. Get Bounding Box from current mask data
        try:
            pil_mask = Image.fromarray(self.cv_mask)
            bbox = pil_mask.getbbox()
            if not bbox:
                QMessageBox.warning(self, "Crop Error", "The current mask is empty. Cannot determine crop area.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Crop Error", f"Could not process mask for cropping:\n{e}")
            return

        # 2. Confirm with user (If enabled)
        if self.chk_confirm_actions.isChecked():
            reply = QMessageBox.question(self, "Confirm Crop", 
                "This will:\n"
                "1. Crop the current image and its mask.\n"
                "2. Overwrite the original files with the cropped versions.\n"
                "3. Move the original full-size files to an 'uncropped' subfolder.\n\n"
                "This action cannot be undone. Proceed?", 
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # 3. Get paths
        current_image_path = self.image_files[self.current_index]
        current_mask_path = self._get_current_mask_path() # Default separate file path
        
        uncropped_dir = os.path.join(self.current_folder, "uncropped")
        os.makedirs(uncropped_dir, exist_ok=True)
        
        uncropped_image_path = os.path.join(uncropped_dir, os.path.basename(current_image_path))
        
        # Handle Subfolder Mask Mode
        if getattr(self, 'mask_subfolder_mode', False):
            folder = os.path.dirname(current_image_path)
            base_name = os.path.splitext(os.path.basename(current_image_path))[0]
            current_mask_path = os.path.join(folder, "masks", f"{base_name}.png")
            
            uncropped_masks_dir = os.path.join(uncropped_dir, "masks")
            if not os.path.exists(uncropped_masks_dir):
                os.makedirs(uncropped_masks_dir, exist_ok=True)
            uncropped_mask_path = os.path.join(uncropped_masks_dir, f"{base_name}.png")
            
        elif current_mask_path:
            uncropped_mask_path = os.path.join(uncropped_dir, os.path.basename(current_mask_path))
        else:
            uncropped_mask_path = None

        # 4. Perform operations
        try:
            # Load original full-size image from disk
            pil_image = Image.open(current_image_path)
            pil_image = ImageOps.exif_transpose(pil_image)

            # Alpha Channel Mode Logic
            if self.is_alpha_mask_mode:
                pil_image = pil_image.convert("RGBA")
                
                # Crop
                cropped_image = pil_image.crop(bbox)
                
                # Backup Original
                pil_image.save(uncropped_image_path, compress_level=1)
                
                # Save Cropped (Overwrite)
                cropped_image.save(current_image_path, compress_level=1)
                
                self.log_msg.emit(f"✂️ Cropped '{os.path.basename(current_image_path)}' (Alpha) and moved original to 'uncropped/'.")
                
            else:
                # Separate File Mode
                # Crop both PIL objects
                cropped_image = pil_image.crop(bbox)
                cropped_mask = pil_mask.crop(bbox)

                # Save originals to 'uncropped' folder
                pil_image.save(uncropped_image_path, compress_level=1)
                if uncropped_mask_path:
                    pil_mask.save(uncropped_mask_path, compress_level=1)

                # Overwrite current files with cropped versions
                cropped_image.save(current_image_path, compress_level=1)
                if current_mask_path:
                    cropped_mask.save(current_mask_path, compress_level=1)
                
                self.log_msg.emit(f"✂️ Cropped '{os.path.basename(current_image_path)}' and moved original to 'uncropped/'.")
            
            # 5. Reload the view with the new cropped data
            self.load_image_data(current_image_path)
            self.update_image_display()
            self._update_brush_cursor_size()

        except Exception as e:
            error_msg = f"An error occurred during the crop operation: {e}"
            self.log_msg.emit(f"❌ {error_msg}")
            QMessageBox.critical(self, "Crop Failed", error_msg)

    def delete_current_image(self):
        """Moves the current image, mask, and caption to an 'unused' folder."""
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return

        # --- FIX: Stop auto-save timer immediately ---
        if self.save_timer.isActive():
            self.save_timer.stop()
        # ---------------------------------------------

        # 1. Get paths and filename
        img_path = self.image_files[self.current_index]
        filename = os.path.basename(img_path)
        
        base_path, ext = os.path.splitext(img_path)
        txt_path = base_path + ".txt"
        mask_path = base_path + "-masklabel.png"

        # 2. Ask the user for confirmation if needed
        if self.chk_confirm_actions.isChecked():
            reply = QMessageBox.question(
                self, 
                "Confirm Move", 
                f"Are you sure you want to move <b>{filename}</b> to the 'unused' folder?", 
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # 3. Prepare 'unused' folder
        unused_dir = os.path.join(self.current_folder, "unused")
        if not os.path.exists(unused_dir):
            try:
                os.makedirs(unused_dir)
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Could not create 'unused' directory:\n{e}")
                return
        
        # --- FIX: Prevent stale mask save ---
        # Explicitly clear "dirty" flag so on_row_changed doesn't try to save the OLD mask onto the NEW image.
        self.mask_is_dirty = False 
        self.cv_mask = None
        self.cv_img_original = None
        self._update_mask_button_states()

        # 4. Move Files
        files_to_move = [img_path]
        if os.path.exists(txt_path): files_to_move.append(txt_path)
        
        # Only move mask if it exists as a separate file
        if os.path.exists(mask_path): 
            files_to_move.append(mask_path)
        
        # Check for subfolder mask
        folder = os.path.dirname(img_path)
        base_name = os.path.splitext(filename)[0]
        mask_sub_path = os.path.join(folder, "masks", f"{base_name}.png")
        if os.path.exists(mask_sub_path):
            files_to_move.append(mask_sub_path)

        moved_count = 0
        files_moved_records = [] # Store tuple (original_path, new_path)
        try:
            for src in files_to_move:
                # Check if src is in a subfolder (e.g. "masks")
                rel_path = os.path.relpath(src, self.current_folder)
                dst = os.path.join(unused_dir, rel_path)
                
                # Ensure destination directory exists (for masks/ subdir)
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                
                # Handle duplicate filenames in unused folder
                if os.path.exists(dst):
                    base, ex = os.path.splitext(dst)
                    # Add a random number to avoid overwriting existing files in 'unused'
                    dst = f"{base}_dup_{np.random.randint(1000)}{ex}"
                
                shutil.move(src, dst)
                moved_count += 1
                
                # Verify move was successful for undo tracking
                if os.path.exists(dst):
                     files_moved_records.append((src, dst))
            
            # Push to stack
            if files_moved_records:
                self.deleted_files_stack.append({
                    "index": self.current_index,
                    "files": files_moved_records
                })
                self.btn_undo_delete.setEnabled(True)
                self.btn_undo_delete.setText(f"↩️ Undo Delete ({len(self.deleted_files_stack)})")
            
            self.log_msg.emit(f"🗑️ Moved {filename} to 'unused'.")

        except Exception as e:
            QMessageBox.critical(self, "Move Failed", f"Error moving files:\n{e}")
            return

        # 5. Calculate the next index to select after the refresh
        # (Stay on the same index, unless we deleted the very last item)
        target_index = min(self.current_index, len(self.image_files) - 2)
        target_index = max(0, target_index)

        # 6. Perform a full refresh of the file list to ensure sync with the folder
        self.list_widget.blockSignals(True)
        self.refresh_file_list()
        self.list_widget.blockSignals(False)

        # 7. Select the next image
        if len(self.image_files) > 0:
            self.list_widget.setCurrentRow(target_index)
            # Force UI update for the newly selected image
            if self.list_widget.currentRow() == target_index:
                self.on_row_changed(target_index)
        else:
            # Folder is now empty
            self.current_index = -1
            self.lbl_image.set_image(None)
            self.txt_caption.setPlainText("")
            self.lbl_info.setText("0 / 0")


    def fill_mask(self, color):
        if self.cv_img_original is None: return
        
        # Confirm with user (If enabled)
        if self.cv_mask is not None and self.chk_confirm_actions.isChecked():
            reply = QMessageBox.question(self, "Confirm Overwrite", "This will overwrite the current mask. Are you sure?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No: return
            
        # Create mask if it doesn't exist so we can save 'empty' state to undo
        h, w = self.cv_img_original.shape[:2]
        if self.cv_mask is None:
            self.cv_mask = np.zeros((h, w), dtype=np.uint8)
            
        self._push_undo_state()

        self.cv_mask = np.full((h, w), color, dtype=np.uint8)
        self.mask_is_dirty = True
        self.log_msg.emit(f"Filled mask with {'white' if color == 255 else 'black'}.")
        self._update_mask_button_states()
        self.update_image_display()

    def _force_discard(self):
        if not self.mask_is_dirty: return
        self.log_msg.emit("Discarding mask changes.")
        self.load_image_data(self.image_files[self.current_index])
        self.update_image_display()

    def discard_mask_changes(self):
        if not self.mask_is_dirty: return
        
        # Confirm with user (If enabled)
        if self.chk_confirm_actions.isChecked():
            reply = QMessageBox.question(self, "Confirm Discard", "Are you sure you want to discard all unsaved changes to this mask?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes: self._force_discard()
        else:
            self._force_discard()

    def check_unsaved_changes(self):
        if not self.mask_is_dirty: return True
        
        # If confirmations are disabled, auto-save and proceed
        if not self.chk_confirm_actions.isChecked():
            self.save_current_mask()
            return True
            
        reply = QMessageBox.question(self, "Unsaved Changes", "The current mask has unsaved changes. Do you want to save them?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if reply == QMessageBox.Save:
            self.save_current_mask()
            return True
        elif reply == QMessageBox.Discard:
            return True
        else:
            return False

    def on_opacity_changed(self, value):
        self.lbl_opacity_val.setText(f"{value}%")
        self._refresh_dimmed_cache()
        self.update_image_display()
        
    def pick_overlay_color(self):
        color = QColorDialog.getColor(self.mask_overlay_color, self)
        if color.isValid():
            self.mask_overlay_color = color
            self._update_color_button_style()
            self._refresh_dimmed_cache()
            self.update_image_display()

    def _update_color_button_style(self):
        self.btn_color_picker.setStyleSheet(f"background-color: {self.mask_overlay_color.name()}; border: 1px solid #888;")

    def set_recursive(self, recursive):
        self.recursive = recursive
        self.refresh_file_list()

    def update_folder(self, folder, recursive=None):
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

        self.current_folder = folder
        if recursive is not None:
            self.recursive = recursive
        self.mask_is_dirty = False
        self.undo_snapshot = {}
        self.btn_undo.setEnabled(False)
        self.refresh_file_list()

    def update_stats(self):
        if not self.current_folder: return
        num_images = len(self.image_files)
        num_txts = sum(1 for f in self.image_files if os.path.exists(os.path.splitext(f)[0] + ".txt"))
        try:
            num_masks = sum(1 for f in self.image_files if os.path.exists(os.path.splitext(f)[0] + "-masklabel.png"))
        except:
            num_masks = 0
            
        self.lbl_stat_img.setText(f"Images: {num_images}")
        self.lbl_stat_txt.setText(f"Captions: {num_txts}")
        self.lbl_stat_mask.setText(f"Masks: {num_masks}")
        
        if num_txts == num_images and num_images > 0: self.lbl_stat_txt.setStyleSheet("color: #a3be8c; font-weight: bold;")
        elif num_txts == 0: self.lbl_stat_txt.setStyleSheet("color: #ebcb8b; font-weight: bold;")
        else: self.lbl_stat_txt.setStyleSheet("color: #bf616a; font-weight: bold;")
        
        if num_masks == num_images and num_images > 0: self.lbl_stat_mask.setStyleSheet("color: #a3be8c; font-weight: bold;")
        elif num_masks == 0: self.lbl_stat_mask.setStyleSheet("color: #666666; font-weight: bold;")
        else: self.lbl_stat_mask.setStyleSheet("color: #bf616a; font-weight: bold;")

    def refresh_file_list(self):
        if not self.current_folder:
            self.list_widget.clear()
            return

        # Prioritize internal index state, then UI selection
        current_selection = None
        if 0 <= self.current_index < len(self.image_files):
             current_selection = get_display_name(self.image_files[self.current_index], self.current_folder, self.recursive)
        elif self.list_widget.currentItem():
             current_selection = self.list_widget.currentItem().text()
        self.list_widget.clear()
        self.image_files = []

        self.image_files = find_media_files(self.current_folder, exts=ALL_MEDIA_EXTS,
                                            recursive=self.recursive)

        for f in self.image_files:
            self.list_widget.addItem(get_display_name(f, self.current_folder, self.recursive))

        self.slider.setRange(0, max(0, len(self.image_files) - 1))
        self.update_stats()

        if self.image_files:
            # Determine which file to select, prioritizing the one from settings on startup
            file_to_select = self.last_selected_file or current_selection

            if file_to_select:
                items = self.list_widget.findItems(file_to_select, Qt.MatchExactly)
                if items:
                    self.list_widget.setCurrentItem(items[0])
                    # We've used the setting, clear it so it doesn't get re-used on subsequent refreshes
                    self.last_selected_file = None
                    return

            # If nothing was selected or found, default to the first item
            self.list_widget.setCurrentRow(0)
            self.last_selected_file = None # Also clear if the file wasn't found
        else:
            self.lbl_image.set_image(None)
            self.txt_caption.setPlainText("")
            self.lbl_info.setText("0 / 0")

    def on_row_changed(self, row):
        if not self.check_unsaved_changes():
            self.list_widget.blockSignals(True)
            self.list_widget.setCurrentRow(self.current_index)
            self.list_widget.blockSignals(False)
            return
            
        if self.save_timer.isActive():
            self.save_timer.stop()
            self.save_current_caption(force_index=self.current_index)
            
        self.current_index = row
        if row < 0 or row >= len(self.image_files): return
        
        self.slider.blockSignals(True)
        self.slider.setValue(row)
        self.slider.blockSignals(False)
        
        self.load_image_data(self.image_files[row])
        
        # Update Info Label with Resolution
        display_name = get_display_name(self.image_files[row], self.current_folder, self.recursive)
        res_info = ""
        if self.cv_img_original is not None:
             h, w = self.cv_img_original.shape[:2]
             res_info = f"    |    {w} x {h}"

        self.lbl_info.setText(f"{row + 1} / {len(self.image_files)}    |    {display_name}{res_info}")
        self.update_image_display()
        self.load_caption(row)
        self._update_brush_cursor_size()

    def load_image_data(self, f_path):
        # 1. Cleanup previous video
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        
        self.mask_is_dirty = False
        self.cv_img_original = None
        self.cv_mask = None
        
        # Reset UI visibility/state
        self.wid_video_controls.setVisible(False)
        self.chk_show_mask.setEnabled(True)
        # Restore user preference for images
        self.chk_show_mask.blockSignals(True)
        self.chk_show_mask.setChecked(self.show_mask_state)
        self.chk_show_mask.blockSignals(False)
        
        self.btn_tool_brush.setEnabled(True)
        self.btn_tool_bucket.setEnabled(True)
        
        self.mask_undo_stack.clear()
        self.mask_redo_stack.clear()
        self._update_undo_redo_buttons()
        
        try:
            # Check if file is a video
            _, ext = os.path.splitext(f_path)
            video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
            
            if ext.lower() in video_exts:
                self.video_cap = cv2.VideoCapture(f_path)
                
                if not self.video_cap.isOpened():
                    raise IOError("Could not open video file.")
                
                # Get Frame Count
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Setup Slider
                self.slider_video.blockSignals(True)
                self.slider_video.setRange(0, max(0, self.total_frames - 1))
                self.slider_video.setValue(0)
                self.slider_video.blockSignals(False)
                self.slider_video.setEnabled(True)
                
                self.wid_video_controls.setVisible(True)
                self.lbl_frame_info.setText(f"Frame: 0 / {self.total_frames}")

                # Disable Masking Tools for Video
                self.chk_show_mask.setEnabled(False)
                self.chk_show_mask.blockSignals(True)
                self.chk_show_mask.setChecked(False)
                self.chk_show_mask.blockSignals(False)
                
                self.btn_tool_brush.setEnabled(False)
                self.btn_tool_bucket.setEnabled(False)
                
                # Read First Frame
                ret, frame = self.video_cap.read()
                if ret:
                    self.cv_img_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Standard Image Loading
                pil_img = Image.open(f_path)
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
                self.cv_img_original = np.array(pil_img)
            
            # --- Mask Loading (Only if NOT a video) ---
            if self.video_cap is None:
                base, ext = os.path.splitext(f_path)
                mask_path_std = f"{base}-masklabel.png"
                
                # Check for "masks" subfolder path
                folder = os.path.dirname(f_path)
                filename = os.path.basename(f_path)
                base_name = os.path.splitext(filename)[0]
                mask_path_sub = os.path.join(folder, "masks", f"{base_name}.png")
                
                # Reset modes
                self.is_alpha_mask_mode = False
                self.mask_subfolder_mode = False
                
                final_mask_path = None
                
                # 1. Check for Masks Subfolder (Highest Priority)
                if os.path.exists(mask_path_sub):
                    final_mask_path = mask_path_sub
                    self.mask_subfolder_mode = True
                    self.chk_show_mask.setText(f"Show Mask (Submasks, {base_name}.png)")
                    
                # 2. Check for Separate Mask File (Secondary Priority)
                elif os.path.exists(mask_path_std):
                    final_mask_path = mask_path_std
                    self.chk_show_mask.setText(f"Show Mask (File, {os.path.basename(mask_path_std)})")

                if final_mask_path:
                    # Load the mask from file
                    mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE)
                    h, w = self.cv_img_original.shape[:2]
                    if mask is not None:
                         if mask.shape[:2] != (h, w):
                             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                         self.cv_mask = mask
                
                # 3. Check for Alpha Channel (Lowest Priority, only if no other mask found)
                else:
                    # Re-open with Pillow to check mode/alpha
                    try:
                        with Image.open(f_path) as img_check:
                             if img_check.mode in ('RGBA', 'LA') or (img_check.mode == 'P' and 'transparency' in img_check.info):
                                # Load Alpha Channel
                                img_rgba = img_check.convert("RGBA")
                                alpha_channel = np.array(img_rgba.split()[-1])
                                
                                # Check if alpha is actually used (not all 255)
                                if np.mean(alpha_channel) < 255:
                                     _, mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
                                     self.cv_mask = mask
                                     self.is_alpha_mask_mode = True
                                     self.chk_show_mask.setText("Show Mask (Alpha Channel)")
                                else:
                                     self.chk_show_mask.setText("Show Mask (None)")
                             else:
                                 self.chk_show_mask.setText("Show Mask (None)")
                    except Exception as e:
                        print(f"Error checking alpha: {e}")
                        self.chk_show_mask.setText("Show Mask (None)")
            else:
                self.chk_show_mask.setText("Show Mask (N/A for Video)")
                
            self._refresh_dimmed_cache()
                
        except Exception as e:
            print(f"Error loading image data for {f_path}: {e}")
            self.cv_img_original = None
            self.cv_dimmed_cache = None
            
        self._update_mask_button_states()

    def update_image_display(self):
        if self.cv_img_original is None:
            self.lbl_image.set_image(None)
            return
            
        # Optimization: Use cached background if available, else calc it
        if self.chk_show_mask.isChecked() and self.cv_mask is not None:
            if self.cv_dimmed_cache is None:
                self._refresh_dimmed_cache()
            
            # 1. Start with a copy of the pre-dimmed background
            #    (This acts as the 'masked out' area)
            final_img = self.cv_dimmed_cache.copy()
            
            # 2. Copy original pixels WHERE the mask is white
            #    np.copyto is significantly faster than boolean indexing with assignment
            #    We need [..., None] to broadcast the 2D mask (H,W) to 3D image (H,W,3)
            np.copyto(final_img, self.cv_img_original, where=self.cv_mask[..., None] > 0)
            
        else:
            # Mask hidden or doesn't exist? Just show original
            final_img = self.cv_img_original.copy()
            
        height, width, channel = final_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(final_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.lbl_image.current_pixmap = QPixmap.fromImage(q_img)
        self.lbl_image.update_view()

    def seek_video(self, frame_idx):
        if self.video_cap is None or not self.video_cap.isOpened():
            return
            
        # Set property to seek
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        
        if ret:
            # Update internal image
            self.cv_img_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.lbl_frame_info.setText(f"Frame: {frame_idx} / {self.total_frames}")
            
            # Force display update
            self.update_image_display()

    def load_caption(self, row):
        if row < 0 or row >= len(self.image_files): return
        
        f_path = self.image_files[row]
        txt_path = os.path.splitext(f_path)[0] + ".txt"
        
        self.txt_caption.blockSignals(True)
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    self.txt_caption.setPlainText(f.read().strip())
            except:
                self.txt_caption.setPlainText("")
        else:
            self.txt_caption.setPlainText("")
        self.txt_caption.blockSignals(False)

    def on_slider_changed(self, val):
        if val != self.current_index:
            self.list_widget.setCurrentRow(val)

    def on_text_changed(self):
        self.save_timer.start() 

    def save_current_caption(self, force_index=None):
        idx = force_index if force_index is not None else self.current_index
        if idx < 0 or idx >= len(self.image_files): return
        
        f_path = self.image_files[idx]
        txt_path = os.path.splitext(f_path)[0] + ".txt"
        content = self.txt_caption.toPlainText().strip()
        
        try:
            if not content:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    self.log_msg.emit(f"🗑️ Deleted empty caption: {os.path.basename(txt_path)}")
                    self.update_stats()
            else:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_msg.emit(f"Saved: {os.path.basename(txt_path)}")
                self.update_stats()
        except Exception as e:
            self.log_msg.emit(f"Error saving {txt_path}: {e}")

    def get_regex_pattern(self):
        find_text = self.txt_find.text()
        if not find_text: return None, None
        
        pattern = re.escape(find_text)
        flags = 0
        if self.chk_whole.isChecked():
            pattern = r"\b" + pattern + r"\b"
        if not self.chk_case.isChecked():
            flags = re.IGNORECASE
        return pattern, flags

    def apply_preset(self, index):
        data = self.combo_presets.currentData()
        if data:
            self.txt_find.setText(data.get("find", ""))
            self.txt_replace.setText(data.get("replace", ""))
            self.chk_case.setChecked(data.get("match_case", False))
            self.chk_whole.setChecked(data.get("match_whole", False))

    def apply_replace(self):
        pattern, flags = self.get_regex_pattern()
        replace_text = self.txt_replace.text()
        
        if not pattern:
            return QMessageBox.warning(self, "Input Error", "Please enter text to find.")
            
        if self.save_timer.isActive():
            self.save_timer.stop()
            self.save_current_caption()
            
        files_to_change, original_contents = {}, {}
        for f_path in self.image_files:
            txt_path = os.path.splitext(f_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        old_content = f.read()
                    new_content, count = re.subn(pattern, replace_text, old_content, flags=flags)
                    if count > 0:
                        files_to_change[txt_path] = new_content
                        original_contents[txt_path] = old_content
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")
                    
        if not files_to_change:
            return QMessageBox.information(self, "No Matches", "No matches found.")
            
        reply = QMessageBox.question(self, "Confirm Replace", f"This will modify {len(files_to_change)} files.\nProceed?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
        
        success_count = 0
        try:
            for path, new_text in files_to_change.items():
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_text)
                success_count += 1
            self.undo_snapshot = original_contents
            self.btn_undo.setEnabled(True)
            self.btn_undo.setText(f"Undo ({len(files_to_change)} files)")
            self.log_msg.emit(f"✅ Replaced text in {success_count} files.")
            self.load_caption(self.current_index)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while writing files:\n{e}")

    def undo_last_replace(self):
        if not self.undo_snapshot: return
        
        reply = QMessageBox.question(self, "Confirm Undo", f"Revert changes to {len(self.undo_snapshot)} files?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
        
        restored_count = 0
        try:
            for path, old_text in self.undo_snapshot.items():
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(old_text)
                restored_count += 1
            self.undo_snapshot = {}
            self.btn_undo.setEnabled(False)
            self.btn_undo.setText("Undo Last Replace")
            self.btn_undo.setText("Undo Last Replace")
            self.log_msg.emit(f"↩️ Undid changes to {restored_count} files.")
            self.load_caption(self.current_index)
        except Exception as e:
            QMessageBox.critical(self, "Undo Error", f"Failed to revert some files:\n{e}")

    def on_show_mask_toggled(self):
        """Update persistent state only on user interaction."""
        if self.chk_show_mask.isEnabled():
            self.show_mask_state = self.chk_show_mask.isChecked()

    def image_wheel_event(self, event):
        self.navigate(1 if event.angleDelta().y() < 0 else -1)

    def modify_mask(self, amount):
        """
        Expands (dilate) or Contracts (erode) the mask by 'amount' pixels.
        amount > 0: Expand
        amount < 0: Contract
        """
        if self.cv_mask is None: return
        
        self._push_undo_state()
        
        kernel_size = abs(amount) * 2 + 1 # e.g., 1 -> 3x3 kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if amount > 0:
            self.cv_mask = cv2.dilate(self.cv_mask, kernel, iterations=1)
            self.log_msg.emit("➕ Expanded Mask")
        else:
            self.cv_mask = cv2.erode(self.cv_mask, kernel, iterations=1)
            self.log_msg.emit("➖ Contracted Mask")
            
        self.mask_is_dirty = True
        self._update_mask_button_states()
        self.update_image_display()

    def navigate(self, steps):
        new_row = self.current_index + steps
        if 0 <= new_row < len(self.image_files):
            self.list_widget.setCurrentRow(new_row)

    def undo_delete(self):
        if not self.deleted_files_stack: return
        
        record = self.deleted_files_stack.pop()
        restored_files = []
        
        try:
            for original_path, current_path in record["files"]:
                if os.path.exists(current_path):
                    # Move back to original location
                    # Ensure directory exists (it should, but safety first)
                    os.makedirs(os.path.dirname(original_path), exist_ok=True)
                    shutil.move(current_path, original_path)
                    restored_files.append(original_path)
            
            # Identify the main image file from restored files
            # (It's the one that matches our supported extensions and isn't a mask/txt)
            exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.mp4', '.mkv', '.avi', '.mov', '.webm')
            img_file = None
            for f in restored_files:
                if f.lower().endswith(exts) and not f.lower().endswith("-masklabel.png"):
                    img_file = f
                    break
            
            if img_file:
                user_index = record["index"]
                
                # Insert back into list
                self.image_files.insert(user_index, img_file)
                self.list_widget.insertItem(user_index, os.path.basename(img_file))
                
                # Select it
                self.list_widget.setCurrentRow(user_index)
                self.log_msg.emit(f"♻️ Restored {os.path.basename(img_file)}")
            else:
                self.log_msg.emit("⚠️ Restored files but could not identify main image to select.")
                self.refresh_file_list()

        except Exception as e:
            QMessageBox.critical(self, "Restore Failed", f"Error restoring files:\n{e}")
            self.log_msg.emit(f"❌ Error restoring files: {e}")
        
        # Update button state
        if not self.deleted_files_stack:
            self.btn_undo_delete.setEnabled(False)
            self.btn_undo_delete.setText("↩️ Undo Delete")
        else:
            self.btn_undo_delete.setText(f"↩️ Undo Delete ({len(self.deleted_files_stack)})")
        if 0 <= new_row < len(self.image_files):
            self.list_widget.setCurrentRow(new_row)

    def _push_undo_state(self):
        """Saves a copy of the current mask to the undo stack."""
        if self.cv_mask is None: return
        
        # Limit stack size to avoid memory issues (e.g., 20 steps)
        if len(self.mask_undo_stack) > 20:
            self.mask_undo_stack.pop(0)
            
        self.mask_undo_stack.append(self.cv_mask.copy())
        self.mask_redo_stack.clear() # New action clears redo history
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self):
        self.btn_mask_undo.setEnabled(len(self.mask_undo_stack) > 0)
        self.btn_mask_redo.setEnabled(len(self.mask_redo_stack) > 0)

    def undo_mask_action(self):
        if not self.mask_undo_stack: return
        
        # Save current state to redo
        if self.cv_mask is not None:
            self.mask_redo_stack.append(self.cv_mask.copy())
        
        # Pop from undo
        self.cv_mask = self.mask_undo_stack.pop()
        
        # Update UI
        self.mask_is_dirty = True
        self._update_mask_button_states()
        self._update_undo_redo_buttons()
        self.update_image_display()

    def redo_mask_action(self):
        if not self.mask_redo_stack: return
        
        # Save current state to undo
        if self.cv_mask is not None:
            self.mask_undo_stack.append(self.cv_mask.copy())
            
        # Pop from redo
        self.cv_mask = self.mask_redo_stack.pop()
        
        # Update UI
        self.mask_is_dirty = True
        self._update_mask_button_states()
        self._update_undo_redo_buttons()
        self.update_image_display()