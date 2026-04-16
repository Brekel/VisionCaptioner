import os
import sys
import subprocess
import cv2
from PySide6.QtWidgets import QLabel, QLineEdit, QDialog, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QImageReader, QPainter, QPen, QColor


class ShutdownCountdownDialog(QDialog):
    """Modal dialog that counts down before shutting the computer down."""

    def __init__(self, seconds=60, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shutdown")
        self.setModal(True)
        self.setFixedSize(340, 130)
        self.remaining = seconds

        layout = QVBoxLayout(self)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.btn_cancel = QPushButton("Cancel Shutdown")
        self.btn_cancel.clicked.connect(self._cancel)
        layout.addWidget(self.btn_cancel)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        self._update_label()
        self.timer.start()

    def _update_label(self):
        self.label.setText(f"Shutting down in {self.remaining} seconds...")

    def _tick(self):
        self.remaining -= 1
        if self.remaining <= 0:
            self.timer.stop()
            self._do_shutdown()
        else:
            self._update_label()

    def _cancel(self):
        self.timer.stop()
        self.reject()

    def _do_shutdown(self):
        if sys.platform == "win32":
            subprocess.Popen(["shutdown", "/s", "/t", "0"])
        elif sys.platform == "darwin":
            subprocess.Popen(["osascript", "-e",
                              'tell app "System Events" to shut down'])
        else:
            subprocess.Popen(["systemctl", "poweroff"])
        self.accept()

class DropLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Select Image Folder (or drag & drop here)...")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self.setText(path)
            else:
                self.setText(os.path.dirname(path))

class ResizableImageLabel(QLabel):
    paint_start = Signal(int, int, Qt.MouseButton)
    paint_move = Signal(int, int, Qt.MouseButton)
    paint_end = Signal(Qt.MouseButton)
    view_resized = Signal()
    middle_click = Signal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #222; border: 1px solid #444; border-radius: 5px;")
        self.current_pixmap = None
        self.setMinimumHeight(100)
        self.is_painting = False
        
        self.setMouseTracking(True)
        self.cursor_pos = None
        self.cursor_size_px = 0

    def _widget_to_image_coords(self, widget_pos):
        if not self.pixmap() or self.pixmap().isNull() or not self.current_pixmap:
            return None
            
        img_w = self.current_pixmap.width()
        img_h = self.current_pixmap.height()
        
        scaled_pixmap = self.pixmap()
        scaled_w = scaled_pixmap.width()
        scaled_h = scaled_pixmap.height()
        
        # Calculate offsets caused by KeepAspectRatio
        offset_x = (self.width() - scaled_w) / 2
        offset_y = (self.height() - scaled_h) / 2
        
        # Calculate coordinates relative to the top-left of the displayed image
        pixmap_x = widget_pos.x() - offset_x
        pixmap_y = widget_pos.y() - offset_y
        
        # We removed the bounds check here. 
        # Even if pixmap_x is -5 (outside left) or > width (outside right),
        # we calculate the relative coordinate so OpenCV can draw the edge of the brush.
        
        ratio_x = img_w / scaled_w
        ratio_y = img_h / scaled_h
        
        image_x = int(pixmap_x * ratio_x)
        image_y = int(pixmap_y * ratio_y)
        
        return (image_x, image_y)

    def set_brush_outline_size(self, size_px):
        if self.cursor_size_px != size_px:
            self.cursor_size_px = size_px
            if self.cursor_pos: self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.middle_click.emit()
            event.accept()
            return

        coords = self._widget_to_image_coords(event.pos())
        if coords:
            button = event.button()
            if button == Qt.LeftButton or button == Qt.RightButton:
                self.is_painting = True
                self.grabMouse() # Capture mouse to prevent triggering other widgets (like context menus)
                self.paint_start.emit(coords[0], coords[1], button); event.accept()

    def mouseMoveEvent(self, event):
        self.cursor_pos = event.pos()
        self.update()
        if self.is_painting:
            coords = self._widget_to_image_coords(event.pos())
            if coords:
                buttons = event.buttons()
                button = Qt.LeftButton if buttons & Qt.LeftButton else Qt.RightButton
                self.paint_move.emit(coords[0], coords[1], button); event.accept()

    def leaveEvent(self, event):
        self.cursor_pos = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_painting:
            self.is_painting = False
            # Delay release to consume any subsequent ContextMenu events directed at the image
            QTimer.singleShot(0, self.releaseMouse)
            self.paint_end.emit(event.button()); event.accept()

    def contextMenuEvent(self, event):
        event.accept()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.cursor_pos and self.cursor_size_px > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            radius = self.cursor_size_px / 2
            painter.drawEllipse(self.cursor_pos, radius, radius)

    def set_image(self, image_path):
        self.setText("") 
        if not image_path or not os.path.exists(image_path):
            self.setText("No Image Selected"); self.current_pixmap = None; self.update_view(); return
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
            try:
                cap = cv2.VideoCapture(image_path); ret, frame = cap.read(); cap.release()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch = frame.shape
                    bytes_per_line = ch * w; q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.current_pixmap = QPixmap.fromImage(q_img)
                else: self.setText(f"VIDEO ERROR:\n{os.path.basename(image_path)}"); self.current_pixmap = None
            except Exception as e:
                print(f"Error previewing video: {e}"); self.setText(f"VIDEO ERROR:\n{os.path.basename(image_path)}"); self.current_pixmap = None
        else:
            reader = QImageReader(image_path); reader.setAutoTransform(True); img = reader.read()
            if not img.isNull(): self.current_pixmap = QPixmap.fromImage(img)
            else: self.current_pixmap = QPixmap(image_path)
        self.update_view()

    def resizeEvent(self, event):
        self.update_view()
        self.view_resized.emit()
        super().resizeEvent(event)

    def update_view(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            scaled = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
        elif self.text() == "": self.setText("No Image")