import os
import json
import shutil
import sys
import re
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
                               QTableWidgetItem, QPushButton, QLabel, QHeaderView, 
                               QMessageBox, QLineEdit, QTextEdit, QProgressBar, 
                               QGroupBox, QWidget, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor, QTextCursor, QCursor

# Try to import huggingface_hub
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

MODELS_FILE = "models.json"

class OutputRedirector(QObject):
    """Redirects stdout/stderr to a signal"""
    text_written = Signal(str)
    def write(self, text):
        if text and text.strip():
            self.text_written.emit(text)
    def flush(self):
        pass

class DownloadWorker(QThread):
    finished = Signal(bool, str)
    
    def __init__(self, repo_id, folder_name, token=None):
        super().__init__()
        self.repo_id = repo_id
        self.folder_name = folder_name
        self.token = token
        self.dest_path = os.path.join(os.getcwd(), "models", self.folder_name)

    def run(self):
        try:
            os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)

            snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.dest_path,
                token=self.token,
                local_dir_use_symlinks=False, 
                resume_download=True
            )
            self.finished.emit(True, f"Successfully downloaded {self.folder_name}")
        except Exception as e:
            self.finished.emit(False, str(e))

class DeleteWorker(QThread):
    finished = Signal(str)

    def __init__(self, folder_name):
        super().__init__()
        self.path = os.path.join(os.getcwd(), "models", folder_name)

    def run(self):
        try:
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            self.finished.emit("Deleted")
        except Exception as e:
            self.finished.emit(f"Error: {e}")

class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Manager")
        self.resize(950, 750) 
        self.models_data = []
        self.worker = None
        
        self.original_stderr = sys.stderr
        self.redirector = OutputRedirector()
        self.redirector.text_written.connect(self.append_log)

        self.setup_ui()
        self.load_db()
        
        if not HF_AVAILABLE:
            QMessageBox.critical(self, "Missing Library", "huggingface_hub is not installed.\nPlease run: pip install huggingface_hub")
            self.table.setEnabled(False)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # --- HuggingFace Group ---
        hf_group = QGroupBox("HuggingFace Authentication")
        hf_layout = QVBoxLayout(hf_group)

        # 1. Toggle Button for Help
        self.btn_help = QPushButton("❓ Need a Token? Click here for instructions")
        self.btn_help.setCheckable(True) # Acts as a toggle switch
        self.btn_help.setChecked(False)  # Collapsed by default
        self.btn_help.setFlat(True)      # Minimal look
        self.btn_help.setStyleSheet("text-align: left; font-weight: bold; color: #4da6ff;")
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.toggled.connect(self.toggle_help)
        hf_layout.addWidget(self.btn_help)

        # 2. Collapsible Instructions Container
        self.help_container = QWidget()
        help_layout = QVBoxLayout(self.help_container)
        help_layout.setContentsMargins(10, 0, 0, 10) # Indent slightly

        instr_text = (
            "<b>Some models (like SAM3) are 'Gated'. To download them:</b><br>"
            "1. Create an account at <a href='https://huggingface.co/join'>huggingface.co</a><br>"
            "2. Go to the model page (link in table below) and <b>Accept the License Agreement</b>.<br>"
            "3. Generate a 'Read' token here: <a href='https://huggingface.co/settings/tokens'>huggingface.co/settings/tokens</a><br>"
            "4. Paste the token below."
        )
        self.lbl_instr = QLabel(instr_text)
        self.lbl_instr.setOpenExternalLinks(True)
        self.lbl_instr.setWordWrap(True)
        self.lbl_instr.setStyleSheet("color: #ccc; background-color: #222; padding: 10px; border-radius: 5px;")
        
        help_layout.addWidget(self.lbl_instr)
        
        self.help_container.setVisible(False) # Start hidden
        hf_layout.addWidget(self.help_container)

        # 3. Token Input
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Token:"))
        
        self.txt_token = QLineEdit()
        self.txt_token.setEchoMode(QLineEdit.Password)
        self.txt_token.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        
        # Check environment variable
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            self.txt_token.setPlaceholderText("Found HF_TOKEN in environment variables (Leave empty to use)")
            self.txt_token.setToolTip("System HF_TOKEN detected.")

        token_layout.addWidget(self.txt_token)
        hf_layout.addLayout(token_layout)
        
        layout.addWidget(hf_group)
        # -------------------------------------------

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Model Name (Link)", "Description", "Status", "Gated", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #111; color: #ddd; font-family: Consolas, Monospace; font-size: 11px;")
        self.txt_log.setFixedHeight(150)
        self.txt_log.setLineWrapMode(QTextEdit.NoWrap) 
        layout.addWidget(self.txt_log)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        layout.addWidget(self.btn_close)

    def toggle_help(self, checked):
        self.help_container.setVisible(checked)
        if checked:
            self.btn_help.setText("🔽 Hide Instructions")
        else:
            self.btn_help.setText("❓ Need a Token? Click here for instructions")

    def load_db(self):
        if not os.path.exists(MODELS_FILE):
            self.append_log(f"Error: {MODELS_FILE} not found.")
            return

        try:
            with open(MODELS_FILE, 'r') as f:
                self.models_data = json.load(f)
            self.refresh_table()
        except Exception as e:
            self.append_log(f"Error loading JSON: {e}")

    def refresh_table(self):
        self.table.setRowCount(0)
        models_dir = os.path.join(os.getcwd(), "models")
        
        for i, m in enumerate(self.models_data):
            self.table.insertRow(i)
            
            repo_url = f"https://huggingface.co/{m['repo_id']}"
            link_html = f"<a href='{repo_url}' style='color: #4da6ff; text-decoration: underline;'>{m['name']}</a>"
            
            lbl_link = QLabel(link_html)
            lbl_link.setOpenExternalLinks(True)
            lbl_link.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lbl_link.setCursor(QCursor(Qt.PointingHandCursor))
            lbl_link.setStyleSheet("QLabel { margin-left: 5px; }")
            self.table.setCellWidget(i, 0, lbl_link)

            self.table.setItem(i, 1, QTableWidgetItem(m.get('description', '')))
            
            target_path = os.path.join(models_dir, m['folder'])
            is_installed = os.path.exists(target_path) and os.path.isdir(target_path)
            
            status_item = QTableWidgetItem("Installed" if is_installed else "Not Found")
            status_item.setForeground(QColor("#a3be8c") if is_installed else QColor("#bf616a"))
            status_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 2, status_item)

            gated = m.get('gated', False)
            gated_item = QTableWidgetItem("Yes" if gated else "No")
            gated_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 3, gated_item)

            btn = QPushButton()
            if is_installed:
                btn.setText("Delete")
                btn.setStyleSheet("background-color: #d73a49; color: white;")
                btn.clicked.connect(lambda checked, idx=i: self.delete_model(idx))
            else:
                btn.setText("Download")
                btn.setStyleSheet("background-color: #2da44e; color: white;")
                btn.clicked.connect(lambda checked, idx=i: self.download_model(idx))
            
            self.table.setCellWidget(i, 4, btn)

    def append_log(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        
        if not clean_text:
            return

        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.End)

        if '\r' in clean_text:
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            clean_text = clean_text.replace('\r', '').strip()
            
            if not clean_text: 
                return
            
            cursor.insertText(clean_text)
        else:
            cursor.insertText(clean_text)

        self.txt_log.setTextCursor(cursor)
        self.txt_log.ensureCursorVisible()

    def download_model(self, index):
        data = self.models_data[index]
        token = self.txt_token.text().strip() or None
        
        if data.get('gated', False) and not token:
            if "HF_TOKEN" not in os.environ:
                ret = QMessageBox.warning(self, "Token Required?", 
                                    f"{data['name']} is a gated model.\nIf you haven't logged in via CLI, please enter a Token above.\nTry anyway?",
                                    QMessageBox.Yes | QMessageBox.No)
                if ret == QMessageBox.No: return

        status_item = self.table.item(index, 2)
        status_item.setText("Downloading...")
        status_item.setForeground(QColor("#ebcb8b")) 

        self.toggle_interface(False)
        self.txt_log.clear()
        self.append_log(f"Starting download for {data['name']}...\n")
        self.append_log("Please wait, this may take a while. Progress is shown below:\n")

        sys.stderr = self.redirector

        self.worker = DownloadWorker(data['repo_id'], data['folder'], token)
        self.worker.finished.connect(self.on_download_finished)
        self.worker.start()

    def on_download_finished(self, success, msg):
        sys.stderr = self.original_stderr
        
        self.toggle_interface(True)
        if success:
            self.append_log(f"\nSUCCESS: {msg}\n")
            QMessageBox.information(self, "Done", "Download Complete!")
        else:
            self.append_log(f"\nERROR: {msg}\n")
            if "401" in msg or "403" in msg:
                 QMessageBox.critical(self, "Auth Error", "Authentication failed.\nDid you provide a valid HuggingFace Token for this gated model?\nHave you accepted the license agreement on the website?")
            else:
                QMessageBox.critical(self, "Error", f"Download failed:\n{msg}")
        
        self.refresh_table()

    def delete_model(self, index):
        data = self.models_data[index]
        ret = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete {data['name']}?\nThis cannot be undone.", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.toggle_interface(False)
            self.append_log(f"Deleting {data['folder']}...\n")
            
            self.worker = DeleteWorker(data['folder'])
            self.worker.finished.connect(self.on_delete_finished)
            self.worker.start()

    def on_delete_finished(self, msg):
        self.toggle_interface(True)
        self.append_log(f"{msg}\n")
        self.refresh_table()

    def toggle_interface(self, enabled):
        self.table.setEnabled(enabled)
        self.btn_close.setEnabled(enabled)

    def closeEvent(self, event):
        sys.stderr = self.original_stderr
        super().closeEvent(event)