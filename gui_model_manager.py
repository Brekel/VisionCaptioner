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

from model_probe import ModelProbe


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

from gui_workers import ScanWorker
class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Manager")
        self.resize(1100, 750) 
        self.models_data = [] # JSON data
        self.scan_data = {}   # Probe data
        self.worker = None

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
        self.btn_help.setCheckable(True) 
        self.btn_help.setChecked(False) 
        self.btn_help.setFlat(True) 
        self.btn_help.setStyleSheet("text-align: left; font-weight: bold; color: #4da6ff;")
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.toggled.connect(self.toggle_help)
        hf_layout.addWidget(self.btn_help)

        # 2. Collapsible Instructions Container
        self.help_container = QWidget()
        help_layout = QVBoxLayout(self.help_container)
        help_layout.setContentsMargins(10, 0, 0, 10) 

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
        self.help_container.setVisible(False) 
        hf_layout.addWidget(self.help_container)

        # 3. Token Input
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Token:"))
        self.txt_token = QLineEdit()
        self.txt_token.setEchoMode(QLineEdit.Password)
        self.txt_token.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            self.txt_token.setPlaceholderText("Found HF_TOKEN in environment variables (Leave empty to use)")
            self.txt_token.setToolTip("System HF_TOKEN detected.")
        token_layout.addWidget(self.txt_token)
        hf_layout.addLayout(token_layout)
        layout.addWidget(hf_group)

        # --- Actions Group ---
        action_layout = QHBoxLayout()
        self.btn_scan = QPushButton("🔍 Scan Local Models")
        self.btn_scan.setToolTip("Refresh list and probe local models/GGUFs")
        self.btn_scan.clicked.connect(self.start_scan)
        action_layout.addWidget(self.btn_scan)
        action_layout.addStretch()
        layout.addLayout(action_layout)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(5) # Reverted to 5
        self.table.setHorizontalHeaderLabels(["Model Name", "Description", "Status", "Gated", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)
        
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
            print(f"Error: {MODELS_FILE} not found.")
            return

        try:
            with open(MODELS_FILE, 'r') as f:
                self.models_data = json.load(f)
            # Initial scan
            self.start_scan() 
        except Exception as e:
            print(f"Error loading JSON: {e}")
            
    def start_scan(self):
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("Scanning...")
        self.worker = ScanWorker()
        self.worker.finished.connect(self.on_scan_finished)
        self.worker.start()
        
    def on_scan_finished(self, results):
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("🔍 Scan Local Models")
        self.scan_data = results
        self.refresh_table()
        print(f"Scan complete. Found {len(results)} local models.")

    def refresh_table(self):
        self.table.setRowCount(0)
        
        # 1. Combine Known JSON models + Unknown Local models
        # Create a set of folder names from JSON to identify "Extras"
        json_folders = {m['folder'] for m in self.models_data}
        
        display_list = []
        
        # Add JSON models first
        for m in self.models_data:
            item = m.copy()
            item['is_extra'] = False
            # Check if installed using scan data
            folder = m['folder']
            if folder in self.scan_data:
                item['installed'] = True
                item['probe_info'] = self.scan_data[folder]
            else:
                item['installed'] = False
            display_list.append(item)
            
        # Add Extra Local Models (GGUFs or Folders not in JSON)
        for name, info in self.scan_data.items():
            if name not in json_folders:
                display_list.append({
                    'name': name,
                    'folder': name, # File or Folder name
                    'description': "Locally discovered model",
                    'gated': False,
                    'is_extra': True,
                    'installed': True,
                    'probe_info': info
                })
        
        self.table.setRowCount(len(display_list))
        self.current_list = display_list

        for i, m in enumerate(display_list):
            # Name / Link
            if not m.get('is_extra', False):
                repo_url = f"https://huggingface.co/{m.get('repo_id', '')}"
                link_html = f"<a href='{repo_url}' style='color: #4da6ff; text-decoration: underline;'>{m['name']}</a>"
                lbl_link = QLabel(link_html)
                lbl_link.setOpenExternalLinks(True)
            else:
                lbl_link = QLabel(m['name'])
                if m['name'].endswith(".gguf"):
                     lbl_link.setText(f"📦 {m['name']}")
            
            lbl_link.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lbl_link.setStyleSheet("QLabel { margin-left: 5px; }")
            self.table.setCellWidget(i, 0, lbl_link)
            
            # Reverted columns: No Type/Vision
            self.table.setItem(i, 1, QTableWidgetItem(m.get('description', '')))
            
            # Status
            status_text = "Installed" if m['installed'] else "Not Found"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QColor("#a3be8c") if m['installed'] else QColor("#bf616a"))
            status_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 2, status_item)

            gated = m.get('gated', False)
            gated_item = QTableWidgetItem("Yes" if gated else "No")
            gated_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 3, gated_item)

            # Action
            btn = QPushButton()
            if m['installed']:
                btn.setText("Delete")
                btn.setStyleSheet("background-color: #d73a49; color: white;")
                # Fix closure with default argument
                btn.clicked.connect(lambda checked, f=display_list[i]['folder']: self.delete_model(f)) 
            else:
                btn.setText("Download")
                btn.setStyleSheet("background-color: #2da44e; color: white;")
                # Fix closure with default argument
                btn.clicked.connect(lambda checked, idx=i: self.download_model(idx))
                if m.get('is_extra', False):
                    btn.setEnabled(False) 
            
            self.table.setCellWidget(i, 4, btn)

    def download_model(self, display_index):
        item = self.current_list[display_index]
        
        token = self.txt_token.text().strip() or None
        if item.get('gated', False) and not token:
            if "HF_TOKEN" not in os.environ:
                ret = QMessageBox.warning(self, "Token Required?", 
                                    f"{item['name']} is a gated model.\nIf you haven't logged in via CLI, please enter a Token above.\nTry anyway?",
                                    QMessageBox.Yes | QMessageBox.No)
                if ret == QMessageBox.No: return

        # UI Updates
        btn = self.table.cellWidget(display_index, 4)
        btn.setEnabled(False)
        btn.setText("Downloading...")
        # Yellowish background, black text
        btn.setStyleSheet("background-color: #ebcb8b; color: black; font-weight: bold;")
        
        self.toggle_interface(False)
        print(f"Starting download for {item['name']}...")
        
        self.worker = DownloadWorker(item['repo_id'], item['folder'], token)
        self.worker.finished.connect(self.on_download_finished)
        self.worker.start()

    def delete_model(self, folder_name):
        ret = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete {folder_name}?\nThis cannot be undone.", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.toggle_interface(False)
            print(f"Deleting {folder_name}...")
            self.worker = DeleteWorker(folder_name)
            self.worker.finished.connect(self.on_delete_finished)
            self.worker.start()

    def on_delete_finished(self, msg):
        self.toggle_interface(True)
        print(f"{msg}")
        self.start_scan() # Refresh after delete
    
    def on_download_finished(self, success, msg):
        self.toggle_interface(True)
        if success:
            print(f"SUCCESS: {msg}")
            QMessageBox.information(self, "Done", "Download Complete!")
        else:
            print(f"ERROR: {msg}")
            if "401" in msg or "403" in msg:
                 QMessageBox.critical(self, "Auth Error", "Authentication failed.\nDid you provide a valid Token?")
            else:
                QMessageBox.critical(self, "Error", f"Download failed:\n{msg}")
        self.start_scan() # Refresh

    def toggle_interface(self, enabled):
        self.table.setEnabled(enabled)
        self.btn_close.setEnabled(enabled)
        self.btn_scan.setEnabled(enabled)

    def closeEvent(self, event):
        super().closeEvent(event)
