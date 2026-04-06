"""
conftest.py - Shared test fixtures and sys.modules patching.

Heavy ML dependencies (torch, transformers, PySide6, etc.) are replaced with
MagicMock objects at the sys.modules level BEFORE any source module is imported.
This lets us test business logic without GPU, models, or a running GUI.
"""

import sys
import os
import json
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Pre-patch heavy dependencies so importing source modules doesn't fail
# ---------------------------------------------------------------------------

# Minimal real base classes for PySide6 widgets that source code inherits from.
class _FakeQLabel:
    def __init__(self, *a, **kw): pass
    def setAlignment(self, *a): pass
    def setStyleSheet(self, *a): pass
    def setMinimumHeight(self, *a): pass
    def setMouseTracking(self, *a): pass
    def pixmap(self): return None
    def width(self): return 0
    def height(self): return 0
    def text(self): return ""
    def setText(self, *a): pass
    def setPixmap(self, *a): pass
    def size(self): return MagicMock(width=MagicMock(return_value=0), height=MagicMock(return_value=0))
    def update(self): pass
    def grabMouse(self): pass
    def releaseMouse(self): pass
    def setAcceptDrops(self, *a): pass
    def setPlaceholderText(self, *a): pass


class _FakeQLineEdit(_FakeQLabel):
    pass


class _FakeQThread:
    def __init__(self, *a, **kw): pass
    def start(self): pass
    def wait(self): pass


class _FakeSignal:
    """Mimics PySide6.QtCore.Signal as a callable descriptor."""
    def __init__(self, *a, **kw): pass
    def emit(self, *a, **kw): pass
    def connect(self, *a): pass
    def disconnect(self, *a): pass


class _FakeQt:
    AlignCenter = 0
    KeepAspectRatio = 0
    SmoothTransformation = 0
    LeftButton = 1
    RightButton = 2
    MiddleButton = 4
    DashLine = 0
    NoBrush = 0
    MouseButton = int


# Build mock modules
_qt_widgets = MagicMock()
_qt_widgets.QLabel = _FakeQLabel
_qt_widgets.QLineEdit = _FakeQLineEdit
_qt_widgets.QWidget = type("QWidget", (), {"__init__": lambda *a, **kw: None})

_qt_core = MagicMock()
_qt_core.Qt = _FakeQt
_qt_core.Signal = _FakeSignal
_qt_core.QThread = _FakeQThread
_qt_core.QTimer = MagicMock()
_qt_core.QByteArray = MagicMock()

_qt_gui = MagicMock()

# Modules to patch
_MOCK_MODULES = {
    # PyTorch
    "torch": MagicMock(),
    "torch.cuda": MagicMock(),
    "torch.backends": MagicMock(),
    "torch.backends.mps": MagicMock(),
    "torch._dynamo": MagicMock(),
    "torchvision": MagicMock(),
    # Transformers
    "transformers": MagicMock(),
    "transformers.AutoModelForImageTextToText": MagicMock(),
    "transformers.AutoProcessor": MagicMock(),
    "transformers.StoppingCriteria": MagicMock(),
    "transformers.StoppingCriteriaList": MagicMock(),
    "transformers.BitsAndBytesConfig": MagicMock(),
    # Qwen / Vision
    "qwen_vl_utils": MagicMock(),
    "flash_attn": MagicMock(),
    # Llama
    "llama_cpp": MagicMock(),
    "llama_cpp.llama_chat_format": MagicMock(),
    # GGUF
    "gguf": MagicMock(),
    # Video
    "decord": MagicMock(),
    # Progress
    "tqdm": MagicMock(),
    # PySide6
    "PySide6": MagicMock(),
    "PySide6.QtWidgets": _qt_widgets,
    "PySide6.QtCore": _qt_core,
    "PySide6.QtGui": _qt_gui,
    # MediaPipe
    "mediapipe": MagicMock(),
    "mediapipe.tasks": MagicMock(),
    "mediapipe.tasks.vision": MagicMock(),
    # SAM3
    "sam3": MagicMock(),
    "sam3.model_builder": MagicMock(),
    "sam3.model": MagicMock(),
    "sam3.model.sam3_image_processor": MagicMock(),
}

for mod_name, mock_obj in _MOCK_MODULES.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock_obj

# Configure torch mock defaults
_torch = sys.modules["torch"]
_torch.cuda.is_available.return_value = False
_torch.backends.mps.is_available.return_value = False
_torch.float16 = "float16"
_torch.bfloat16 = "bfloat16"
_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

# Make tqdm.tqdm act as a pass-through iterable wrapper
_tqdm = sys.modules["tqdm"]
_tqdm.tqdm = lambda iterable=None, *a, **kw: iterable if iterable is not None else MagicMock()

# Make StoppingCriteria a real base class so StopTrigger can inherit from it
sys.modules["transformers"].StoppingCriteria = type("StoppingCriteria", (), {})
sys.modules["transformers"].StoppingCriteriaList = list

# ---------------------------------------------------------------------------
# 2. Add project root to sys.path so `import backend`, `import cli` etc. work
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 3. Shared pytest fixtures
# ---------------------------------------------------------------------------
import pytest


@pytest.fixture
def sample_settings(tmp_path):
    """Creates a minimal settings.json and returns its path."""
    data = {
        "folder": str(tmp_path),
        "generate_tab": {
            "model_index": 2,
            "quantization": "FP16 (Half Precision)",
            "resolution_idx": 3,
            "batch_size": 8,
            "frames": 4,
            "tokens": 512,
            "trigger": "photo",
            "prompt_text": "Describe this image in detail.",
            "suffix": "Be concise.",
            "skip_existing": True
        },
        "mask_tab": {
            "prompt": "person",
            "max_res": 768,
            "expand_percent": 5.0,
            "skip_existing": False,
            "crop_to_mask": True
        },
        "video_tab": {
            "step": 15,
            "start_frame": 10,
            "end_frame": 500,
            "res": 768,
            "conf": 0.3,
            "expand": 3.0,
            "crop": True
        }
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(data))
    return str(settings_path)


