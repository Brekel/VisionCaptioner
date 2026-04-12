"""
Detect system environment and install the correct llama-cpp-python wheel
from JamePeng's GitHub releases.

GitHub: https://github.com/JamePeng/llama-cpp-python/releases

The wheel selection is based on:
  - Python version  (e.g. cp312)
  - Operating system (win / linux / macos)
  - CUDA version    (12.4 / 12.6 / 12.8 / 13.0) or Metal on macOS
"""

import sys
import platform
import subprocess
import json
import re
from urllib.request import urlopen, Request
from urllib.error import URLError

GITHUB_API = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases"
GITHUB_PAGE = "https://github.com/JamePeng/llama-cpp-python/releases"

# CUDA version -> release tag token
CUDA_TAG_MAP = {
    "12.4": "cu124",
    "12.6": "cu126",
    "12.8": "cu128",
    "13.0": "cu130",
}

# Ordered from newest to oldest — used when exact match unavailable
CUDA_VERSIONS_ORDERED = ["13.0", "12.8", "12.6", "12.4"]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect_python_tag():
    """Return the cpython tag, e.g. 'cp312'."""
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def detect_platform():
    """Return 'win', 'linux', or 'macos'."""
    s = sys.platform
    if s.startswith("win"):
        return "win"
    elif s.startswith("linux"):
        return "linux"
    elif s == "darwin":
        return "macos"
    return "unknown"


def detect_cuda_version():
    """
    Return the CUDA version string from PyTorch (e.g. '12.8') or None.
    Falls back to nvidia-smi if torch is not available.
    """
    # Try torch first (already a project dependency)
    try:
        import torch
        cuda_str = getattr(torch.version, "cuda", None)
        if cuda_str:
            # torch.version.cuda is like "12.8"
            return cuda_str
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        )
        if out.strip():
            # nvidia-smi doesn't directly give CUDA toolkit version,
            # but we can try the CUDA version line from full output
            full = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, text=True)
            m = re.search(r"CUDA Version:\s*(\d+\.\d+)", full)
            if m:
                return m.group(1)
    except Exception:
        pass

    return None


def _best_cuda_tag(cuda_version):
    """
    Map an exact CUDA version (e.g. '12.8') to the best matching release tag.
    Picks the highest release tag that does not exceed the installed CUDA version.
    Returns a tag like 'cu128', or None.
    """
    if not cuda_version:
        return None

    # Parse major.minor
    m = re.match(r"(\d+)\.(\d+)", cuda_version)
    if not m:
        return None
    installed = (int(m.group(1)), int(m.group(2)))

    # Walk from newest to oldest, pick the first one <= installed
    for ver_str in CUDA_VERSIONS_ORDERED:
        parts = ver_str.split(".")
        candidate = (int(parts[0]), int(parts[1]))
        if candidate <= installed:
            return CUDA_TAG_MAP[ver_str]

    return None


def detect_system():
    """
    Return a dict summarising the detected environment:
      python_tag, platform, cuda_version, cuda_tag
    """
    plat = detect_platform()
    cuda_ver = detect_cuda_version() if plat != "macos" else None
    cuda_tag = _best_cuda_tag(cuda_ver) if plat != "macos" else "Metal"

    return {
        "python_tag": detect_python_tag(),
        "platform": plat,
        "cuda_version": cuda_ver,
        "cuda_tag": cuda_tag,
    }


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _fetch_json(url):
    """Fetch JSON from a URL."""
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def find_matching_wheel(system_info, log=None):
    """
    Query GitHub releases and find the best matching wheel URL.

    Returns (wheel_url, wheel_filename, release_tag) or raises RuntimeError.
    The optional *log* callback receives status strings.
    """
    def _log(msg):
        if log:
            log(msg)

    py_tag = system_info["python_tag"]
    plat = system_info["platform"]
    cuda_tag = system_info["cuda_tag"]

    if not cuda_tag:
        raise RuntimeError(
            "Could not determine CUDA version.\n"
            "Please install the correct wheel manually from:\n"
            f"  {GITHUB_PAGE}"
        )

    # Build expected fragments for matching
    if plat == "macos":
        plat_fragment = "macos"
        wheel_plat = "macosx"
    elif plat == "win":
        plat_fragment = "win"
        wheel_plat = "win_amd64"
    elif plat == "linux":
        plat_fragment = "linux"
        wheel_plat = "linux_x86_64"
    else:
        raise RuntimeError(f"Unsupported platform: {plat}")

    _log(f"Fetching releases from JamePeng/llama-cpp-python on GitHub...")

    releases = _fetch_json(GITHUB_API)

    # Find the newest release whose tag matches our cuda + platform
    for release in releases:
        tag = release.get("tag_name", "")
        # Tag format: v0.3.35-cu128-Basic-win-20260406  or  v0.3.35-Metal-macos-20260406
        tag_lower = tag.lower()

        if cuda_tag.lower() not in tag_lower:
            continue
        if plat_fragment not in tag_lower:
            continue

        # Found a matching release — now find the right wheel
        for asset in release.get("assets", []):
            name = asset["name"]
            if not name.endswith(".whl"):
                continue
            # Match python version and platform in the wheel filename
            if py_tag in name and wheel_plat in name:
                url = asset["browser_download_url"]
                _log(f"Found matching wheel: {name}")
                _log(f"  Release: {tag}")
                return url, name, tag

    raise RuntimeError(
        f"No matching wheel found for {py_tag} / {plat} / {cuda_tag}.\n"
        f"Please check manually at:\n  {GITHUB_PAGE}"
    )


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install_wheel(wheel_url, log=None):
    """
    Install a wheel from a URL using pip.
    The optional *log* callback receives status strings.
    Returns True on success, raises RuntimeError on failure.
    """
    def _log(msg):
        if log:
            log(msg)

    _log(f"Installing: pip install {wheel_url}")
    _log("This may take a minute...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", wheel_url],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pip install failed (exit code {result.returncode}):\n{err}")

    _log("pip install completed successfully.")
    return True


def run_install(log=None):
    """
    Full flow: detect system → find wheel → install.
    Returns (success: bool, message: str).
    The optional *log* callback receives verbose status strings.
    """
    def _log(msg):
        if log:
            log(msg)

    try:
        _log("--- llama-cpp-python Installer ---")
        _log(f"Source: JamePeng/llama-cpp-python")
        _log(f"        {GITHUB_PAGE}")
        _log("")

        info = detect_system()

        _log(f"Detected environment:")
        _log(f"  Python:   {info['python_tag']}  ({sys.version.split()[0]})")
        _log(f"  Platform: {info['platform']}  ({platform.platform()})")
        if info["platform"] == "macos":
            _log(f"  Backend:  Metal (Apple GPU)")
        else:
            _log(f"  CUDA:     {info['cuda_version'] or 'not detected'}")
            if info["cuda_tag"]:
                _log(f"  Wheel:    {info['cuda_tag']}  (best match for CUDA {info['cuda_version']})")
        _log("")

        url, filename, tag = find_matching_wheel(info, log=log)

        _log("")
        _log(f"Selected package:")
        _log(f"  {filename}")
        _log(f"  from release: {tag}")
        _log("")

        install_wheel(url, log=log)

        _log("")
        _log("llama-cpp-python installed successfully!")
        _log("Please reload the model to use GGUF files.")

        return True, "llama-cpp-python installed successfully. Please reload the model."

    except URLError as e:
        msg = f"Network error: {e}"
        _log(f"ERROR: {msg}")
        return False, msg
    except RuntimeError as e:
        _log(f"ERROR: {e}")
        return False, str(e)
    except Exception as e:
        msg = f"Unexpected error: {e}"
        _log(f"ERROR: {msg}")
        return False, msg
