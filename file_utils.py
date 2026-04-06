"""
file_utils.py - Shared file discovery utilities for VisionCaptioner.

Centralizes image/video file discovery logic used across backend, CLI, and UI tabs.
"""

import os
import glob


IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
VIDEO_EXTS = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.webm', '*.flv']
ALL_MEDIA_EXTS = IMAGE_EXTS + VIDEO_EXTS

EXCLUDED_DIRS = {'unused', 'uncropped', 'masks', '__pycache__', '.git'}


def find_media_files(folder_path, exts=None, recursive=False, exclude_masks=True):
    """
    Central file discovery function.

    Args:
        folder_path: Root folder to scan.
        exts: List of glob patterns (e.g. ['*.jpg', '*.png']).
              Defaults to IMAGE_EXTS.
        recursive: If True, scan subdirectories recursively.
        exclude_masks: If True, filter out files containing 'masklabel' in their name.

    Returns:
        Sorted list of unique absolute file paths.
    """
    if exts is None:
        exts = IMAGE_EXTS

    files = []
    if recursive:
        for ext in exts:
            pattern = os.path.join(folder_path, '**', ext)
            files.extend(glob.glob(pattern, recursive=True))
            pattern_upper = os.path.join(folder_path, '**', ext.upper())
            files.extend(glob.glob(pattern_upper, recursive=True))
        files = [f for f in files if not _is_in_excluded_dir(f, folder_path)]
    else:
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
            files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if exclude_masks:
        files = [f for f in files if 'masklabel' not in os.path.basename(f).lower()]

    return sorted(set(files))


def get_display_name(filepath, root_folder, recursive=False):
    """
    Return a display name for file lists.

    When recursive, shows the relative path (e.g. 'subdir/image.jpg').
    When flat, shows just the basename (e.g. 'image.jpg').
    """
    if recursive:
        return os.path.relpath(filepath, root_folder).replace(os.sep, '/')
    return os.path.basename(filepath)


def _is_in_excluded_dir(filepath, root):
    """Check if filepath is inside an excluded subdirectory relative to root."""
    rel = os.path.relpath(filepath, root)
    parts = rel.split(os.sep)
    for part in parts[:-1]:
        if part.lower() in EXCLUDED_DIRS or part.startswith('.'):
            return True
    return False
