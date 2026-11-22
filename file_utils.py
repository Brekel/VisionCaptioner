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


MASKLABEL_SUFFIX = '-masklabel.png'
MASK_SUBDIR = 'masks'


def find_stray_sidecars(folder_path, recursive=False):
    """
    Find caption and mask files that have no matching media file.

    Sidecars checked per directory:
      - <stem>.txt               (caption)
      - <stem>-masklabel.png     (separate mask)
      - masks/<stem>.png         (subfolder mask, paired with the parent dir)

    Matching is case-insensitive and accepts any media extension (images and
    videos, any letter case). Excluded dirs (unused/, uncropped/, hidden) are
    never scanned. Returns a sorted list of absolute paths.
    """
    media_exts = {e.lstrip('*').lower() for e in ALL_MEDIA_EXTS}

    if recursive:
        dirs = []
        for dirpath, dirnames, _ in os.walk(folder_path):
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in EXCLUDED_DIRS and not d.startswith('.')]
            dirs.append(dirpath)
    else:
        dirs = [folder_path]

    strays = []
    for d in dirs:
        try:
            names = [n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n))]
        except OSError:
            continue

        media_stems = set()
        for n in names:
            stem, ext = os.path.splitext(n)
            if ext.lower() in media_exts and not n.lower().endswith(MASKLABEL_SUFFIX):
                media_stems.add(stem.lower())

        for n in names:
            low = n.lower()
            if low.endswith(MASKLABEL_SUFFIX):
                stem = n[:-len(MASKLABEL_SUFFIX)]
            elif low.endswith('.txt'):
                stem = n[:-4]
            else:
                continue
            if stem.lower() not in media_stems:
                strays.append(os.path.join(d, n))

        # masks/ subfolder (any case), paired against this directory's media
        for sub in os.listdir(d):
            sub_path = os.path.join(d, sub)
            if sub.lower() != MASK_SUBDIR or not os.path.isdir(sub_path):
                continue
            for n in os.listdir(sub_path):
                stem, ext = os.path.splitext(n)
                if ext.lower() == '.png' and os.path.isfile(os.path.join(sub_path, n)) \
                        and stem.lower() not in media_stems:
                    strays.append(os.path.join(sub_path, n))

    return sorted(strays)


def _is_in_excluded_dir(filepath, root):
    """Check if filepath is inside an excluded subdirectory relative to root."""
    rel = os.path.relpath(filepath, root)
    parts = rel.split(os.sep)
    for part in parts[:-1]:
        if part.lower() in EXCLUDED_DIRS or part.startswith('.'):
            return True
    return False
