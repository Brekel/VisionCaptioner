"""Tests for file_utils.py - Shared file discovery utilities."""

import os
import pytest

from file_utils import find_media_files, find_stray_sidecars, get_display_name, _is_in_excluded_dir, IMAGE_EXTS, VIDEO_EXTS, ALL_MEDIA_EXTS


# ===========================================================================
# TestFindMediaFiles
# ===========================================================================
class TestFindMediaFiles:

    def test_finds_images_flat(self, tmp_path):
        for name in ["a.jpg", "b.png", "c.webp"]:
            (tmp_path / name).write_bytes(b"fake")

        result = find_media_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "a.jpg" in basenames
        assert "b.png" in basenames
        assert "c.webp" in basenames

    def test_finds_images_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.jpg").write_bytes(b"fake")
        (sub / "nested.jpg").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), recursive=True)
        basenames = [os.path.basename(f) for f in result]
        assert "root.jpg" in basenames
        assert "nested.jpg" in basenames

    def test_flat_does_not_find_subdirs(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.jpg").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), recursive=False)
        assert len(result) == 0

    def test_excludes_masklabel(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "photo-masklabel.png").write_bytes(b"fake")
        (tmp_path / "photo_masklabel.png").write_bytes(b"fake")

        result = find_media_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "photo.jpg" in basenames
        assert "photo-masklabel.png" not in basenames
        assert "photo_masklabel.png" not in basenames

    def test_includes_masklabel_when_disabled(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "photo-masklabel.png").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), exclude_masks=False)
        basenames = [os.path.basename(f) for f in result]
        assert "photo.jpg" in basenames
        assert "photo-masklabel.png" in basenames

    def test_excludes_special_dirs_recursive(self, tmp_path):
        for dirname in ["unused", "masks", ".hidden"]:
            d = tmp_path / dirname
            d.mkdir()
            (d / "img.jpg").write_bytes(b"fake")
        (tmp_path / "root.jpg").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), recursive=True)
        basenames = [os.path.basename(f) for f in result]
        assert "root.jpg" in basenames
        assert len(result) == 1

    def test_sorted_and_deduplicated(self, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"fake")

        result = find_media_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert basenames == sorted(basenames)

    def test_uppercase_extensions(self, tmp_path):
        (tmp_path / "PHOTO.JPG").write_bytes(b"fake")
        (tmp_path / "SCAN.PNG").write_bytes(b"fake")

        result = find_media_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "PHOTO.JPG" in basenames
        assert "SCAN.PNG" in basenames

    def test_empty_folder(self, tmp_path):
        result = find_media_files(str(tmp_path))
        assert result == []

    def test_custom_exts_videos_only(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"fake")
        (tmp_path / "photo.jpg").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), exts=VIDEO_EXTS)
        basenames = [os.path.basename(f) for f in result]
        assert "clip.mp4" in basenames
        assert "photo.jpg" not in basenames

    def test_nested_subdirs_recursive(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.jpg").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), recursive=True)
        assert len(result) == 1
        assert os.path.basename(result[0]) == "deep.jpg"

    def test_mixed_media_exts(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "clip.mp4").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), exts=ALL_MEDIA_EXTS)
        basenames = [os.path.basename(f) for f in result]
        assert "photo.jpg" in basenames
        assert "clip.mp4" in basenames

    def test_ignores_non_media_files(self, tmp_path):
        (tmp_path / "readme.txt").write_bytes(b"text")
        (tmp_path / "data.csv").write_bytes(b"csv")
        (tmp_path / "real.jpg").write_bytes(b"img")

        result = find_media_files(str(tmp_path))
        assert len(result) == 1
        assert os.path.basename(result[0]) == "real.jpg"

    def test_excludes_uncropped_dir_recursive(self, tmp_path):
        d = tmp_path / "uncropped"
        d.mkdir()
        (d / "img.jpg").write_bytes(b"fake")
        (tmp_path / "root.jpg").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), recursive=True)
        assert len(result) == 1
        assert os.path.basename(result[0]) == "root.jpg"

    def test_nested_excluded_dir_recursive(self, tmp_path):
        """Excluded dirs are filtered even when nested inside normal dirs."""
        d = tmp_path / "photos" / "unused"
        d.mkdir(parents=True)
        (d / "img.jpg").write_bytes(b"fake")
        (tmp_path / "photos" / "good.jpg").write_bytes(b"fake")

        result = find_media_files(str(tmp_path), recursive=True)
        basenames = [os.path.basename(f) for f in result]
        assert "good.jpg" in basenames
        assert "img.jpg" not in basenames


# ===========================================================================
# TestGetDisplayName
# ===========================================================================
class TestGetDisplayName:

    def test_flat_returns_basename(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "image.jpg")
        assert get_display_name(filepath, str(tmp_path), recursive=False) == "image.jpg"

    def test_recursive_returns_relative(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "subdir", "image.jpg")
        assert get_display_name(filepath, str(tmp_path), recursive=True) == "subdir/image.jpg"

    def test_recursive_root_file(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "image.jpg")
        assert get_display_name(filepath, str(tmp_path), recursive=True) == "image.jpg"

    def test_recursive_nested_path(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "a", "b", "image.jpg")
        assert get_display_name(filepath, str(tmp_path), recursive=True) == "a/b/image.jpg"


# ===========================================================================
# TestIsInExcludedDir
# ===========================================================================
class TestIsInExcludedDir:

    def test_excluded_dir_unused(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "unused", "img.jpg")
        assert _is_in_excluded_dir(filepath, str(tmp_path)) is True

    def test_excluded_dir_masks(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "masks", "img.png")
        assert _is_in_excluded_dir(filepath, str(tmp_path)) is True

    def test_hidden_dir(self, tmp_path):
        filepath = os.path.join(str(tmp_path), ".hidden", "img.jpg")
        assert _is_in_excluded_dir(filepath, str(tmp_path)) is True

    def test_normal_dir(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "photos", "img.jpg")
        assert _is_in_excluded_dir(filepath, str(tmp_path)) is False

    def test_nested_excluded(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "sub", "unused", "img.jpg")
        assert _is_in_excluded_dir(filepath, str(tmp_path)) is True

    def test_root_file_not_excluded(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "img.jpg")
        assert _is_in_excluded_dir(filepath, str(tmp_path)) is False

    def test_excluded_dir_case_insensitive(self, tmp_path):
        filepath = os.path.join(str(tmp_path), "Unused", "img.jpg")
        assert _is_in_excluded_dir(filepath, str(tmp_path)) is True


# ===========================================================================
# TestFindStraySidecars
# ===========================================================================


def _names(paths, root):
    return sorted(os.path.relpath(p, root).replace(os.sep, '/') for p in paths)


class TestFindStraySidecars:

    def test_paired_files_not_flagged(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "a.txt").write_text("cap")
        (tmp_path / "a-masklabel.png").write_bytes(b"x")
        (tmp_path / "masks").mkdir()
        (tmp_path / "masks" / "a.png").write_bytes(b"x")
        assert find_stray_sidecars(str(tmp_path)) == []

    def test_video_caption_not_flagged(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"x")
        (tmp_path / "clip.txt").write_text("cap")
        (tmp_path / "clip2.flv").write_bytes(b"x")
        (tmp_path / "clip2.txt").write_text("cap")
        assert find_stray_sidecars(str(tmp_path)) == []

    def test_case_mismatch_not_flagged(self, tmp_path):
        (tmp_path / "Photo.JpG").write_bytes(b"x")
        (tmp_path / "photo.txt").write_text("cap")
        (tmp_path / "PHOTO-MaskLabel.png").write_bytes(b"x")
        assert find_stray_sidecars(str(tmp_path)) == []

    def test_orphans_flagged(self, tmp_path):
        (tmp_path / "keep.png").write_bytes(b"x")
        (tmp_path / "gone.txt").write_text("cap")
        (tmp_path / "gone-masklabel.png").write_bytes(b"x")
        (tmp_path / "masks").mkdir()
        (tmp_path / "masks" / "gone.png").write_bytes(b"x")
        (tmp_path / "masks" / "keep.png").write_bytes(b"x")
        assert _names(find_stray_sidecars(str(tmp_path)), tmp_path) == [
            "gone-masklabel.png", "gone.txt", "masks/gone.png"]

    def test_masklabel_is_not_media_for_its_own_caption(self, tmp_path):
        # foo-masklabel.png must not count as an image that "owns" foo-masklabel.txt
        (tmp_path / "foo-masklabel.png").write_bytes(b"x")
        (tmp_path / "foo-masklabel.txt").write_text("cap")
        assert _names(find_stray_sidecars(str(tmp_path)), tmp_path) == [
            "foo-masklabel.png", "foo-masklabel.txt"]

    def test_excluded_dirs_never_scanned(self, tmp_path):
        for d in ["unused", "uncropped", ".hidden"]:
            (tmp_path / d).mkdir()
            (tmp_path / d / "old.txt").write_text("cap")
            (tmp_path / d / "old-masklabel.png").write_bytes(b"x")
        assert find_stray_sidecars(str(tmp_path), recursive=True) == []

    def test_flat_ignores_subdirs(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "orphan.txt").write_text("cap")
        assert find_stray_sidecars(str(tmp_path), recursive=False) == []

    def test_recursive_pairs_per_directory(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.txt").write_text("cap")   # image lives in root, not sub
        (tmp_path / "sub" / "b.jpg").write_bytes(b"x")
        (tmp_path / "sub" / "b.txt").write_text("cap")
        assert _names(find_stray_sidecars(str(tmp_path), recursive=True), tmp_path) == ["sub/a.txt"]
