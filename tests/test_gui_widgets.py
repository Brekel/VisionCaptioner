"""Tests for gui_widgets.py - Coordinate conversion math."""

import pytest
from unittest.mock import MagicMock, PropertyMock

from gui_widgets import ResizableImageLabel


class TestWidgetToImageCoords:
    """Tests for ResizableImageLabel._widget_to_image_coords()"""

    def _make_label(self, img_w, img_h, scaled_w, scaled_h, widget_w, widget_h):
        """Create a ResizableImageLabel with controlled dimensions."""
        label = ResizableImageLabel.__new__(ResizableImageLabel)
        label.is_painting = False
        label.cursor_pos = None
        label.cursor_size_px = 0

        # Mock current_pixmap (the original full-res image)
        mock_original = MagicMock()
        mock_original.width.return_value = img_w
        mock_original.height.return_value = img_h
        label.current_pixmap = mock_original

        # Mock pixmap() (the scaled version currently displayed)
        mock_scaled = MagicMock()
        mock_scaled.width.return_value = scaled_w
        mock_scaled.height.return_value = scaled_h
        mock_scaled.isNull.return_value = False
        label.pixmap = MagicMock(return_value=mock_scaled)

        # Mock widget dimensions
        label.width = MagicMock(return_value=widget_w)
        label.height = MagicMock(return_value=widget_h)

        return label

    def test_1_to_1_center_click(self):
        """Image 100x100, widget 200x200, scaled 100x100 (centered).
        offset = (200-100)/2 = 50. Click at (100,100) -> pixmap(50,50) -> image(50,50)."""
        label = self._make_label(100, 100, 100, 100, 200, 200)

        pos = MagicMock()
        pos.x.return_value = 100
        pos.y.return_value = 100

        result = label._widget_to_image_coords(pos)
        assert result == (50, 50)

    def test_scaled_coordinates(self):
        """Image 1000x500, widget 200x100, scaled fills widget exactly.
        Click at center (100,50). ratio=5x, image coords = (500, 250)."""
        label = self._make_label(1000, 500, 200, 100, 200, 100)

        pos = MagicMock()
        pos.x.return_value = 100
        pos.y.return_value = 50

        result = label._widget_to_image_coords(pos)
        assert result == (500, 250)

    def test_aspect_ratio_offset(self):
        """Image 1000x500, widget 400x400. Scaled pixmap 400x200 (aspect preserved).
        offset_y = (400-200)/2 = 100. Click at (200, 200):
        pixmap_y = 200-100 = 100, ratio_y = 500/200 = 2.5, image_y = 250.
        pixmap_x = 200-0 = 200, ratio_x = 1000/400 = 2.5, image_x = 500."""
        label = self._make_label(1000, 500, 400, 200, 400, 400)

        pos = MagicMock()
        pos.x.return_value = 200
        pos.y.return_value = 200

        result = label._widget_to_image_coords(pos)
        assert result == (500, 250)

    def test_returns_none_when_no_pixmap(self):
        label = ResizableImageLabel.__new__(ResizableImageLabel)
        label.is_painting = False
        label.cursor_pos = None
        label.cursor_size_px = 0
        label.current_pixmap = None
        label.pixmap = MagicMock(return_value=None)

        pos = MagicMock()
        pos.x.return_value = 50
        pos.y.return_value = 50

        result = label._widget_to_image_coords(pos)
        assert result is None

    def test_out_of_bounds_still_returns(self):
        """Clicks outside the image area should still return coordinates
        (for brush edge drawing at image borders)."""
        label = self._make_label(100, 100, 100, 100, 200, 200)

        pos = MagicMock()
        pos.x.return_value = 0  # left edge of widget, outside image area
        pos.y.return_value = 0

        result = label._widget_to_image_coords(pos)
        assert result is not None
        # offset = 50, so pixmap_x = 0-50 = -50, ratio=1, image_x = -50
        assert result[0] == -50
        assert result[1] == -50

    def test_returns_none_when_pixmap_is_null(self):
        """pixmap().isNull() returning True should return None."""
        label = ResizableImageLabel.__new__(ResizableImageLabel)
        label.is_painting = False
        label.cursor_pos = None
        label.cursor_size_px = 0
        label.current_pixmap = MagicMock()  # current_pixmap exists...

        mock_pixmap = MagicMock()
        mock_pixmap.isNull.return_value = True  # ...but pixmap() is null
        label.pixmap = MagicMock(return_value=mock_pixmap)

        pos = MagicMock()
        pos.x.return_value = 50
        pos.y.return_value = 50

        result = label._widget_to_image_coords(pos)
        assert result is None

    def test_top_left_corner(self):
        """Click exactly at the image origin (top-left of displayed image)."""
        label = self._make_label(200, 100, 200, 100, 200, 100)

        pos = MagicMock()
        pos.x.return_value = 0
        pos.y.return_value = 0

        result = label._widget_to_image_coords(pos)
        # No offset (scaled == widget), ratio 1:1
        assert result == (0, 0)

    def test_bottom_right_corner(self):
        """Click at the bottom-right pixel of the displayed image."""
        label = self._make_label(200, 100, 200, 100, 200, 100)

        pos = MagicMock()
        pos.x.return_value = 199
        pos.y.return_value = 99

        result = label._widget_to_image_coords(pos)
        assert result == (199, 99)
