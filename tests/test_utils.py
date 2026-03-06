import math
import os

import numpy as np
import pytest

from utils import (
    decode_cfa,
    find_best_figsize,
    find_best_layout,
    get_exif_metadata,
    get_git_root,
    save_ndarray_as_jpg,
)


class TestGetGitRoot:
    def test_get_git_root_success(self, tmp_path):
        """Test that get_git_root works when .git directory exists."""
        # Create a temporary directory with .git
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        working_dir = tmp_path / "src"
        working_dir.mkdir()

        os.chdir(working_dir)
        root = get_git_root()

        assert root == tmp_path

    def test_get_git_root_no_git(self, tmp_path):
        """Test that get_git_root raises FileNotFoundError when no .git exists."""
        # Create a temporary directory without .git
        os.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="No .git directory found"):
            get_git_root()


class TestDecodeCFA:
    """
    There are 4 CFA pattern for traditional RGB sensors.

    RGGB: Red-Green, Green-Blue (The most common standard).
    BGGR: Blue-Green, Green-Red.
    GRBG: Green-Red, Blue-Green.
    GBRG: Green-Blue, Green-Red.
    """

    def test_decode_rggb_pattern(self):
        """Test decoding RGGB pattern."""
        raw_pattern = np.array([
            [0, 1],
            [3, 2],
        ])

        result = decode_cfa("RGBG", raw_pattern)
        expected = ["R", "Gr", "Gb", "B"]

        assert result == expected

    def test_decode_with_bggr(self):
        """Test decoding BGGR pattern."""
        raw_pattern = np.array([
            [2, 3],
            [1, 0],
        ])

        result = decode_cfa("RGBG", raw_pattern)
        expected = ["B", "Gb", "Gr", "R"]

        assert result == expected

    def test_decode_with_grbg(self):
        """Test decoding GRBG pattern."""
        raw_pattern = np.array([
            [1, 0],
            [2, 3],
        ])

        result = decode_cfa("RGBG", raw_pattern)
        expected = ["Gr", "R", "B", "Gb"]

        assert result == expected

    def test_decode_with_gbrg(self):
        """Test decoding GBRG pattern."""
        raw_pattern = np.array([
            [3, 2],
            [0, 1],
        ])

        result = decode_cfa("RGBG", raw_pattern)
        expected = ["Gb", "B", "R", "Gr"]

        assert result == expected

    def test_decode_invalid_length(self):
        """Test decoding with mismatched length."""
        raw_pattern = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="The length of color_description must match"):
            decode_cfa("rggb", raw_pattern)


class TestFindBestLayout:
    def test_find_layout_single_image(self):
        """Test layout for single image."""
        nrow, ncol = find_best_layout(1)
        assert nrow == 1
        assert ncol == 1

    def test_find_layout_two_images(self):
        """Test layout for two images."""
        nrow, ncol = find_best_layout(2)
        assert nrow == 1
        assert ncol == 2

    def test_find_layout_three_images(self):
        """Test layout for three images."""
        nrow, ncol = find_best_layout(3)
        assert nrow == 1
        assert ncol == 3

    def test_find_layout_four_images(self):
        """Test layout for four images (should be 2x2)."""
        nrow, ncol = find_best_layout(4)
        assert nrow == 2
        assert ncol == 2

    def test_find_layout_five_images(self):
        """Test layout for five images."""
        nrow, ncol = find_best_layout(5)
        assert nrow == 2
        assert ncol == 3

    def test_find_layout_with_max_per_row(self):
        """Test layout with custom max_per_row."""
        nrow, ncol = find_best_layout(7, max_per_row=2)
        # Should be 4 rows x 2 columns (8 slots, 1 empty)
        assert nrow == 4
        assert ncol == 2


class TestFindBestFigsize:
    def test_find_figsize_default(self):
        """Test finding figure size with default width."""
        images = [np.random.rand(100, 100)]
        fig_width, fig_height = find_best_figsize(images, nrow=1, ncol=1)

        assert fig_width == 6.0  # Default base width
        assert fig_height == 6.0

    def test_find_figsize_custom_width(self):
        """Test finding figure size with custom width."""
        images = [np.random.rand(100, 100)]
        fig_width, fig_height = find_best_figsize(images, nrow=2, ncol=2, inch_width_pre_image=4)

        assert fig_width == 8.0  # 4 * 2 columns
        assert fig_height == 8.0  # 4 * 1 aspect ratio * 2 rows

    def test_find_figsize_different_aspect_ratio(self):
        """Test finding figure size with different aspect ratio."""
        images = [np.random.rand(200, 100)]  # Aspect ratio 2:1
        fig_width, fig_height = find_best_figsize(images, nrow=1, ncol=1)

        assert fig_width == 6.0
        assert math.isclose(fig_height, 12.0)  # 6 * 2 aspect ratio


class TestSaveNdarrayAsJpg:
    def test_save_normalized_image(self, tmp_path):
        """Test saving normalized image."""
        img = np.random.rand(100, 100).astype(np.float32)
        path = tmp_path / "test.jpg"

        save_ndarray_as_jpg(img, path)

        assert path.exists()

    def test_save_unnormalized_image(self, tmp_path):
        """Test saving unnormalized image (should be normalized automatically)."""
        img = np.random.rand(100, 100) * 2.0  # Values > 1.0
        path = tmp_path / "test.jpg"

        save_ndarray_as_jpg(img, path)

        assert path.exists()

    def test_save_negative_values(self, tmp_path):
        """Test saving image with negative values."""
        img = np.random.rand(100, 100).astype(np.float32) - 0.5  # Values in [-0.5, 0.5]
        path = tmp_path / "test.jpg"

        save_ndarray_as_jpg(img, path)

        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test that save creates parent directories if needed."""
        img = np.random.rand(100, 100).astype(np.float32)
        path = tmp_path / "subdir" / "test.jpg"

        save_ndarray_as_jpg(img, path)

        assert path.exists()


class TestGetExifMetadata:
    def test_get_exif_metadata(self, tmp_path):
        """Test getting EXIF metadata from a file."""
        # Create a simple image
        img = np.random.rand(100, 100).astype(np.float32)
        path = tmp_path / "test.jpg"

        # Save the image
        save_ndarray_as_jpg(img, path)

        # Try to get EXIF metadata (may fail if exiftool not installed)
        try:
            metadata = get_exif_metadata(path)
            assert isinstance(metadata, list)
            assert len(metadata) == 1
            assert isinstance(metadata[0], dict)
            assert metadata[0] != {}
        except RuntimeError:
            # Expected if exiftool is not installed
            pytest.raises(RuntimeError, match="ExifTool needs to be installed")

    def test_get_exif_metadata_sequence(self, tmp_path):
        """Test getting EXIF metadata from multiple files."""
        paths = []
        for i in range(3):
            img = np.random.rand(100, 100).astype(np.float32)
            path = tmp_path / f"test_{i}.jpg"
            save_ndarray_as_jpg(img, path)
            paths.append(path)

        try:
            metadata = get_exif_metadata(paths)
            assert isinstance(metadata, list)
            assert len(metadata) == 3
            assert all(isinstance(mtd, dict) for mtd in metadata)
            assert all(mtd != {} for mtd in metadata)
        except RuntimeError:
            # Expected if exiftool is not installed
            pytest.raises(RuntimeError, match="ExifTool needs to be installed")
