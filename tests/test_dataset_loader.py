from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ispfoundry.datasets.dataset_loader import DatasetLoader


@pytest.fixture
def mock_folder(tmp_path):
    """Creates a temporary folder with dummy filenames."""
    folder = tmp_path / "test_data"
    folder.mkdir()
    # Create dummy files to be discovered by glob
    (folder / "payload_001.dng").touch()
    (folder / "payload_002.dng").touch()
    (folder / "lens_shading_map_001.tiff").touch()
    (folder / "lens_shading_map_002.tiff").touch()
    return folder


class TestDatasetLoader:
    def test_init_path_handling(self, mock_folder):
        """Tests that paths and dtypes are correctly initialized."""
        loader = DatasetLoader(mock_folder, dtype=np.float64)
        assert len(loader.dng_file_paths) == 2
        assert loader.dtype == np.float64
        assert isinstance(loader.folder_path, Path)

    @patch("rawpy.imread")
    def test_get_raw_images_dtype(self, mock_rawpy, mock_folder):
        """Tests that raw images are loaded and cast to the correct dtype."""
        # Setup mock rawpy object
        mock_raw_obj = MagicMock()
        mock_raw_obj.raw_image = np.array([[10, 20], [30, 40]], dtype=np.uint16)
        mock_rawpy.return_value.__enter__.return_value = mock_raw_obj

        loader = DatasetLoader(mock_folder, dtype=np.float32)
        images = loader.get_raw_images()

        assert images.shape == (2, 2, 2)
        assert images.dtype == np.float32
        assert images[0, 0, 0] == 10.0

    @patch("ispfoundry.datasets.dataset_loader.get_exif_metadata")
    @patch("rawpy.imread")
    def test_get_metadata_conditional_fill(self, mock_rawpy, mock_exif, mock_folder):
        """Tests that black/white levels are only added if missing."""
        # 1. Mock EXIF data: first image has levels, second is missing them
        mock_exif.return_value = [
            {
                "black_level": [1, 1, 1, 1],
                "white_level": 255,
            },
            {
                "some_other_key": "val",
            },
        ]

        # 2. Mock rawpy data
        mock_raw_obj = MagicMock()
        mock_raw_obj.color_desc = b"RGBG"
        mock_raw_obj.raw_pattern = [[0, 1], [1, 2]]
        mock_raw_obj.black_level_per_channel = [50, 50, 50, 50]
        mock_raw_obj.white_level = 1000
        mock_rawpy.return_value.__enter__.return_value = mock_raw_obj

        loader = DatasetLoader(mock_folder)
        metadata = loader.get_metadata()

        # Image 0: Should keep original EXIF values
        assert metadata[0]["black_level"] == [1, 1, 1, 1]
        assert metadata[0]["white_level"] == 255

        # Image 1: Should pull from rawpy because they were missing
        assert metadata[1]["black_level"] == [50, 50, 50, 50]
        assert metadata[1]["white_level"] == 1000
        assert metadata[1]["color_desc"] == "RGBG"

    @patch("tifffile.imread")
    def test_get_lsc_maps_alignment(self, mock_tifffile, mock_folder):
        """Tests that missing LSC maps return None and maintain index alignment."""
        # Delete one of the LSC maps to test the 'missing file' logic
        (mock_folder / "lens_shading_map_002.tiff").unlink()

        mock_tifffile.return_value = np.ones((4, 4), dtype=np.uint16)

        loader = DatasetLoader(mock_folder, dtype=np.float32)
        maps = loader.get_lens_shading_correction_maps()

        assert len(maps) == 2
        assert isinstance(maps[0], np.ndarray)
        assert maps[0].dtype == np.float32
        assert maps[1] is None  # Second map was deleted

    @patch("rawpy.imread")
    @patch("tifffile.imread")
    @patch("ispfoundry.datasets.dataset_loader.get_exif_metadata")
    def test_load_data_integration(self, mock_exif, mock_tiff, mock_rawpy, mock_folder):
        """Tests that load_data populates all class attributes."""
        # Minimal mocks to allow the calls to succeed
        mock_exif.return_value = [{}, {}]
        mock_rawpy.return_value.__enter__.return_value = MagicMock()
        mock_tiff.return_value = np.zeros((2, 2))

        loader = DatasetLoader(mock_folder)
        loader.load_data()

        assert loader.raw_images is not None
        assert len(loader.metadata) == 2
        assert len(loader.lsc_maps) == 2
