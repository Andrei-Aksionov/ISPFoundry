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

    @patch("rawpy.imread")
    @patch("tifffile.imread")
    @patch("ispfoundry.datasets.dataset_loader.extract_metadata")
    def test_load_data_integration(self, mock_metadata, mock_tiff, mock_rawpy, mock_folder):
        """Tests that load_data populates all class attributes."""
        # Minimal mocks to allow the calls to succeed
        mock_metadata.return_value = [{}, {}]
        mock_rawpy.return_value.__enter__.return_value = MagicMock()
        mock_tiff.return_value = np.zeros((2, 2))

        loader = DatasetLoader(mock_folder)
        loader.load_data()

        assert loader.raw_images is not None
        assert len(loader.metadata) == 2
        assert len(loader.lsc_maps) == 2

    @patch("tifffile.imread")
    def test_get_lsc_maps_success(self, mock_tifffile, mock_folder):
        """Tests successful loading of all LSC maps."""
        # Mock the return of tifffile.imread
        mock_tifffile.return_value = np.ones((4, 4), dtype=np.uint16)

        loader = DatasetLoader(mock_folder, dtype=np.float32)
        maps = loader.get_lens_shading_correction_maps()

        assert len(maps) == 2
        assert all(isinstance(m, np.ndarray) for m in maps)
        assert maps[0].dtype == np.float32

    @patch("tifffile.imread")
    def test_get_lsc_maps_raises_error(self, mock_tifffile, mock_folder):
        """Tests that a missing LSC map raises a FileNotFoundError."""
        # Only create one DNG but NO tiff file
        (mock_folder / "payload_999.dng").touch()

        loader = DatasetLoader(mock_folder)

        # Verify that the specific error is raised
        with pytest.raises(FileNotFoundError, match="LSC map not found"):
            loader.get_lens_shading_correction_maps()
