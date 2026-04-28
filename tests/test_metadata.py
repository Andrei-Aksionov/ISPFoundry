from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ispfoundry.datasets import Metadata, extract_metadata


@pytest.fixture
def valid_params() -> dict[str, Any]:
    """Provides a dictionary of valid parameters for Metadata construction."""
    return {
        "file_path": Path("test.dng"),
        "image_width": 1920,
        "image_height": 1080,
        "black_levels": np.array([64.0, 64.0, 64.0, 64.0], dtype=np.float32),
        "white_level": 1023,
        "color_description": "RGGB",
        "raw_pattern": np.array([[0, 1], [1, 2]]),
        "exposure_time": 0.01,
        "iso": 100,
        "camera_model_name": "TestCamera",
        "cfa_plane_color": "Red,Green,Blue",
        "noise_profile": np.array([1.0, 0.1, 1.0, 0.1, 1.0, 0.1], dtype=np.float32),
    }


class TestAutomatedStructuralChecks:
    """Tests the generic 'type-based' validation in __post_init__."""

    def test_mandatory_none_check(self, valid_params):
        """Ensures non-optional fields cannot be None."""
        valid_params["image_width"] = None
        with pytest.raises(TypeError, match="mandatory but received 'None'"):
            Metadata(**valid_params)

    def test_optional_none_allowed(self, valid_params):
        """Ensures fields marked | None CAN be None."""
        valid_params["noise_profile"] = None
        # Should not raise any error
        metadata = Metadata(**valid_params)
        assert metadata.noise_profile is None

    def test_empty_string_check(self, valid_params):
        """Ensures strings aren't just whitespace."""
        valid_params["color_description"] = "   "
        with pytest.raises(ValueError, match="cannot be an empty or whitespace-only string"):
            Metadata(**valid_params)

    def test_numpy_type_enforcement(self, valid_params):
        """
        Verifies that passing a list instead of ndarray raises TypeError.
        This specifically catches errors when using replace().
        """
        valid_params["raw_pattern"] = [[0, 1], [1, 2]]  # List instead of np.ndarray
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            Metadata(**valid_params)

    def test_empty_numpy_array_check(self, valid_params):
        """Ensures numpy arrays have actual data."""
        valid_params["raw_pattern"] = np.array([])
        with pytest.raises(ValueError, match="is an empty NumPy array"):
            Metadata(**valid_params)


class TestImmutability:
    """Tests that the 'frozen' status and 'readonly' flags are enforced."""

    def test_frozen_attributes(self, valid_params):
        """Verifies that field values cannot be reassigned (FrozenInstanceError)."""
        metadata = Metadata(**valid_params)
        with pytest.raises(FrozenInstanceError):
            metadata.iso = 200  # ty:ignore[invalid-assignment]

    def test_numpy_readonly_flag(self, valid_params):
        """Verifies that internal array data cannot be modified."""
        metadata = Metadata(**valid_params)

        assert metadata.black_levels.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            metadata.black_levels[0] = 100.0

    def test_replace_validation_bypass_prevention(self, valid_params):
        """Ensures replace() still triggers __post_init__ and catches bad types."""
        metadata = Metadata(**valid_params)

        # Attempting to 'smuggle' a list through replace()
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            replace(metadata, raw_pattern=[[0, 0], [0, 0]])


class TestDomainValidation:
    """Tests image-processing specific logic (Geometry, Levels, ISP Requirements)."""

    @pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-1, 100)])
    def test_invalid_dimensions(self, valid_params, width, height):
        valid_params["image_width"] = width
        valid_params["image_height"] = height
        with pytest.raises(ValueError, match="Invalid dimensions"):
            Metadata(**valid_params)

    @pytest.mark.parametrize("white_level", (100, 101))
    def test_black_level_white_white(self, valid_params, white_level):
        """Tests saturation/normalization safety."""
        valid_params["white_level"] = white_level
        valid_params["black_levels"] = np.array([101, 101, 101, 101])
        with pytest.raises(ValueError, match="strictly less than white level"):
            Metadata(**valid_params)

    def test_invalid_cfa_plane(self, valid_params):
        """Checks ISP-specific hardware layout requirement."""
        valid_params["cfa_plane_color"] = "Cyan,Magenta,Yellow"
        with pytest.raises(ValueError, match="ISP expects 'Red,Green,Blue'"):
            Metadata(**valid_params)

    def test_invalid_noise_profile_shape(self, valid_params):
        """Noise profile must be exactly 6 elements (3 channels * [scale, offset])."""
        valid_params["noise_profile"] = np.array([1.0, 0.1])
        with pytest.raises(ValueError, match="must contain 6 values"):
            Metadata(**valid_params)


@patch("ispfoundry.datasets.metadata.get_exif_metadata")
@patch("rawpy.imread")
class TestMetadataFactory:
    """Tests the extract_metadata I/O and merging logic."""

    def test_successful_extraction(self, mock_rawpy, mock_get_exif, valid_params):
        """Verifies data merging from EXIF and RawPy."""
        # Setup Mocks
        mock_get_exif.return_value = [{"ImageWidth": 4000, "ExposureTime": "1/125", "WhiteLevel": 16383, "ISO": 400}]

        raw_inst = mock_rawpy.return_value.__enter__.return_value
        raw_inst.sizes.height = 3000  # Falling back for height
        raw_inst.black_level_per_channel = [128, 128, 128, 128]
        raw_inst.color_desc = b"RGBG"
        raw_inst.raw_pattern = [[0, 1], [1, 2]]

        metadata = extract_metadata(Path("dummy.dng"))

        assert metadata.image_width == 4000
        assert metadata.image_height == 3000
        assert metadata.exposure_time == 0.008
        assert metadata.iso == 400
        assert metadata.color_description == "RGBG"

    def test_exposure_time_parsing_variants(self, mock_rawpy, mock_get_exif):
        """Tests that varied exposure string formats don't crash the factory."""
        mock_raw_inst = mock_rawpy.return_value.__enter__.return_value
        # Set minimal valid raw_obj attributes to pass initialization
        mock_raw_inst.sizes.width = 100
        mock_raw_inst.sizes.height = 100
        mock_raw_inst.black_level_per_channel = [0, 0, 0, 0]
        mock_raw_inst.white_level = 1000
        mock_raw_inst.color_desc = b"RGBG"
        mock_raw_inst.raw_pattern = [[0, 1], [1, 2]]

        # Test decimal string
        mock_get_exif.return_value = [{"ExposureTime": "0.01"}]
        metadata = extract_metadata(Path("dummy.dng"))
        assert metadata.exposure_time == 0.01

        # Test fraction string
        mock_get_exif.return_value = [{"ExposureTime": "1/100"}]
        metadata = extract_metadata(Path("dummy.dng"))
        assert metadata.exposure_time == 0.01

    def test_iso_missing_fallback_warning(self, mock_rawpy, mock_get_exif):
        """Tests the ISO default behavior and logger warning."""
        mock_get_exif.return_value = [{"ExposureTime": "0.01"}]  # ISO missing

        # Setup minimal raw_obj
        mock_raw_inst = mock_rawpy.return_value.__enter__.return_value
        mock_raw_inst.sizes.width = 100
        mock_raw_inst.sizes.height = 100
        mock_raw_inst.black_level_per_channel = [0, 0, 0, 0]
        mock_raw_inst.white_level = 1000
        mock_raw_inst.color_desc = b"RGBG"
        mock_raw_inst.raw_pattern = [[0, 1], [1, 2]]

        metadata = extract_metadata(Path("dummy.dng"))
        assert metadata.iso == 100

    def test_factory_invalid_exposure_fraction(self, mock_rawpy, mock_get_exif):
        """Tests error handling for zero-division in EXIF fractions."""
        mock_get_exif.return_value = [{"BlackLevel": np.array([101, 101, 101, 101]), "ExposureTime": "1/0"}]
        with pytest.raises(ValueError, match="Invalid exposure time fraction"):
            extract_metadata(Path("dummy.dng"))

    def test_factory_unsupported_black_level_type(self, mock_rawpy, mock_get_exif):
        """Tests error handling when BlackLevel is an unexpected object type."""
        mock_get_exif.return_value = [{"BlackLevel": {"invalid": "dict"}}]
        with pytest.raises(TypeError, match="Unsupported BlackLevel type"):
            extract_metadata(Path("dummy.dng"))
