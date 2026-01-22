import numpy as np
import pytest

from pipeline_steps.black_level_subtraction import normalize_image, retrieve_black_levels, subtract_black_levels


@pytest.fixture
def sample_raw_image():
    """Fixture for a sample RGGB Bayer raw image (4x4 for simplicity)."""
    # RGGB pattern: R Gr, Gb, B
    return np.array(
        [
            [100, 110, 120, 130],
            [140, 150, 160, 170],
            [180, 190, 200, 210],
            [220, 230, 240, 250],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_metadata():
    """Fixture for sample metadata with BlackLevel and WhiteLevel."""
    return {
        "BlackLevel": [50, 60, 70, 80],  # R, Gr, Gb, B
        "WhiteLevel": 1000,
    }


class TestRetrieveBlackLevels:
    def test_retrieve_from_metadata_list(self, sample_raw_image, sample_metadata):
        black_levels = retrieve_black_levels(sample_raw_image, sample_metadata)
        expected = np.array([50, 60, 70, 80], dtype=np.float32)
        np.testing.assert_array_equal(black_levels, expected)

    def test_retrieve_from_metadata_string(self, sample_raw_image):
        metadata = {"BlackLevel": "50 60 70 80", "WhiteLevel": 1000}
        black_levels = retrieve_black_levels(sample_raw_image, metadata)
        expected = np.array([50, 60, 70, 80], dtype=float)
        np.testing.assert_array_equal(black_levels, expected)

    def test_retrieve_from_metadata_array(self, sample_raw_image):
        metadata = {"BlackLevel": np.array([50, 60, 70, 80]), "WhiteLevel": 1000}
        black_levels = retrieve_black_levels(sample_raw_image, metadata)
        expected = np.array([50, 60, 70, 80], dtype=np.float32)
        np.testing.assert_array_equal(black_levels, expected)

    def test_estimate_from_cfa_minima(self, sample_raw_image):
        metadata = {"WhiteLevel": 1000}  # No BlackLevel
        black_levels = retrieve_black_levels(sample_raw_image, metadata)
        # For RGGB: R min=100, Gr min=110, Gb min=140, B min=150
        expected = np.array([100, 110, 140, 150], dtype=np.float32)
        np.testing.assert_array_equal(black_levels, expected)

    def test_estimate_from_cfa_minima_all_zero(self, sample_raw_image):
        metadata = {"BlackLevel": [0, 0, 0, 0], "WhiteLevel": 1000}
        black_levels = retrieve_black_levels(sample_raw_image, metadata)
        expected = np.array([100, 110, 140, 150], dtype=np.float32)
        np.testing.assert_array_equal(black_levels, expected)

    def test_invalid_size(self, sample_raw_image):
        metadata = {"BlackLevel": [50, 60, 70]}  # Only 3
        with pytest.raises(ValueError, match="Expected 4 black level values"):
            retrieve_black_levels(sample_raw_image, metadata)

    def test_black_level_too_high(self, sample_raw_image):
        metadata = {"BlackLevel": [50, 60, 70, 1001], "WhiteLevel": 1000}
        with pytest.raises(ValueError, match="Black levels cannot be larger"):
            retrieve_black_levels(sample_raw_image, metadata)

    def test_invalid_white_level_none(self, sample_raw_image):
        metadata = {"BlackLevel": [50, 60, 70, 80]}
        with pytest.raises(ValueError, match="Metadata should contain a valid WhiteLevel"):
            normalize_image(sample_raw_image, metadata)

    def test_invalid_white_level_zero(self, sample_raw_image):
        metadata = {"BlackLevel": [50, 60, 70, 80], "WhiteLevel": 0}
        with pytest.raises(ValueError, match="Metadata should contain a valid WhiteLevel"):
            normalize_image(sample_raw_image, metadata)


class TestSubtractBlackLevels:
    def test_subtract_inplace_false(self, sample_raw_image, sample_metadata):
        original = sample_raw_image.copy()
        result = subtract_black_levels(sample_raw_image, sample_metadata, inplace=False)
        assert not np.shares_memory(result, sample_raw_image)
        # Check subtraction
        expected = original.copy()
        black_levels = sample_metadata["BlackLevel"]
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] -= black_level
        np.testing.assert_array_almost_equal(result, expected)

    def test_subtract_inplace_true(self, sample_raw_image, sample_metadata):
        original = sample_raw_image.copy()
        result = subtract_black_levels(sample_raw_image, sample_metadata, inplace=True)
        assert np.shares_memory(result, sample_raw_image)
        expected = original.copy()
        black_levels = sample_metadata["BlackLevel"]
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] -= black_level

        np.testing.assert_array_almost_equal(result, expected)

    def test_unsigned_int_error(self):
        image = np.array([[100, 110], [140, 150]], dtype=np.uint16)
        metadata = {"BlackLevel": [50, 60, 70, 80]}
        with pytest.raises(ValueError, match="Raw image must not be of unsigned integer type"):
            subtract_black_levels(image, metadata)


class TestNormalizeImage:
    def test_normalize_inplace_false(self, sample_raw_image, sample_metadata):
        # First subtract black levels
        subtracted = subtract_black_levels(sample_raw_image, sample_metadata, inplace=False)
        normalized = normalize_image(subtracted, sample_metadata, inplace=False)
        assert not np.shares_memory(normalized, subtracted)
        # Normalize: (value) / (white - black)
        expected = subtracted.copy()
        black_levels = sample_metadata["BlackLevel"]
        white_level = sample_metadata["WhiteLevel"]
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] /= white_level - black_level

        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_inplace_true(self, sample_raw_image, sample_metadata):
        subtracted = subtract_black_levels(sample_raw_image, sample_metadata, inplace=False)
        original_subtracted = subtracted.copy()
        normalized = normalize_image(subtracted, sample_metadata, inplace=True)
        assert np.shares_memory(normalized, subtracted)
        expected = original_subtracted.copy()
        black_levels = sample_metadata["BlackLevel"]
        white_level = sample_metadata["WhiteLevel"]
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] /= white_level - black_level
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_invalid_white_level_none(self, sample_raw_image):
        metadata = {"BlackLevel": [50, 60, 70, 80]}
        with pytest.raises(ValueError, match="Metadata should contain a valid WhiteLevel"):
            normalize_image(sample_raw_image, metadata)

    def test_invalid_white_level_zero(self, sample_raw_image):
        metadata = {"BlackLevel": [50, 60, 70, 80], "WhiteLevel": 0}
        with pytest.raises(ValueError, match="Metadata should contain a valid WhiteLevel"):
            normalize_image(sample_raw_image, metadata)
