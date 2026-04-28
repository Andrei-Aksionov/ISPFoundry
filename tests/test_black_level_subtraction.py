from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ispfoundry.datasets import Metadata
from ispfoundry.pipeline_steps.black_level_subtraction import (
    apply_black_level_subtraction,
    normalize_image,
    retrieve_black_levels,
    subtract_black_levels,
)


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
    """Fixture for sample metadata."""
    return Metadata(
        file_path=Path("test_file_path"),
        image_width=2,
        image_height=2,
        black_levels=np.array([50, 60, 70, 80]),  # R, Gr, Gb, B
        white_level=1000,
        color_description="RGBG",
        raw_pattern=np.array([[2, 2], [1, 0]]),
        exposure_time=0.1,
        iso=100,
        cfa_plane_color="Red,Green,Blue",
        noise_profile=None,
        camera_model_name="test_camera",
    )


class TestRetrieveBlackLevels:
    def test_estimate_from_cfa_minima(self, sample_raw_image, sample_metadata):
        metadata = replace(sample_metadata, black_levels=np.zeros(4))
        black_levels = retrieve_black_levels(sample_raw_image, metadata)
        # For RGGB: R min=100, Gr min=110, Gb min=140, B min=150
        expected = np.array([100, 110, 140, 150], dtype=np.float32)
        np.testing.assert_array_equal(black_levels, expected)

    def test_estimate_from_cfa_minima_all_zero(self, sample_raw_image, sample_metadata):
        metadata = replace(sample_metadata, black_levels=np.zeros(4))
        black_levels = retrieve_black_levels(sample_raw_image, metadata)
        expected = np.array([100, 110, 140, 150], dtype=np.float32)
        np.testing.assert_array_equal(black_levels, expected)


class TestSubtractBlackLevels:
    def test_subtract_inplace_false(self, sample_raw_image, sample_metadata):
        original = sample_raw_image.copy()
        result = subtract_black_levels(sample_raw_image, sample_metadata, inplace=False)
        assert not np.shares_memory(result, sample_raw_image)
        # Check subtraction
        expected = original.copy()
        black_levels = sample_metadata.black_levels
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] -= black_level
        np.testing.assert_array_equal(result, expected)

    def test_subtract_inplace_true(self, sample_raw_image, sample_metadata):
        original = sample_raw_image.copy()
        result = subtract_black_levels(sample_raw_image, sample_metadata, inplace=True)
        assert np.shares_memory(result, sample_raw_image)
        expected = original.copy()
        black_levels = sample_metadata.black_levels
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] -= black_level

        np.testing.assert_array_equal(result, expected)

    def test_unsigned_int_error(self, sample_metadata):
        image = np.array([[100, 110], [140, 150]], dtype=np.uint16)
        with pytest.raises(ValueError, match="Raw image must not be of unsigned integer type"):
            subtract_black_levels(image, sample_metadata)

    def test_negative_values_are_preserved(self, sample_raw_image, sample_metadata):
        # Ensuring at least one channel value is negative after subtraction
        red_bl = sample_raw_image[::2, ::2].max() + 10
        new_black_levels = np.append(red_bl, sample_metadata.black_levels[1:])
        metadata = replace(sample_metadata, black_levels=new_black_levels)
        result = subtract_black_levels(sample_raw_image, metadata)
        expected = sample_raw_image.copy()
        black_levels = metadata.black_levels
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] -= black_level

        np.testing.assert_array_equal(result, expected)
        assert np.any(result < 0)

    def test_negative_values_are_preserved_custom_data(self, sample_metadata):
        metadata = replace(sample_metadata, black_levels=np.array([110, 120, 150, 160]))
        raw_image = np.array(
            [
                [100, 110, 120, 130],
                [140, 150, 160, 170],
                [180, 190, 200, 210],
                [220, 230, 240, 250],
            ],
            dtype=np.float32,
        )
        result = subtract_black_levels(raw_image, metadata)
        expected = np.array(
            [
                [-10.0, -10.0, 10.0, 10.0],
                [-10.0, -10.0, 10.0, 10.0],
                [70.0, 70.0, 90.0, 90.0],
                [70.0, 70.0, 90.0, 90.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(result, expected)


class TestNormalizeImage:
    def test_normalize_inplace_false(self, sample_raw_image, sample_metadata):
        # First subtract black levels
        subtracted = subtract_black_levels(sample_raw_image, sample_metadata, inplace=False)
        normalized = normalize_image(subtracted, sample_metadata, inplace=False)
        assert not np.shares_memory(normalized, subtracted)
        expected = subtracted.copy()
        black_levels = sample_metadata.black_levels
        white_level = sample_metadata.white_level
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] /= white_level - black_level

        np.testing.assert_array_equal(normalized, expected)

    def test_normalize_inplace_true(self, sample_raw_image, sample_metadata):
        subtracted = subtract_black_levels(sample_raw_image, sample_metadata, inplace=False)
        original_subtracted = subtracted.copy()
        normalized = normalize_image(subtracted, sample_metadata, inplace=True)
        assert np.shares_memory(normalized, subtracted)
        expected = original_subtracted.copy()
        black_levels = sample_metadata.black_levels
        white_level = sample_metadata.white_level
        for idx, black_level in enumerate(black_levels):
            row_offset, col_offset = divmod(idx, 2)
            expected[row_offset::2, col_offset::2] /= white_level - black_level
        np.testing.assert_array_equal(normalized, expected)


class TestApplyBlackLevelSubtraction:
    def test_apply_black_level_subtraction(self, sample_raw_image, sample_metadata):
        result_images = apply_black_level_subtraction(sample_raw_image[None, ...], [sample_metadata])
        expected_normalized = normalize_image(subtract_black_levels(sample_raw_image, sample_metadata), sample_metadata)
        np.testing.assert_array_equal(result_images[0], expected_normalized)

    def test_apply_black_level_subtraction_inplace_true(self, sample_raw_image, sample_metadata):
        result_images = apply_black_level_subtraction(sample_raw_image[None, ...], [sample_metadata], inplace=True)
        assert np.shares_memory(sample_raw_image, result_images[0])

    def test_apply_black_level_subtraction_inplace_false(self, sample_raw_image, sample_metadata):
        result_images = apply_black_level_subtraction(sample_raw_image[None, ...], [sample_metadata], inplace=False)
        assert not np.shares_memory(sample_raw_image, result_images[0])

    def test_apply_black_level_subtraction_multiple_images(self, sample_metadata):
        raw_images = np.array(
            [
                [
                    [100, 110, 120, 130],
                    [140, 150, 160, 170],
                    [180, 190, 200, 210],
                    [220, 230, 240, 250],
                ],
                [
                    [200, 210, 220, 230],
                    [240, 250, 260, 270],
                    [280, 290, 300, 310],
                    [320, 330, 340, 350],
                ],
            ],
            dtype=np.float32,
        )

        metadata = [
            replace(
                sample_metadata,
                black_levels=np.array([50, 60, 70, 80]),  # R, Gr, Gb, B
                white_level=1_000,
            ),
            replace(
                sample_metadata,
                black_levels=np.array([90, 100, 110, 120]),  # R, Gr, Gb, B
                white_level=1_500,
            ),
        ]
        result_images = apply_black_level_subtraction(raw_images, metadata)
        expected_normalized_1 = normalize_image(subtract_black_levels(raw_images[0], metadata[0]), metadata[0])
        expected_normalized_2 = normalize_image(subtract_black_levels(raw_images[1], metadata[1]), metadata[1])
        np.testing.assert_array_equal(result_images[0], expected_normalized_1)
        np.testing.assert_array_equal(result_images[1], expected_normalized_2)
