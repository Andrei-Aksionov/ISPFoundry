import numpy as np
import pytest

from pipeline_steps.lens_shading_correction import (
    align_cfa_pattern,
    apply_lens_shading_correction,
    apply_single_image,
    interpolate,
)


@pytest.fixture
def sample_images():
    """Fixture for a sample list of images."""
    return [np.array([[100, 110], [120, 130]], dtype=np.float32), np.array([[140, 150], [160, 170]], dtype=np.float32)]


@pytest.fixture
def sample_metadata():
    """Fixture for sample metadata with color description and raw pattern."""
    return [
        {"color_desc": "RGBG", "raw_pattern": np.array([[2, 2], [1, 0]]), "ImageHeight": 2, "ImageWidth": 2},
        {"color_desc": "RGBG", "raw_pattern": np.array([[2, 2], [1, 0]]), "ImageHeight": 2, "ImageWidth": 2},
    ]


@pytest.fixture
def sample_lsc_maps():
    """Fixture for a sample list of lens shading maps."""
    return [
        np.array([[[1, 2, 3, 4]]], dtype=np.float32),
        np.array([[[5, 6, 7, 8]]], dtype=np.float32),
    ]


class TestAlignCFAPattern:
    def test_align_cfa_pattern(self, sample_lsc_maps, sample_metadata):
        aligned_maps = align_cfa_pattern(sample_lsc_maps, sample_metadata)
        expected = [
            np.array([[[4, 3, 2, 1]]], dtype=np.float32),
            np.array([[[8, 7, 6, 5]]], dtype=np.float32),
        ]
        for aligned_map, expected_map in zip(aligned_maps, expected):
            np.testing.assert_array_equal(aligned_map, expected_map)


class TestInterpolate:
    def test_interpolate(self, sample_lsc_maps, sample_metadata):
        interpolated_maps = [interpolate(lsc_map, mt) for lsc_map, mt in zip(sample_lsc_maps, sample_metadata)]
        expected = [
            np.array([[[1, 2, 3, 4]]], dtype=np.float32),
            np.array([[[5, 6, 7, 8]]], dtype=np.float32),
        ]
        for interpolated_map, expected_map in zip(interpolated_maps, expected):
            np.testing.assert_array_equal(interpolated_map, expected_map)


class TestApplySingleImage:
    def test_apply_single_image(self, sample_images, sample_lsc_maps):
        corrected_images = [apply_single_image(img, lsc_map) for img, lsc_map in zip(sample_images, sample_lsc_maps)]
        expected = [
            np.array([[100, 220], [360, 520]], dtype=np.float32),
            np.array([[700, 900], [1120, 1360]], dtype=np.float32),
        ]
        for corrected_img, expected_img in zip(corrected_images, expected):
            np.testing.assert_array_equal(corrected_img, expected_img)

    def test_apply_single_image_inplace_true(self, sample_images, sample_metadata, sample_lsc_maps):
        corrected_images = [
            apply_single_image(img, lsc_map, inplace=True) for img, lsc_map in zip(sample_images, sample_lsc_maps)
        ]
        for sample_img, result_img in zip(corrected_images, sample_images):
            assert np.shares_memory(sample_img, result_img)

    def test_apply_single_image_inplace_false(self, sample_images, sample_metadata, sample_lsc_maps):
        corrected_images = [
            apply_single_image(img, lsc_map, inplace=False) for img, lsc_map in zip(sample_images, sample_lsc_maps)
        ]
        for sample_img, result_img in zip(corrected_images, sample_images):
            assert not np.shares_memory(sample_img, result_img)

    def test_apply_single_image_values_not_clipped(self, sample_lsc_maps):
        """
        LSC shouldn't clip values to [0, 1] range.

        When an input is mostly normalized to [0, 1] and images contain negative values from previous steps
        (e.g. from black level subtraction), or after applying lens shading correction
        (multiplying an image with a gain map), some values might exceed 1. That's still an important
        statistics for next steps.

        These values shouldn't be clipped to [0, 1].
        """
        sample_images = [
            np.array([[0.23, 0.87], [-0.01, 0.95]], dtype=np.float32),
            np.array([[0.17, -0.06], [0.87, 0.53]], dtype=np.float32),
        ]
        corrected_images = [apply_single_image(img, lsc_map) for img, lsc_map in zip(sample_images, sample_lsc_maps)]
        expected = [
            np.array([[0.23, 1.74], [-0.03, 3.8]], dtype=np.float32),
            np.array([[0.85, -0.36], [6.09, 4.24]], dtype=np.float32),
        ]
        for corrected_img, expected_img in zip(corrected_images, expected):
            np.testing.assert_array_almost_equal(corrected_img, expected_img)


class TestApplyLensShadingCorrection:
    def test_apply_lens_shading_correction(self, sample_images, sample_metadata, sample_lsc_maps):
        result_images = apply_lens_shading_correction(sample_images, sample_metadata, sample_lsc_maps)
        expected = [
            np.array([[400, 330], [240, 130]], dtype=np.float32),
            np.array([[1120, 1050], [960, 850]], dtype=np.float32),
        ]
        assert len(result_images) == len(sample_images)
        for result_img, expected_img in zip(result_images, expected):
            np.testing.assert_array_equal(result_img, expected_img)

    def test_apply_lens_shading_correction_inplace_true(self, sample_images, sample_metadata, sample_lsc_maps):
        result_images = apply_lens_shading_correction(sample_images, sample_metadata, sample_lsc_maps, inplace=True)
        for sample_img, result_img in zip(result_images, sample_images):
            assert np.shares_memory(sample_img, result_img)

    def test_apply_lens_shading_correction_inplace_false(self, sample_images, sample_metadata, sample_lsc_maps):
        result_images = apply_lens_shading_correction(sample_images, sample_metadata, sample_lsc_maps, inplace=False)
        for sample_img, result_img in zip(result_images, sample_images):
            assert not np.shares_memory(sample_img, result_img)

    def test_apply_lens_shading_correction_values_not_clipped(self, sample_metadata, sample_lsc_maps):
        """
        LSC shouldn't clip values to [0, 1] range.

        When an input is mostly normalized to [0, 1] and images contain negative values from previous steps
        (e.g. from black level subtraction), or after applying lens shading correction
        (multiplying an image with a gain map), some values might exceed 1. That's still an important
        statistics for next steps.

        These values shouldn't be clipped to [0, 1].
        """
        sample_images = [
            np.array([[0.23, 0.87], [-0.01, 0.95]], dtype=np.float32),
            np.array([[0.17, -0.06], [0.87, 0.53]], dtype=np.float32),
        ]
        corrected_images = apply_lens_shading_correction(sample_images, sample_metadata, sample_lsc_maps)
        expected = [
            np.array([[0.92, 2.61], [-0.02, 0.95]], dtype=np.float32),
            np.array([[1.36, -0.42], [5.22, 2.65]], dtype=np.float32),
        ]
        for corrected_img, expected_img in zip(corrected_images, expected):
            np.testing.assert_array_almost_equal(corrected_img, expected_img)
