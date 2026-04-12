from unittest.mock import patch

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pipeline_steps.align_and_merge import (
    compute_tile_sad,
    downsample_luma_proxy,
    estimate_noise_profile,
    find_best_float_offset,
    find_best_integer_offset,
    find_best_offset,
    find_sharpest_image_idx,
    find_subpixel_shift,
    get_hann_window_2d,
    get_luma_proxy,
    get_noise_profile,
    get_photometric_scalers,
    merge_images,
    merge_tile,
    sample_raw_bilinear,
)

# --------------------------------- Luma Proxy & Exposure Functions ----------------------------------


class TestGetLumaProxy:
    def test_get_luma_proxy_dimensions(self):
        """Test that the function handles odd dimensions correctly using & ~1."""
        # Create a 5x5 image (odd dimensions)
        raw_image = np.ones((5, 5), dtype=np.uint16)
        metadata = {
            "color_desc": "RGBG",
            "raw_pattern": [[0, 1], [3, 2]],  # Standard RGGB
        }

        proxy = get_luma_proxy(raw_image, metadata)

        # Expected output should be (5//2, 5//2) -> (2, 2)
        assert proxy.shape == (2, 2)

    def test_get_luma_proxy_math_rggb(self):
        """Test a 2x2 block with known values to verify weights are applied correctly."""
        # RGGB Pattern: R=100, Gr=200, Gb=300, B=400
        raw_image = np.array([[100, 200], [300, 400]], dtype=np.uint16)

        metadata = {
            "color_desc": "RGBG",  # R=index0, G=index1, B=index2, G=index3
            "raw_pattern": [[0, 1], [3, 2]],  # R, Gr, Gb, B
        }

        # Calculation: (100 * 0.15) + (200 * 0.35) + (300 * 0.35) + (400 * 0.15)  # noqa: ERA001
        # 15 + 70 + 105 + 60 = 250
        expected_value = 250.0

        proxy = get_luma_proxy(raw_image, metadata)
        np.testing.assert_allclose(proxy[0, 0], expected_value)

    def test_get_luma_proxy_bayer_invariance(self):
        """Test that different Bayer patterns (BGGR vs RGGB) yield different (correct) results."""
        # Raw block
        raw_image = np.array([[100, 0], [0, 0]], dtype=np.uint16)

        # Case 1: Top-left is Red (Weight 0.15)
        meta_rggb = {"color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]]}
        proxy_r = get_luma_proxy(raw_image, meta_rggb)

        # Case 2: Top-left is Green (Weight 0.35)
        # We simulate this by changing the pattern so index 0 is Green
        meta_grbg = {"color_desc": "GRBG", "raw_pattern": [[0, 1], [3, 2]]}
        proxy_g = get_luma_proxy(raw_image, meta_grbg)

        assert proxy_r[0, 0] == 15.0
        assert proxy_g[0, 0] == 35.0

    def test_get_luma_proxy_output_type(self):
        """Ensure the output is always float32 as specified."""
        raw_image = np.zeros((10, 10), dtype=np.uint16)
        metadata = {"color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]]}

        proxy = get_luma_proxy(raw_image, metadata)
        assert proxy.dtype == np.float32

    def test_get_luma_proxy_normalization(self):
        """Verify that a pure white 2x2 quad results in 1.0 (unit energy conservation)."""
        raw_image = np.ones((2, 2), dtype=np.float32)
        metadata = {
            "color_desc": "RGBG",
            "raw_pattern": [[0, 1], [3, 2]],  # RGGB
        }

        proxy = get_luma_proxy(raw_image, metadata)

        # 0.15(R) + 0.35(G) + 0.35(G) + 0.15(B) = 1.0
        np.testing.assert_allclose(proxy[0, 0], 1.0)

    def test_get_luma_proxy_large_burst(self):
        """Test with a larger synthetic image to ensure einsum scales correctly."""
        raw_image = np.random.rand(128, 128).astype(np.float32)
        metadata = {"color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]]}

        proxy = get_luma_proxy(raw_image, metadata)
        assert proxy.shape == (64, 64)
        assert not np.any(np.isnan(proxy))

    def test_invalid_color_desc(self):
        """Ensure it raises KeyError if a color in pattern isn't in color_weights."""
        raw_image = np.zeros((2, 2), dtype=np.float32)
        # 'X' is not in color_weights dict
        metadata = {"color_desc": "RGBX", "raw_pattern": [[0, 1], [3, 2]]}

        with pytest.raises(KeyError):
            get_luma_proxy(raw_image, metadata)


class TestDownsampleLumaProxy:
    def test_downsample_odd_dimensions(self):
        """Verify that odd-dimensioned arrays are truncated to even before downsampling."""
        # Input 5x7 -> Truncated to 4x6 -> Result should be 2x3
        proxy = np.ones((5, 7), dtype=np.float32)
        downsampled = downsample_luma_proxy(proxy)

        assert downsampled.shape == (2, 3)

    def test_downsample_box_averaging_math(self):
        """Verify that a 2x2 block is averaged correctly."""
        # Create a 2x2 block where the mean is exactly 2.5
        proxy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        expected = np.array([[2.5]], dtype=np.float32)
        result = downsample_luma_proxy(proxy)

        np.testing.assert_allclose(result, expected)
        assert result.shape == (1, 1)

    def test_downsample_spatial_consistency(self):
        """
        Verify that different 2x2 quads in a larger image are averaged independently.

        [[1., 1., 2., 2.],
         [1., 1., 2., 2.],
         [3., 3., 4., 4.],
         [3., 3., 4., 4.]]
        """
        # 4x4 image with distinct values in each quadrant
        proxy = np.zeros((4, 4), dtype=np.float32)
        proxy[0:2, 0:2] = 1.0  # Top-left average: 1.0
        proxy[0:2, 2:4] = 2.0  # Top-right average: 2.0
        proxy[2:4, 0:2] = 3.0  # Bottom-left average: 3.0
        proxy[2:4, 2:4] = 4.0  # Bottom-right average: 4.0

        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        result = downsample_luma_proxy(proxy)
        np.testing.assert_allclose(result, expected)

    def test_downsample_identity_preservation(self):
        """A uniform field should remain the same value after downsampling."""
        val = 0.75
        proxy = np.full((10, 10), val, dtype=np.float32)
        result = downsample_luma_proxy(proxy)

        np.testing.assert_equal(result, val)
        assert result.dtype == np.float32

    def test_downsample_empty_or_too_small(self):
        """Handle cases where dimensions are smaller than a 2x2 block."""
        proxy = np.ones((1, 1), dtype=np.float32)

        result = downsample_luma_proxy(proxy)
        assert result.size == 1
        assert result.shape == (1, 1)


class TestGetExposureScalers:
    # TODO (andrei aksionau): add a test with different ISOs
    def test_scalers_math_simple(self):
        """Verify that a 4x longer exposure results in a 0.25 scaler."""
        metadata = [
            {"ExposureTime": 0.01},  # Shortest (Reference)
            {"ExposureTime": 0.04},  # 4x longer
            {"ExposureTime": 0.02},  # 2x longer
        ]

        expected = np.array([1.0, 0.25, 0.5], dtype=np.float32)
        result = get_photometric_scalers(metadata)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_fractional_string_parsing(self):
        """Verify that '1/100' style strings are parsed and scaled correctly."""
        metadata = [
            {"ExposureTime": "1/100"},  # 0.01
            {"ExposureTime": "1/50"},  # 0.02
            {"ExposureTime": "1/400"},  # 0.0025 (Shortest Reference)
        ]

        # Calculation -> (1/400) / (1/100) = 0.25
        expected = np.array([0.25, 0.125, 1.0], dtype=np.float32)
        result = get_photometric_scalers(metadata)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_mixed_types_scaling(self):
        """Ensure integers, floats, and strings are handled in a single burst."""
        metadata = [
            {"ExposureTime": 1},  # Integer
            {"ExposureTime": 0.5},  # Float (Shortest)
            {"ExposureTime": "2/1"},  # String
        ]

        expected = np.array([0.5, 1.0, 0.25], dtype=np.float32)
        result = get_photometric_scalers(metadata)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_different_exposures_and_same_iso(self):
        metadata = [
            {"ExposureTime": "1/100", "ISO": 100},
            {"ExposureTime": "1/50", "ISO": 100},
            {"ExposureTime": "1/400", "ISO": 100},
        ]

        expected = np.array([0.25, 0.125, 1.0], dtype=np.float32)
        result = get_photometric_scalers(metadata)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_different_exposures_and_different_iso(self):
        metadata = [
            # These 3 should have the same brightness
            {"ExposureTime": "1/50", "ISO": 100},  # 8x brighter
            {"ExposureTime": "1/100", "ISO": 200},  # 8x brighter
            {"ExposureTime": "1/400", "ISO": 800},  # 8x brighter
            # These 3 should have different brightness
            {"ExposureTime": "1/50", "ISO": 400},  # 32x brighter
            {"ExposureTime": "1/100", "ISO": 100},  # 4x brighter
            {"ExposureTime": "1/400", "ISO": 100},  # 1x
        ]

        expected = np.array([0.125, 0.125, 0.125, 0.03125, 0.25, 1.0], dtype=np.float32)
        result = get_photometric_scalers(metadata)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_identical_exposures_uniformity(self):
        """All scalers must be 1.0 when exposure times are identical."""
        metadata = [{"ExposureTime": 0.0333}] * 3
        result = get_photometric_scalers(metadata)

        expected = np.ones(3, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_output_metadata_consistency(self):
        """Verify output array properties: length, type, and min value."""
        metadata = [{"ExposureTime": 0.1}, {"ExposureTime": 0.5}, {"ExposureTime": 0.2}]
        result = get_photometric_scalers(metadata)

        assert result.dtype == np.float32
        assert result.shape == (3,)
        assert np.max(result) == 1.0  # Shortest must be 1.0

    def test_parsing_error_on_invalid_string(self):
        """Ensure the function raises a ValueError for un-parsable ExposureTime."""
        metadata = [{"ExposureTime": "not_a_number"}]
        with pytest.raises(ValueError, match="convert string to float"):
            get_photometric_scalers(metadata)


class TestFindSharpestImageIdx:
    @pytest.fixture
    def base_metadata(self):
        # We now include the required 'ExposureTime' key
        # 'color_desc' and 'raw_pattern' are needed by get_luma_proxy
        return {"color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]], "ISO": 100, "ExposureTime": "1/1000"}

    def test_selects_sharpest_within_short_exposure_group(self, base_metadata):
        """Test that the sharpest image is chosen among frames with the same short exposure."""
        sharp = np.zeros((32, 32), dtype=np.uint16)
        row, col = np.indices(sharp.shape)
        mask = (row // 8 + col // 8) % 2 == 0
        sharp[mask] = 1.0

        blurry = gaussian_filter(sharp.astype(float), sigma=4.0).astype(np.uint16)

        images = np.stack([blurry, sharp])
        # Both are "short" exposures
        metadata = [base_metadata.copy() for _ in range(2)]

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 1

    def test_prefers_short_exposure_over_sharp_long_exposure(self, base_metadata):
        """Test that the logic correctly excludes long exposures from selection, even if the long exposure appears 'sharper' due to higher signal/contrast."""
        # 1. Create a "Long Exposure" image: very sharp, high contrast
        long_exp_img = np.zeros((32, 32), dtype=np.float32)
        row, col = np.indices(long_exp_img.shape)
        mask = (row // 8 + col // 8) % 2 == 0
        long_exp_img[mask] = 1.0  # High signal
        long_metadata = base_metadata.copy()
        long_metadata["ExposureTime"] = "1/250"  # 4x longer than base

        # 2. Create a "Short Exposure" image: slightly blurry, lower contrast
        # In a real burst, this might happen due to slight hand shake
        short_exp_img = np.zeros((32, 32), dtype=np.float32)
        short_exp_img[mask] = 0.7  # Lower signal
        short_exp_img = gaussian_filter(short_exp_img.astype(float), sigma=0.8).astype(np.float32)
        short_metadata = base_metadata.copy()
        short_metadata["ExposureTime"] = "1/1000"

        # Even though index 0 (long) is mathematically "sharper" (higher variance),
        # index 1 (short) MUST be selected to avoid clipped highlights.
        images = np.stack([long_exp_img, short_exp_img])
        metadata = [long_metadata, short_metadata]

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 1

    def test_noise_robustness(self, base_metadata):
        """Test that Gaussian smoothing prevents sensor noise from being mistaken for sharpness."""
        # Pure noise
        noisy = np.random.uniform(0.4, 0.6, (32, 32)).astype(np.float32)

        # Actual structure
        structural = np.zeros((32, 32), dtype=np.float32)
        structural[:16, :] = 0.8

        images = np.stack([noisy, structural])
        metadata = [base_metadata.copy() for _ in range(2)]

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 1

    def test_handles_mixed_string_and_float_exposure(self, base_metadata):
        """Ensure the parser handles both fractional strings and floats in metadata."""
        img = np.full((32, 32), 0.5, dtype=np.float32)

        m1 = base_metadata.copy()
        m1["ExposureTime"] = "1/500"

        m2 = base_metadata.copy()
        m2["ExposureTime"] = 0.002  # Equivalent to 1/500

        images = np.stack([img, img])
        metadata = [m1, m2]

        # Both are considered "short" (minimum), should default to first index if identical
        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 0

    def test_fallback_when_no_short_exposures_found(self, base_metadata):
        """Ensure the function returns a valid index from the whole burst even if exposure_scalers logic fails or metadata is uniform."""
        # Create two identical images
        img = np.random.rand(32, 32).astype(np.float32)
        images = np.stack([img, img])

        # Metadata with same exposure (all will be 'short' by definition, but we test the logic's ability to handle the list)
        metadata = [base_metadata.copy() for _ in range(2)]

        best_idx = find_sharpest_image_idx(images, metadata)

        assert best_idx in [0, 1]
        assert isinstance(best_idx, (int, np.integer))

    def test_sharpness_with_normalized_floats(self, base_metadata):
        """Verify the function works with [0, 1] float32 images as specified in the ISP pipeline requirements."""
        # Create a sharp edge in [0, 1] range
        sharp_img = np.zeros((32, 32), dtype=np.float32)
        sharp_img[:16, :] = 1.0

        # Create a blurry edge
        blurry_img = gaussian_filter(sharp_img, sigma=2.0)

        images = np.stack([blurry_img, sharp_img])
        metadata = [base_metadata.copy() for _ in range(2)]

        best_idx = find_sharpest_image_idx(images, metadata)

        # Index 1 should be significantly sharper (higher Laplacian variance)
        assert best_idx == 1

    def test_all_black_images(self, base_metadata):
        """Edge case: If images are completely black (underexposed), the function should still return an index rather than crashing."""
        black_img = np.zeros((32, 32), dtype=np.float32)
        images = np.stack([black_img, black_img])
        metadata = [base_metadata.copy() for _ in range(2)]

        # Should not raise ZeroDivisionError or similar
        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 0

    def test_single_image_burst(self, base_metadata):
        """Verify behavior with a burst of size 1."""
        img = np.random.rand(32, 32).astype(np.float32)
        images = img[None, ...]
        metadata = [base_metadata]

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 0


# ------------------------------------ Noise Estimation Functions ------------------------------------


class TestGetNoiseProfile:
    @pytest.fixture
    def sample_metadata(self):
        return {
            "CFAPlaneColor": "Red,Green,Blue",
            # 3 pairs of (Scale, Offset): R=(0.01, 0.001), G=(0.02, 0.002), B=(0.03, 0.003)
            "NoiseProfile": "0.01 0.001 0.02 0.002 0.03 0.003",
            "color_desc": "RGBG",
            "raw_pattern": [[0, 1], [3, 2]],  # RGGB: 0=R, 1=G, 3=G, 2=B
        }

    def test_parse_dng_noise_profile_mapping(self, sample_metadata):
        """Verify that NoiseProfile string is correctly mapped to the 2x2 Bayer grid."""
        image = np.zeros((10, 10), dtype=np.float32)

        scales, offsets = get_noise_profile(image, sample_metadata)

        # Expected Scales (RGGB): R=0.01, G=0.02, G=0.02, B=0.03
        expected_scales = np.array([[0.01, 0.02], [0.02, 0.03]], dtype=np.float32)

        # Expected Offsets (RGGB): R=0.001, G=0.002, G=0.002, B=0.003
        expected_offsets = np.array([[0.001, 0.002], [0.002, 0.003]], dtype=np.float32)

        np.testing.assert_allclose(scales, expected_scales, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(offsets, expected_offsets, rtol=1e-6, atol=1e-6)
        assert scales.dtype == np.float32
        assert offsets.dtype == np.float32

    def test_different_bayer_phase_mapping(self, sample_metadata):
        """Verify mapping works correctly for a different Bayer phase (e.g., BGGR)."""
        image = np.zeros((10, 10), dtype=np.float32)
        # Change pattern to BGGR: 2=B, 1=G, 3=G, 0=R
        sample_metadata["raw_pattern"] = [[2, 1], [3, 0]]

        scales, offsets = get_noise_profile(image, sample_metadata)

        # Top-left should now be Blue scale (0.03)
        assert scales[0, 0] == 0.03
        # Bottom-right should be Red scale (0.01)
        assert scales[1, 1] == 0.01
        np.testing.assert_allclose(scales, [[0.03, 0.02], [0.02, 0.01]])
        np.testing.assert_allclose(offsets, [[0.003, 0.002], [0.002, 0.001]])

    def test_invalid_cfa_plane_color(self, sample_metadata):
        """Ensure ValueError is raised if CFAPlaneColor is not Red,Green,Blue."""
        sample_metadata["CFAPlaneColor"] = "Cyan,Magenta,Yellow"
        image = np.zeros((4, 4))

        with pytest.raises(ValueError, match="The code expects that the matrix layout is Red Green Blue"):
            get_noise_profile(image, sample_metadata)

    def test_fallback_to_estimation(self, sample_metadata):
        """Verify that estimate_noise_profile is called if NoiseProfile is missing."""
        # Remove NoiseProfile from metadata
        del sample_metadata["NoiseProfile"]
        image = np.random.rand(10, 10).astype(np.float32)

        # Mocking the estimation function to verify it's reached
        with patch("pipeline_steps.align_and_merge.estimate_noise_profile") as mock_estimate:
            mock_estimate.return_value = (np.ones((2, 2)), np.zeros((2, 2)))

            get_noise_profile(image, sample_metadata)
            mock_estimate.assert_called_once_with(image)

    def test_malformed_noise_profile_string(self, sample_metadata):
        """Ensure it raises an error if the NoiseProfile string is incomplete."""
        sample_metadata["NoiseProfile"] = "0.01 0.001"  # Only 2 values instead of 6
        image = np.zeros((4, 4))

        with pytest.raises(IndexError):
            get_noise_profile(image, sample_metadata)


class TestEstimateNoiseProfile:
    def test_estimate_noise_fallback_on_flat_field(self):
        """Verify that the function returns default fallback values when it can't find enough variance bins (e.g., a perfectly flat constant image)."""
        # A perfectly flat image has 0 variance, linregress will fail or
        # there won't be enough unique bins.
        flat_img = np.full((128, 128), 0.5, dtype=np.float32)

        scales, offsets = estimate_noise_profile(flat_img, patch_size=8)

        # Fallback values from the code: scale=1e-4, offset=1e-6
        expected_scales = np.full((2, 2), 1e-4, dtype=np.float32)
        expected_offsets = np.full((2, 2), 1e-6, dtype=np.float32)

        np.testing.assert_allclose(scales, expected_scales, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(offsets, expected_offsets, rtol=1e-6, atol=1e-6)

    def test_estimate_noise_clipping_safety(self):
        """Ensure that the function handles negative intercepts (physically impossible) by clipping them to the defined minimums (1e-7 and 1e-9)."""
        # Create an image where variance decreases as signal increases (impossible noise)
        # This should force a negative slope/intercept in a naive regression.
        h, w = 256, 256
        img = np.random.rand(h, w).astype(np.float32)

        scales, offsets = estimate_noise_profile(img, patch_size=16)

        # Check that we are at least at the minimum floor
        assert np.all(scales >= 1e-7)
        assert np.all(offsets >= 1e-9)

    def test_estimate_noise_patch_size_consistency(self):
        """Test that different patch sizes still return the correct grid shape."""
        img = np.random.rand(256, 256).astype(np.float32)

        scales, _ = estimate_noise_profile(img, patch_size=16)
        assert scales.shape == (2, 2)

        scales_small, _ = estimate_noise_profile(img, patch_size=4)
        assert scales_small.shape == (2, 2)

    def test_estimate_noise_center_crop_logic(self):
        """Verify the function doesn't crash on small images where center crop might be very tiny."""
        # Minimal image that allows for center cropping and tiling
        # 32x32 -> center crop 16x16 -> 2x2 planes of 8x8 -> 1 patch of 8x8
        small_img = np.random.rand(32, 32).astype(np.float32)

        # This will likely hit the fallback because 1 patch < 10 required
        scales, _ = estimate_noise_profile(small_img, patch_size=8)
        assert scales.shape == (2, 2)


# ---------------------------------------- Aligning Functions ----------------------------------------


class TestFindSubpixelShift:
    def test_perfect_center(self):
        """If the center is much lower than its neighbors and they are balanced, shift should be 0."""
        grid = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        dy, dx = find_subpixel_shift(grid)

        # dy = (1 - 1) / (2 * (1 + 1 - 0)) = 0
        np.testing.assert_allclose(dy, 0.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dx, 0.0, rtol=1e-6, atol=1e-6)

    def test_clamping_logic(self):
        """If the neighbors suggest a vertex far outside the center pixel, it should clamp to [-0.5, 0.5]."""
        # Heavily asymmetrical: Center=10, Left=11, Right=100
        # denom = 2 * (11 + 100 - 20) = 182
        # offset = (11 - 100) / 182 = -89 / 182 approx -0.489
        # Let's force an even more extreme case:
        grid = np.array(
            [
                [0, 10, 0],
                [10, 10, 50],  # Right is very high, suggesting vertex is way to the left
                [0, 10, 0],
            ],
            dtype=np.float32,
        )

        # denom = 2 * (10 + 50 - 20) = 80
        # dx = (10 - 50) / 80 = -0.5
        dy, dx = find_subpixel_shift(grid)

        assert -0.5 <= dy <= 0.5
        assert -0.5 <= dx <= 0.5

    def test_divide_by_zero_robustness(self):
        """If the surface is flat (denominator is 0), it should return 0 offset."""
        grid = np.full((3, 3), 10.0, dtype=np.float32)

        dy, dx = find_subpixel_shift(grid)

        np.testing.assert_allclose(dy, 0.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dx, 0.0, rtol=1e-6, atol=1e-6)

    def test_output_types(self):
        """Ensure the function returns a tuple of standard floats."""
        grid = np.random.rand(3, 3).astype(np.float32)
        dy, dx = find_subpixel_shift(grid)

        assert isinstance(dy, float)
        assert isinstance(dx, float)

    def test_known_fractional_shift(self):
        """Test with a grid where the minimum is mathematically at a known offset. For a parabola y = ax^2 + bx + c, the vertex is at -b / 2a."""
        # Create a 1D parabola: f(x) = (x - 0.25)^2
        # Points at x = -1, 0, 1:
        # f(-1) = 1.5625 (Up/Left)
        # f(0)  = 0.0625 (Center)
        # f(1)  = 0.5625 (Down/Right)
        grid = np.zeros((3, 3), dtype=np.float32)
        grid[1, 1] = 0.0625  # Center

        # Vertical shift of +0.25 (towards 'Down')
        grid[0, 1] = 1.5625  # Up
        grid[2, 1] = 0.5625  # Down

        # Horizontal shift of -0.25 (towards 'Left')
        # f(x) = (x + 0.25)^2
        # f(-1) = 0.5625, f(0) = 0.0625, f(1) = 1.5625
        grid[1, 0] = 0.5625  # Left
        grid[1, 2] = 1.5625  # Right

        dr, dc = find_subpixel_shift(grid)

        # Note: In the code, offset_row = (up - down) / denom
        # For our vertical setup: (1.5625 - 0.5625) / (2 * (1.5625 + 0.5625 - 2*0.0625))
        # 1.0 / (2 * (2.125 - 0.125)) = 1.0 / 4.0 = 0.25
        assert np.isclose(dr, 0.25)
        assert np.isclose(dc, -0.25)

    def test_asymmetric_noise_robustness(self):
        """Verify that diagonal elements don't affect the output. We change a corner (diagonal) and the result should remain identical."""
        grid_a = np.array([[0.0, 1.0, 0.0], [1.0, 0.5, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        grid_b = grid_a.copy()
        grid_b[0, 0] = 99.9  # Modify corner

        res_a = find_subpixel_shift(grid_a)
        res_b = find_subpixel_shift(grid_b)

        assert res_a == res_b

    def test_linear_slope(self):
        """If the surface is a linear plane (not a parabola), the denominator logic should handle it or the clamp should catch it."""
        # Vertical is linear: Up=3, Center=2, Down=1
        # denom = 2 * (3 + 1 - 2*2) = 0. This should trigger the 1e-7 check.
        grid = np.array([[0, 3, 0], [0, 2, 0], [0, 1, 0]], dtype=np.float32)

        dr, dc = find_subpixel_shift(grid)
        assert dr == 0.0
        assert dc == 0.0


class TestComputeTileSad:
    def test_perfect_alignment_zero_sad(self):
        """If images are identical and aligned, SAD should be 0.0."""
        ref = np.full((100, 100), 0.5, dtype=np.float32)
        tgt = np.full((100, 100), 0.5, dtype=np.float32)

        # Tile at (10, 10), size 16, no offset
        score = compute_tile_sad(0, 0, ref, tgt, 10, 10, 16, 1.0)

        assert score == 0.0

    def test_sad_normalization_with_inv_sigma(self):
        """Verify that inv_sigma and area normalization are applied correctly."""
        tile_size = 4
        inv_sigma = 10.0  # sigma = 0.1
        diff = 0.2

        # Create 4x4 tile where every pixel has a difference of 0.2
        ref = np.zeros((10, 10), dtype=np.float32)
        tgt = np.full((10, 10), diff, dtype=np.float32)

        score = compute_tile_sad(0, 0, ref, tgt, 0, 0, tile_size, inv_sigma)

        # Sum of Abs Diff = 0.2 * 16 pixels = 3.2
        # Area Normalization = 3.2 / 16 = 0.2
        # Sigma Normalization = 0.2 * 10 = 2.0
        np.testing.assert_allclose(score, 2.0, rtol=1e-6)

    def test_saturation_skipping(self):
        """Pixels above the threshold should be ignored in the calculation."""
        # 2x2 tile
        ref = np.array([[0.1, 0.99], [0.1, 0.1]], dtype=np.float32)  # 0.99 is saturated
        tgt = np.array([[0.2, 0.99], [0.2, 0.2]], dtype=np.float32)

        # threshold = 0.95. Only 3 pixels should count.
        # SAD = |0.1-0.2| + |0.1-0.2| + |0.1-0.2| = 0.3
        # Count is 3
        # Expected = (0.3 * 1.0) / 3 = 0.1
        score = compute_tile_sad(0, 0, ref, tgt, 0, 0, 2, 1.0, saturation_threshold=0.95)

        np.testing.assert_allclose(score, 0.1, rtol=1e-6)

    def test_insufficient_data_returns_none(self):
        """If more than 75% of pixels are saturated, return None."""
        ref = np.full((4, 4), 0.99, dtype=np.float32)
        tgt = np.full((4, 4), 0.99, dtype=np.float32)

        score = compute_tile_sad(0, 0, ref, tgt, 0, 0, 4, 1.0, saturation_threshold=0.95)

        assert score is None

    def test_boundary_clipping_and_offset(self):
        """Test that the function handles tiles partially outside the image via offsets. Use values below 0.95 to avoid the saturation check returning None."""
        # 20x20 images with valid (non-saturated) data
        ref = np.full((20, 20), 0.5, dtype=np.float32)
        tgt = np.full((20, 20), 0.5, dtype=np.float32)

        # Tile size 8, Offset -2.
        # Intersection is 6 rows x 8 cols = 48 valid pixels.
        score = compute_tile_sad(
            row_offset=-2,
            col_offset=0,
            reference_proxy=ref,
            target_proxy=tgt,
            row_start=0,
            col_start=0,
            tile_size=8,
            inv_sigma=1.0,
        )

        # Now score should be 0.0 (perfect match of 0.5 values)
        assert score is not None
        np.testing.assert_allclose(score, 0.0, atol=1e-6)

    def test_partial_overlap_math(self):
        """Verify SAD math when only a portion of the tile overlaps the image."""
        # Create a 10x10 image
        ref = np.zeros((10, 10), dtype=np.float32)
        tgt = np.zeros((10, 10), dtype=np.float32)

        # Set a diff in the overlapping region
        # Offset +8 means Ref[0,0] matches Tgt[8,0]
        ref[0, 0] = 0.4
        tgt[8, 0] = 0.0  # Diff of 0.4

        # Tile size 4 at (0,0). With row_offset 8, only 2 rows overlap (8,9)
        # Intersection size: 2 rows * 4 cols = 8 pixels
        # Non clipped count: 8
        # sad = 0.4 (from the one pixel diff)
        # Expected = 0.4 / 8 = 0.05
        score = compute_tile_sad(8, 0, ref, tgt, 0, 0, 4, 1.0)

        assert score is not None
        np.testing.assert_allclose(score, 0.05, atol=1e-6)

    def test_exact_overlap_logic(self):
        """Verification of the coordinate mapping using values below saturation."""
        ref = np.zeros((10, 10), dtype=np.float32)
        tgt = np.zeros((10, 10), dtype=np.float32)

        # Use 0.8 to stay below the default 0.95 saturation threshold
        val = 0.8
        ref[5, 5] = val
        tgt[6, 6] = val

        # 1. Perfect offset: (1, 1) shift aligns Ref[5,5] with Tgt[6,6]
        score_perfect = compute_tile_sad(1, 1, ref, tgt, 4, 4, 4, 1.0)
        np.testing.assert_allclose(score_perfect, 0.0, atol=1e-6)

        # 2. Bad offset: (0, 0) shift compares Ref[5,5]=0.8 with Tgt[5,5]=0.0
        # AND Ref[6,6]=0.0 with Tgt[6,6]=0.8
        score_bad = compute_tile_sad(0, 0, ref, tgt, 4, 4, 4, 1.0)

        # Sum of Abs Diff should be |0.8 - 0.0| + |0.0 - 0.8| = 1.6
        # Normalized by 16 pixels = 0.1
        assert score_bad is not None
        assert score_bad > 0
        np.testing.assert_allclose(score_bad, 0.1, atol=1e-6)

    def test_completely_out_of_bounds(self):
        """If the offset pushes the tile entirely off the image, return None."""
        ref = np.ones((10, 10), dtype=np.float32)
        tgt = np.ones((10, 10), dtype=np.float32)

        # Offset > image height
        score = compute_tile_sad(20, 0, ref, tgt, 0, 0, 4, 1.0)
        assert score is None


class TestFindBestIntegerOffset:
    def test_finds_correct_offset_within_radius(self):
        """Verify the search finds a known shift within the radius."""
        # 32x32 images
        ref = np.full((32, 32), 0.2, dtype=np.float32)
        tgt = np.full((32, 32), 0.2, dtype=np.float32)

        # Place a unique feature (a small 2x2 square) in ref at (10, 10)
        ref[10:12, 10:12] = 0.8
        # Place it in tgt at (12, 13) -> Shift is dy=2, dx=3
        tgt[12:14, 13:15] = 0.8

        # Search around hint (0,0) with radius 4
        # The tile being checked is at (8, 8), size 8
        best_dy, best_dx, min_sad = find_best_integer_offset(
            ref, tgt, row_start=8, col_start=8, tile_size=8, hint_dy=0, hint_dx=0, search_radius=4, inv_sigma=1.0
        )

        assert best_dy == 2
        assert best_dx == 3
        np.testing.assert_allclose(min_sad, 0.0, atol=1e-6)

    def test_uses_hint_as_center(self):
        """Verify the search window is centered on the hint, not (0,0)."""
        ref = np.full((32, 32), 0.3, dtype=np.float32)
        tgt = np.full((32, 32), 0.3, dtype=np.float32)

        # Feature at (10, 10) in ref, (20, 20) in tgt -> True shift is 10, 10
        ref[10, 10] = 0.7
        tgt[20, 20] = 0.7

        # Hint is (9, 9), radius is 2. Search range is [7, 11].
        # (10, 10) is inside this range.
        best_dy, best_dx, _ = find_best_integer_offset(
            ref, tgt, 10, 10, 4, hint_dy=9, hint_dx=9, search_radius=2, inv_sigma=1.0
        )

        assert best_dy == 10
        assert best_dx == 10

    def test_handles_all_none_sad_gracefully(self):
        """If every position in the search window returns None (e.g., all saturated), it should return the initial hint and the high dummy SAD value."""
        # All saturated images (1.0 > 0.95 threshold)
        ref = np.ones((16, 16), dtype=np.float32)
        tgt = np.ones((16, 16), dtype=np.float32)

        best_dy, best_dx, min_sad = find_best_integer_offset(
            ref, tgt, 4, 4, 4, hint_dy=2, hint_dx=2, search_radius=1, inv_sigma=1.0
        )

        # Should stay at hint
        assert best_dy == 2
        assert best_dx == 2
        assert min_sad == 1e20

    def test_respects_search_radius_boundary(self):
        """Verify it doesn't find a better match that is outside the radius."""
        ref = np.zeros((32, 32), dtype=np.float32)
        tgt = np.zeros((32, 32), dtype=np.float32)

        # True match is at (5, 5)
        ref[10, 10] = 0.5
        tgt[15, 15] = 0.5

        # Search with hint (0,0) and radius 2.
        # (5, 5) is unreachable.
        best_dy, best_dx, min_sad = find_best_integer_offset(
            ref, tgt, 10, 10, 4, hint_dy=0, hint_dx=0, search_radius=2, inv_sigma=1.0
        )

        assert abs(best_dy) <= 2
        assert abs(best_dx) <= 2
        assert min_sad > 0  # Should not find the perfect match

    def test_tie_breaking(self):
        """In case of equal SAD scores, it should ideally keep the first one found (stability)."""
        # Uniform images, every offset is a 'perfect' 0.0 match
        ref = np.full((16, 16), 0.4, dtype=np.float32)
        tgt = np.full((16, 16), 0.4, dtype=np.float32)

        hint_y, hint_x = 1, 1
        radius = 2
        # Start search at hint_dy - radius = 1 - 2 = -1
        best_dy, best_dx, _ = find_best_integer_offset(
            ref, tgt, 4, 4, 4, hint_dy=hint_y, hint_dx=hint_x, search_radius=radius, inv_sigma=1.0
        )

        # Since the first SAD calculated (at the start of the loops) is 0.0,
        # and the condition is `sad < min_sad`, it should return the very first offset checked.
        assert best_dy == hint_y - radius
        assert best_dx == hint_x - radius


class TestFindBestFloatOffset:
    @pytest.fixture
    def mock_proxies(self):
        return np.zeros((32, 32)), np.zeros((32, 32))

    def test_assembles_correct_final_offset1(self, mock_proxies):
        """Verify that integer shift and sub-pixel refinement are summed correctly."""
        ref, tgt = mock_proxies

        # We mock compute_tile_sad to return a grid that results in a known shift.
        # If neighbors are symmetric, find_subpixel_shift returns (0, 0).
        with patch("pipeline_steps.align_and_merge.compute_tile_sad", return_value=10.0):
            # .py_func calls the raw Python logic, ignoring the @njit decorator
            dy, dx, sad = find_best_float_offset.py_func(
                ref, tgt, 0, 0, 8, best_dy_int=5, best_dx_int=-3, min_sad=5.0, inv_sigma=1.0
            )

            assert dy == 5.0
            assert dx == -3.0
            assert sad == 5.0

    def test_subpixel_refinement_math(self, mock_proxies):
        """Test that an asymmetrical error surface shifts the float result correctly."""
        ref, tgt = mock_proxies

        # Define SAD values for: [Up, Down, Left, Right]
        # Up=2.0, Down=4.0, Center=1.0 -> Vertical shift should be -0.25
        sad_values = {
            (4, 5): 2.0,  # Up (best_dy-1, best_dx)
            (6, 5): 4.0,  # Down (best_dy+1, best_dx)
            (5, 4): 3.0,  # Left (best_dy, best_dx-1)
            (5, 6): 3.0,  # Right (best_dy, best_dx+1)
        }

        def side_effect(dy, dx, *args, **kwargs):
            return sad_values.get((dy, dx), 3.0)

        with patch("pipeline_steps.align_and_merge.compute_tile_sad", side_effect=side_effect):
            dy, dx, _ = find_best_float_offset.py_func(
                ref, tgt, 0, 0, 8, best_dy_int=5, best_dx_int=5, min_sad=1.0, inv_sigma=1.0
            )

            # 5.0 (int) + (-0.25) (sub) = 4.75
            assert dy == 4.75
            assert dx == 5.0

    def test_boundary_mirroring_logic(self, mock_proxies):
        """If a neighbor is None (out of bounds), the function should mirror the opposite neighbor's value to stay stable."""
        ref, tgt = mock_proxies

        # Setup: Up is out of bounds (None), Down is 15.0, Center is 10.0
        def side_effect(dy, dx, *args, **kwargs):
            if dy == 4:
                return None  # Up
            return 15.0  # All others

        with patch("pipeline_steps.align_and_merge.compute_tile_sad", side_effect=side_effect):
            dy, dx, _ = find_best_float_offset.py_func(
                ref, tgt, 0, 0, 8, best_dy_int=5, best_dx_int=5, min_sad=10.0, inv_sigma=1.0
            )

            # If Up mirrored Down (15.0), the grid becomes [15, 10, 15]
            # vertically, which results in 0.0 sub-pixel shift.
            assert dy == 5.0
            assert dx == 5.0

    def test_mirroring_convexity_safety(self, mock_proxies):
        """Verify that mirrored values are at least as large as min_sad to prevent creating a 'fake' minimum at the edge."""
        ref, tgt = mock_proxies

        # Down is 5.0, Center is 10.0 (an 'impossible' case where center isn't min)
        # Up is None. Mirroring should take max(Down, Center) = 10.0.
        def side_effect(dy, dx, *args, **kwargs):
            if dy == 4:
                return None  # Up
            if dy == 6:
                return 5.0  # Down
            return 12.0

        with patch("pipeline_steps.align_and_merge.compute_tile_sad", side_effect=side_effect):
            # This shouldn't crash and should produce a clamped/stable float
            dy, dx, _ = find_best_float_offset.py_func(
                ref, tgt, 0, 0, 8, best_dy_int=5, best_dx_int=5, min_sad=10.0, inv_sigma=1.0
            )
            assert isinstance(dy, float)

    def test_output_types_and_precision(self, mock_proxies):
        """Ensure outputs are standard floats for downstream processing."""
        ref, tgt = mock_proxies
        dy, dx, sad = find_best_float_offset(ref, tgt, 0, 0, 4, 0, 0, 1.0, 1.0)

        assert isinstance(dy, float)
        assert isinstance(dx, float)
        assert isinstance(sad, float)


class TestFindBestOffset:
    @pytest.fixture
    def dummy_pyramid(self):
        """Creates dummy luma proxies for 3 pyramid levels."""
        return (
            np.zeros((32, 32), dtype=np.float32),  # Level 0 (1x)
            np.zeros((16, 16), dtype=np.float32),  # Level 1 (0.5x)
            np.zeros((8, 8), dtype=np.float32),  # Level 2 (0.25x)
        )

    @pytest.fixture
    def noise_params(self):
        return {"scales": np.array([0.01]), "offsets": np.array([0.001]), "scaler": 1.0}

    def test_coordinate_scaling_and_flow(self, dummy_pyramid, noise_params):
        """Verify that coarse-level results are correctly multiplied by 2 when passed as hints to the next level."""
        L0, L1, L2 = dummy_pyramid

        # We need to mock both the integer search and the final float refinement
        with (
            patch("pipeline_steps.align_and_merge.find_best_integer_offset") as mock_int,
            patch("pipeline_steps.align_and_merge.find_best_float_offset") as mock_float,
        ):
            # Setup mock returns: (dy, dx, sad)
            # Level 2 returns (1, 1, 0.5)
            # Level 1 should then receive hint (2, 2)
            # Level 0 should then receive hint (Level 1 result * 2)
            mock_int.side_effect = [
                (1, 1, 0.5),  # Level 2 result
                (2, 2, 0.3),  # Level 1 result
                (4, 4, 0.1),  # Level 0 result
            ]
            mock_float.return_value = (4.25, 4.25, 0.1)

            # Use .py_func to bypass JIT for the coordinator function
            result = find_best_offset.py_func(
                L0,
                L1,
                L2,
                L0,
                L1,
                L2,
                row_start=16,
                col_start=16,
                tile_size=8,
                search_radius=16,
                noise_scales=noise_params["scales"],
                noise_offsets=noise_params["offsets"],
                exposure_scaler=noise_params["scaler"],
            )

            # Check Level 2 call (row_start // 4 = 4)
            assert mock_int.call_args_list[0].kwargs["row_start"] == 4

            # Check Level 1 hint scaling (L2 result * 2)
            assert mock_int.call_args_list[1].kwargs["hint_dy"] == 2

            # Check Level 0 hint scaling (L1 result * 2)
            assert mock_int.call_args_list[2].kwargs["hint_dy"] == 4

            assert result == (4.25, 4.25, 0.1)

    def test_noise_floor_math_stability(self, dummy_pyramid, noise_params):
        """Verify inv_sigma calculation doesn't crash with zero inputs and applies the proxy_variance_scale."""
        L0, L1, L2 = dummy_pyramid  # All zeros

        with (
            patch("pipeline_steps.align_and_merge.find_best_integer_offset") as mock_int,
            patch("pipeline_steps.align_and_merge.find_best_float_offset") as mock_float,
        ):
            mock_int.return_value = (0, 0, 0.0)
            mock_float.return_value = (0.0, 0.0, 0.0)

            # Test with very high noise to see if it propagates to inv_sigma
            find_best_offset.py_func(
                L0,
                L1,
                L2,
                L0,
                L1,
                L2,
                0,
                0,
                8,
                4,
                noise_scales=np.array([1.0]),
                noise_offsets=np.array([1.0]),
                exposure_scaler=1.0,
            )

            # Retrieve the inv_sigma passed to the first call
            passed_inv_sigma = mock_int.call_args_list[0].kwargs["inv_sigma"]

            # With mean 0, var = (1*0 + 1) + (1*1*0 + 1^2*1) = 2.0
            # sigma_sq = 2.0 * 0.29 (proxy_variance_scale) = 0.58
            # inv_sigma = 1 / sqrt(0.58) approx 1.31
            assert 1.3 < passed_inv_sigma < 1.32


# ----------------------------------------- Merging Functions -----------------------------------------


class TestGetHannWindow2D:
    def test_hann_window_dimensions(self):
        """Ensure the window matches the requested tile size."""
        tile_size = 32
        window = get_hann_window_2d(tile_size)
        assert window.shape == (tile_size, tile_size)
        assert window.dtype == np.float32

    def test_hann_window_symmetry(self):
        """Ensure the window is symmetric along both axes."""
        tile_size = 16
        window = get_hann_window_2d(tile_size)

        # Check horizontal and vertical symmetry
        assert np.allclose(window, np.flipud(window))
        assert np.allclose(window, np.fliplr(window))
        # Check diagonal symmetry (it's a square outer product)
        assert np.allclose(window, window.T)

    def test_hann_window_unity_gain(self):
        """
        The most critical test: check the Partition of Unity property.

        When overlapped by 50% (stride = size // 2), weights must sum to 1.0.
        """
        tile_size = 32
        stride = tile_size // 2
        window = get_hann_window_2d(tile_size)

        # Simulate a 2x2 grid of overlapping tiles
        # The overlap area (the center of the grid) should sum to exactly 1.0
        canvas_size = tile_size + stride
        weight_map = np.zeros((canvas_size, canvas_size))

        offsets = [0, stride]
        for row_offset in offsets:
            for col_offset in offsets:
                weight_map[row_offset : row_offset + tile_size, col_offset : col_offset + tile_size] += window

        # Check the central 'stride x stride' region where all 4 tiles overlap
        overlap_region = weight_map[stride:tile_size, stride:tile_size]

        # In a 2D Hann window with 50% overlap, the sum is 1.0
        assert np.allclose(overlap_region, 1.0)

    def test_hann_window_boundary_values(self):
        """Check that the window tapers toward zero at the very edges."""
        tile_size = 8
        window = get_hann_window_2d(tile_size)

        # With the 0.5 offset, the values aren't exactly 0 at indices 0 and N-1,
        # but they should be small and identical.
        np.testing.assert_almost_equal(window[0, 0], window[7, 7])
        assert window[0, 0] < 0.05

        # The center of the window should have the highest weight
        center = tile_size // 2
        assert window[center, center] > window[0, 0]


class TestSampleRawBilinear:
    @pytest.fixture
    def bayer_grid(self):
        """
        Creates a 10x10 synthetic Bayer-like grid.

        Even rows/cols (0,0): 100.0 (e.g., Red)
        Even rows, Odd cols (0,1): 50.0  (e.g., Green 1)
        Odd rows, Even cols (1,0): 25.0  (e.g., Green 2)
        Odd rows, Odd cols (1,1): 10.0   (e.g., Blue)
        """
        grid = np.zeros((10, 10), dtype=np.float32)
        grid[0::2, 0::2] = 100.0
        grid[0::2, 1::2] = 50.0
        grid[1::2, 0::2] = 25.0
        grid[1::2, 1::2] = 10.0
        return grid

    def test_integer_multiples_of_two(self, bayer_grid):
        """Verify that offsets of 2, 4, etc., return exact pixel values (Fast Path)."""
        row_base, col_base = 2, 2
        # Shift by 2 pixels vertically and 4 pixels horizontally
        val = sample_raw_bilinear(bayer_grid, row_base, col_base, 2.0, 4.0)

        # Expected: bayer_grid[4, 6] which is an 'Even, Even' site (100.0)
        assert val == 100.0
        assert isinstance(val, (float, np.float32))

    def test_fractional_midpoint(self, bayer_grid):
        """Verify that a 1.0 offset (midpoint of a 2-pixel stride) returns a 50/50 mix."""
        row_base, col_base = 2, 2
        # row_offset 1.0 is halfway between index 2 and index 4
        # bayer_grid[2, 2] = 100.0, bayer_grid[4, 2] = 100.0
        # result should be 100.0
        val_vertical = sample_raw_bilinear(bayer_grid, row_base, col_base, 1.0, 0.0)
        assert np.isclose(val_vertical, 100.0)

        # Now test across a gradient
        # Let's modify a pixel to create a difference
        bayer_grid[2, 4] = 200.0
        # Halfway between bayer_grid[2, 2] (100.0) and bayer_grid[2, 4] (200.0)
        val_horizontal = sample_raw_bilinear(bayer_grid, row_base, col_base, 0.0, 1.0)
        assert np.isclose(val_horizontal, 150.0)

    def test_subpixel_bilinear_quad(self, bayer_grid):
        """Verify interpolation within a 2x2 same-color quad."""
        # Setup a 2x2 grid of the same color (Red sites at 2,2; 2,4; 4,2; 4,4)
        # We'll set them to different values to test the bilinear mix
        row_base, col_base = 2, 2
        bayer_grid[2, 2] = 10.0
        bayer_grid[2, 4] = 20.0
        bayer_grid[4, 2] = 30.0
        bayer_grid[4, 4] = 40.0

        # Sample at offset (0.5, 0.5)
        # In our 2-pixel stride, 0.5 is 25% of the way to the next neighbor.
        # lerp_weight = 0.5 / 2.0 = 0.25
        val = sample_raw_bilinear(bayer_grid, row_base, col_base, 0.5, 0.5)

        # Expected calculation:
        # top_mix = 10 + 0.25 * (20 - 10) = 12.5
        # bottom_mix = 30 + 0.25 * (40 - 30) = 32.5
        # final = 12.5 + 0.25 * (32.5 - 12.5) = 17.5
        assert np.isclose(val, 17.5)

    def test_boundary_clamping_preserves_color(self, bayer_grid):
        """Verify that sampling near the edge doesn't pull signal from the wrong Bayer color (phase) due to naive index clamping."""
        height, width = bayer_grid.shape
        # Base is the last Red site (8, 8) in a 10x10 grid.
        row_base, col_base = 8, 8

        # If we sample with an offset of 0.5, row_bottom would be 10 (OOB).
        # Our phase-aware logic should clamp row_bottom to row_top (8).
        val = sample_raw_bilinear(bayer_grid, row_base, col_base, 0.5, 0.5)

        # If clamping works, it should ignore the fractional distance because
        # there is no 'next' Red neighbor to interpolate with.
        # Expected: exactly the value at [8, 8]
        assert np.isclose(val, bayer_grid[8, 8])

    def test_negative_offsets(self, bayer_grid):
        """Verify that negative offsets (moving 'up' or 'left') work as expected."""
        # Start in the middle
        row_base, col_base = 4, 4
        bayer_grid[2, 4] = 50.0
        bayer_grid[4, 4] = 100.0

        # Shift 'up' by 2 pixels
        val = sample_raw_bilinear(bayer_grid, row_base, col_base, -2.0, 0.0)
        assert val == 50.0

    def test_negative_fractional_offset(self, bayer_grid):
        """
        Verify sub-pixel interpolation with negative fractional offsets.

        This tests the transition logic when moving 'backwards' on the grid.
        """
        # Start at a known center point
        row_base, col_base = 4, 4

        # Setup the 2x2 same-color quad that sits 'above' and 'left' of our base
        # Neighbors are at (2,2), (2,4), (4,2), (4,4)
        bayer_grid[2, 2] = 100.0  # Top-Left
        bayer_grid[2, 4] = 200.0  # Top-Right
        bayer_grid[4, 2] = 300.0  # Bottom-Left
        bayer_grid[4, 4] = 400.0  # Bottom-Right (the original base)

        # We want to sample halfway between row 2 and 4, and halfway between col 2 and 4.
        # This corresponds to a shift of -1.0 on both axes relative to (4,4).
        # row_offset = -1.0 -> row_shift_base = floor(-1.0 / 2) * 2 = -2
        # row_lerp = (-1.0 - (-2)) / 2 = 0.5 (exact midpoint)
        val_mid = sample_raw_bilinear(bayer_grid, row_base, col_base, -1.0, -1.0)

        # Expected midpoint calculation:
        # (100 + 200 + 300 + 400) / 4 = 250.0
        assert np.isclose(val_mid, 250.0)

        # Test a more granular fractional shift: -0.5
        # row_shift_base = floor(-0.5 / 2) * 2 = -2
        # row_lerp = (-0.5 - (-2)) / 2 = 1.5 / 2 = 0.75
        # This means we are 75% of the way from row 2 toward row 4.
        val_granular = sample_raw_bilinear(bayer_grid, row_base, col_base, -0.5, -0.5)

        # Calculation:
        # top_mix = 100 + 0.75 * (200 - 100) = 175
        # bottom_mix = 300 + 0.75 * (400 - 300) = 375
        # final = 175 + 0.75 * (375 - 175) = 325.0
        assert np.isclose(val_granular, 325.0)


class TestMergeTile:
    @pytest.fixture
    def accumulators(self):
        # 16x16 buffers
        merged = np.zeros((16, 16), dtype=np.float32)
        weights = np.zeros((16, 16), dtype=np.float32)
        return merged, weights

    @pytest.fixture
    def window(self):
        # Simple flat window (all 1s) for easier math verification
        return np.ones((8, 8), dtype=np.float32)

    def test_reference_frame_merge(self, accumulators, window):
        """The reference frame should be merged with a robustness of 1.0 and no SNR scaling."""
        merged_acc, weights_acc = accumulators
        target = np.full((16, 16), 0.5, dtype=np.float32)

        # Merge an 8x8 tile at (0,0) as reference
        # Note: we use .py_func if merge_tile is jitted
        merge_tile.py_func(
            merged_acc,
            weights_acc,
            target,
            row_start=0,
            col_start=0,
            row_offset=0.0,
            col_offset=0.0,
            sad_score=0.0,
            tile_size=8,
            blending_window=window,
            k=1.0,
            exposure_scaler=1.0,
            is_reference=True,
        )

        # Reference pixels should be exactly 0.5, weights exactly 1.0
        assert np.all(weights_acc[:8, :8] == 1.0)
        assert np.all(merged_acc[:8, :8] == 0.5)

    def test_exposure_normalization(self, accumulators, window):
        """Target frame at 0.25 brightness with scaler 2.0 should result in 0.5 in accumulator."""
        merged_acc, weights_acc = accumulators
        target = np.full((16, 16), 0.25, dtype=np.float32)

        # exposure_scaler=2.0 means target is 1 stop darker than reference
        # robustness k is high so exp(-0.1/10) is approx 1.0
        merge_tile.py_func(
            merged_acc,
            weights_acc,
            target,
            row_start=0,
            col_start=0,
            row_offset=0.0,
            col_offset=0.0,
            sad_score=0.0,
            tile_size=8,
            blending_window=window,
            k=100.0,
            exposure_scaler=2.0,
            is_reference=False,
        )

        # SNR weight for darker frame (scaler 2.0) = 1/2 = 0.5
        # Pixel value added = (0.25 * 2.0) * 0.5 = 0.25
        # Total weight = 0.5
        # Resulting average (merged/weight) would be 0.5.
        assert weights_acc[0, 0] == 0.5
        assert merged_acc[0, 0] == 0.25

    def test_saturation_soft_threshold(self, accumulators, window):
        """Pixels near saturation should have their weights tapered off."""
        merged_acc, weights_acc = accumulators
        # 0.93 is in the 'danger zone' (threshold 0.95 - softness 0.05 = 0.90)
        target = np.full((16, 16), 0.93, dtype=np.float32)

        merge_tile.py_func(
            merged_acc,
            weights_acc,
            target,
            row_start=0,
            col_start=0,
            row_offset=0.0,
            col_offset=0.0,
            sad_score=0.0,
            tile_size=8,
            blending_window=window,
            k=100.0,
            exposure_scaler=1.0,
            saturation_threshold=0.95,
            is_reference=False,
        )

        # Fade calculation: (0.95 - 0.93) / 0.05 = 0.4
        # Since snr_weight=1 and robustness=1, total weight should be 0.4
        np.testing.assert_allclose(weights_acc[0, 0], 0.4, atol=1e-5)

    def test_robustness_rejection(self, accumulators, window):
        """Tiles with very high SAD scores (mismatches) should not be merged."""
        merged_acc, weights_acc = accumulators
        target = np.ones((16, 16), dtype=np.float32)

        # SAD=10 with k=1.0 -> exp(-10) = 0.000045 (below 1e-4 threshold)
        merge_tile.py_func(
            merged_acc,
            weights_acc,
            target,
            row_start=0,
            col_start=0,
            row_offset=0.0,
            col_offset=0.0,
            sad_score=10.0,
            tile_size=8,
            blending_window=window,
            k=1.0,
            exposure_scaler=1.0,
            is_reference=False,
        )

        assert np.all(weights_acc == 0)
        assert np.all(merged_acc == 0)

    def test_bilinear_sampling_integration(self, accumulators, window):
        """Verify that the function correctly calls the bilinear sampler with offsets."""
        merged_acc, weights_acc = accumulators
        target = np.zeros((16, 16), dtype=np.float32)

        # We mock the sampler to ensure it's receiving the correct float offsets
        with patch("pipeline_steps.align_and_merge.sample_raw_bilinear", return_value=0.7) as mock_sample:
            merge_tile.py_func(
                merged_acc,
                weights_acc,
                target,
                row_start=2,
                col_start=2,
                row_offset=0.5,
                col_offset=0.5,
                sad_score=0.0,
                tile_size=4,
                blending_window=np.ones((4, 4)),
                k=1.0,
                exposure_scaler=1.0,
            )

            # Should be called for every pixel in the 4x4 tile
            assert mock_sample.call_count == 16
            # Verify the offset was passed through
            args, _ = mock_sample.call_args
            assert args[3] == 0.5  # row_offset


class TestMergeImages:
    @pytest.fixture
    def mock_burst(self):
        # Two 64x64 dummy images
        img1 = np.full((64, 64), 0.2, dtype=np.float32)
        img2 = np.full((64, 64), 0.2, dtype=np.float32)
        return [img1, img2]

    @pytest.fixture
    def mock_metadata(self):
        return [
            {"ExposureTime": 0.01, "ISO": 100, "BlackLevel": 0, "color_desc": "RGBG", "raw_pattern": [[2, 3], [1, 0]]},
            {"ExposureTime": 0.01, "ISO": 100, "BlackLevel": 0, "color_desc": "RGBG", "raw_pattern": [[2, 3], [1, 0]]},
        ]

    def test_input_validation(self, mock_burst, mock_metadata):
        """Ensure the function catches bad inputs before starting the heavy lifting."""
        # Test: Only one image
        with pytest.raises(ValueError, match="At least two images"):
            merge_images(mock_burst[:1], mock_metadata[:1])

        # Test: Bad search radius
        with pytest.raises(ValueError, match="multiple of 8"):
            merge_images(mock_burst, mock_metadata, max_search_radius=7)

    def test_sharpest_image_selection_logic(self, mock_burst, mock_metadata):
        # 1. Ensure unique objects
        img_target = mock_burst[0]
        img_ref = mock_burst[1]

        # 2. Force index 1 to be the "sharpest"
        with (
            patch("pipeline_steps.align_and_merge.find_sharpest_image_idx", return_value=1),
            patch("pipeline_steps.align_and_merge._parallel_tile_processor") as mock_proc,
            patch("pipeline_steps.align_and_merge.get_luma_proxy", return_value=np.zeros((32, 32))),
            patch("pipeline_steps.align_and_merge.get_noise_profile", return_value=(np.zeros(1), np.zeros(1))),
            patch("pipeline_steps.align_and_merge.get_photometric_scalers", return_value=[1.0, 1.0]),
            patch("pipeline_steps.align_and_merge.get_hann_window_2d", return_value=np.ones((32, 32))),
        ):
            # No .py_func needed if merge_images isn't @njit
            merge_images(mock_burst, mock_metadata)

            calls = mock_proc.call_args_list
            assert len(calls) == 2

            # Since the loop goes 0 -> 1:
            # Call 0 should be index 0 (Target)
            # Call 1 should be index 1 (Reference)

            # Inspect Call for index 0
            args0, kwargs0 = calls[0]
            assert kwargs0["is_reference"] is False
            assert kwargs0["target_image"] is img_target  # Checking identity

            # Inspect Call for index 1
            args1, kwargs1 = calls[1]
            assert kwargs1["is_reference"] is True
            assert kwargs1["target_image"] is img_ref  # Checking identity

    def test_normalization_step(self, mock_burst, mock_metadata):
        """Ensure the final division by weights_accumulator happens correctly."""
        # Create a tiny 16x16 scenario
        img = np.full((16, 16), 0.5, dtype=np.float32)
        burst = [img, img]

        # We mock the parallel processor to manually fill the accumulators
        def mock_fill(**kwargs):
            kwargs["merged_accumulator"] += 1.0  # Add 1.0 to every pixel
            kwargs["weights_accumulator"] += 2.0  # Total weight is 2.0

        with (
            patch("pipeline_steps.align_and_merge._parallel_tile_processor", side_effect=mock_fill),
            patch("pipeline_steps.align_and_merge.get_luma_proxy", return_value=np.zeros((8, 8))),
            patch("pipeline_steps.align_and_merge.get_noise_profile", return_value=(np.array([0]), np.array([0]))),
        ):
            result = merge_images(burst, mock_metadata, tile_size=8)

            # Final result should be 1.0 / 2.0 = 0.5 across the whole image
            assert np.allclose(result, 0.5)

    def test_k_adaptive_scaling(self, mock_burst, mock_metadata):
        """Verify that higher ISO results in a larger (more trusting) k_adaptive."""
        # ISO 100 -> stops = 0 -> k = 1.0
        # ISO 400 -> stops = 2 -> k = 1.0 + 0.5*2 = 2.0
        mock_metadata[0]["ISO"] = 400

        with (
            patch("pipeline_steps.align_and_merge._parallel_tile_processor") as mock_proc,
            patch("pipeline_steps.align_and_merge.get_luma_proxy", return_value=np.zeros((32, 32))),
            patch("pipeline_steps.align_and_merge.get_noise_profile", return_value=(np.array([0]), np.array([0]))),
        ):
            merge_images(mock_burst, mock_metadata)

            # Check the k_adaptive passed to the processor
            passed_k = mock_proc.call_args.kwargs["k_adaptive"]
            assert passed_k == 2.0

    def test_merge_no_motion_reduces_noise_vs_ground_truth(self):
        """
        Verifies that merging a noisy burst reduces noise relative to the known ground-truth image.
        Uses a local generator for determinism and aligns noise generation with metadata.
        """
        # 1. Setup local RNG for bit-perfect determinism
        rng = np.random.default_rng(42)

        # 2. Define ground truth and noise parameters
        # Matches "0.01 0.0001" in NoiseProfile: var = 0.01 * 0.5 + 0.0001
        base = np.full((128, 128), 0.5, dtype=np.float32)
        noise_var = 0.01 * 0.5 + 0.0001
        noise_sigma = np.sqrt(noise_var)

        # 3. Create noisy burst (no motion)
        burst = [np.clip(base + rng.normal(0, noise_sigma, base.shape), 0, 1).astype(np.float32) for _ in range(5)]

        # 4. Metadata (Aligned with generated noise)
        metadata = [
            {
                "ExposureTime": "1/100",
                "ISO": 100,
                "color_desc": ["R", "G", "G", "B"],
                "raw_pattern": [[0, 1], [2, 3]],
                "NoiseProfile": "0.01 0.0001 0.01 0.0001 0.01 0.0001",
                "CFAPlaneColor": "Red,Green,Blue",
            }
        ] * len(burst)

        # 5. Execute Merge
        merged = merge_images(burst, metadata)

        # 6. Evaluation
        input_noise = np.mean([np.std(img - base) for img in burst])
        output_noise = np.std(merged - base)

        assert merged.shape == base.shape
        assert np.isfinite(merged).all()

        # Assert noise reduction
        # We allow a small epsilon for clipping effects, but it should be significantly lower.
        assert output_noise < input_noise

    def test_merge_with_motion_reduces_noise(self):
        """
        Verifies that merging a noisy burst with motion (shifting) still reduces noise.
        This tests the pipeline's ability to either align or robustly handle pixel shifts.
        """
        # 1. Setup local RNG
        rng = np.random.default_rng(24)

        # 2. Define ground truth (use a simple pattern so motion is detectable)
        # A flat field is bad for motion testing; let's add a simple gradient or block
        base = np.zeros((128, 128), dtype=np.float32)
        base[32:96, 32:96] = 0.5  # A gray square in the middle

        noise_var = 0.01 * 0.5 + 0.0001
        noise_sigma = np.sqrt(noise_var)

        # 3. Create noisy burst with motion (shifting each frame by 1 pixel)
        burst = []
        for i in range(5):
            # Shift the base image by 'i' pixels in both x and y
            shifted_base = np.roll(base, shift=(i, i), axis=(0, 1))

            # Add noise
            noisy_frame = shifted_base + rng.normal(0, noise_sigma, base.shape)
            burst.append(np.clip(noisy_frame, 0, 1).astype(np.float32))

        # 4. Metadata (ISO 1600 to keep k_adaptive high enough for merge)
        metadata = [
            {
                "ExposureTime": "1/100",
                "ISO": 1600,
                "color_desc": ["R", "G", "G", "B"],
                "raw_pattern": [[0, 1], [2, 3]],
                "NoiseProfile": "0.01 0.0001 0.01 0.0001 0.01 0.0001",
                "CFAPlaneColor": "Red,Green,Blue",
            }
        ] * len(burst)

        # 5. Execute Merge
        merged = merge_images(burst, metadata)

        # 6. Evaluation
        # Input noise: standard deviation of (Frame - Ground Truth)
        input_noise = np.mean([np.std(img - base) for img in burst])
        output_noise = np.std(merged - base)

        # Check for basic validity
        assert merged.shape == base.shape
        assert np.isfinite(merged).all()

        # If alignment works, output_noise should be significantly lower than input_noise
        # If alignment is off but robustness is on, output_noise will be roughly equal to input_noise
        # (because it defaults to the reference frame).
        assert output_noise < input_noise

    def test_merge_images_no_motion_improves_snr(self):
        """Verifies that merging improves signal-to-noise ratio (SNR) compared to individual noisy frames with bit-perfect determinism."""

        # Use a local generator for guaranteed determinism
        rng = np.random.default_rng(0)

        # 1. Setup base signal
        base = np.ones((128, 128), dtype=np.float32) * 0.5

        # 2. Generate noisy burst
        # Standard deviation: sqrt(0.01 * 0.5 + 0.0001) ≈ 0.0714
        noise_sigma = np.sqrt(0.01 * base + 0.0001)

        # Use the generator for the normal distribution
        burst = [np.clip(base + rng.normal(0, noise_sigma), 0, 1).astype(np.float32) for _ in range(5)]

        # 3. Metadata setup
        # Ensure NoiseProfile string matches the params used in generation
        metadata = [
            {
                "ExposureTime": "1/100",
                "ISO": 100,
                "color_desc": ["R", "G", "G", "B"],
                "raw_pattern": [[0, 1], [2, 3]],
                "NoiseProfile": "0.01 0.0001 0.01 0.0001 0.01 0.0001",
                "CFAPlaneColor": "Red,Green,Blue",
            }
        ] * 5

        # 4. Process
        merged = merge_images(burst, metadata)

        # 5. Metrics Calculation
        # Input noise: standard deviation of (Frame - Ground Truth)
        input_noise = np.mean([np.std(img - base) for img in burst])
        output_noise = np.std(merged - base)

        # SNR
        snr_in = 0.5 / (input_noise + 1e-8)
        snr_out = 0.5 / (output_noise + 1e-8)

        # Assert improvement
        # With 5 frames, we expect roughly sqrt(5) ≈ 2.2x improvement in a perfect world
        assert snr_out > snr_in

    def test_merge_with_motion_improves_snr(self):
        """
        Verifies that merging a noisy burst with motion (shifting) still reduces noise.
        This tests the pipeline's ability to either align or robustly handle pixel shifts.
        """
        # 1. Setup local RNG
        rng = np.random.default_rng(24)

        # 2. Define ground truth (use a simple pattern so motion is detectable)
        # A flat field is bad for motion testing; let's add a simple gradient or block
        base = np.zeros((128, 128), dtype=np.float32)
        base[32:96, 32:96] = 0.5  # A gray square in the middle

        noise_var = 0.01 * 0.5 + 0.0001
        noise_sigma = np.sqrt(noise_var)

        # 3. Create noisy burst with motion (shifting each frame by 1 pixel)
        burst = []
        for i in range(5):
            # Shift the base image by 'i' pixels in both x and y
            shifted_base = np.roll(base, shift=(i, i), axis=(0, 1))

            # Add noise
            noisy_frame = shifted_base + rng.normal(0, noise_sigma, base.shape)
            burst.append(np.clip(noisy_frame, 0, 1).astype(np.float32))

        # 4. Metadata (ISO 1600 to keep k_adaptive high enough for merge)
        metadata = [
            {
                "ExposureTime": "1/100",
                "ISO": 1600,
                "color_desc": ["R", "G", "G", "B"],
                "raw_pattern": [[0, 1], [2, 3]],
                "NoiseProfile": "0.01 0.0001 0.01 0.0001 0.01 0.0001",
                "CFAPlaneColor": "Red,Green,Blue",
            }
        ] * len(burst)

        # 5. Execute Merge
        merged = merge_images(burst, metadata)

        # 6. Evaluation
        # Input noise: standard deviation of (Frame - Ground Truth)
        input_noise = np.mean([np.std(img - base) for img in burst])
        output_noise = np.std(merged - base)

        # SNR
        snr_in = 0.5 / (input_noise + 1e-8)
        snr_out = 0.5 / (output_noise + 1e-8)

        # Assert improvement
        # With 5 frames, we expect roughly sqrt(5) ≈ 2.2x improvement in a perfect world
        assert snr_out > snr_in

    def test_merge_parallel_tile_processor_jit_no_jit_consistency(self):
        """Verifies that the Numba-accelerated (parallel) version and the pure Python version produce identical results."""
        # 1. Setup local RNG
        rng = np.random.default_rng(24)

        # 2. Define ground truth
        base = np.zeros((128, 128), dtype=np.float32)
        base[32:96, 32:96] = 0.5  # A gray square

        noise_var = 0.01 * 0.5 + 0.0001
        noise_sigma = np.sqrt(noise_var)

        # 3. Create noisy burst with motion
        burst = []
        for i in range(5):
            shifted_base = np.roll(base, shift=(i, i), axis=(0, 1))
            noisy_frame = shifted_base + rng.normal(0, noise_sigma, base.shape)
            burst.append(np.clip(noisy_frame, 0, 1).astype(np.float32))

        metadata = [
            {
                "ExposureTime": "1/100",
                "ISO": 1600,
                "color_desc": "RGBG",  # Adjusted for typical metadata structure
                "raw_pattern": [[0, 1], [3, 2]],
                "NoiseProfile": "0.01 0.0001 0.01 0.0001 0.01 0.0001",
                "CFAPlaneColor": "Red,Green,Blue",
            }
        ] * len(burst)

        # --- 4. Verify Numba Configuration ---
        # We check the dispatcher for the parallel tile processor
        # (assuming it's the core JIT function)
        from pipeline_steps.align_and_merge import _parallel_tile_processor

        target_parallel = _parallel_tile_processor.targetoptions.get("parallel", False)
        target_fastmath = _parallel_tile_processor.targetoptions.get("fastmath", False)

        assert target_parallel is True, "Numba parallelism is not enabled!"
        assert target_fastmath is True, "Numba fastmath is not enabled!"

        # --- 5. Execute and Compare ---

        # a) Execute Parallel Numba variant
        # This uses the decorated function as defined in your code
        merged_numba = merge_images(burst, metadata)

        # b) Execute Pure Python variant
        # We use .py_func to bypass the Numba JIT wrapper entirely
        # We temporarily swap the JIT function with its Python original
        import pipeline_steps.align_and_merge

        original_processor = pipeline_steps.align_and_merge._parallel_tile_processor
        try:
            pipeline_steps.align_and_merge._parallel_tile_processor = original_processor.py_func
            merged_python = pipeline_steps.align_and_merge.merge_images(burst, metadata)
        finally:
            # Restore the JIT version
            pipeline_steps.align_and_merge._parallel_tile_processor = original_processor

        # 6. Evaluation
        # We check for bit-exact identity.
        # Note: If fastmath is on, we might allow a very small epsilon (1e-7)
        # due to reordering of floating point additions.
        np.testing.assert_allclose(
            merged_numba, merged_python, atol=1e-7, err_msg="Numba parallel and Python results diverged!"
        )

    def test_merge_parallel_tile_processor_verify_no_racing_condition(self):
        """
        Stress tests the parallel implementation to catch race conditions.
        Runs the same merge multiple times and asserts bit-exact (or epsilon) identity.
        """
        # 1. Setup - Use a larger image to increase the window for collisions
        # 256x256 provides significantly more tiles and overlap points
        rng = np.random.default_rng(42)
        shape = (256, 256)

        # Create a base with some structure
        base = np.zeros(shape, dtype=np.float32)
        base[64:192, 64:192] = 0.8

        # 2. Create a noisy burst (10 frames to increase accumulation work)
        burst = []
        for i in range(10):
            # Add random sub-pixel shifts to force different alignment offsets
            shift = rng.uniform(-2, 2, size=2)
            # Using a simple roll for the test, or a real shift if available
            frame = np.roll(base, shift=shift.astype(int), axis=(0, 1))
            noise = rng.normal(0, 0.05, shape)
            burst.append(np.clip(frame + noise, 0, 1).astype(np.float32))

        metadata = [
            {
                "ExposureTime": "1/100",
                "ISO": 1600,
                "color_desc": "RGBG",  # Adjusted for typical metadata structure
                "raw_pattern": [[0, 1], [3, 2]],
                "NoiseProfile": "0.01 0.0001 0.01 0.0001 0.01 0.0001",
                "CFAPlaneColor": "Red,Green,Blue",
            }
        ] * len(burst)

        # 3. Execution - Run multiple iterations
        # Race conditions are probabilistic; running 10+ times increases catch rate.
        results = []
        iterations = 10

        for i in range(iterations):
            # We call the actual JIT-decorated function here
            results.append(merge_images(burst, metadata))

        # 4. Verification
        # Compare every run against the first run
        reference = results[0]
        for i in range(1, iterations):
            # Use a very tight tolerance.
            # Even with fastmath, the phased approach should be deterministic
            # because the order of operations within each phase is consistent.
            try:
                np.testing.assert_allclose(
                    reference,
                    results[i],
                    atol=1e-9,
                    err_msg=f"Race condition detected! Iteration {i} differs from Iteration 0.",
                )
            except AssertionError as e:
                # Calculate how many pixels actually differed to help debugging
                diff_mask = ~np.isclose(reference, results[i], atol=1e-9)
                num_diffs = np.sum(diff_mask)
                print(f"\n[DEBUG] Failure in iteration {i}: {num_diffs} pixels differed.")
                raise e
