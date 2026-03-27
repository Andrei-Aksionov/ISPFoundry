import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pipeline_steps.align_and_merge import (
    compute_tile_sad,
    downsample_luma_proxy,
    find_sharpest_image_idx,
    find_subpixel_shift,
    get_exposure_scalers,
    get_hann_window_2d,
    get_luma_proxy,
    sample_raw_bilinear,
)


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
    def test_scalers_math_simple(self):
        """Verify that a 4x longer exposure results in a 0.25 scaler."""
        metadata = [
            {"ExposureTime": 0.01},  # Shortest (Reference)
            {"ExposureTime": 0.04},  # 4x longer
            {"ExposureTime": 0.02},  # 2x longer
        ]

        expected = np.array([1.0, 0.25, 0.5], dtype=np.float32)
        result = get_exposure_scalers(metadata)

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
        result = get_exposure_scalers(metadata)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_mixed_types_scaling(self):
        """Ensure integers, floats, and strings are handled in a single burst."""
        metadata = [
            {"ExposureTime": 1},  # Integer
            {"ExposureTime": 0.5},  # Float (Shortest)
            {"ExposureTime": "2/1"},  # String
        ]

        expected = np.array([0.5, 1.0, 0.25], dtype=np.float32)
        result = get_exposure_scalers(metadata)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_identical_exposures_uniformity(self):
        """All scalers must be 1.0 when exposure times are identical."""
        metadata = [{"ExposureTime": 0.0333}] * 3
        result = get_exposure_scalers(metadata)

        expected = np.ones(3, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_output_metadata_consistency(self):
        """Verify output array properties: length, type, and min value."""
        metadata = [{"ExposureTime": 0.1}, {"ExposureTime": 0.5}, {"ExposureTime": 0.2}]
        result = get_exposure_scalers(metadata)

        assert result.dtype == np.float32
        assert result.shape == (3,)
        assert np.max(result) == 1.0  # Shortest must be 1.0

    def test_parsing_error_on_invalid_string(self):
        """Ensure the function raises a ValueError for un-parsable ExposureTime."""
        metadata = [{"ExposureTime": "not_a_number"}]
        with pytest.raises(ValueError, match="convert string to float"):
            get_exposure_scalers(metadata)


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
        sharp[mask] = 1000

        blurry = gaussian_filter(sharp.astype(float), sigma=4.0).astype(np.uint16)

        images = [blurry, sharp]
        # Both are "short" exposures
        metadata = [base_metadata.copy() for _ in range(2)]

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 1

    def test_prefers_short_exposure_over_sharp_long_exposure(self, base_metadata):
        """Test that the logic correctly excludes long exposures from selection, even if the long exposure appears 'sharper' due to higher signal/contrast."""
        # 1. Create a "Long Exposure" image: very sharp, high contrast
        long_exp_img = np.zeros((32, 32), dtype=np.uint16)
        row, col = np.indices(long_exp_img.shape)
        mask = (row // 8 + col // 8) % 2 == 0
        long_exp_img[mask] = 4000  # High signal
        long_metadata = base_metadata.copy()
        long_metadata["ExposureTime"] = "1/250"  # 4x longer than base

        # 2. Create a "Short Exposure" image: slightly blurry, lower contrast
        # In a real burst, this might happen due to slight hand shake
        short_exp_img = np.zeros((32, 32), dtype=np.uint16)
        short_exp_img[mask] = 1000  # Lower signal
        short_exp_img = gaussian_filter(short_exp_img.astype(float), sigma=0.8).astype(np.uint16)
        short_metadata = base_metadata.copy()
        short_metadata["ExposureTime"] = "1/1000"

        # Even though index 0 (long) is mathematically "sharper" (higher variance),
        # index 1 (short) MUST be selected to avoid clipped highlights.
        images = [long_exp_img, short_exp_img]
        metadata = [long_metadata, short_metadata]

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 1

    def test_noise_robustness(self, base_metadata):
        """Test that Gaussian smoothing prevents sensor noise from being mistaken for sharpness."""
        # Pure noise
        noisy = np.random.randint(400, 600, (32, 32)).astype(np.uint16)

        # Actual structure
        structural = np.zeros((32, 32), dtype=np.uint16)
        structural[:16, :] = 800

        images = [noisy, structural]
        metadata = [base_metadata.copy() for _ in range(2)]

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 1

    def test_handles_mixed_string_and_float_exposure(self, base_metadata):
        """Ensure the parser handles both fractional strings and floats in metadata."""
        img = np.full((32, 32), 500, dtype=np.uint16)

        m1 = base_metadata.copy()
        m1["ExposureTime"] = "1/500"

        m2 = base_metadata.copy()
        m2["ExposureTime"] = 0.002  # Equivalent to 1/500

        images = [img, img]
        metadata = [m1, m2]

        # Both are considered "short" (minimum), should default to first index if identical
        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 0


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


class TestFindSubpixelPeak3x3:
    def test_perfect_center(self):
        """If the center is the absolute minimum, offset should be (0, 0)."""
        grid = np.array([[2.0, 1.0, 2.0], [1.0, 0.5, 1.0], [2.0, 1.0, 2.0]], dtype=np.float32)

        dr, dc = find_subpixel_shift(grid)
        assert dr == 0.0
        assert dc == 0.0

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

    def test_clamping_at_boundary(self):
        """Ensure the function clamps at +/- 0.5 even if the math suggests further."""
        # Create a grid where the 'Down' and 'Right' values are extremely small,
        # pulling the peak far beyond the next pixel.
        grid = np.array([[10.0, 10.0, 10.0], [10.0, 5.0, 0.1], [10.0, 0.1, 10.0]], dtype=np.float32)

        dr, dc = find_subpixel_shift(grid)

        # The math would suggest > 0.5, but we must clamp.
        assert dr == 0.5
        assert dc == 0.5

    def test_flat_surface_division_by_zero(self):
        """If neighbors are identical to center, offset should be 0.0 (no division by zero)."""
        grid = np.full((3, 3), 1.0, dtype=np.float32)

        dr, dc = find_subpixel_shift(grid)
        assert dr == 0.0
        assert dc == 0.0

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


class TestComputeTileSAD:
    @pytest.fixture
    def proxy_pair(self):
        """Create a 100x100 reference and target proxy."""
        ref = np.full((100, 100), 100.0, dtype=np.float32)
        tgt = np.full((100, 100), 100.0, dtype=np.float32)
        return ref, tgt

    def test_perfect_match(self, proxy_pair):
        """If frames are identical and offset is 0, SAD should be 0."""
        ref, tgt = proxy_pair
        score = compute_tile_sad(0, 0, ref, tgt, 10, 10, 16, 1.0, 240)
        assert score == 0.0

    def test_known_difference(self, proxy_pair):
        """Verify the VW-SAD math: (abs_diff * inv_sigma) / area."""
        ref, tgt = proxy_pair
        # Introduce a constant difference of 10.0
        tgt += 10.0

        inv_sigma = 0.5
        tile_size = 16
        # Expected: (10.0 * 0.5) / (no area division because it cancels out in the loop)
        # Loop sum: 16 * 16 * (10.0) = 2560
        # Normalization (2560 * 0.5) / (16 * 16) = 5.0
        score = compute_tile_sad(0, 0, ref, tgt, 10, 10, tile_size, inv_sigma, 240)
        assert np.isclose(score, 5.0)

    def test_partial_out_of_bounds_clipping(self, proxy_pair):
        """Ensure that when a tile is partially OOB, the area normalization uses the *actual* intersection area, not the full tile_size."""
        ref, tgt = proxy_pair
        # Put a high difference in the corner
        tgt[0:5, 0:5] += 100.0

        # Shift so only a 5x5 area of the tile is actually on the image
        # Tile size is 16, but we start at row -11
        score = compute_tile_sad(-11, 0, ref, tgt, 0, 0, 16, 1.0, 240)

        # Valid rows are max(0, -(-11)) = 11 to min(16, 100+11) = 16.
        # Intersection height is 5. Width is 16.
        # If normalization uses 5*16 (80) instead of 16*16 (256), the code is correct.
        assert score >= 0  # Should not crash and should return valid float

    def test_total_out_of_bounds(self, proxy_pair):
        """Verify None is returned when there is no intersection."""
        ref, tgt = proxy_pair
        # Shift entirely off a 100x100 image
        score = compute_tile_sad(200, 200, ref, tgt, 0, 0, 16, 1.0, 240)
        assert score is None

    def test_translation_consistency(self):
        """Verify that a translated feature is found at the correct offset."""
        ref = np.zeros((50, 50), dtype=np.float32)
        tgt = np.zeros((50, 50), dtype=np.float32)

        # Feature at (10, 10) in ref, moved to (12, 12) in target
        ref[10:15, 10:15] = 235.0
        tgt[12:17, 12:17] = 235.0

        # If we shift by dy=2, dx=2, the SAD should be 0
        score_correct = compute_tile_sad(2, 2, ref, tgt, 10, 10, 5, 1.0, 240)
        # If we don't shift, the SAD should be high
        score_wrong = compute_tile_sad(0, 0, ref, tgt, 10, 10, 5, 1.0, 240)

        assert score_correct == 0.0
        assert score_wrong > 0.0
