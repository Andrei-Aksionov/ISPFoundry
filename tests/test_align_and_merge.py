import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pipeline_steps.align_and_merge import find_sharpest_image_idx, get_luma_proxy


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
        assert np.isclose(proxy[0, 0], expected_value)

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


class TestFindSharpestImageIdx:
    @pytest.fixture
    def base_metadata(self):
        return {"color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]], "ISO": 100}

    def test_selects_sharpest_image(self, base_metadata):
        """Test that a sharp image is preferred over a blurred one."""
        # Image 0: Sharp edges (8x8 blocks so they survive 2x2 downsampling)
        sharp = np.zeros((32, 32), dtype=np.uint16)
        row, col = np.indices(sharp.shape)
        mask = (row // 8 + col // 8) % 2 == 0
        sharp[mask] = 1000

        # Using a heavy blur on the sharp image to create a realistic 'blurry' frame
        blurry = gaussian_filter(sharp.astype(float), sigma=4.0).astype(np.uint16)

        images = [blurry, sharp]
        metadata = [base_metadata] * 2

        best_idx = find_sharpest_image_idx(images, metadata)

        # The sharp image has high-contrast transitions that survive the proxy
        # downsampling and the sigma=0.5 smoothing.
        assert best_idx == 1

    def test_noise_robustness(self, base_metadata):
        """Test that the Gaussian blur prevents high-frequency noise from winning."""
        # Image 0: Pure random noise (high variance, but not 'sharp' detail)
        noisy = np.random.randint(400, 600, (32, 32)).astype(np.uint16)

        # Image 1: Clean image with actual structural edges
        # We make a strong edge that survives the sigma=0.5 blur
        structural = np.zeros((32, 32), dtype=np.uint16)
        structural[:16, :] = 800

        images = [noisy, structural]
        metadata = [base_metadata] * 2

        best_idx = find_sharpest_image_idx(images, metadata)

        # With sigma=0.5, the structural edge should yield higher Laplacian variance
        # than the suppressed random noise.
        assert best_idx == 1

    def test_identical_images(self, base_metadata):
        """Test that it returns the first index if all images are identical."""
        img = np.full((32, 32), 500, dtype=np.uint16)
        images = [img, img, img]
        metadatas = [base_metadata] * 3

        best_idx = find_sharpest_image_idx(images, metadatas)
        assert best_idx == 0

    def test_handles_different_exposure_proxies(self, base_metadata):
        """Test that it handles images with different brightness levels correctly."""
        # Brightness alone shouldn't necessarily dictate sharpness,
        # but higher contrast edges should score higher.

        low_contrast = np.zeros((32, 32), dtype=np.uint16)
        row, col = np.indices(low_contrast.shape)
        mask = (row // 8 + col // 8) % 2 == 0
        low_contrast[mask] = 100

        high_contrast = low_contrast.copy()
        high_contrast[mask] *= 10

        images = [low_contrast, high_contrast]
        metadata = [base_metadata] * 2

        best_idx = find_sharpest_image_idx(images, metadata)
        assert best_idx == 1
