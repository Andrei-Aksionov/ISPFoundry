import numpy as np

from pipeline_steps.align_and_merge import get_luma_proxy


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

        # Calculation: (100 * 0.15) + (200 * 0.35) + (300 * 0.35) + (400 * 0.15)
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
