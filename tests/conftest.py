from pathlib import Path

import numpy as np
import pytest

from ispfoundry.datasets import Metadata


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
    return Metadata(
        file_path=Path("test_file_path"),
        black_levels=np.array([50, 60, 70, 80]),  # R, Gr, Gb, B
        white_level=1000,
    )
