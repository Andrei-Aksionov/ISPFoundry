from typing import Sequence

import numpy as np
from loguru import logger

from ispfoundry import ISPStep, register_step
from ispfoundry.datasets import Metadata


def retrieve_black_levels(raw_image: np.ndarray, metadata: Metadata) -> np.ndarray:
    """
    Retrieves black level values from metadata or, if absent, calculates them from the image.

    Args:
        raw_image: The input raw image.
        metadata: Metadata class containing metadata of the image.

    Returns:
        Array of black level values.

    """

    black_levels = metadata.black_levels
    if black_levels is None or np.allclose(black_levels, 0):
        logger.info("Metadata missing per-channel black levels - estimating from CFA minima.")

        black_levels = np.zeros(4, dtype=np.float32)
        for idx in range(4):
            row_offset, col_offset = divmod(idx, 2)
            black_levels[idx] = raw_image[row_offset::2, col_offset::2].min()

    return black_levels


def subtract_black_levels(raw_image: np.ndarray, metadata: Metadata, inplace: bool = False) -> np.ndarray:
    """
    Subtracts black levels from the raw image.

    Raw sensor data contains an offset introduced by the camera’s sensor and readout electronics.
    Even when no light hits the sensor, the output signal is not zero due to this offset, known as the black level.

    Args:
        raw_image: The input raw image as a NumPy array in float32.
        metadata: Metadata class containing metadata of the image.
        inplace: Whether to perform the operation in-place.

    Returns:
        The raw image with black levels subtracted.

    Raises:
        ValueError: If Raw image dtype is an unsigned integer

    """

    if np.issubdtype(raw_image.dtype, np.unsignedinteger):
        raise ValueError(
            "Raw image must not be of unsigned integer type to avoid underflow issues after subtraction. "
            f"Expected to be dtype of float32, but got `{raw_image.dtype}`"
        )

    raw_image = raw_image if inplace else raw_image.copy()

    black_levels = retrieve_black_levels(raw_image, metadata)

    for idx, black_level in enumerate(black_levels):
        row_offset, col_offset = divmod(idx, 2)
        raw_image[row_offset::2, col_offset::2] -= black_level

    return raw_image


def normalize_image(raw_image: np.ndarray, metadata: Metadata, inplace: bool = False) -> np.ndarray:
    """
    Normalizes the raw image into range [0, 1] using black and white levels.

    Args:
        raw_image: The input raw image.
        metadata: Metadata class containing metadata of the image.
        inplace: Whether to perform the operation in-place.

    Returns:
        The normalized image.

    """

    raw_image = raw_image if inplace else raw_image.copy()

    white_level = metadata.white_level
    black_levels = retrieve_black_levels(raw_image, metadata)

    for idx, black_level in enumerate(black_levels):
        row_offset, col_offset = divmod(idx, 2)
        denominator = white_level - black_level
        raw_image[row_offset::2, col_offset::2] /= denominator

    return raw_image


@register_step(ISPStep.BLACK_LEVEL_SUBTRACTION)
def apply_black_level_subtraction(
    image_inputs: np.ndarray,
    metadata: Sequence[Metadata],
    inplace: bool = False,
) -> np.ndarray:
    """
    Subtracts black levels from the raw image and normalizes into range [0, 1] using black and white levels.

    Args:
        image_inputs: 3D Numpy array of shape (N, H, W) containing input images.
        metadata: Sequence of Metadata classes containing metadata of input images.
        inplace: Whether to perform the operation in-place.

    Returns:
        Images after black level subtraction and normalization to range [0, 1]. Shape is (N, H, W).

    """

    processed_images = image_inputs if inplace else image_inputs.copy()

    for img, mtd in zip(processed_images, metadata):
        subtract_black_levels(img, mtd, inplace=True)
        normalize_image(img, mtd, inplace=True)

    return processed_images
