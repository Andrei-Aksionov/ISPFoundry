import typing

import numpy as np
from loguru import logger


def retrieve_black_levels(raw_image: np.ndarray, metadata: dict[str, typing.Any]) -> np.ndarray:
    """Retrieves black level values from metadata or, if absent, calculates them from the image.

    Args:
        raw_image (np.ndarray): The input raw image.
        metadata (dict[str, any]): Dictionary containing metadata.

    Returns:
        np.ndarray: Array of black level values.

    Raises:
        ValueError: If black levels are larger than the image's maximum pixel value.

    """

    # 1. Find black level values
    black_levels = metadata.get("BlackLevel")
    if isinstance(black_levels, str):
        black_levels = np.array([float(x) for x in black_levels.split()])
    elif black_levels is not None:
        black_levels = np.asarray(black_levels, dtype=np.float32)

    if black_levels is None or np.allclose(black_levels, 0):
        logger.info("Metadata missing per-channel black levels - estimating from CFA minima.")

        black_levels = np.zeros(4, dtype=np.float32)
        for idx in range(4):
            row_offset, col_offset = divmod(idx, 2)
            black_levels[idx] = raw_image[row_offset::2, col_offset::2].min()

    if black_levels.size != 4:
        raise ValueError(f"Expected 4 black level values (e.g. RGGB), got: {black_levels}")

    # 2. Verify black level values
    if any(bl > raw_image.max() for bl in black_levels):
        raise ValueError(
            "Something is wrong with either raw image data or black levels. "
            f"Max value for RAW image is `{raw_image.max()}` and black levels are `{black_levels}`. "
            "Black levels cannot be larger than images max pixel value."
        )

    return black_levels


def subtract_black_levels(raw_image: np.ndarray, metadata: dict[str, typing.Any], inplace: bool = False) -> np.ndarray:
    """Subtracts black levels from the raw image.

    Raw sensor data contains an offset introduced by the camera’s sensor and readout electronics.
    Even when no light hits the sensor, the output signal is not zero due to this offset, known as the black level.

    Args:
        raw_image (np.ndarray): The input raw image as a NumPy array in float32.
        metadata (dict[str, any]): Dictionary containing metadata, including "BlackLevel".
        inplace (bool): Whether to perform the operation in-place.

    Returns:
        np.ndarray: The raw image with black levels subtracted.

    """

    if np.issubdtype(raw_image.dtype, np.unsignedinteger):
        raise ValueError(
            "Raw image must not be of unsigned integer type to avoid underflow issues after subtraction. "
            f"Expected to be dtype of float32, but got `{raw_image.dtype}`"
        )

    if not inplace:
        raw_image = raw_image.copy()

    black_levels = retrieve_black_levels(raw_image, metadata)

    for idx, black_level in enumerate(black_levels):
        row_offset, col_offset = divmod(idx, 2)
        raw_image[row_offset::2, col_offset::2] -= black_level

    return raw_image


def normalize_image(raw_image: np.ndarray, metadata: dict[str, typing.Any], inplace: bool = False) -> np.ndarray:
    """Normalizes the raw image into range [0, 1] using black and white levels.

    Args:
        raw_image (np.ndarray): The input raw image.
        metadata (dict[str, any]): Dictionary containing metadata.
        inplace (bool): Whether to perform the operation in-place.

    Returns:
        np.ndarray: The normalized image.

    """

    if not inplace:
        raw_image = raw_image.copy()

    white_level = metadata.get("WhiteLevel")
    if white_level is None or white_level == 0:
        raise ValueError(f"Metadata should contain a valid WhiteLevel, but instead got: `{white_level}`")

    black_levels = retrieve_black_levels(raw_image, metadata)

    for idx, black_level in enumerate(black_levels):
        row_offset, col_offset = divmod(idx, 2)
        denominator = white_level - black_level
        if denominator == 0:
            raise ValueError(f"WhiteLevel equals BlackLevel for channel {idx}")
        raw_image[row_offset::2, col_offset::2] /= denominator

    return raw_image
