from typing import Any

import numpy as np
from loguru import logger
from numba import njit
from tqdm import trange

from base import ISPStep, register_step


def downscale_bayer_to_grayscale(raw_image: np.ndarray) -> np.ndarray:
    """
    Downsample Bayer RAW to grayscale by averaging each 2x2 block.

    Converts a Bayer-mosaic RAW image to a lower-resolution grayscale image by averaging each 2x2 block.
    This is useful for burst alignment, as it reduces noise and resolution, making alignment more robust and efficient.

    Args:
        raw_image (np.ndarray): Bayer RAW sensor data, shape (height, width)

    Returns:
        np.ndarray: Grayscale image, shape (height//2, width//2)

    """
    height, width = raw_image.shape
    if height % 2 != 0 or width % 2 != 0:
        logger.info("At least one of the dimensions is not even. Will proceed with cropped raw image.")
        height = (height // 2) * 2
        width = (width // 2) * 2
        raw_image = raw_image[:height, :width]

    # Reshape and average 2x2 blocks
    return raw_image.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))


@njit
def find_best_offset(
    reference_tile: np.ndarray,
    target_image: np.ndarray,
    tile_row: int,
    tile_col: int,
    height: int,
    width: int,
    tile_size: int,
    max_offset: int,
) -> tuple:
    """
    Find the best offset (dy, dx) for aligning a target tile to a reference tile using SAD (Sum of Absolute Differences).

    SAD stands for "Sum of Absolute Differences". It is a simple metric used to measure how similar two image blocks are.
    For two blocks of the same size, SAD is computed by summing the absolute value of the difference between corresponding pixels.
    A lower SAD value means the blocks are more similar (better aligned), while a higher value means they are less similar.

    Args:
        reference_tile (np.ndarray): Reference tile from the first (reference) image
        target_image (np.ndarray): Target image to align
        tile_row (int): Row index of the tile (in downsampled space)
        tile_col (int): Column index of the tile (in downsampled space)
        height (int): Height of the downsampled image
        width (int): Width of the downsampled image
        tile_size (int): Size of the tile (in downsampled space)
        max_offset (int): Maximum search offset for alignment

    Returns:
        tuple: (best_dy, best_dx, min_sad)

    """
    # Bounds and shape checks
    assert reference_tile.shape == (tile_size, tile_size)
    assert tile_row + tile_size <= height
    assert tile_col + tile_size <= width

    min_sad = float("inf")
    best_dy = 0
    best_dx = 0

    # Search over a window of offsets
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            row = max(0, min(tile_row + dy, height - tile_size))
            col = max(0, min(tile_col + dx, width - tile_size))

            target_tile = target_image[row : row + tile_size, col : col + tile_size]
            sad = 0.0
            # Compute sum of absolute differences (SAD) for this offset
            for row_idx in range(tile_size):
                for col_idx in range(tile_size):
                    sad += abs(reference_tile[row_idx, col_idx] - target_tile[row_idx, col_idx])

            if sad < min_sad:
                min_sad = sad
                best_dy = row - tile_row
                best_dx = col - tile_col

    return best_dy, best_dx, min_sad


@njit
def merge_tiles(
    merged_image: np.ndarray,
    weight_matrix: np.ndarray,
    burst_images: np.ndarray,
    offsets: np.ndarray,
    sads: np.ndarray,
    tile_row: int,
    tile_col: int,
    sad_threshold: float,
    tile_size: int,
    height: int,
    width: int,
) -> None:
    """
    Merge a tile region from a burst of images into the merged image using weighted averaging.

    Weighting is used to give more influence to well-aligned pixels (those with lower alignment error/SAD)
    when merging the burst frames. Pixels from frames that align better with the reference are given higher
    weights, so their values contribute more to the final merged result. This helps suppress noise and
    artifacts from misaligned or poorly matched frames, leading to a cleaner, higher-quality merged image.

    Args:
        merged_image (np.ndarray): Output merged image (accumulated sum)
        weight_matrix (np.ndarray): Output weight matrix (for normalization)
        burst_images (np.ndarray): Input burst images, shape (N, H, W)
        offsets (np.ndarray): Alignment offsets for each image, shape (N, 2)
        sads (np.ndarray): SAD values for each image, shape (N,)
        tile_row (int): Tile row index (downsampled space)
        tile_col (int): Tile column index (downsampled space)
        sad_threshold (float): SAD threshold for robust merging
        tile_size (int): Size of the tile (downsampled space)
        height (int): Height of the full-res image
        width (int): Width of the full-res image

    """
    # Bounds and shape checks
    assert merged_image.shape == weight_matrix.shape == (height, width)
    assert burst_images.shape[1:3] == (height, width)
    assert offsets.shape[0] == sads.shape[0] == burst_images.shape[0]
    assert tile_row * 2 + tile_size * 2 <= height + 1
    assert tile_col * 2 + tile_size * 2 <= width + 1

    for row_offset in range(tile_size * 2):
        for col_offset in range(tile_size * 2):
            row_idx = tile_row * 2 + row_offset
            col_idx = tile_col * 2 + col_offset

            if row_idx >= height or col_idx >= width:
                continue

            # Always include the reference frame (weight 1)
            pixel_sum = burst_images[0][row_idx, col_idx]
            weight_sum = 1.0

            # For each other frame, include if well-aligned
            for img_idx in range(1, burst_images.shape[0]):
                if sads[img_idx] < sad_threshold:
                    dy = offsets[img_idx, 0] * 2
                    dx = offsets[img_idx, 1] * 2
                    ref_row = max(0, min(row_idx + dy, height - 1))
                    ref_col = max(0, min(col_idx + dx, width - 1))

                    # Higher weight for better alignment
                    weight = 1.0 / (sads[img_idx] + 1e-8)
                    pixel_sum += burst_images[img_idx, ref_row, ref_col] * weight
                    weight_sum += weight

            if weight_sum > 0:
                merged_image[row_idx, col_idx] += pixel_sum / weight_sum  # Weighted average
                weight_matrix[row_idx, col_idx] += 1.0  # Count for normalization


@register_step(ISPStep.ALIGN_AND_MERGE)
def merge_images(burst_images: np.ndarray | list[np.ndarray], *args: Any, **kwargs: Any) -> np.ndarray:  # noqa: ARG001
    """
    Aligns and merges a burst of RAW images to reduce noise and improve image quality.

    Uses block-based alignment on downsampled grayscale images, then merges aligned pixels with robust weighting.

    Weighting is used to give more influence to well-aligned pixels (those with lower alignment error/SAD)
    when merging the burst frames. Pixels from frames that align better with the reference are given higher
    weights, so their values contribute more to the final merged result. This helps suppress noise and
    artifacts from misaligned or poorly matched frames, leading to a cleaner, higher-quality merged image.

    SAD stands for "Sum of Absolute Differences". It is a simple metric used to measure how similar two image blocks are.
    For two blocks of the same size, SAD is computed by summing the absolute value of the difference between corresponding pixels.
    A lower SAD value means the blocks are more similar (better aligned), while a higher value means they are less similar.

    Args:
        burst_images (np.ndarray): Array of RAW images, shape (N, H, W) or a list of raw images of shape (H, W)
        *args: positional arguments that are passed by the pipeline but not needed here
        **kwargs: keyword arguments that are passed by the pipeline but not needed here

    Returns:
        np.ndarray: Merged RAW image, shape (H, W)

    Raises:
        ValueError: if any of theth image in the burst is of a different shape to others.

    """
    if isinstance(burst_images, list) and not isinstance(burst_images, np.ndarray):
        burst_images = np.array(burst_images)

    if burst_images.shape == 0:
        return np.zeros_like(burst_images[0])
    if len(burst_images) == 1:
        return burst_images[0]
    if not all(burst_images[0].shape == burst_image.shape for burst_image in burst_images[1:]):
        raise ValueError("All images in the burst must have the same shape.")

    num_images = len(burst_images)
    height, width = burst_images[0].shape

    merged_image = np.zeros((height, width), dtype=np.float32)
    weight_matrix = np.zeros((height, width), dtype=np.float32)

    # Build downsampled grayscale for all frames
    # TODO (andrei aksionau): using np.stack and then calling this function in a loop for a large burst will eat up RAM.
    # Since numba is used, this logic might be moved directly into a JIT-compiled kernel to process images on the fly
    gray_images = np.stack([downscale_bayer_to_grayscale(img) for img in burst_images])

    # Parameters for block matching
    tile_size = 16  # Size of each block (in downsampled space)
    stride = 8  # Stride between blocks
    max_offset = 4  # Maximum search offset for alignment
    sad_threshold = 0.15 * tile_size**2  # SAD threshold for robust merging
    half_height, half_width = height // 2, width // 2

    # For each tile in the downsampled reference image
    for tile_row in trange(0, half_height, stride, desc="Iterating over tiles"):
        for tile_col in range(0, half_width, stride):
            # Ensure tile does not go out of bounds
            row_idx = min(tile_row, half_height - tile_size)
            col_idx = min(tile_col, half_width - tile_size)

            ref_tile = gray_images[0, row_idx : row_idx + tile_size, col_idx : col_idx + tile_size]

            offsets = np.zeros((num_images, 2), dtype=np.int32)
            sads = np.zeros(num_images, dtype=np.float32)

            # For each frame, find the best offset for this tile
            for img_idx in range(1, num_images):
                best_dy, best_dx, min_sad = find_best_offset(
                    ref_tile,
                    gray_images[img_idx],
                    row_idx,
                    col_idx,
                    half_height,
                    half_width,
                    tile_size,
                    max_offset,
                )
                offsets[img_idx] = (best_dy, best_dx)  # Best offset for this frame
                sads[img_idx] = min_sad  # Best SAD for this frame

            # For each pixel in the tile (full-res)
            merge_tiles(
                merged_image,
                weight_matrix,
                burst_images,
                offsets,
                sads,
                row_idx,
                col_idx,
                sad_threshold,
                tile_size,
                height,
                width,
            )

    # Normalize by weight to blend overlapping tiles
    mask = weight_matrix > 0
    merged_image[mask] /= weight_matrix[mask]

    return merged_image

    return merged_image
