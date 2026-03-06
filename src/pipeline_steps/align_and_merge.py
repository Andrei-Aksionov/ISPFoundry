from typing import Any

import numpy as np
from numba import njit
from tqdm import trange

from base import ISPStep, register_step


def get_luma_proxy(raw_image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """
    Converts a Bayer RAW image to a half-resolution grayscale proxy for alignment.

    This function implements a weighted luma conversion directly on RAW data.
    Unlike standard Rec.601/709 conversions, it uses a Green-heavy weighting scheme (70% Green, 15% Red, 15% Blue) to
    maximize the Signal-to-Noise Ratio (SNR) for motion estimation, while remaining robust to pure Red or Blue
    high-contrast edges where Green signal may be absent.

    Args:
        raw_image: Bayer RAW sensor data of shape (H, W).
        metadata: Dictionary containing:
            - 'color_desc': String describing the CFA colors (e.g., "RGBG").
            - 'raw_pattern': 2x2 list/array mapping CFA indices to the physical pixel grid

    Returns:
        np.ndarray: Grayscale luma proxy of shape (H//2, W//2) as float32.
            Dimensions are truncated to the nearest even multiple before processing.

    """
    # 1. Even dimensions check
    height, width = raw_image.shape
    height, width = height & ~1, width & ~1  # down to the nearest multiple of 2
    raw_image = raw_image[:height, :width]

    # 2. Map color names to weights
    color_weights = {"R": 0.15, "G": 0.35, "Gr": 0.35, "Gb": 0.35, "B": 0.15}

    # 3. Build a 2x2 weight mask
    desc = metadata["color_desc"]  # e.g., "RGBG"
    pattern = metadata["raw_pattern"]  # e.g., [[2, 3], [1, 0]]
    # This creates a 2x2 array of weights, e.g., [[0.15, 0.35], [0.35, 0.15]]
    weights_map = np.array([[color_weights[desc[idx]] for idx in row] for row in pattern])

    # 4. Apply using the Reshape Trick
    quads = raw_image.reshape(height // 2, 2, width // 2, 2)
    # For every 2x2 block in the image, multiply it element-wise by the weight map and sum the results into a single pixel.
    return np.einsum("hiwj,ij->hw", quads, weights_map).astype(np.float32)


@njit
def find_best_offset(
    reference_proxy: np.ndarray,
    target_proxy: np.ndarray,
    row_start: int,
    col_start: int,
    tile_size: int,
    max_offset: int,
) -> tuple[int, int, float]:
    """
    Finds the integer translation (dy, dx) that minimizes the SAD between tiles.

    Args:
        reference_proxy: The reference grayscale image (usually the first frame).
        target_proxy: The target grayscale image to be aligned.
        row_start: Top row index of the tile in the proxy coordinate system.
        col_start: Left column index of the tile in the proxy coordinate system.
        tile_size: The width/height of the tile in proxy pixels.
        max_offset: The maximum pixels to search in any direction.

    Returns:
        A tuple of (best_dy, best_dx, minimum_sad).

    """

    height, width = target_proxy.shape
    min_sad = 1e20  # A large float for Numba compatibility
    best_dy, best_dx = 0, 0

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            # Calculate the intersection of the Target Tile (Ref + offset) and the Target Image
            # These must be clipped so we don't index out of bounds
            r_start = max(row_start, -dy, 0)
            r_end = min(row_start + tile_size, height - dy, height)

            c_start = max(col_start, -dx, 0)
            c_end = min(col_start + tile_size, width - dx, width)

            if r_start >= r_end or c_start >= c_end:
                continue

            # Slicing: The target slice must be offset by `dy` and `dx` because it is the 'shifted' version of the reference
            ref_view = reference_proxy[r_start:r_end, c_start:c_end]
            tgt_view = target_proxy[r_start + dy : r_end + dy, c_start + dx : c_end + dx]

            # Width Numba per-pixel calculation is faster
            # np.sum(np.abs(...)) leads to temporary array allocations
            sad = 0.0
            rows, cols = ref_view.shape
            for r in range(rows):
                for c in range(cols):
                    # Direct subtraction and absolute value
                    sad += abs(ref_view[r, c] - tgt_view[r, c])

            # Normalization: If tiles are partially off-image, a smaller area will naturally have a lower SAD.
            # We divide by area to find the true best match per pixel.
            sad = sad / (rows * cols)

            if sad < min_sad:
                min_sad = sad
                best_dy = dy
                best_dx = dx

    return best_dy, best_dx, min_sad


@njit
def merge_tiles(
    merged_accumulator: np.ndarray,
    weights_accumulator: np.ndarray,
    burst_images: list[np.ndarray],
    offsets: np.ndarray,
    sad_scores: np.ndarray,
    row_start: int,
    col_start: int,
    sad_threshold: float,
    tile_size: int,
) -> None:
    """
    Performs weighted accumulation of a tile across the burst into the final image.

    Frames with lower SAD scores (better alignment) are given higher weights.
    Frames exceeding the sad_threshold are ignored for that specific tile to
    prevent ghosting artifacts.

    Args:
        merged_accumulator: The full-res buffer accumulating weighted pixel values.
        weights_accumulator: The full-res buffer accumulating weights for normalization.
        burst_images: The original stack of RAW images (N, H, W).
        offsets: Integer offsets scaled to full resolution (N, 2).
        sad_scores: SAD values from the proxy alignment step (N,).
        row_start: Top row index of the tile in full-res coordinates.
        col_start: Left column index of the tile in full-res coordinates.
        sad_threshold: Threshold above which a frame is considered misaligned.
        tile_size: The width/height of the tile in full-res pixels.

    """

    num_images = len(burst_images)
    height, width = merged_accumulator.shape

    for tile_ridx in range(tile_size):
        for tile_cidx in range(tile_size):
            image_ridx = row_start + tile_ridx
            image_cidx = col_start + tile_cidx

            if image_ridx >= height or image_cidx >= width:
                continue

            # Frame 0 is our reference; anchor it with a weight of 1.0
            pixel_sum = burst_images[0][image_ridx, image_cidx]
            weight_sum = 1.0

            for img_idx in range(1, num_images):
                if sad_scores[img_idx] < sad_threshold:
                    dy = offsets[img_idx, 0]
                    dx = offsets[img_idx, 1]

                    # Find the corresponding pixel in the target frame
                    target_ridx = max(0, min(image_ridx + dy, height - 1))
                    target_cidx = max(0, min(image_cidx + dx, width - 1))

                    # Weight is inversely proportional to alignment error (SAD)
                    weight = 1.0 / (sad_scores[img_idx] + 1e-8)
                    pixel_sum += burst_images[img_idx][target_ridx, target_cidx] * weight
                    weight_sum += weight

            if weight_sum > 0:
                merged_accumulator[image_ridx, image_cidx] += pixel_sum / weight_sum  # Weighted average
                weights_accumulator[image_ridx, image_cidx] += 1.0  # Count for normalization


@register_step(ISPStep.ALIGN_AND_MERGE)
def merge_images(
    burst_images: list[np.ndarray],
    metadata: list[dict[str, Any]],
    tile_size: int = 32,
    tile_stride: int = 16,
    max_search_offset: int = 8,
    *args: Any,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> np.ndarray:
    """
    Performs block-based alignment and temporal merging of a RAW image burst.

    This function reduces noise by aligning subsequent frames to a reference (the first
    frame) and merging them using a weighted average. Alignment is performed on
    downsampled grayscale 'proxies' to improve speed and robustness against noise,
    while the final merge occurs at the original RAW resolution.

    Args:
        burst_images: A list 2D numpy arrays.
        metadata: A list of dictionaries containing per-frame sensor metadata (e.g.,
            exposure time, black level).
        tile_size: The width/height of the processing tile in full-resolution pixels.
            Must be a multiple of the downsampling factor (usually 2).
        tile_stride: The step size between tiles in full-resolution pixels. Smaller
            strides increase overlap and reduce tiling artifacts but increase compute time.
        max_search_offset: The maximum distance (in full-res pixels) the algorithm
            will search for a matching block in any direction.
        *args: Additional positional arguments for ISP pipeline compatibility.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The merged RAW image of shape (H, W), normalized by tile weights.

    Raises:
        ValueError: If fewer than two images are provided.
        ValueError: If the images in the burst have inconsistent shapes.
        RuntimeError: If tile parameters are incompatible with the downsampling factor.

    """

    if len(burst_images) <= 1:
        raise ValueError(f"At least two images needed for Align&Merge, but got {len(burst_images)}.")

    if len({x.shape for x in burst_images}) != 1:
        raise ValueError("All images in the burst must have the same shape.")

    num_images = len(burst_images)
    image_height, image_width = burst_images[0].shape

    # 1. Initialize accumulation buffers
    merged_accumulator = np.zeros((image_height, image_width), dtype=np.float32)
    weights_accumulator = np.zeros((image_height, image_width), dtype=np.float32)

    # 2. Generate grayscale proxies for alignment
    # TODO (andrei aksionau): using np.stack and then calling this function in a loop for a large burst will eat up RAM.
    # Since numba is used, this logic might be moved directly into a JIT-compiled kernel to process images on the fly
    luma_proxies = [get_luma_proxy(img, mtd) for img, mtd in zip(burst_images, metadata)]
    proxy_height, proxy_width = luma_proxies[0].shape
    if image_height // proxy_height != image_width // proxy_width:
        raise RuntimeError("Downsampling scale for luma proxy is uneven for width and height.")

    # Calculate scaling factor between proxy and full-res space
    size_scaler = image_height // proxy_height

    # 3. Parameters for block matching
    sad_threshold = 0.15 * tile_size**2  # SAD threshold for robust merging
    proxy_tile_size = tile_size // size_scaler
    proxy_stride = tile_stride // size_scaler
    proxy_max_offset = max_search_offset // size_scaler

    # 4. Main loop: block-matching and merging
    for proxy_row in trange(0, proxy_height, proxy_stride, desc="Aligning&Merging Burst"):
        for proxy_col in range(0, proxy_width, proxy_stride):
            offsets_proxy = np.zeros((num_images, 2), dtype=np.int32)
            sad_scores = np.zeros(num_images, dtype=np.float32)

            # Align target frames to the reference proxy
            for img_idx in range(1, num_images):
                dy, dx, score = find_best_offset(
                    reference_proxy=luma_proxies[0],
                    target_proxy=luma_proxies[img_idx],
                    row_start=proxy_row,
                    col_start=proxy_col,
                    tile_size=proxy_tile_size,
                    max_offset=proxy_max_offset,
                )
                offsets_proxy[img_idx] = (dy, dx)
                sad_scores[img_idx] = score

            # Scale alignment results back to the original RAW resolution
            offsets_full_res = offsets_proxy * size_scaler

            # Merge the tiles into the full-resolution accumulator
            merge_tiles(
                merged_accumulator=merged_accumulator,
                weights_accumulator=weights_accumulator,
                burst_images=burst_images,
                offsets=offsets_full_res,
                sad_scores=sad_scores,
                row_start=proxy_row * size_scaler,
                col_start=proxy_col * size_scaler,
                sad_threshold=sad_threshold,
                tile_size=tile_size,
            )

    # 5. Final normalization (weighted average across overlapping tiles)
    mask = weights_accumulator > 0
    merged_accumulator[mask] /= weights_accumulator[mask]

    return merged_accumulator
