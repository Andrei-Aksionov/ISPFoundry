from typing import Any

import numpy as np
from loguru import logger
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


def get_noise_params_2x2(metadata: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses the NoiseProfile from metadata into 2x2 grids aligned with the Bayer pattern.

    The NoiseProfile (typically from DNG/EXIF) provides pairs of (scale, offset) values
    representing shot noise and read noise respectively. This function maps those
    values to the specific 2x2 CFA layout of the sensor.

    Args:
        metadata: A list of metadata dictionaries for the burst. Only the first
            entry is used for the profile, but all are checked for consistency.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - scales_grid: A 2x2 float32 array of shot noise scales.
            - offsets_grid: A 2x2 float32 array of read noise offsets.

    Raises:
        ValueError: If the CFAPlaneColor does not match the expected "Red,Green,Blue"
            sequence or if the NoiseProfile format is invalid.

    """

    if "NoiseProfile" in metadata[0] and not all(
        metadata[0]["NoiseProfile"] == mtd["NoiseProfile"] for mtd in metadata
    ):
        logger.warning(
            "Images in the burst have different NoiseProfile which means that shots were taken "
            "with different parameters. Merging multi-exposure frames is not yet supported/tested."
        )
    mtd = metadata[0]

    if "CFAPlaneColor" in mtd and mtd["CFAPlaneColor"] != "Red,Green,Blue":
        raise ValueError(f"The code expects that the matrix layout is Red Green Blue, but got {mtd['CFAPlaneColor']}")

    # NoiseProfile contains 6 values: 3 pairs of Scale and Offset for each color
    if "NoiseProfile" in mtd:
        noise_profile = [float(x) for x in mtd["NoiseProfile"].split()]
    else:
        logger.warning(f"NoiseProfile missing for {mtd.get('Model', 'Unknown')}. Using generic defaults.")
        # TODO (andrei aksionau): make it camera and ISO agnostic
        # Generic defaults for a 10-bit sensor (scaled to [0, 1] range)
        # These are placeholders; real values for Nexus 6 at ISO 40 are very small.
        noise_profile = [1e-5, 1e-6, 1e-5, 1e-6, 1e-5, 1e-6]

    color_map = {}
    for idx, color_name in enumerate(("R", "G", "B")):
        color_map[color_name] = (noise_profile[idx * 2], noise_profile[idx * 2 + 1])

    desc = mtd["color_desc"]  # e.g., "RGBG"
    pattern = mtd["raw_pattern"]  # e.g., [[2, 3], [1, 0]]

    # Create 2x2 grids for G and C
    scales_grid = np.array([[color_map[desc[idx]][0] for idx in row] for row in pattern], dtype=np.float32)
    offsets_grid = np.array([[color_map[desc[idx]][1] for idx in row] for row in pattern], dtype=np.float32)

    return scales_grid, offsets_grid


@njit
def merge_tiles(
    merged_accumulator: np.ndarray,
    weights_accumulator: np.ndarray,
    burst_images: list[np.ndarray],
    offsets: np.ndarray,
    sad_scores: np.ndarray,
    row_start: int,
    col_start: int,
    tile_size: int,
    noise_scales: np.ndarray,
    noise_offsets: np.ndarray,
    k: float = 4.0,
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
        tile_size: The width/height of the tile in full-res pixels.
        noise_scales: 2x2 matrix of shot noises per-channel
        noise_offsets: 2x2 matrix of matrix read noise per-channel
        k: higher = more blurring, lower = more ghosting rejection. A value of k=4 to k=12 is usually a good starting point.

    """

    num_images = len(burst_images)
    height, width = merged_accumulator.shape

    # Statistical correction: Luma proxy averaging reduces variance by ~4x
    # We must account for this so the SAD (from proxy) matches the Noise Model
    # Since luma proxy weighs greens more, the actual value is
    # 0.15**2 + 0.35**2 + 0.35**2 + 0.15**2 = 0.29
    proxy_variance_scale = 0.29

    def get_sigma_square(row: int, col: int, reference_values: float) -> float:
        """
        Helper function to get noise floor for the pixel given per-channel value in NoiseProfile from metadata.

        Given the NoiseProfile from the metadata, the noise model is:

        sigma^2 = g * x + c

        g (Scale): shot noise
        c (Offset): read noise
        """  # noqa: DOC201

        # Determine which part of the Bayer 2x2 we are in
        bayer_row = row % 2
        bayer_col = col % 2

        # Get the specific noise params for this color channel
        scale = noise_scales[bayer_row, bayer_col]
        offset = noise_offsets[bayer_row, bayer_col]

        # Variance for this specific pixel
        # Because we are comparing a target frame to a reference frame, the variance of the difference is the sum of their variances
        # Assumes that Var(Reference) ≈ Var(Target)
        sigma_sq = max(1e-9, 2.0 * (scale * reference_values + offset))
        # Adjust sigma to match the scale of the SAD score (calculated on proxy)
        return sigma_sq * proxy_variance_scale

    for tile_ridx in range(tile_size):
        for tile_cidx in range(tile_size):
            image_ridx = row_start + tile_ridx
            image_cidx = col_start + tile_cidx

            if image_ridx >= height or image_cidx >= width:
                continue

            pixel_sum = burst_images[0][image_ridx, image_cidx]
            weight_sum = 1.0
            sigma_sq = get_sigma_square(image_ridx, image_cidx, pixel_sum)

            for img_idx in range(1, num_images):
                dy = offsets[img_idx, 0]
                dx = offsets[img_idx, 1]

                # Find the corresponding pixel in the target frame
                target_ridx = max(0, min(image_ridx + dy, height - 1))
                target_cidx = max(0, min(image_cidx + dx, width - 1))

                # Robust Exponential Kernel Weighting
                # Weighting formula: exp( - (SAD^2) / (k * sigma_sq) )
                # We square the L1-based SAD to create a sharper rejection "cliff."
                # This aggressively suppresses misaligned tiles to prevent ghosting, effectively behaving like a robust Gaussian weight.
                weight = np.exp(-(sad_scores[img_idx] ** 2) / (k * sigma_sq + 1e-8))

                pixel_sum += burst_images[img_idx][target_ridx, target_cidx] * weight
                weight_sum += weight

            merged_accumulator[image_ridx, image_cidx] += pixel_sum
            weights_accumulator[image_ridx, image_cidx] += weight_sum


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
    if size_scaler % 2 != 0:
        logger.warning("Luma proxy is scaled by an odd number: merging can cause catastrophic color artifacts.")

    # 3. Parameters for block matching
    proxy_tile_size = tile_size // size_scaler
    proxy_stride = tile_stride // size_scaler
    proxy_max_offset = max_search_offset // size_scaler
    noise_scales, noise_offsets = get_noise_params_2x2(metadata)

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
                tile_size=tile_size,
                noise_scales=noise_scales,
                noise_offsets=noise_offsets,
                k=4.0,
            )

    # 5. Final normalization (weighted average across overlapping tiles)
    mask = weights_accumulator > 0
    merged_accumulator[mask] /= weights_accumulator[mask]

    return merged_accumulator
