from typing import Any

import numpy as np
from loguru import logger
from numba import njit
from scipy import stats
from scipy.ndimage import convolve, gaussian_filter
from tqdm import trange

from base import ISPStep, register_step

# ---------------------------------------- Utility functions -----------------------------------------


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


def downsample_luma_proxy(proxy: np.ndarray) -> np.ndarray:
    """
    Downsamples a luma proxy by a factor of 2 using box averaging.

    This function prepares the next level of an image pyramid. It ensures the
    input dimensions are even to maintain consistent 2x2 block alignment,
    averaging each quad into a single pixel at the coarser resolution.

    Args:
        proxy: A 2D grayscale array (luma proxy) at the current pyramid level.

    Returns:
        A 2D array with half the width and height of the input, representing
        the next coarsest level in the pyramid.

    """
    height, width = proxy.shape
    # Ensure even dimensions for 2x2 reshaping
    height, width = height & ~1, width & ~1
    proxy = proxy[:height, :width]

    # Reshape to (H', 2, W', 2) and average over the 2x2 blocks
    return proxy.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))


# TODO (andrei aksionau): write a docstring
def get_exposure_scalers(metadata: list[dict[str, Any]]) -> np.ndarray:

    def parse_expr(value: str) -> float:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/")
            return float(numerator) / float(denominator)
        return float(value)

    # Find the shortest exposure
    exposures = np.array([parse_expr(mtd["ExposureTime"]) for mtd in metadata], dtype=np.float32)
    ref_exposure = exposures.min()

    # Normalize other exposures
    return ref_exposure / exposures


def find_sharpest_image_idx(images: list[np.ndarray], metadata: list[dict[str, Any]]) -> int:
    """
    Identifies the sharpest frame in a burst to serve as the alignment base.

    This 'Lucky Imaging' selection helps minimize the propagation of motion blur
    through the pipeline. It uses a Noise-Robust Laplacian Variance method:
    1. Generates a luma proxy for each frame.
    2. Applies a light Gaussian blur (sigma=0.5) to prevent sensor noise from
       being misidentified as sharp edge detail.
    3. Convolves with a Laplacian kernel and calculates variance to estimate
       high-frequency content (edge strength).

    Args:
        images: List of Bayer RAW images (H, W).
        metadata: List of per-frame metadata dictionaries.

    Returns:
        int: The index of the frame with the highest relative sharpness.

    """
    unique_exposures = {mtd["ExposureTime"] for mtd in metadata}
    if len(unique_exposures) > 1:
        logger.warning("The code is tested to work with up to 2 different exposures, but got %s" % unique_exposures)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)  # Simple 3x3 Laplacian kernel

    # Identify the "short" exposure group (the most frequent one)
    # This ensures we pick the best from the 'standard' frames
    # TODO (andrei aksionau): ensure that we are searching only from the short exposure images
    exp_times = [mtd["ExposureTime"] for mtd in metadata]
    most_common_exp = max(set(exp_times), key=exp_times.count)

    scores = []
    for idx in range(len(images)):
        # Only consider frames from the majority exposure group to avoid brightness-bias in the Laplacian variance
        if metadata[idx]["ExposureTime"] != most_common_exp:
            continue
        # 1. Recalculate on-the-go to reduce memory footprint
        luma_proxy = get_luma_proxy(images[idx], metadata[idx])
        # 2. Light blur to suppress high-frequency noise that mimics sharpness
        # This ensures we measure actual structural edges, not sensor grain.
        smoothed = gaussian_filter(luma_proxy, sigma=0.5)
        # 3. Calculate Laplacian variance
        scores.append((idx, convolve(smoothed, kernel).var()))

    best_idx, best_score = max(scores, key=lambda x: x[1])
    avg_score = sum(s[1] for s in scores) / len(scores)
    improvement = (best_score / (avg_score + 1e-9) - 1) * 100

    logger.info(f"Lucky Imaging: Selected frame `{best_idx}` ({improvement:+.1f}% sharper than burst average)")
    return best_idx


def get_noise_params_2x2(
    burst_images: list[np.ndarray], metadata: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses the NoiseProfile from metadata into 2x2 grids aligned with the Bayer pattern.

    The NoiseProfile (typically from DNG/EXIF) provides pairs of (scale, offset) values
    representing shot noise and read noise respectively. This function maps those
    values to the specific 2x2 CFA layout of the sensor.

    If NoiseProfile is missing in metadata, it falls back to a Mean-Variance (Photon Transfer Curve)
    estimation performed on the reference frame.

    Args:
        burst_images: A list of RAW images (2D numpy arrays).
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
    ref_image = burst_images[0]
    mtd = metadata[0]

    if "CFAPlaneColor" in mtd and mtd["CFAPlaneColor"] != "Red,Green,Blue":
        raise ValueError(f"The code expects that the matrix layout is Red Green Blue, but got {mtd['CFAPlaneColor']}")

    # Case A: Use DNG-standard NoiseProfile if available
    if "NoiseProfile" in mtd:
        noise_profile = [float(x) for x in mtd["NoiseProfile"].split()]
        color_map = {}
        # NoiseProfile contains 6 values: 3 pairs of Scale and Offset for each color
        for idx, color_name in enumerate(("R", "G", "B")):
            color_map[color_name] = (noise_profile[idx * 2], noise_profile[idx * 2 + 1])

        desc = mtd["color_desc"]  # e.g., "RGBG"
        pattern = mtd["raw_pattern"]  # e.g., [[2, 3], [1, 0]]

        # Create 2x2 grids for G and C
        scales_grid = np.array([[color_map[desc[idx]][0] for idx in row] for row in pattern], dtype=np.float32)
        offsets_grid = np.array([[color_map[desc[idx]][1] for idx in row] for row in pattern], dtype=np.float32)

        return scales_grid, offsets_grid

    # Case B: Fallback to empirical estimation
    logger.warning("NoiseProfile missing for %s. Estimating noise from reference frame." % mtd.get("Model", "Unknown"))
    return estimate_noise_profile(ref_image)


def estimate_noise_profile(image: np.ndarray, patch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates shot and read noise parameters using Mean-Variance analysis.

    The function implements a Photon Transfer Curve (PTC) approach:
    1. Sub-samples the image into 4 Bayer phases (R, Gr, Gb, B).
    2. Tiles each phase into small patches and calculates local mean and variance.
    3. Filters for 'flat' patches across brightness levels to isolate noise from texture.
    4. Performs linear regression: Variance = g * Mean + c.

    Args:
        image: The reference RAW image (normalized [0, 1], black-level subtracted, LSC applied).
        patch_size: Size of tiles used for local statistics (applied to sub-sampled channels).

    Returns:
        tuple[np.ndarray, np.ndarray]: 2x2 grids of (slope/shot) and (intercept/read) noise.

    """

    # By cropping to the center, we ensure the linear regression isn't trying to fit a "tilted"
    # plane where the read noise is amplified by different LSC gain factors at the edges.
    ch, cw = [x // 4 for x in image.shape]
    center_crop = image[ch:-ch, cw:-cw]

    scales_grid = np.zeros((2, 2), dtype=np.float32)
    offsets_grid = np.zeros((2, 2), dtype=np.float32)

    # Analyze each of the 4 CFA positions independently
    for idx in range(4):
        row_offset, col_offset = divmod(idx, 2)
        plane = center_crop[row_offset::2, col_offset::2]

        plane_height, plane_width = plane.shape
        h_tiles, w_tiles = plane_height // patch_size, plane_width // patch_size
        # Tile the channel into patches for statistical sampling
        patches = plane[: h_tiles * patch_size, : w_tiles * patch_size].reshape(
            h_tiles, patch_size, w_tiles, patch_size
        )

        means = np.mean(patches, axis=(1, 3)).flatten()
        vars = np.var(patches, axis=(1, 3)).flatten()

        # Isolate noise by binning variances by brightness and selecting
        # the lowest 10% (the 'flattest' patches) in each bin.
        filtered_means = []
        filtered_vars = []

        # Ignore extreme blacks/highlights to avoid clipping bias
        bins = np.linspace(np.quantile(means, 0.05), np.quantile(means, 0.95), 15)
        for bin_idx in range(len(bins) - 1):
            idx = (means >= bins[bin_idx]) & (means < bins[bin_idx + 1])
            if np.count_nonzero(idx) > 5:
                bin_vars = vars[idx]
                thresh = np.percentile(bin_vars, 10)
                filtered_means.extend(means[idx][bin_vars <= thresh])
                filtered_vars.extend(bin_vars[bin_vars <= thresh])

        if len(filtered_means) < 10:
            # Default values for a typical clean sensor if estimation fails
            scales_grid[row_offset, col_offset] = 1e-4
            offsets_grid[row_offset, col_offset] = 1e-6
            continue

        # Linear regression yields the noise model: sigma^2 = slope * signal + intercept
        slope, intercept, _, _, _ = stats.linregress(filtered_means, filtered_vars)

        # Clip to ensure physical validity (noise parameters must be positive)
        scales_grid[row_offset, col_offset] = max(slope, 1e-7)
        offsets_grid[row_offset, col_offset] = max(intercept, 1e-9)

    return scales_grid, offsets_grid


def get_hann_window_2d(tile_size: int) -> np.ndarray:
    """
    Generates a 2D Hann (raised cosine) window for seamless tile blending.

    The window tapers to zero at the edges, ensuring that when tiles with 50% overlap are summed,
    the spatial weights add up to a constant 1.0.
    The 0.5 pixel offset aligns the cosine curve to pixel centers.

    Args:
        tile_size: The width and height of the square tile in pixels.

    Returns:
        A 2D array of shape (tile_size, tile_size) containing the
        normalized blending weights.

    """

    pos = np.arange(tile_size, dtype=np.float32)
    # The (pos + 0.5) ensures the window is centered on pixels
    w_1d = 0.5 * (1 - np.cos(2 * np.pi * (pos + 0.5) / tile_size))
    # Create 2D map via outer product
    return np.outer(w_1d, w_1d).astype(np.float32)


@njit(fastmath=True)
def find_subpixel_shift(grid: np.ndarray) -> tuple[float, float]:
    """
    Finds the subpixel shift (delta_y, delta_x) by fitting 1D parabolas to a 3x3 score grid.

    This function performs a "separable" quadratic fit. It treats the horizontal and
    vertical axes independently, finding the vertex of the parabola formed by the
    three points on each axis (e.g., [Up, Center, Down]).

    The grid is expected to be a 3x3 array of alignment costs (e.g., VW-SAD):
    [  _      Up      _   ]
    [ Left  Center  Right ]
    [  _     Down     _   ]

    Args:
        grid: 3x3 numpy array of alignment scores.

    Returns:
        (offset_row, offset_col): The fractional shift relative to the center pixel.
        Values are clamped to the range [-0.5, 0.5].

    """

    # Extract values for readability
    center = grid[1, 1]
    left = grid[1, 0]
    right = grid[1, 2]
    up = grid[0, 1]
    down = grid[2, 1]

    # 1D Quadratic fit for each axis independently:
    # Formula for offset = (f(x-1) - f(x+1)) / (2 * (f(x-1) + f(x+1) - 2*f(x)))

    # Row (Vertical) offset
    denom_row = 2 * (up + down - 2 * center)
    offset_row = (up - down) / denom_row if abs(denom_row) > 1e-7 else 0.0

    # Column (Horizontal) offset
    denom_col = 2 * (left + right - 2 * center)
    offset_col = (left - right) / denom_col if abs(denom_col) > 1e-7 else 0.0

    # Robustness clamp: If the offset is > 0.5, the integer search likely
    # landed on the wrong pixel. Clamping prevents inconsistent alignment.
    offset_r = max(-0.5, min(0.5, offset_row))
    offset_c = max(-0.5, min(0.5, offset_col))

    return float(offset_r), float(offset_c)


@njit(fastmath=True)
def sample_raw_bilinear(
    raw_image: np.ndarray, row_base: int, col_base: int, row_offset: float, col_offset: float
) -> np.float32:
    """
    Performs sub-pixel sampling on Bayer RAW data while preserving color integrity.

    In a Bayer sensor, adjacent pixels are different colors (e.g., Red next to Green).
    Standard bilinear interpolation would mix these colors, causing severe artifacts.
    This function "jumps" with a 2-pixel stride to ensure it only interpolates
    between pixels of the same color phase.

    Sub-pixel Logic Examples:
    -------------------------
    - Shift = 0.4: Interpolates between phase 0 and phase 2.
    - Shift = 1.0: Interpolates exactly 50/50 between phase 0 and phase 2.
    - Shift = 1.8: Interpolates heavily toward phase 2.
    - Shift = 2.0: Triggers the "Fast Path" and takes the pixel at index 2 directly.

    Args:
        raw_image: The 2D sensor data array (H, W).
        row_base: The integer row index in the reference frame.
        col_base: The integer column index in the reference frame.
        row_offset: The vertical shift to apply (can be fractional).
        col_offset: The horizontal shift to apply (can be fractional).

    Returns:
        np.float32: The color-accurate interpolated pixel value.

    """

    height, width = raw_image.shape

    # 1. Fast Path: If the offset is exactly a multiple of 2, we are aligned
    # perfectly with a Bayer pixel of the same color. We skip math to avoid blur.
    r_off_int, c_off_int = round(row_offset), round(col_offset)
    is_integer_shift = abs(row_offset - r_off_int) < 1e-5 and abs(col_offset - c_off_int) < 1e-5

    if is_integer_shift and (r_off_int % 2 == 0) and (c_off_int % 2 == 0):
        # Direct access to the same-color pixel
        return np.float32(raw_image[row_base + r_off_int, col_base + c_off_int])

    # 2. Identify the "sampling quad": the 4 nearest pixels of the same color.
    # To keep the Bayer phase, we move in steps of 2 pixels.
    # We find the top-left neighbor by flooring the offset to the nearest even number.
    row_shift_base = int(np.floor(row_offset / 2.0) * 2)
    col_shift_base = int(np.floor(col_offset / 2.0) * 2)

    # Coordinates for the 2x2 same-color grid
    row_top = row_base + row_shift_base
    col_left = col_base + col_shift_base
    # Stay in bounds while maintaining Bayer phase
    row_bottom = row_top + 2 if row_top + 2 < height else row_top
    col_right = col_left + 2 if col_left + 2 < width else col_left

    # 3. Boundary Guard: If the 2-pixel stride goes off-sensor,
    # we clamp to the base pixel to prevent crashes and maintain parity.
    row_top = max(0, min(row_top, height - 1))
    row_bottom = max(0, min(row_bottom, height - 1))

    # 4. Calculate fractional distances [0.0, 1.0] within the 2-pixel gap.
    # Example: if row_offset is 0.5, row_lerp is 0.25 (a quarter of the 2-pixel jump).
    row_lerp = (row_offset - row_shift_base) / 2.0
    col_lerp = (col_offset - col_shift_base) / 2.0

    # 5. Fetch the 4 same-color neighbors
    top_left = raw_image[row_top, col_left]
    top_right = raw_image[row_top, col_right]
    bottom_left = raw_image[row_bottom, col_left]
    bottom_right = raw_image[row_bottom, col_right]

    # 6. Bilinear Interpolation (Standard Lerp)
    # Interpolate horizontally across the top and bottom pairs
    top_mix = top_left + col_lerp * (top_right - top_left)
    bottom_mix = bottom_left + col_lerp * (bottom_right - bottom_left)

    # Interpolate vertically between the two horizontal results
    return np.float32(top_mix + row_lerp * (bottom_mix - top_mix))


@njit(fastmath=True)
def _compute_tile_sad(
    row_offset: int,
    col_offset: int,
    reference_proxy: np.ndarray,
    target_proxy: np.ndarray,
    row_start: int,
    col_start: int,
    tile_size: int,
    inv_sigma: float,
    saturation_threshold: float = 0.95,
) -> float | None:
    """
    Computes the area-normalized, Variance-Weighted Sum of Absolute Differences (VW-SAD).

    This helper handles the spatial translation between frames and clips the
    calculation to the valid intersection of the tile and the image boundaries.

    Args:
        row_offset: Integer vertical shift to apply to the target frame.
        col_offset: Integer horizontal shift to apply to the target frame.
        reference_proxy: Grayscale luma proxy of the reference frame.
        target_proxy: Grayscale luma proxy of the frame being aligned.
        row_start: Top-most coordinate of the tile in the reference frame.
        col_start: Left-most coordinate of the tile in the reference frame.
        tile_size: Size (width/height) of the square tile.
        inv_sigma: Pre-calculated inverse noise floor (1/sigma) for normalization.
        saturation_threshold: A value above this threshold is considered saturated and skipped
            in SAD calculation

    Returns:
        float or None: The normalized SAD score. A value of ~1.0 indicates differences
               consistent with the expected noise floor. None if offsets are out of bounds.

    """

    height, width = reference_proxy.shape

    # Calculate the intersection of the Target Tile (Ref + offset) and the Target Image
    # These must be clipped so we don't index out of bounds
    r_start = max(row_start, -row_offset, 0)
    r_end = min(row_start + tile_size, height - row_offset, height)
    c_start = max(col_start, -col_offset, 0)
    c_end = min(col_start + tile_size, width - col_offset, width)

    if r_start >= r_end or c_start >= c_end:
        return None

    # Slicing: The target slice must be offset by `dy` and `dx` because it is the 'shifted' version of the reference
    ref_view = reference_proxy[r_start:r_end, c_start:c_end]
    tgt_view = target_proxy[r_start + row_offset : r_end + row_offset, c_start + col_offset : c_end + col_offset]

    # Width Numba per-pixel calculation is faster
    # np.sum(np.abs(...)) leads to temporary array allocations
    sad = 0.0
    non_clipped_count = 0
    rows, cols = ref_view.shape
    for r in range(rows):
        for c in range(cols):
            ref_val, tgt_val = ref_view[r, c], tgt_view[r, c]
            # Only count pixels that are valid (not clipped) in both frames
            if ref_val < saturation_threshold and tgt_val < saturation_threshold:
                sad += abs(ref_val - tgt_val)
                non_clipped_count += 1

    # If 75%+ of a tile is pure white (clipped), there isn't enough texture left to determine an offset.
    if non_clipped_count < (rows * cols) // 4:
        return None

    # Normalization: If tiles are partially off-image, a smaller area will naturally have a lower SAD.
    # Divide by area to find the true best match per pixel.
    return (sad * inv_sigma) / non_clipped_count


# ----------------------------------------- Merging functions -----------------------------------------


@njit(fastmath=True)
def find_best_integer_offset(
    reference_proxy: np.ndarray,
    target_proxy: np.ndarray,
    row_start: int,
    col_start: int,
    tile_size: int,
    hint_dy: int,
    hint_dx: int,
    search_radius: int,
    inv_sigma: float,
) -> tuple[int, int, float]:
    """
    Performs a local exhaustive search for the best integer translation.

    Args:
        reference_proxy: Grayscale reference image at a specific pyramid level.
        target_proxy: Grayscale target image to be aligned.
        row_start: Top row index of the tile in the proxy coordinate system.
        col_start: Left column index of the tile in the proxy coordinate system.
        tile_size: Size of the tile (pixels).
        hint_dy: Initial row offset guess (usually from a coarser level).
        hint_dx: Initial column offset guess (usually from a coarser level).
        search_radius: Number of pixels to search around the hint.
        inv_sigma: Pre-calculated inverse noise standard deviation for normalization.

    Returns:
        tuple: (best_dy, best_dx, minimum_sad_score)

    """

    min_sad = 1e20
    best_dy, best_dx = hint_dy, hint_dx

    # Search a square window centered on the hint
    for dy in range(hint_dy - search_radius, hint_dy + search_radius + 1):
        for dx in range(hint_dx - search_radius, hint_dx + search_radius + 1):
            # _compute_tile_sad handles boundary checks and returns None if out of bounds
            sad = _compute_tile_sad(dy, dx, reference_proxy, target_proxy, row_start, col_start, tile_size, inv_sigma)
            sad = 1e20 if (sad is None) else sad
            if sad < min_sad:
                min_sad = sad
                best_dy, best_dx = dy, dx

    return best_dy, best_dx, min_sad


@njit(fastmath=True)
def find_best_offset_final(
    reference_proxy: np.ndarray,
    target_proxy: np.ndarray,
    row_start: int,
    col_start: int,
    tile_size: int,
    hint_dy: int,
    hint_dx: int,
    search_radius: int,
    inv_sigma: float,
) -> tuple[float, float, float]:
    """
    Refines a coarse integer shift into a sub-pixel translation using quadratic fitting.

    This is the final step of the pyramid search, operating on the highest-resolution proxy.
    It performs a narrow integer search followed by a 2D parabolic fit on the SAD surface.

    Args:
        reference_proxy: Level 0 (highest resolution) grayscale proxy.
        target_proxy: Level 0 grayscale target image.
        row_start: Top row index of the tile in Level 0 pixels.
        col_start: Left column index of the tile in Level 0 pixels.
        tile_size: Size of the tile in pixels.
        hint_dy: Initial row offset guess (from level 1).
        hint_dx: Initial column offset guess (from level 1).
        search_radius: Local search area (typically 1).
        inv_sigma: Noise-normalization factor.

    Returns:
        tuple: (subpixel_dy, subpixel_dx, normalized_sad_score)

    """

    # --- 1. Local Integer Refinement ---
    best_dy_int, best_dx_int, min_sad = find_best_integer_offset(
        reference_proxy=reference_proxy,
        target_proxy=target_proxy,
        row_start=row_start,
        col_start=col_start,
        tile_size=tile_size,
        hint_dy=hint_dy,
        hint_dx=hint_dx,
        search_radius=search_radius,
        inv_sigma=inv_sigma,
    )

    # --- 2. Sub-pixel Refinement within 3x3 neighborhood ---
    # Centered on the best integer offset to calculate the surface curvature
    neighborhood = np.zeros((3, 3), dtype=np.float32)
    neighborhood[1, 1] = min_sad

    # Up, Down, Left, Right
    cross_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for offset_row, offset_col in cross_offsets:
        dy, dx = best_dy_int + offset_row, best_dx_int + offset_col
        sad = _compute_tile_sad(dy, dx, reference_proxy, target_proxy, row_start, col_start, tile_size, inv_sigma)
        # Use -1 as a sentinel for tiles that fall off the image sensor boundary
        neighborhood[offset_row + 1, offset_col + 1] = -1 if (sad is None) else sad

    # --- 3. Boundary Fallback ---
    # If a neighbor is off-screen, mirror the SAD value from the opposite side
    # to allow the parabola to still be calculated.
    for i, j in cross_offsets:
        row_idx, col_idx = i + 1, j + 1
        if neighborhood[row_idx, col_idx] < 0:
            # Find the opposite neighbor's value
            opp_val = neighborhood[1 - i, 1 - j]
            neighborhood[row_idx, col_idx] = max(opp_val, min_sad)

    # --- 4. Sub-pixel Parabola Fitting ---
    # find_subpixel_shift fits a 1D parabola to rows and columns independently
    sub_dy, sub_dx = find_subpixel_shift(neighborhood)

    return float(best_dy_int) + sub_dy, float(best_dx_int) + sub_dx, min_sad


@njit(fastmath=True)
def find_best_offset_pyramids(
    reference_pyramids: list[np.ndarray],
    target_pyramids: list[np.ndarray],
    row_start: int,  # Coordinates at Level 0
    col_start: int,
    tile_size: int,
    search_radius: int,
    noise_scales: np.ndarray,
    noise_offsets: np.ndarray,
    exposure_scaler: float,
) -> tuple[float, float, float]:
    """
    Performs hierarchical motion estimation across a 3-level image pyramid.

    This function implements a coarse-to-fine search:
    1. Level 2 (1/4 proxy scale): Finds large-scale motion with a wide search.
    2. Level 1 (1/2 proxy scale): Refines the previous guess in a narrow window.
    3. Level 0 (Full proxy scale): Final sub-pixel refinement via find_best_offset_final.

    Args:
        reference_pyramids: List of [Level0, Level1, Level2] reference images.
        target_pyramids: List of [Level0, Level1, Level2] target images.
        row_start: Top row index of the tile in Level 0 pixels (highest resolution).
        col_start: Left column index of the tile in Level 0 pixels (highest resolution).
        tile_size: Pixel size of tiles (constant across levels).
        search_radius: Max search distance in proxy pixels at Level 0.
        noise_scales: 2x2 matrix of shot noises per-channel
        noise_offsets: 2x2 matrix of matrix read noise per-channel

    Returns:
        tuple: (final_dy, final_dx, final_sad_score)

    """

    # --- 1. Noise Floor & Normalization Setup ---

    # We calculate the noise floor based on Level 0 luma mean.
    ref_tile = reference_pyramids[0][row_start : row_start + tile_size, col_start : col_start + tile_size]
    tile_mean = np.mean(ref_tile)

    # Average noise parameters (G and B channels usually share similar noise)
    avg_scale = np.mean(noise_scales)
    avg_offset = np.mean(noise_offsets)

    # Statistical correction: averaging RGGB into Luma reduces variance by ~4x.
    # Must account for this so the SAD (from proxy) matches the Noise Model
    # Since luma proxy weighs greens more, the actual value is
    # 0.15**2 + 0.35**2 + 0.35**2 + 0.15**2 = 0.29
    proxy_variance_scale = 0.29

    # Variance of Reference: Var_ref = g*x + c
    # Variance of Target (Scaled): Var_tgt = (scaler^2) * (g*(x/scaler) + c)
    #                                      = scaler*g*x + (scaler^2)*c
    # Total Variance = Var_ref + Var_tgt
    var_ref = avg_scale * tile_mean + avg_offset
    var_tgt = (exposure_scaler * avg_scale * tile_mean) + (exposure_scaler**2 * avg_offset)

    # Given the NoiseProfile from the metadata, the noise model is: sigma^2 = g * x + c
    sigma_sq = max(1e-9, var_ref + var_tgt) * proxy_variance_scale
    inv_sigma = 1.0 / (np.sqrt(sigma_sq) + 1e-8)

    # --- 2. Coarsest Level (Level 2: 1/4 of Proxy) ---
    # Search radius is scaled down. A 1px shift here = 4px at Level 0.
    best_dy, best_dx, _ = find_best_integer_offset(
        reference_proxy=reference_pyramids[2],
        target_proxy=target_pyramids[2],
        row_start=row_start // 4,
        col_start=col_start // 4,
        tile_size=tile_size,
        hint_dy=0,
        hint_dx=0,
        search_radius=search_radius // 4,
        inv_sigma=inv_sigma,
    )

    # --- 3. Middle Level (Level 1: 1/2 of Proxy) ---
    # Multiply coarsest shift by 2 to align with Level 1 scale.
    best_dy, best_dx, _ = find_best_integer_offset(
        reference_proxy=reference_pyramids[1],
        target_proxy=target_pyramids[1],
        row_start=row_start // 2,
        col_start=col_start // 2,
        tile_size=tile_size,
        hint_dy=best_dy * 2,
        hint_dx=best_dx * 2,
        search_radius=1,
        inv_sigma=inv_sigma,
    )

    # --- 4. Level 0 (Highest Res Proxy) ---
    # Final refinement with sub-pixel interpolation.
    return find_best_offset_final(
        reference_proxy=reference_pyramids[0],
        target_proxy=target_pyramids[0],
        row_start=row_start,
        col_start=col_start,
        tile_size=tile_size,
        hint_dy=best_dy * 2,
        hint_dx=best_dx * 2,
        search_radius=1,
        inv_sigma=inv_sigma,
    )


@njit(fastmath=True)
def merge_tile(
    merged_accumulator: np.ndarray,
    weights_accumulator: np.ndarray,
    target_image: np.ndarray,
    row_start: int,
    col_start: int,
    row_offset: float,
    col_offset: float,
    sad_score: float,
    tile_size: int,
    blending_window: np.ndarray,
    k: float,
    exposure_scaler: float = 1.0,
    saturation_threshold: float = 0.95,
    is_reference: bool = False,
) -> None:
    """
    Performs weighted merging of a tile from target image into the final image.

    Tiles with lower SAD scores (better alignment) are given higher weights.

    Args:
        merged_accumulator: The full-res buffer accumulating weighted pixel values (H, W).
        weights_accumulator: The full-res buffer accumulating weights for normalization (H, W).
        target_image: The image to be merged to the reference one (H, W).
        row_start: Top row index of the tile in full-res coordinates.
        col_start: Left column index of the tile in full-res coordinates.
        row_offset: Offset in row/y/height direction for the target image's tile.
        col_offset: Offset in col/x/width direction for the target image's tile.
        sad_score: SAD value from the proxy alignment step.
        tile_size: The width/height of the tile in full-res pixels.
        blending_window :
            A 2D weight map (e.g., Hann or Gaussian) applied to each tile to
            taper edges and ensure seamless transitions between overlapping
            regions. Should match the tile dimensions.
        k:
           - High k (4.0 - 12.0): Promotes temporal averaging (ideal for high-ISO denoising).
           - Low k (0.5 - 1.0): Promotes frame rejection (ideal for sharp motion/ghosting).

    """

    height, width = merged_accumulator.shape

    # 1. Temporal robustness
    # Determines if the tile matches the reference. If SAD is high (ghosting/motion),
    # the weight drops to near zero, effectively ignoring this specific tile.
    robustness_weight = np.exp(-sad_score / k)
    if robustness_weight < 1e-4:
        return

    # 2. Snr priority
    # In linear space, SNR is proportional to exposure time. We weight the long
    # exposure frame higher (e.g., if 4x longer, it gets 4x weight) so it dominates
    # the shadow/midtone average, significantly reducing visible noise.
    snr_weight = 1.0 / exposure_scaler

    # 3. Transition parameters
    # We want to use the 'clean' long exposure as late as possible.
    edge_softness = 0.05
    lower_bound = saturation_threshold - edge_softness  # e.g., 0.90

    # 4. Boundary-safe intersection
    # We must ensure that [r + row_offset] and [c + col_offset] stay within
    # the target_image bounds. We add a 1-pixel margin for the bilinear interpolation.
    r_start = int(max(row_start, np.ceil(-row_offset), 0))
    r_end = int(min(row_start + tile_size, np.floor(height - row_offset - 1), height))

    c_start = int(max(col_start, np.ceil(-col_offset), 0))
    c_end = int(min(col_start + tile_size, np.floor(width - col_offset - 1), width))

    if r_start >= r_end or c_start >= c_end:
        return

    # 5. Accumulation Loop
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            # Subpixel sampling from the target image
            raw_val = sample_raw_bilinear(target_image, r, c, row_offset, col_offset)
            # Check for saturation of the non-scaled pixel: if clipped, it carries no useful detail
            # A value from reference image should be added anyway
            if not is_reference and raw_val > saturation_threshold:
                continue
            # If the pixel is in the 'danger zone' [0.90, 0.95], we linearly reduce
            # the long exposure's weight. This forces the accumulator to cross-fade
            # to the short-exposure frames, preventing jagged clipping edges.
            if not is_reference and raw_val > lower_bound:
                fade = (saturation_threshold - raw_val) / edge_softness
            else:
                fade = 1.0
            # Reference always has weight 1.0; others are scaled by SNR and the Fade ramp.
            current_snr_w = 1.0 if is_reference else (snr_weight * fade)

            # Spatial blending
            # Combines robustness (motion), SNR (quality), and the Hann window (seams).
            combined_weight = robustness_weight * current_snr_w * blending_window[r - row_start, c - col_start]

            # Linearize and Accumulate
            merged_accumulator[r, c] += (raw_val * exposure_scaler) * combined_weight
            weights_accumulator[r, c] += combined_weight


@register_step(ISPStep.ALIGN_AND_MERGE)
def merge_images(
    burst_images: list[np.ndarray],
    metadata: list[dict[str, Any]],
    tile_size: int = 32,
    max_search_radius: int = 32,
) -> np.ndarray:
    """
    Performs block-based alignment and temporal merging of a RAW image burst.

    This function reduces noise by aligning subsequent frames to a reference (the first
    frame) and merging them using a weighted average. Alignment is performed on
    downsampled grayscale 'proxies' to improve speed and robustness against noise,
    while the final merge occurs at the original RAW resolution.

    Args:
        burst_images: A list of RAW images (2D numpy arrays).
        metadata: A list of dictionaries containing per-frame sensor metadata (e.g.,
            exposure time, black level).
        tile_size: The width/height of the processing tile in full-resolution pixels.
            Must be a multiple of the downsampling factor (usually 2).
        max_search_radius: The maximum distance (in full-res pixels) the algorithm
            will search for a matching block in any direction.
            Should be multiple of 8 and at least 32.

    Returns:
        np.ndarray: The merged RAW image of shape (H, W), normalized by tile weights.

    Raises:
        ValueError: If fewer than two images are provided.
        ValueError: If the images in the burst have inconsistent shapes.
        RuntimeError: If tile parameters are incompatible with the downsampling factor.
        ValueError: If max_search_radius is not a multiple of 8.

    """

    if len(burst_images) <= 1:
        raise ValueError(f"At least two images needed for Align&Merge, but got {len(burst_images)}.")

    if len({x.shape for x in burst_images}) != 1:
        raise ValueError("All images in the burst must have the same shape.")

    if not all(metadata[0]["ExposureTime"] == mtd["ExposureTime"] for mtd in metadata[1:]):
        logger.info(
            "The burst contains images with different exposures: %s." % [mtd["ExposureTime"] for mtd in metadata]
        )

    if max_search_radius % 8 != 0:
        raise ValueError(f"max_search_radius should be multiple of 8, but got {max_search_radius}")

    # 1. Find the image with the highest sharpness
    exposure_scalers = get_exposure_scalers(metadata)
    sharpest_image_idx = find_sharpest_image_idx(burst_images, metadata)
    # The sharpest image will be our reference, other images will be merged into it
    reference_image = burst_images[sharpest_image_idx]
    reference_metadata = metadata[sharpest_image_idx]
    image_height, image_width = reference_image.shape

    # 2. Initialize accumulation buffers
    merged_accumulator = np.zeros((image_height, image_width), dtype=np.float32)
    weights_accumulator = np.zeros((image_height, image_width), dtype=np.float32)

    # 2. Generate reference grayscale proxy for alignment
    reference_luma_proxy = get_luma_proxy(reference_image, reference_metadata)
    proxy_height, proxy_width = reference_luma_proxy.shape
    if image_height // proxy_height != image_width // proxy_width:
        raise RuntimeError("Downsampling scale for luma proxy is uneven for width and height.")

    # Calculate scaling factor between proxy and full-res space
    size_scaler = image_height // proxy_height
    if size_scaler % 2 != 0:
        logger.warning("Luma proxy is scaled by an odd number: merging can cause catastrophic color artifacts.")

    # 3. Parameters for block matching
    tile_stride = tile_size // 2
    proxy_tile_size = tile_size // size_scaler
    proxy_stride = tile_stride // size_scaler
    proxy_max_search_radius = max_search_radius // size_scaler

    # 4. Noise Estimation & Adaptive K Calculation
    noise_scales, noise_offsets = get_noise_params_2x2(burst_images, metadata)

    # Calculates k using a logarithmic ISO scale. Doubling ISO (1 stop) increases k linearly.
    current_iso = metadata[0].get("ISO")
    if not current_iso:
        raise RuntimeError("ISO is not provided in the metadata.")
    stops = np.log2(current_iso / 100.0)
    # VW-SAD scores are in "sigma units", k = 1.0 means we trust differences up to 1 standard deviation.
    k_adaptive = 1.0 + (0.5 * stops)

    # 5. Main loop: block-matching and merging
    hann_window = get_hann_window_2d(tile_size)

    # Pre-calculate tile start positions to ensure we hit the bottom/right edges
    proxy_rows = list(range(0, proxy_height - proxy_tile_size, proxy_stride))
    proxy_rows.append(proxy_height - proxy_tile_size)  # Edge coverage
    proxy_cols = list(range(0, proxy_width - proxy_tile_size, proxy_stride))
    proxy_cols.append(proxy_width - proxy_tile_size)  # Edge coverage

    reference_pyramids = [reference_luma_proxy]  # Level 0
    for _ in range(2):  # Level 1 and 2
        reference_pyramids.append(downsample_luma_proxy(reference_pyramids[-1]))

    # Note: we need to loop through the reference image using the same tiling logic as the target images.
    #       This ensures the spatial weights (the "window sum") are perfectly uniform across the frame.
    for img_idx in trange(len(burst_images), desc="Align&Merge (Images)", ascii=True):
        if img_idx != sharpest_image_idx:
            target_pyramids = [
                get_luma_proxy(burst_images[img_idx], metadata[img_idx]) * exposure_scalers[img_idx]
            ]  # Level 0
            for _ in range(2):  # Level 1 and 2
                target_pyramids.append(downsample_luma_proxy(target_pyramids[-1]))

        for proxy_row in proxy_rows:
            for proxy_col in proxy_cols:
                if img_idx == sharpest_image_idx:
                    row_offset, col_offset, score = 0, 0, 0
                else:
                    row_offset, col_offset, score = find_best_offset_pyramids(
                        reference_pyramids=reference_pyramids,
                        target_pyramids=target_pyramids,
                        row_start=proxy_row,
                        col_start=proxy_col,
                        tile_size=proxy_tile_size,
                        search_radius=proxy_max_search_radius,
                        noise_scales=noise_scales,
                        noise_offsets=noise_offsets,
                        exposure_scaler=exposure_scalers[img_idx],
                    )
                # Merging is done on full-res image, thus size_scaler is used
                merge_tile(
                    merged_accumulator=merged_accumulator,
                    weights_accumulator=weights_accumulator,
                    target_image=burst_images[img_idx],
                    row_offset=row_offset * size_scaler,
                    col_offset=col_offset * size_scaler,
                    sad_score=score,
                    row_start=proxy_row * size_scaler,
                    col_start=proxy_col * size_scaler,
                    tile_size=tile_size,
                    blending_window=hann_window,
                    k=k_adaptive,
                    exposure_scaler=exposure_scalers[img_idx],
                    is_reference=(img_idx == sharpest_image_idx),
                )

    # 6. Final normalization (weighted average across overlapping tiles)
    mask = weights_accumulator > 0
    merged_accumulator[mask] /= weights_accumulator[mask]

    return merged_accumulator
