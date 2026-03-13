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

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)  # Simple 3x3 Laplacian kernel

    scores = []
    for image, mtd in zip(images, metadata):
        # 1. Recalculate on-the-go to reduce memory footprint
        luma_proxy = get_luma_proxy(image, mtd)
        # 2. Light blur to suppress high-frequency noise that mimics sharpness
        # This ensures we measure actual structural edges, not sensor grain.
        smoothed = gaussian_filter(luma_proxy, sigma=0.5)
        # 3. Calculate Laplacian variance
        scores.append(convolve(smoothed, kernel).var())

    best_idx, best_score = max(enumerate(scores), key=lambda x: x[1])
    avg_score = sum(scores) / len(scores)
    improvement = (best_score / avg_score - 1) * 100

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

    pos = np.arange(tile_size)
    # The (pos + 0.5) ensures the window is centered on pixels
    w_1d = 0.5 * (1 - np.cos(2 * np.pi * (pos + 0.5) / tile_size))
    # Create 2D map via outer product
    return np.outer(w_1d, w_1d)


# ----------------------------------------- Merging function -----------------------------------------


@njit(fastmath=True)
def find_best_offset(
    reference_proxy: np.ndarray,
    target_proxy: np.ndarray,
    row_start: int,
    col_start: int,
    tile_size: int,
    max_offset: int,
    noise_scales: np.ndarray,
    noise_offsets: np.ndarray,
) -> tuple[int, int, float]:
    """
    Finds the integer translation (dy, dx) using Variance-Weighted SAD (VW-SAD).

    The score is normalized by the noise floor (sigma) of the tile, meaning a score of 1.0 indicates
    that the average pixel difference matches the expected noise level.

    Args:
        reference_proxy: The reference grayscale image (usually the first frame).
        target_proxy: The target grayscale image to be aligned.
        row_start: Top row index of the tile in the proxy coordinate system.
        col_start: Left column index of the tile in the proxy coordinate system.
        tile_size: The width/height of the tile in proxy pixels.
        max_offset: The maximum pixels to search in any direction.
        noise_scales: 2x2 matrix of shot noises per-channel
        noise_offsets: 2x2 matrix of matrix read noise per-channel

    Returns:
        A tuple of (best_dy, best_dx, minimum_sad).

    """

    # 1. Calculate the noise floor for this tile
    ref_tile = reference_proxy[row_start : row_start + tile_size, col_start : col_start + tile_size]
    tile_mean = np.mean(ref_tile)

    # Average noise parameters for the luma proxy
    avg_scale = np.mean(noise_scales)
    avg_offset = np.mean(noise_offsets)

    # Statistical correction: Luma proxy averaging reduces variance by ~4x
    # Must account for this so the SAD (from proxy) matches the Noise Model
    # Since luma proxy weighs greens more, the actual value is
    # 0.15**2 + 0.35**2 + 0.35**2 + 0.15**2 = 0.29
    proxy_variance_scale = 0.29

    # Variance of the difference: Var(Ref - Tgt) = Var(Ref) + Var(Tgt) ≈ 2 * Var(Ref)
    # Assumes that Var(Reference) ≈ Var(Target)
    # Given the NoiseProfile from the metadata, the noise model is: sigma^2 = g * x + c
    sigma_sq = max(1e-9, 2.0 * (avg_scale * tile_mean + avg_offset)) * proxy_variance_scale
    inv_sigma = 1.0 / (np.sqrt(sigma_sq) + 1e-8)

    height, width = reference_proxy.shape
    min_sad = 1e20
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
            # Divide by area to find the true best match per pixel.
            normalized_sad = (sad * inv_sigma) / (tile_size * tile_size)

            if normalized_sad < min_sad:
                min_sad = normalized_sad
                best_dy = dy
                best_dx = dx

    return best_dy, best_dx, min_sad


@njit(fastmath=True)
def merge_tile(
    merged_accumulator: np.ndarray,
    weights_accumulator: np.ndarray,
    target_image: np.ndarray,
    row_start: int,
    col_start: int,
    row_offset: int,
    col_offset: int,
    sad_score: float,
    tile_size: int,
    blending_window: np.ndarray,
    k: float,
    use_l2_kernel: bool = False,
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
        use_l2_kernel: if False - L1 kernel is used
           - L1 (Laplacian weighting): Best for static, noisy backgrounds.
           - L2 (Gaussian weighting): Best for aggressive ghosting rejection.

    """

    height, width = merged_accumulator.shape

    # Calculate weight: L1 (Laplacian) or L2 (Gaussian)
    weight = np.exp(-(sad_score**2) / k**2) if use_l2_kernel else np.exp(-sad_score / k)
    if weight < 1e-4:
        return

    # Boundary-safe ref and target tiles intersection
    r_start = max(row_start, -row_offset, 0)
    r_end = min(row_start + tile_size, height - row_offset, height)
    c_start = max(col_start, -col_offset, 0)
    c_end = min(col_start + tile_size, width - col_offset, width)

    if r_start >= r_end or c_start >= c_end:
        return

    # In-place accumulation (ideal for Numba)
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            val = target_image[r + row_offset, c + col_offset]
            combined_weight = weight * blending_window[r - row_start, c - col_start]
            merged_accumulator[r, c] += val * combined_weight
            weights_accumulator[r, c] += combined_weight


# TODO (andrei aksionau): perhaps get rid of tile_stride and max_search_offset altogether
@register_step(ISPStep.ALIGN_AND_MERGE)
def merge_images(
    burst_images: list[np.ndarray],
    metadata: list[dict[str, Any]],
    tile_size: int = 32,
    tile_stride: int = 16,
    max_search_offset: int = 8,
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
        tile_stride: The step size between tiles in full-resolution pixels. Smaller
            strides increase overlap and reduce tiling artifacts but increase compute time.
        max_search_offset: The maximum distance (in full-res pixels) the algorithm
            will search for a matching block in any direction.

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

    if not all(metadata[0]["ExposureTime"] != mtd["ExposureTime"] for mtd in metadata[1:]):
        logger.warning("The burst contains images with different exposures. Bracketing is not yet supported. ")

    if not tile_size // tile_stride == 2:
        raise ValueError(
            f"The code assumes 50% tiles overlap, but got tile size of {tile_size} and stride of {tile_stride}"
        )

    # 1. Find the image with the highest sharpness
    sharpest_image_idx = find_sharpest_image_idx(burst_images, metadata)
    # The sharpest image will be our reference, other images will be merged into it
    reference_image = burst_images[sharpest_image_idx]
    reference_metadata = metadata[sharpest_image_idx]
    image_height, image_width = reference_image.shape

    # 2. Initialize accumulation buffers
    merged_accumulator = reference_image.astype(np.float32, copy=True)
    weights_accumulator = np.ones((image_height, image_width), dtype=np.float32)

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
    proxy_tile_size = tile_size // size_scaler
    proxy_stride = tile_stride // size_scaler
    proxy_max_offset = max_search_offset // size_scaler

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

    for img_idx in trange(len(burst_images), desc="Align&Merge (Images)", leave=True, position=0):
        # don't need to merge the sharpest image (reference image) into itself
        if img_idx == sharpest_image_idx:
            continue
        target_luma_proxy = get_luma_proxy(burst_images[img_idx], metadata[img_idx])
        for proxy_row in proxy_rows:
            for proxy_col in proxy_cols:
                row_offset, col_offset, score = find_best_offset(
                    reference_proxy=reference_luma_proxy,
                    target_proxy=target_luma_proxy,
                    row_start=proxy_row,
                    col_start=proxy_col,
                    tile_size=proxy_tile_size,
                    max_offset=proxy_max_offset,
                    noise_scales=noise_scales,
                    noise_offsets=noise_offsets,
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
                )

    # 5. Final normalization (weighted average across overlapping tiles)
    mask = weights_accumulator > 0
    merged_accumulator[mask] /= weights_accumulator[mask]

    return merged_accumulator
