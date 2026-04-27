from typing import Sequence

import cv2
import numpy as np
from loguru import logger

from ispfoundry import ISPStep, register_step
from ispfoundry.configs.config_loader import config
from ispfoundry.datasets import Metadata
from ispfoundry.utils import decode_cfa


def align_cfa_pattern(lsc_maps: Sequence[np.ndarray], metadata: Sequence[Metadata]) -> list[np.ndarray]:
    """
    Aligns the CFA pattern of lens shading maps to match that of the images.

    Args:
        lsc_maps: Sequence of lens shading maps.
        metadata: Sequence of metadata classes containing color description and raw pattern information.

    Returns:
        list: List of lens shading maps with aligned CFA patterns.

    Raises:
        ValueError: If the color description or raw pattern is missing in any metadata.

    """

    # CFA of the lens shading map might be different to the CFA of the image
    lsc_cfa = config.pipeline.lsc_cfa
    lsc_maps_reordered = []

    for idx, (lsc_map, mtd) in enumerate(zip(lsc_maps, metadata)):
        color_description = mtd.color_description
        if not color_description:
            raise ValueError(f"Color description is missing in the metadata[{idx}].")
        raw_pattern = mtd.raw_pattern
        if raw_pattern is None or raw_pattern.size == 0:
            raise ValueError(f"Raw pattern is missing in the metadata[{idx}].")

        image_cfa = decode_cfa(color_description, raw_pattern)
        reordering_indices = [image_cfa.index(ch) for ch in lsc_cfa]
        lsc_maps_reordered.append(lsc_map[:, :, reordering_indices])

    return lsc_maps_reordered


def interpolate(lsc_map: np.ndarray, metadata: Metadata) -> np.ndarray:
    """
    Interpolates the lens shading map to match the dimensions of the original image.

    Args:
        lsc_map: Lens shading map.
        metadata: Metadata class containing width and height information.

    Returns:
        np.ndarray: Interpolated lens shading map.

    """

    width = metadata.image_width
    height = metadata.image_height

    interpolated_lsc_planes = []
    for lsc_plane in np.unstack(lsc_map, axis=-1):
        interpolated_lsc_planes.append(
            cv2.resize(lsc_plane, dsize=(width // 2, height // 2), interpolation=cv2.INTER_LINEAR_EXACT)
        )

    interpolated_lsc_map = np.empty((height // 2, width // 2, 4), dtype=np.float32)
    for idx in range(4):
        interpolated_lsc_map[..., idx] = interpolated_lsc_planes[idx]

    return interpolated_lsc_map


def apply_single_image(img: np.ndarray, lsc_map: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Applies the lens shading map to a single image.

    Args:
        img: Input image.
        lsc_map: Lens shading map.
        inplace: If True, modifies the input image in place; otherwise, creates a copy and returns it.

    Returns:
        np.ndarray: Corrected image with lens shading applied.

    """

    corrected_img = img if inplace else img.copy()
    for idx in range(4):
        row_offset, col_offset = divmod(idx, 2)
        corrected_img[row_offset::2, col_offset::2] *= lsc_map[..., idx]

    return corrected_img


@register_step(ISPStep.LENS_SHADING_CORRECTION)
def apply_lens_shading_correction(
    input_images: np.ndarray,
    metadata: Sequence[Metadata],
    lsc_maps: Sequence[np.ndarray],
    inplace: bool = False,
) -> np.ndarray:
    """
    Applies lens shading correction to a burst of images.

    Args:
        input_images: 3D Numpy array of shape (N, H, W) containing input images.
        metadata: Sequence of metadata classes for each image, containing color description and raw pattern information.
        lsc_maps: Sequence of lens shading maps corresponding to the images.
        inplace: If True, modifies the input images in place; otherwise, creates copies and returns them.

    Returns:
        A numpy array of shape (N, H, W) that contains corrected images with lens shading applied.

    """

    # 1. Checks for equality
    # lsc map is calibrated per device and thus should be identical across the burst
    if np.equal(lsc_maps[0], lsc_maps[1:]).all():
        logger.info("All lens shading maps are identical. Reusing the first one.")
        lsc_maps = [lsc_maps[0]]
        metadata = [metadata[0]]

    # 2. Aligns lsc maps
    lsc_maps = align_cfa_pattern(lsc_maps, metadata)

    # 3. Interpolates
    lsc_maps = [interpolate(lsc_map, mt) for lsc_map, mt in zip(lsc_maps, metadata)]

    # 4. Applies to the burst
    if len(lsc_maps) == 1 and len(lsc_maps) != len(input_images):
        lsc_maps = lsc_maps * len(input_images)

    processed_images = input_images if inplace else input_images.copy()
    for image, lsc_map in zip(processed_images, lsc_maps):
        apply_single_image(image, lsc_map, inplace=True)

    return processed_images
