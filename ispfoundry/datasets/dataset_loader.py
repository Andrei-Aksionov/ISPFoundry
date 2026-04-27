from pathlib import Path
from typing import List, Union

import numpy as np
import numpy.typing as npt
import rawpy
import tifffile
from loguru import logger

from ispfoundry.datasets import Metadata, extract_metadata


class DatasetLoader:
    """
    Handles loading of RAW image data, EXIF/sensor metadata, and lens shading maps.

    This loader specifically targets DNG files following a 'payload_*.dng'
    naming convention and expects corresponding LSC maps in TIFF format.
    """

    def __init__(self, folder_path: Union[str, Path], dtype: npt.DTypeLike = np.float32) -> None:
        """
        Initializes the DatasetLoader.

        Args:
            folder_path (Union[str, Path]): Path to the directory containing DNG and TIFF files.
            dtype (npt.DTypeLike): The target numpy data type for image and map loading.
                Defaults to np.float32.

        """
        self.folder_path = Path(folder_path)
        self.dtype = dtype

        self.dng_file_paths = sorted(self.folder_path.glob("payload_*.dng"))

        self.raw_images: np.ndarray
        self.metadata: List[Metadata]
        self.lsc_maps: List[np.ndarray]

    def load_data(self) -> None:
        """
        Executes the full loading sequence for images, metadata, and LSC maps.

        Populates self.raw_images, self.metadata, and self.lsc_maps.
        """
        self.raw_images = self.get_raw_images()
        self.metadata = self.get_metadata()
        self.lsc_maps = self.get_lens_shading_correction_maps()

    def get_raw_images(self) -> np.ndarray:
        """
        Loads Bayer raw data from DNG files and stacks them into a single array.

        Returns:
            np.ndarray: A 3D array of shape (N, H, W) containing RAW image data
                cast to the specified self.dtype.

        """
        logger.info(f"Loading RAW images as `{np.dtype(self.dtype).name}`")
        raw_images = []
        for dp in self.dng_file_paths:
            with rawpy.imread(str(dp)) as raw_obj:
                # Cast to the user-defined dtype
                raw_image = raw_obj.raw_image.astype(self.dtype)
                raw_images.append(raw_image)

        return np.stack(raw_images)

    def get_metadata(self) -> List[Metadata]:
        """
        Extracts EXIF data and internal sensor constants from the DNG files.

        Adds black_level and white_level only if they are not already present
        in the initial metadata.

        Returns:
            List[Metadata]: A list of Metadata classes containing metadata
                for each image.

        """

        logger.info("Loading metadata")

        return [extract_metadata(path) for path in self.dng_file_paths]

    def get_lens_shading_correction_maps(self) -> List[np.ndarray]:
        """
        Loads lens shading correction (LSC) maps associated with the DNG files.

        Returns:
            List[np.ndarray]: A list of numpy arrays cast to self.dtype
                representing the gain maps.

        Raises:
            FileNotFoundError: If a corresponding LSC TIFF file is missing
                for any discovered DNG file.

        """

        logger.info(f"Loading lens shading correction maps as `{np.dtype(self.dtype).name}`")
        lsc_maps = []
        for dp in self.dng_file_paths:
            lsc_name = dp.name.replace("payload", "lens_shading_map")
            lsc_path = (dp.parent / lsc_name).with_suffix(".tiff")

            if not lsc_path.exists():
                error_msg = f"LSC map not found for {dp.name}. Expected: {lsc_path.name}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            lsc_map = tifffile.imread(lsc_path).astype(self.dtype)
            lsc_maps.append(lsc_map)

        return lsc_maps
