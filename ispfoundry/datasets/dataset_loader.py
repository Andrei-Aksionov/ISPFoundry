from pathlib import Path
from typing import Any

import numpy as np
import rawpy
import tifffile
from loguru import logger

from ispfoundry.utils import get_exif_metadata


class DatasetLoader:
    def __init__(self, folder_path: str | Path) -> None:
        self.folder_path = Path(folder_path)
        self.dng_file_paths = [p for p in sorted(self.folder_path.iterdir()) if p.match("payload_*.dng")]

    def load_data(self) -> None:
        logger.info("Loading RAW images")
        self.raw_images = self.get_raw_images()
        logger.info("Loading metadata")
        self.metadata = self.get_metadata()
        logger.info("Loading lens shading correction maps")
        self.lsc_maps = self.get_lens_shading_correction_maps()

    def get_raw_images(self) -> np.ndarray:
        raw_images = []
        for dp in self.dng_file_paths:
            with rawpy.imread(str(dp)) as raw_obj:
                raw_image = raw_obj.raw_image.astype(np.float32)
                raw_images.append(raw_image)

        return np.stack(raw_images, dtype=np.float32)

    def get_metadata(self) -> list[dict[str, Any]]:
        metadata = get_exif_metadata(self.dng_file_paths)
        for idx, p in enumerate(self.dng_file_paths):
            with rawpy.imread(str(p)) as raw_obj:
                metadata[idx]["color_desc"] = raw_obj.color_desc.decode()
                metadata[idx]["raw_pattern"] = raw_obj.raw_pattern

        return metadata

    def get_lens_shading_correction_maps(self) -> list[np.ndarray]:
        lsc_maps = []
        for dp in self.dng_file_paths:
            lens_shading_map_path = dp.parent / (dp.stem.replace("payload", "lens_shading_map") + ".tiff")
            lens_shading_map = tifffile.imread(lens_shading_map_path)
            lsc_maps.append(lens_shading_map)

        return lsc_maps
