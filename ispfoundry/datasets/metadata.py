from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

from ispfoundry.utils import get_exif_metadata


@dataclass(kw_only=True, slots=True)
class Metadata:
    file_path: Path
    black_levels: np.ndarray | None
    white_level: int


def get_metadata(folder_path: Path) -> list[Metadata]:  # noqa: D103

    # Filter to only .dng files
    dng_paths = list(folder_path.glob("*.dng"))

    exif_data = get_exif_metadata(dng_paths)

    output = []
    for exif_data_entry, dng_path in zip(exif_data, dng_paths):
        raw_obj = rawpy.imread(str(dng_path))
        # Black Levels
        black_levels = exif_data_entry.get("BlackLevel")
        black_levels = black_levels or raw_obj.black_level_per_channel

        if isinstance(black_levels, str):
            black_levels = np.array([float(x) for x in black_levels.split()])
        elif isinstance(black_levels, list):
            black_levels = np.asarray(black_levels, dtype=np.float32)

        if black_levels.size != 4:
            raise ValueError(f"Expected 4 black level values (e.g. RGGB), got: {black_levels}")

        # White Level
        white_level = exif_data_entry.get("WhiteLevel")
        white_level = white_level or raw_obj.white_level
        if white_level is None or white_level == 0:
            raise ValueError(f"Metadata should contain a valid WhiteLevel, but instead got: `{white_level}`")

        if any(bl > white_level for bl in black_levels):
            raise ValueError(
                "Black levels cannot be larger or equal to white level (saturation point), "
                f"but got black level values: {black_levels} and white level: {white_level}."
            )

        if any(bl == white_level for bl in black_levels):
            raise ValueError("WhiteLevel equals on the BlackLevels")

        raw_obj.close()
        metadata = Metadata(file_path=dng_path, black_levels=black_levels, white_level=white_level)
        output.append(metadata)

    return output


if __name__ == "__main__":
    path = Path(
        "/Users/andreiaksionau/Developer/Computational_Photography/ISPFoundry/data/raw/hdrplus_dataset/0006_20160722_115157_431"
    )
    get_metadata(path)
