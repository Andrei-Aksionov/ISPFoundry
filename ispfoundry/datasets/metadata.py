from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

from ispfoundry.utils import get_exif_metadata


@dataclass(kw_only=True, slots=True)
class Metadata:
    file_path: Path
    image_width: int
    image_height: int
    black_levels: np.ndarray | None
    white_level: int
    color_description: str
    raw_pattern: np.ndarray
    exposure_time: float
    iso: int
    cfa_plane_color: str | None
    noise_profile: np.ndarray | None
    camera_model_name: str


def extract_metadata(file_path: Path) -> Metadata:  # noqa: D103

    exif_data = get_exif_metadata(file_path)[0]
    raw_obj = rawpy.imread(str(file_path))

    # Image sizes
    image_width = exif_data.get("ImageWidth")
    if not image_width:
        raise ValueError("ImageWidth is missing in the metadata.")
    image_height = exif_data.get("ImageHeight")
    if not image_height:
        raise ValueError("ImageHeight is missing in the metadata.")

    # Black Levels
    black_levels = exif_data.get("BlackLevel")
    black_levels = black_levels or raw_obj.black_level_per_channel

    if isinstance(black_levels, str):
        black_levels = np.array([float(x) for x in black_levels.split()])
    elif isinstance(black_levels, list):
        black_levels = np.asarray(black_levels, dtype=np.float32)

    if black_levels.size != 4:
        raise ValueError(f"Expected 4 black level values (e.g. RGGB), got: {black_levels}")

    # White Level
    white_level = exif_data.get("WhiteLevel")
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

    # Color description
    color_description = exif_data.get("color_desc")
    color_description = color_description or raw_obj.color_desc
    if isinstance(color_description, bytes):
        color_description = color_description.decode()

    if not color_description:
        raise ValueError(f"Color description cannot be empty of None, but got {color_description}")

    # Raw pattern
    raw_pattern = raw_obj.raw_pattern
    if isinstance(raw_pattern, list):
        raw_pattern = np.array(raw_pattern)
    if raw_pattern is None or raw_pattern.size == 0:
        raise ValueError(f"Raw pattern cannot be None of empty, but got {raw_pattern}.")

    # Exposure time
    exposure_time = exif_data.get("ExposureTime")
    if not exposure_time:
        raise ValueError(f"Exposure time cannot be None or empty, but got {exposure_time}")

    if isinstance(exposure_time, str):
        if "/" in exposure_time:
            numerator, denominator = exposure_time.split("/")
            exposure_time = float(numerator) / float(denominator)
        else:
            exposure_time = float(exposure_time)
    else:
        raise ValueError(f"Exposure time expected to be of type str, but got {type(exposure_time)}")

    # ISO
    iso = exif_data.get("ISO")
    if iso is None:
        raise ValueError("Cannot be None")
    # TODO (andrei aksionau): assert that it is int, if it's not and it's a string - parse it, check
    #    it's greater than 0
    # TODO (andrei aksionau): if anything of this fails - default to 100

    # CFA plane color
    cfa_plane_color = exif_data.get("CFAPlaneColor")
    if cfa_plane_color is not None and cfa_plane_color != "Red,Green,Blue":
        raise ValueError(f"The code expects that the matrix layout is Red Green Blue, but got {cfa_plane_color}")

    # Noise profile
    noise_profile = exif_data.get("NoiseProfile")
    if noise_profile is not None and isinstance(noise_profile, str):
        noise_profile = np.fromstring(noise_profile, sep=" ")
    if noise_profile is not None and noise_profile.size != 6:
        raise ValueError(f"Needs to be 6 values in a noise profile but got {noise_profile.size}")

    # Model name
    camera_model_name = exif_data.get("Model", "Unknown")

    raw_obj.close()

    return Metadata(
        file_path=file_path,
        image_width=image_width,
        image_height=image_height,
        black_levels=black_levels,
        white_level=white_level,
        color_description=color_description,
        raw_pattern=raw_pattern,
        exposure_time=exposure_time,
        iso=iso,
        cfa_plane_color=cfa_plane_color,
        noise_profile=noise_profile,
        camera_model_name=camera_model_name,
    )


if __name__ == "__main__":
    path = Path(
        "/Users/andreiaksionau/Developer/Computational_Photography/ISPFoundry/data/raw/hdrplus_dataset/0006_20160722_115157_431"
    )
    extract_metadata(path)
