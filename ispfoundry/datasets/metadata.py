import types
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import numpy as np
import rawpy
from loguru import logger

from ispfoundry.utils import get_exif_metadata


@dataclass(kw_only=True, slots=True, frozen=True)
class Metadata:
    file_path: Path
    """
    The absolute or relative filesystem path to the source DNG (Digital Negative) file.
    This is used for reference and for re-loading raw data if necessary during later
    stages of the pipeline.
    """

    image_width: int
    """
    The total horizontal pixel count of the image sensor data. In the context of
    raw files, this often refers to the 'active' area width, excluding non-image
    optical black pixels used for internal sensor calibration.
    """

    image_height: int
    """
    The total vertical pixel count of the image sensor data. Similar to width,
    this represents the number of rows of pixels captured by the sensor that
    contain actual light information.
    """

    black_levels: np.ndarray
    """
    A set of values (typically four) representing the 'dark current' or pedestal
    level for each color channel in the Bayer pattern (e.g., R, Gr, Gb, B).
    Since sensors record some electrical noise even in total darkness, these
    values must be subtracted to ensure that 'true black' in the scene results
    in a numerical value of zero.
    """

    white_level: int
    """
    The maximum possible numerical value a pixel can reach before it saturates.
    Any light intensity beyond this point is clipped. This value is critical for
    normalization, allowing the pipeline to map raw sensor readings into a
    standard 0.0 to 1.0 floating-point range.
    """

    color_description: str
    """
    A string (usually 4 characters like 'RGBG' or 'RGGB') that maps the sequence
    of colors in the raw sensor's filter array. This tells the ISP which
    pixel represents Red, Green, or Blue, which is essential for the
    'demosaicing' process that reconstructs a full-color image.
    """

    raw_pattern: np.ndarray
    """
    A 2x2 numerical matrix indicating the physical arrangement of the color
    filters on the sensor. Each number corresponds to a color index. This
    hardware-level description ensures the ISP correctly aligns its processing
    with the sensor's mosaic structure.
    """

    exposure_time: float
    """
    The duration (in seconds) that the sensor was exposed to light. This is a
    fundamental value for 'Align and Merge' algorithms to calculate the relative
    brightness between different frames in a burst and to compensate for
    motion blur.
    """

    iso: int
    """
    The gain or sensitivity setting applied to the sensor's signal. Higher ISO
    values amplify the signal (and the noise). This value is used by noise
    reduction algorithms to estimate the expected variance (noise) in the
    captured image.
    """

    camera_model_name: str
    """
    The string identifier of the camera hardware (e.g., 'Pixel 6' or 'IMX586').
    This is often used to look up factory-calibrated noise profiles or color
    correction matrices specific to that sensor and lens combination.
    """

    cfa_plane_color: str | None = None
    """
    An optional metadata field describing the color layout of the sensor planes.
    The ISP specifically expects this to be 'Red,Green,Blue' to ensure compatibility
    with its internal matrix multiplication and color conversion logic.
    """

    noise_profile: np.ndarray | None = None
    """
    A specialized 6-element array containing 'scale' and 'offset' parameters
    for the sensor's noise model. These parameters allow the ISP to mathematically
    predict how much photon noise and read noise will be present at a given
    signal intensity, enabling more effective denoising.
    """

    def __post_init__(self) -> None:
        """Validates the metadata fields to ensure ISP steps will not fail."""
        # --- Stage 1: Structural & Type Integrity ---
        self._check_non_optional_fields()
        self._check_field_types()

        # --- Stage 2: Specific Value Logic ---
        self._check_string_fields()  # Catches "" or " "
        self._check_path_fields()  # Catches "" or "."
        self._check_numpy_arrays()  # Catches size 0

        # --- Stage 3: ISP Domain Logic ---
        self._validate_geometry()
        self._validate_levels()
        self._validate_isp_requirements()

        # --- Stage 4: Locking ---
        self._make_numpy_arrays_readonly()  # Deep freeze

    def _check_non_optional_fields(self) -> None:
        """Iterates through dataclass fields and raises TypeError if a non-Optional field is None."""  # noqa: DOC501

        for field in fields(self):
            value = getattr(self, field.name)

            # Check if 'None' is a valid type for this field
            # This handles both 'Type | None' and 'Union[Type, None]'
            type_args = getattr(field.type, "__args__", None)
            is_optional = type_args is not None and type(None) in type_args

            if value is None and not is_optional:
                raise TypeError(f"Field '{field.name}' is mandatory but received 'None'. Expected type: {field.type}")

    def _check_field_types(self) -> None:
        """Strictly enforces that assigned values match the types defined in type hints."""  # noqa: DOC501

        for field in fields(self):
            value = getattr(self, field.name)

            # Skip None (Mandatory/Optional logic is handled in _check_non_optional_fields)
            if value is None:
                continue

            # Get the allowed types from the hint
            origin = get_origin(field.type)
            if origin in (Union, getattr(types, "UnionType", None)):
                allowed_types = get_args(field.type)
            else:
                allowed_types = (field.type,)

            # Filter out NoneType from allowed types for this specific check
            actual_allowed = tuple(t for t in allowed_types if t is not type(None))

            if not isinstance(value, actual_allowed):
                raise TypeError(f"Field '{field.name}' must be of type {field.type}, but received {type(value)}.")

    def _check_string_fields(self) -> None:
        """
        Check fields of type str.

        Automatically ensures that any string field (including Optional[str])
        is not just whitespace, provided the value is not None.
        """  # noqa: DOC501

        for field in fields(self):
            value = getattr(self, field.name)

            if isinstance(value, str) and not value.strip():
                raise ValueError(f"Field '{field.name}' cannot be an empty or whitespace-only string.")

    def _check_path_fields(self) -> None:
        """Value validation for Path objects."""  # noqa: DOC501
        for f in fields(self):
            value = getattr(self, f.name)

            # Ensure the path isn't just an empty string passed to Path()
            if isinstance(value, Path) and (str(value).strip() in (".", "")):
                raise ValueError(f"Field '{f.name}' appears to be an empty or invalid Path.")

    def _check_numpy_arrays(self) -> None:
        """
        Checks numpy arrays.

        Automatically ensures that any NumPy array field is not empty (size > 0),
        provided the value is not None.
        """  # noqa: DOC501

        for field in fields(self):
            value = getattr(self, field.name)

            if isinstance(value, np.ndarray) and value.size == 0:
                raise ValueError(f"Field '{field.name}' is an empty NumPy array (size 0).")

    def _make_numpy_arrays_readonly(self) -> None:
        """Sets the WRITEABLE flag to False for all ndarray fields."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def _validate_geometry(self) -> None:
        """Ensures dimensions are positive and non-zero."""  # noqa: DOC501
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(f"Invalid dimensions: {self.image_width}x{self.image_height}")

    def _validate_levels(self) -> None:
        """Ensures black and white levels are consistent for normalization."""  # noqa: DOC501

        if self.black_levels.size != 4:
            raise ValueError(f"Expected 4 black level values, got {self.black_levels.size}")

        # Check for saturation/underflow. Also prevents denominator == 0 in normalization.
        if np.any(self.black_levels >= self.white_level):
            raise ValueError(
                f"Black levels {self.black_levels} must be strictly less than "
                f"white level {self.white_level} to avoid zero/negative denominators."
            )

    def _validate_isp_requirements(self):
        """Validates fields specifically required by Align&Merge and LSC steps."""  # noqa: DOC501

        if not self.color_description or not self.color_description.strip():
            raise ValueError("color_description is missing or empty.")

        if self.raw_pattern.size == 0:
            raise ValueError("raw_pattern cannot be empty.")

        if self.cfa_plane_color is not None and self.cfa_plane_color != "Red,Green,Blue":
            raise ValueError(f"ISP expects 'Red,Green,Blue' layout, got: {self.cfa_plane_color}")

        if self.noise_profile is not None and self.noise_profile.size != 6:
            raise ValueError(f"NoiseProfile must contain 6 values, got {self.noise_profile.size}")

        if self.exposure_time <= 0:
            raise ValueError(f"Exposure time must be positive, got {self.exposure_time}")


def extract_metadata(file_path: Path) -> Metadata:
    """
    Extracts, cleans, and validates image metadata from a DNG file.

    This function acts as the primary ingestion point for the ISP pipeline. It
    coordinates the extraction of hardware-level information (like sensor black
    levels and Bayer patterns) and capture-level information (like exposure
    time and ISO). It prioritizes EXIF data where available, falling back to
    the raw file's internal properties (via rawpy).

    Args:
        file_path: The filesystem path to the .dng file to be processed.

    Returns:
        A strictly validated Metadata instance containing all parameters
        required for subsequent ISP processing steps.

    Raises:
        RuntimeError: If the EXIF metadata extraction fails entirely or the
            file is inaccessible.
        TypeError: If a metadata field (e.g., BlackLevel, ExposureTime, or
            NoiseProfile) is found in an unexpected data format that cannot
            be reliably parsed.
        ValueError: If numerical values are logically inconsistent, such as:
            - A malformed exposure time fraction (e.g., "1/0").
            - Image dimensions that are zero or negative.
            - Black levels that exceed or equal the white level saturation point.
            - A noise profile that does not contain exactly 6 elements.

    Notes:
        - ISO values are handled leniently: if missing or non-positive, the
          system logs a warning and defaults to 100 to allow processing to
          continue.
        - Exposure time is normalized to a float representing seconds,
          handling both decimal strings and fractional strings (e.g., "1/250").

    """

    # Fetch EXIF data
    exif_list = get_exif_metadata(file_path)
    if not exif_list:
        raise RuntimeError(f"Could not extract EXIF data from {file_path}")
    exif: dict[str, Any] = exif_list[0]

    # Context Manager for safe RawPy handle management
    with rawpy.imread(str(file_path)) as raw_obj:
        # 1. Strict Black Level Handling
        black_levels = exif.get("BlackLevel")
        if black_levels is None:
            black_levels = raw_obj.black_level_per_channel

        if isinstance(black_levels, str):
            black_levels = np.fromstring(black_levels, sep=" ")
        elif isinstance(black_levels, (list, np.ndarray)):
            black_levels = np.asarray(black_levels, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported BlackLevel type: {type(black_levels)}. Expected str, list, or ndarray.")

        # 2. Strict Exposure Time Handling
        exposure_time = exif.get("ExposureTime")
        if exposure_time is None:
            raise ValueError("Provided exposure time is None")
        if isinstance(exposure_time, str):
            if not exposure_time.strip():
                raise ValueError("Provided exposure time is an empty string")
            if "/" in exposure_time:
                try:
                    num, den = exposure_time.split("/")
                    exposure_time = float(num) / float(den)
                except (ValueError, ZeroDivisionError) as e:
                    raise ValueError(f"Invalid exposure time fraction '{exposure_time}': {e}") from e
            else:
                exposure_time = float(exposure_time)
        elif isinstance(exposure_time, (int, float)):
            exposure_time = float(exposure_time)
        else:
            raise TypeError(f"Unsupported ExposureTime type: {type(exposure_time)}. Expected str, int, or float.")

        # 3. ISO Handling
        iso_val = exif.get("ISO")
        if iso_val is None:
            logger.warning(f"ISO missing for {file_path.name}. Defaulting to 100.")
            iso = 100
        else:
            try:
                iso = int(iso_val)
                if iso <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                logger.warning(f"Invalid/Non-positive ISO '{iso_val}' for {file_path.name}. Defaulting to 100.")
                iso = 100

        # 4. Noise Profile
        noise_raw = exif.get("NoiseProfile")
        noise_profile = None
        if noise_raw is not None:
            if isinstance(noise_raw, str):
                noise_profile = np.fromstring(noise_raw, sep=" ")
            elif isinstance(noise_raw, (list, np.ndarray)):
                noise_profile = np.asarray(noise_raw, dtype=np.float32)
            else:
                raise TypeError(f"Unsupported NoiseProfile type: {type(noise_raw)}")

        color_desc = raw_obj.color_desc
        if isinstance(color_desc, bytes):
            color_desc = color_desc.decode()

        camera_model_name = f"{exif.get('Make', 'Unknown make')} {exif.get('Model', 'Unknown model')}"

        return Metadata(
            file_path=file_path,
            image_width=int(exif.get("ImageWidth") or raw_obj.sizes.width),
            image_height=int(exif.get("ImageHeight") or raw_obj.sizes.height),
            black_levels=black_levels,
            white_level=int(exif.get("WhiteLevel") or raw_obj.white_level),
            color_description=color_desc,
            raw_pattern=np.asarray(raw_obj.raw_pattern),
            exposure_time=exposure_time,
            iso=iso,
            camera_model_name=camera_model_name,
            cfa_plane_color=exif.get("CFAPlaneColor"),
            noise_profile=noise_profile,
        )
