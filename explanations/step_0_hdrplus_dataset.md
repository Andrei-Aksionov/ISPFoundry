# HDR+ Dataset Overview

A curated collection of raw sensor data designed for computational photography and high dynamic range research.

The [HDR+ dataset](https://hdrplusdata.org/dataset.html) dataset serves as a benchmark for the ISP pipeline, providing the raw materials necessary to reconstruct professional-grade imagery from mobile sensor data. It centers on the concept of burst photography - capturing a rapid sequence of frames to overcome the physical limitations of small sensors.

<br>

## Burst Methodology

Rather than a single long exposure, the dataset utilizes a burst of Digital Negative (DNG) files. This approach provides three primary advantages for digital signal processing:

**Signal-to-Noise Improvement**<br>
By aligning and merging multiple frames, temporal noise is mathematically reduced. This increases the Signal-to-Noise Ratio (SNR) and effectively widens the usable dynamic range.

**Highlight Preservation**<br>
Frames are intentionally underexposed to protect highlight data from clipping. The lost shadows are later recovered through the merging process.

**Motion Robustness**<br>
Short exposure times minimize the impact of camera shake and subject movement, ensuring that the alignment stage remains accurate.

<br>

## Technical Components

Each burst directory contains the specific metadata and raw data required for the reconstruction process.

| Resource | Description |
| :--- | :--- |
| **`payload_*.dng`** | The raw sensor images containing the Bayer-pattern data. |
| **`lens_shading_map_*.tiff`** | Four-channel maps used to correct vignetting and color shading. |
| **`rgb2rgb.txt`** | A 3x3 CCM for transforming sensor RGB into linear sRGB space. |
| **`timing.txt`** | Execution logs for the Align, Merge, and Finish stages. |

<br>

## Pipeline Execution

The transition from raw data to a finished image is handled in three distinct phases.

**1. Align**<br>
Identification of a reference frame followed by the spatial alignment of the remaining burst.

**2. Merge**<br>
The mathematical averaging of aligned frames to suppress noise while maintaining detail.

**3. Finish**<br>
The final ISP application: Black Level, White Balance, Demosaic, CCM, and Tone Mapping.
