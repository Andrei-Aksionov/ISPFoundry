# HDR+ Dataset Overview

For the custom ISP pipeline I'll be using images from the HDR+ dataset. Let me explain what it is.

The [HDR+ dataset](https://hdrplusdata.org/dataset.html) is a collection of raw sensor data and processed images captured using Google's HDR+ computational photography pipeline. It is primarily used for research in high dynamic range (HDR) imaging and low-light photography.

## Burst Photography

A single scene in this dataset consists of multiple **DNG (Digital Negative)** files, forming a "burst". This approach is used for:

* **SNR Improvement:** By aligning and merging multiple frames, temporal noise is reduced, increasing the Signal-to-Noise Ratio (SNR) and widening the usable dynamic range.
* **Highlight Preservation:** Images are often captured with short exposure times (underexposed) to prevent highlights (like the sky) from clipping.
* **Motion Robustness:** Short exposures minimize motion blur and camera shake, making it easier to align frames.

## Included Files

In a typical burst directory, you will find:

* **`payload_*.dng`**: The raw sensor images.
* **`lens_shading_map_*.tiff`**: Low-resolution 4-channel maps used to correct for **vignetting** and color shading.
* **`rgb2rgb.txt`**: A 3x3 **Color Correction Matrix (CCM)** that transforms sensor-specific RGB values into a standard linear sRGB color space.
* **`timing.txt`**: Logs providing execution times for the three main pipeline stages: **Align**, **Merge**, and **Finish**.

## HDR+ Pipeline Stages

1. **Align:** Identifying a reference frame and aligning the burst.
2. **Merge:** Averaging aligned frames to reduce noise.
3. **Finish:** Applying the ISP pipeline (Black Level, White Balance, Demosaic, CCM, and Tone Mapping).
