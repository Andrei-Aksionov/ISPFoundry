# Black Level Subtraction

## What is Black Level?

The **black level** (also known as dark level or pedestal) is a constant voltage offset introduced by the camera sensor's readout electronics. Even when no light reaches the sensor, the output signal is not zero due to:

- **Electronic bias**: ADCs require a positive voltage for proper operation.
- **Amplifier offsets**: Signal amplifiers add DC offsets in the readout chain.
- **Dark current**: Thermally generated electrons create a baseline signal.
- **Manufacturing variations**: Each camera has unique electronic characteristics.

This offset ensures the sensor operates in a valid voltage range but must be removed for accurate image processing.

## Why Subtract Black Level?

Black level subtraction is essential for several reasons:

### 1. Photometric Accuracy

Pixel values should directly represent light intensity. A value of 0 must correspond to zero photons. The black level shifts all values upward, violating this principle.

### 2. Linear Sensor Response

The raw sensor output follows:

$$ RAW(x,y) = BL + k \cdot I(x,y) + n(x,y) $$

Where:

- $RAW(x,y)$: Raw pixel value at position $(x,y)$
- $BL$: Electronic offset (constant per channel)
- $k$: Sensor gain (conversion factor)
- $I(x,y)$: Light intensity (photons)
- $n(x,y)$: Sensor noise

To recover the true light intensity, we must subtract the black level:

$$ I'(x,y) = RAW(x,y) - BL $$

### 3. Correct Shadow Rendering

Without subtraction:

- Blacks appear as mid-gray.
- Contrast is reduced.
- Shadows lose detail due to incorrect offset.

### 4. Enables Downstream Processing

Operations like white balance, color correction, and gamma mapping assume a true zero point. Applying them to offset data causes:

- Color casts in dark areas.
- Incorrect tone curves.
- Artifacts in low-light regions.

## Pipeline Order

Black level subtraction **must be the first step** in any ISP pipeline:

1. ⭐ **Black Level Subtraction** (mandatory)
2. Lens Shading Correction
3. White Balance
4. Align and Merge burst of images
5. Demosaicing
6. ... Remaining steps

### Why First?

- Corrects a fundamental sensor artifact present in raw ADC data.
- Ensures all subsequent operations (multiplicative or additive) work on true light values.
- Prevents errors like spatially varying offsets from lens shading or color-dependent shifts from white balance.

**Example**: White balance scales channels multiplicatively. On offset data:
$$P' = k \cdot (BL + I) = k \cdot BL + k \cdot I$$
This adds a color-dependent offset, corrupting neutrality.

## Mathematical Details

### Per-Channel Subtraction

For RGGB Bayer CFA, black levels differ per color channel:

$$BL = \begin{bmatrix} BL_R & BL_{G_r} \\ BL_{G_b} & BL_B \end{bmatrix}$$

Applied per pixel based on its $(x, y)$ position in the Bayer grid:
$$P(x, y) = RAW(x, y) - BL_c$$

Where $c \in \{R, G_r, G_b, B\}$.

### Normalization (Optional)

After subtraction, the image is often normalized to a `[0, 1]` range using the **White Level** ($WL$), which represents the sensor's saturation point:

$$P_{norm}(x, y) = \frac{P(x, y)}{WL - BL_c}$$

This ensures that the "useful" signal range (from black to saturation) is mapped to unit scale.

But subtraction alone is often sufficient for linear pipelines.

## How to Obtain Black Levels

- **From metadata**: EXIF/DNG tags like "BlackLevel" (string or array).
- **Fallback**: Estimate from image minima per CFA channel (less accurate).
- **Never use**: Global image minimum, as it's scene-dependent.

## Common Misconceptions

### "Subtract the image's minimum value instead"

**Incorrect**. The minimum pixel is influenced by noise, hot pixels, and scene content. Black level is a calibrated constant from sensor characterization, not per-image.

### "Only needed for scientific imaging"

**Incorrect**. Mandatory for all raw processing: photography, video, CV, ML. Without it, colors shift, dynamic range compresses, and algorithms fail.

### "Apply white balance before subtraction"

**Incorrect**. White balance is multiplicative; on offset data, it scales the black level, introducing color casts. Always subtract first.

### "Black levels are identical across channels"

**Incorrect**. Channels often differ due to separate amplifiers and filter variations. Use per-channel values from metadata when available.

### "Negative values after subtraction are bad"

**Incorrect**. Noise is zero-mean; negatives represent valid fluctuations below the mean dark level. Preserve them for accurate statistics in denoising/merging.

## Summary

Black Level Subtraction:

- Removes electronic offset from sensor readout
- Ensures zero pixel value = zero light
- Must be applied per-channel (R, Gr, Gb, B)
- First operation in the ISP pipeline (mandatory)
- Prerequisite for all color and tone processing
