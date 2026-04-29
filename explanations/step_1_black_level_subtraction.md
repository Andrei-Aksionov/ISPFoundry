# Black Level Subtraction

A fundamental calibration step that removes electronic offsets to ensure photometric linear accuracy.

The [HDR+ dataset](https://hdrplusdata.org/dataset.html) provides specific per-channel black level constants. Without removing this "pedestal," every subsequent calculation in the ISP, from white balance to merging, will be mathematically corrupted by a constant error term.

<br>

## The Nature of Black Level

The black level (pedestal) is a constant voltage offset introduced by the sensor's readout electronics. Even in total darkness, the sensor output is non-zero due to a combination of physical and electronic factors:

**Electronic Bias** <br>
Analog-to-Digital Converters (ADCs) require a baseline positive voltage for stable operation.

**Dark Current** <br>
Thermally generated electrons that accumulate in the pixels, creating a baseline signal independent of light.

**Readout Artifacts** <br>
Amplifier offsets and manufacturing variations unique to each sensor's electronic characteristics.

<br>

## Visual Impact

Failing to subtract the black level results in immediate and visible degradation of the final image. Because the "zero point" is misaligned, the entire tonal range is shifted.

**Shadow Rendering** <br>
Pure blacks appear as mid-gray, leading to a "washed out" look where shadows lose depth and fine detail is buried in the incorrect offset.

**Contrast and Dynamic Range** <br>
The effective contrast is reduced because the bottom of the histogram is artificially lifted, compressing the available space for real light data.

**Downstream Corruption** <br>
Operations like white balance, color correction, and gamma mapping assume a true zero point. Applying them to offset data causes color casts in dark areas, incorrect tone curves, and artifacts in low-light regions.

<br>

## Mathematical Foundation

The raw sensor output is a linear combination of signal, offset, and noise. To recover true light intensity, we must isolate the signal component.

<br>

$$\large RAW(x,y) = BL + k \cdot I(x,y) + n(x,y) $$

<br>

Where:

* $BL$: Electronic offset (the black level)
* $k$: Sensor gain (conversion factor)
* $I(x,y)$: True light intensity (photons)
* $n(x,y)$: Sensor noise

To normalize the data for the rest of the pipeline, we apply per-channel subtraction followed by an optional scale to a $[0, 1]$ range:

<br>

$$\large P_{norm}(x, y) = \frac{RAW(x, y) - BL_c}{WL - BL_c}$$

<br>

*Note:* $\large WL$ *represents the White Level (saturation point), ensuring the usable signal range is mapped to a unit scale.*

<br>

## Pipeline Priority

Black level subtraction **must be the first step** in the ISP pipeline. Because many downstream operations are multiplicative, applying them before subtraction creates non-linear artifacts.

1. ⇨ **Black Level Subtraction** (Mandatory) ⇦
2. Lens Shading Correction
3. Align and Merge
4. White Balance
5. Demosaicing

**The Multiplication Error** <br>
If White Balance ($k$) is applied before subtraction, the offset itself is scaled, introducing a permanent color cast:

<br>

$$\large P' = k \cdot (BL + I) = k \cdot BL + k \cdot I$$

<br>

## Implementation Details

For an RGGB Bayer CFA, black levels are rarely uniform across the sensor. They are typically stored as a $2 \times 2$ matrix and applied based on the pixel's $(x, y)$ coordinate:

<br>

$$\large BL = \begin{bmatrix} BL_R & BL_{G_r} \\\\ BL_{G_b} & BL_B \end{bmatrix}$$

<br>

**Data Sourcing** <br>
Always prioritize metadata (EXIF/DNG tags) over image-based estimation. Black levels are calibrated constants from the manufacturer, not variables derived from the scene's minimum pixel value.

<br>

## Technical Clarifications

**Scene Minimums** <br>
Never use the image's minimum value as the black level. This value is influenced by noise and scene content; the black level is a hardware constant.

**Negative Values** <br>
Do not clip negative values that appear after subtraction. Noise is zero-mean; these negative fluctuations are essential for maintaining statistical accuracy during the **Merge** and **Denoise** stages.

**Color Neutrality** <br>
Applying white balance before subtraction is a common error. Since channels have different gains, scaling the offset will shift the "dark point" of the image toward a specific color.

**Global vs. Per-Channel** <br>
Modern sensors utilize separate amplifiers for different color channels. Treating the black level as a single global value usually results in a faint green or magenta tint in the shadows.

<br>

---

### Summary

* Subtraction is the absolute first operation.
* Values are pulled from DNG/EXIF metadata.
* Subtraction is performed per-channel ($R, G_r, G_b, B$).
* Negative values are preserved for downstream merging.
