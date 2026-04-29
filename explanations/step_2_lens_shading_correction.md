# Lens Shading Correction

A spatial calibration process that compensates for optical brightness and color falloff from the center of the sensor to its periphery.

In the [HDR+ dataset](https://hdrplusdata.org/dataset.html), lens shading is provided as a low-resolution metadata grid. Without this correction, the corners of the image would not only appear darker (vignetting) but would also exhibit significant color tints due to the wavelength-dependent nature of light attenuation through glass.

<br>

## The Nature of Lens Shading

Lens shading, or vignetting, is the gradual reduction in intensity and color accuracy toward the edges of the frame. It is the cumulative result of several physical and optical phenomena:

**Natural Vignetting** <br>
Derived from the $cos^4 \theta$ law, where light hitting the sensor at an angle travels a longer distance and spreads over a larger area, reducing its energy density.

**Mechanical Vignetting** <br>
Physical obstructions within the optical path, such as lens barrels or hoods, that block light rays arriving at extreme angles.

**Pixel Vignetting** <br>
The Chief Ray Angle (CRA) effect, where the sensor's microlenses become less efficient at capturing photons that arrive at steep angles relative to the pixel surface.

<br>

## Visual and Analytical Impact

Correcting lens shading is not merely an aesthetic choice; it is a prerequisite for the mathematical assumptions made by the rest of the ISP.

**Uniform Illumination** <br>
LSC equalizes the signal so that a uniformly lit scene (such as a white wall) is represented by uniform pixel values across the entire frame.

**Spatially Consistent Color** <br>
Because different wavelengths attenuate at different rates, corners often exhibit greenish or magenta tints. LSC applies per-channel gains to restore color neutrality.

**Downstream Accuracy** <br>
Operations like White Balance and Color Correction (CCM) assume the sensor response is spatially uniform. Without LSC, white balance may only be accurate in the center of the image.

<br>

## Mathematical Foundation

We model the observed pixel value $P(x,y)$ as the true intensity $I(x,y)$ modulated by a spatially varying vignetting factor $V(x,y)$. To recover the true signal, we multiply by the **Lens Shading Map (LSM)**.

<br>

$$\large I(x,y) = P(x,y) \cdot LSM(x,y), \quad \text{where } LSM(x,y) = \frac{1}{V(x,y)}$$

<br>

For a Bayer sensor, this correction must be applied to all four color channels ($R, G_r, G_b, B$) independently to account for color-specific shading:

<br>

$$\large \begin{bmatrix} R' \\\\ G_r' \\\\ G_b' \\\\ B' \end{bmatrix} = \begin{bmatrix} LSM_R & 0 & 0 & 0 \\\\ 0 & LSM_{G_r} & 0 & 0 \\\\ 0 & 0 & LSM_{G_b} & 0 \\\\ 0 & 0 & 0 & LSM_B \end{bmatrix} \cdot \begin{bmatrix} R \\\\ G_r \\\\ G_b \\\\ B \end{bmatrix}$$

<br>

## Acquisition and Calibration

Obtaining an accurate Lens Shading Map (LSM) requires isolating the sensor's spatial response from the scene content.

**Factory Calibration** <br>
Sensors are pointed at a calibrated uniform light source (an integrating sphere). The gain required to "flatten" the resulting image is calculated and stored as a reference grid.

**DNG Metadata** <br>
Professional raw files often embed a "GainMap" or "OpcodeList2." These contain the specific grid needed to correct that specific lens-sensor combination at that moment of capture.

**Flat-Fielding** <br>
A manual calibration technique involving photographing a neutral gray card or a clear, uniform sky to generate a reference frame for gain calculation.

**Synthetic Modeling** <br>
When no map is available, radial polynomials ($1 + ar^2 + br^4$) are used to approximate the falloff based on the distance from the optical center.

<br>

## Implementation Details

Because vignetting is a smooth, low-frequency phenomenon, storing a full-resolution gain map is inefficient. Instead, maps are stored as low-resolution grids (e.g., $17 \times 13$).

**Upsampling** <br>
To apply the map to the raw image, the grid must be upsampled to full resolution. **Bilinear interpolation** is the standard approach, providing smooth transitions and preventing the introduction of high-frequency artifacts into the raw data.

<br>

## Pipeline Priority

Lens shading correction **must occur after black level subtraction** but before the global white balance is applied.

1. Black Level Subtraction
2. ⇨ **Lens Shading Correction** ⇦
3. Align and Merge
4. White Balance
5. Demosaicing

**The Offset Error** <br>
LSC is a multiplicative gain. If applied before subtracting the black level (an additive offset), you scale the electronic noise pedestal, creating a spatially varying black level that is nearly impossible to correct later:

<br>

$$\large P_{wrong} = LSM(x,y) \cdot (I(x,y) + BL) = LSM \cdot I + LSM \cdot BL$$

<br>

## Technical Clarifications

**Noise Amplification** <br>
LSC increases the signal in the corners, which inherently boosts the noise floor in those regions. This is why corners often appear grainier than the center in low-light shots.

**Demosaicing Order** <br>
LSC should never be performed after demosaicing. Shading is a raw Bayer-level artifact; demosaicing uncorrected data "smears" the spatial errors across color channels, leading to irreversible fringing.

**Optical Dependencies** <br>
A single map is rarely sufficient for all conditions. Shading characteristics change significantly based on the **Aperture** and, in the case of zoom lenses, the **Focal Length**.

<br>

---

### Summary

* Compensates for optical brightness and color falloff (vignetting).
* Uses a multiplicative 4-channel gain map (LSM).
* **Must** be applied after black level subtraction and before white balance.
* Requires bilinear upsampling for smooth, per-pixel application.
* Essential for maintaining spatial color consistency and uniform exposure.
