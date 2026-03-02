# Lens Shading Correction

## What is Lens Shading?

**Lens shading** (also known as vignetting or shading rolloff) is the gradual reduction in brightness and color accuracy from the center of an image toward its edges and corners. This occurs because the lens and sensor do not capture light uniformly across the entire frame.

The effect is caused by:

* **Law (Natural Vignetting)**: Light hitting the sensor at an angle travels further and spreads over a larger area, reducing intensity.
* **Mechanical Vignetting**: Physical obstructions like lens barrels or hoods block light at extreme angles.
* **Pixel Vignetting (CRA)**: Microlenses on the sensor are less efficient at capturing light arriving at a high Chief Ray Angle (CRA).
* **Wavelength Dependence**: Different colors (R, G, B) attenuate at different rates, leading to spatially varying color casts.

---

## Why Correct Lens Shading?

### 1. Uniform Illumination

A uniformly lit scene (like a white wall) should appear uniform in the raw data. Lens Shading Correction (LSC) equalizes the signal so the corners match the center brightness.

### 2. Spatially Consistent Color

Because shading is wavelength-dependent, the corners often exhibit "color tints" (e.g., greenish or magenta corners). LSC applies per-channel gains to ensure color neutrality across the frame.

### 3. Improves Downstream Accuracy

Operations like **White Balance**, **Demosaicing**, and **Color Correction (CCM)** assume the sensor's response is spatially uniform. Without LSC:

* White balance might only be accurate in the center.
* Demosaicing can introduce color fringing in dark corners.

### 4. Maximizes Dynamic Range

By "boosting" the signal in the corners, LSC ensures that the full bit-depth of the sensor is utilized effectively across the entire image area.

---

## Pipeline Order

Lens shading correction **must occur after black level subtraction** but before white balance:

1. Black Level Subtraction (mandatory)
2. ⭐ **Lens Shading Correction**
3. White Balance
4. Demosaicing
5. ... Remaining steps

### Why After Black Level Subtraction?

LSC is a **multiplicative gain**. If applied before subtracting the black level (an additive offset), you scale the offset itself:

$$P_{wrong} = LSM(x,y) \cdot (I(x,y) + BL) = LSM(x,y) \cdot I(x,y) + LSM(x,y) \cdot BL$$

This creates a **spatially varying black level**, which is nearly impossible to remove later and causes severe color shifts in shadows.

### Why Before White Balance?

LSC corrects for **hardware/optical** non-uniformity, while White Balance corrects for **lighting** conditions. Correcting the sensor's spatial response first provides a "flat" field for the global white balance gains to work on.

---

## Mathematical Details

### The Correction Model

We model the observed pixel value $P(x,y)$ as the true intensity $I(x,y)$ multiplied by a vignetting factor $V(x,y)$:

$$P(x,y) = I(x,y) \cdot V(x,y)$$

To recover the true signal, we multiply by the **Lens Shading Map (LSM)**, which is the inverse of the vignetting:

$$I(x,y) = P(x,y) \cdot LSM(x,y), \quad \text{where } LSM(x,y) = \frac{1}{V(x,y)}$$

### Per-Channel Gain

For a Bayer sensor, the correction is applied to all four channels independently:

$$\begin{bmatrix} R' \\\\ Gr' \\\\ Gb' \\\\ B' \end{bmatrix} = \begin{bmatrix} LSM_R & 0 & 0 & 0 \\\\ 0 & LSM_{Gr} & 0 & 0 \\\\ 0 & 0 & LSM_{Gb} & 0 \\\\ 0 & 0 & 0 & LSM_B \end{bmatrix} \cdot \begin{bmatrix} R \\\\ Gr \\\\ Gb \\\\ B \end{bmatrix}$$

---

## The Lens Shading Map (LSM)

### Low-Resolution Storage

Vignetting is a low-frequency, smooth phenomenon. Therefore, LSMs are typically stored as low-resolution grids (e.g., $17 \times 13 \times 4$) to save space.

### Upsampling

To apply the map to a full-resolution image, the LSM must be **upsampled** to the image dimensions. **Bilinear interpolation** is the standard method, as it ensures smooth transitions without introducing ringing artifacts.

---

## How to Obtain LSMs

* **Calibration (Factory)**: Point the camera at a calibrated uniform light source (integrating sphere) and calculate the gain needed to make the image flat.
* **DNG Metadata**: Professional RAW files often embed the "OpcodeList2" or "GainMap" containing the correction grid.
* **Flat-Fielding**: Photographing a neutral gray card or a clear sky can provide a reference for manual calibration.
* **Synthetic Models**: Using radial polynomials ($1 + ar^2 + br^4$) to approximate the falloff when no map is available.

---

## Common Misconceptions

### "LSC can be done after demosaicing"

**Incorrect**. Shading happens at the raw Bayer level and is color-channel specific. If you demosaic first, you mix uncorrected color values, making accurate correction much harder.

### "LSC increases noise"

**True, but necessary**. Since you are multiplying (boosting) the signal in the corners, you are also boosting the noise floor. This is why corner regions often appear noisier than the center.

### "A single map works for all lenses"

**Incorrect**. Each lens has unique optical properties. Even for the same lens, shading changes based on **Aperture (f-stop)** and **Focal Length (zoom)**.

---

## Summary

Lens Shading Correction:

* Compensates for optical brightness and color falloff (vignetting).
* Uses a multiplicative 4-channel gain map (LSM).
* **Must** be applied after black level subtraction.
* Requires bilinear upsampling for smooth, per-pixel application.
* Is essential for spatial color consistency and uniform exposure.
