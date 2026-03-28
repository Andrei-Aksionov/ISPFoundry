# Align and Merge (Burst Photography)

## What is Align and Merge?

**Align and Merge** is a computational technique used to combine a sequence of images—called a **burst**—into a single, high-quality photograph. Modern sensors, especially those in smartphones, are physically small and limited in how much light they can capture in a single shot. Burst photography overcomes these hardware constraints by using software to "stack" multiple samples of the same scene.

The process consists of two primary theoretical challenges:

1. **Alignment**: Calculating the precise movement between frames so that every object in the scene sits at the exact same coordinate across the entire burst.
2. **Merging**: Combining these aligned pixels while ignoring "outliers" (like a moving car or a walking person) to prevent visual artifacts.

---

## Why Align and Merge?

### 1. Improving the Signal-to-Noise Ratio (SNR)

**Signal-to-Noise Ratio (SNR)** is a measure used to describe how much "clean" image information (Signal) there is compared to the grainy "interference" (Noise). In low-light conditions, noise usually overwhelms the signal.

By averaging $N$ independent shots, the signal stays the same, but the random noise partially cancels itself out. Mathematically, the noise is reduced by a factor of $\sqrt{N}$.

**The "9x Sensor" Analogy**: If you merge **9 frames**, your noise is reduced by $\sqrt{9} = 3$. A sensor that is physically **9 times larger** in surface area also collects 9 times more photons. Since SNR also scales with the square root of the number of photons, that giant sensor would have the same "cleanliness" (3x better SNR) as your small sensor after merging 9 frames. In effect, burst photography allows software to simulate a much larger, more expensive sensor.

### 2. Extending Dynamic Range

A single exposure often forces a choice: capture detail in the dark shadows (making the sky "blow out" to pure white) or preserve the bright sky (making the shadows pitch black).

* **Zero-shutter-lag (All Short)**: Capturing many identical short, underexposed frames. Highlights are never clipped, and the merging process cleans up the noisy shadows.
* **Hybrid Burst (Short + Long)**: Some pipelines capture a burst of short frames followed by one "long" exposure. The short frames provide the highlight detail and alignment data, while the long frame provides a high-quality "anchor" for the shadows.

### 3. "Lucky" Imaging

In handheld photography, your hand is constantly shaking. In a burst of 10 frames, 1 or 2 will likely be captured during a moment of relative stability. The pipeline identifies this "sharpest" frame and uses it as the **Reference Frame** to which all others are aligned.

---

## Pipeline Order

The Align and Merge step must be carefully placed:

1. **Black Level Subtraction (BLS)**: Mandatory. Alignment depends on comparing pixel values. If there is a "fake" brightness offset from the sensor, it biases the math.
2. **Lens Shading Correction (LSC)**: Highly Recommended. LSC removes the "vignetting" (dark corners). Without it, a tile moving from the dark corner to the bright center would look "different" to the computer, even if the scene didn't change, causing alignment to fail.
3. ⭐ **Align and Merge**
4. **White Balance (WB)**: Performed after merging.

### Why before White Balance?

White Balance applies different multipliers to the Red, Green, and Blue channels. If you White Balance first, you change the noise characteristics and the relative "weight" of the channels. Merging in the **Linear RAW** space (before WB) ensures that the noise follows a predictable physical model, which makes the **Ghost Rejection** math much more accurate.

## The Core Mechanisms

### Why Tile-Based Alignment?

We do not align the "whole image" at once because real-world motion is **non-rigid**. If you move your phone, different parts of the image shift differently due to:

* **Parallax**: Objects close to the lens move faster than objects far away.
* **Scene Motion**: A tree swaying in the wind moves independently of the background.
* **Rolling Shutter**: CMOS sensors capture the image row-by-row; if the camera moves during capture, the image actually "bends" or "skews."

**Tile-based alignment** divides the image into a grid (e.g., $16 \times 16$ pixel blocks). By finding a unique motion vector for every tile, we can "warp" the frames to account for complex distortions that a single global shift could never fix.

### Speed: The Hierarchical Search

Finding where a $16 \times 16$ tile moved in a 12-megapixel image is computationally expensive. If a person moved 100 pixels, you'd have to check thousands of possibilities. **Hierarchical (Coarse-to-Fine) Search** solves this:

1. **Coarse Level**: Downsample the image (e.g., to 1/4 or 1/16 size). A 100-pixel move becomes a tiny 6-pixel move. We find the "rough" location quickly.
2. **Fine Level**: Use the rough location as a starting point on the full-resolution image and search only a tiny surrounding area (e.g., $\pm 1$ or $\pm 2$ pixels) to find the exact match.
This reduces the mathematical complexity from an "impossible" amount of work to something a smartphone can do in milliseconds.

### Precision: Sub-pixel Refinement

Pixels are discrete blocks, but light and motion are continuous. An object might move **1.4 pixels** to the right. If we only align to the nearest whole pixel (1 or 2), we introduce a "misalignment error." When we average these slightly-off frames, the result looks blurry or "soft."

**Theory of Implementation**: We calculate the alignment error (usually **SAD - Sum of Absolute Differences**) for the best pixel and its immediate neighbors. By fitting a **Quadratic Curve** (a parabola) to these error values, we can mathematically find the "bottom of the valley"—the true sub-pixel peak—where the error is at its absolute minimum.

## Mathematical Details of Align and Merge

This section provides the formal mathematical framework for the burst merging process.

---

### 1. Signal-to-Noise Ratio (SNR) and Dynamic Range

In a digital image, any pixel value $I$ is composed of the true scene signal $S$ and random noise $N$:
$$I = S + N$$

#### SNR Improvement

When we average $M$ frames, we assume the signal $S$ is constant (perfectly aligned), while the noise $N$ is random with a mean of zero and a standard deviation of $\sigma$.

* **Merged Signal**: The average of $M$ identical signals remains $S$.
* **Merged Noise**: According to the **Central Limit Theorem**, when adding independent random variables, their variances add. The new noise $\sigma_{merged}$ is:
$$\sigma_{merged} = \frac{1}{N} \sqrt{\sum_{i=1}^{N} \sigma^2} = \frac{\sqrt{N \cdot \sigma^2}}{N} = \frac{\sigma}{\sqrt{N}}$$

**Explanation**: Dividing the single-frame noise by $\sqrt{M}$ significantly cleans the image. For example, merging **9 frames** reduces noise by a factor of **3**.

#### Dynamic Range (DR) Extension

Dynamic Range is the ratio between the brightest signal the sensor can record ($S_{max}$) and the lowest signal distinguishable from noise (the **noise floor**, $\sigma$). It is measured in decibels (dB):
$$DR = 20 \log_{10} \left( \frac{S_{max}}{\sigma} \right)$$

**Explanation**: By merging frames, we reduce the noise floor $\sigma$ to $\sigma_{merged}$. Because the denominator becomes smaller, the total range increases. This allows us to "see" further into the dark shadows where the signal was previously buried under grain, effectively widening the gap between the darkest and brightest parts of the image.

---

### 2. Alignment: Integer and Sub-pixel "Flow"

Alignment is the process of finding a motion vector (or "flow") that maps a target frame $T$ onto a reference frame $R$.

#### Integer Alignment (SAD)

To find the best whole-pixel movement, we calculate the **Sum of Absolute Differences (SAD)** for a given tile. We test various offsets $(\Delta x, \Delta y)$ and look for the minimum value:
$$SAD(\Delta x, \Delta y) = \sum_{i,j \in \text{Tile}} |R(i, j) - T(i + \Delta x, j + \Delta y)|$$

* **$R(i, j)$**: Pixel value in the reference frame.
* **$T(i + \Delta x, j + \Delta y)$**: Corresponding pixel in the target frame, shifted by the test offset.
* **Minimum SAD**: The $(\Delta x, \Delta y)$ that results in the lowest sum is considered the best integer alignment.

#### Sub-pixel Refinement (Flow Estimation)

To find motion at a precision smaller than one pixel, we treat the SAD scores of the best integer match and its 8 neighbors as a 3D surface. We fit a **2D Quadratic Function** to these points:
$$f(x, y) = Ax^2 + By^2 + Cxy + Dx + Ey + F$$

**Explanation of the Variables**:

* **$Ax^2, By^2, Cxy$**: These terms define the curvature (the shape of the "bowl").
* **$Dx, Ey$**: These terms define the slope.
* **$F$**: The vertical offset.

By taking the partial derivatives $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$ and setting them to zero, we solve for the exact coordinates $(x, y)$ of the "bottom of the bowl." This gives us the **sub-pixel flow vector**, allowing the pipeline to align frames with a precision of 0.1 pixels or better.

---

### 3. Robust Merging (Ghost Rejection)

To combine frames without creating "ghosts" of moving objects, we use a weighted average. The merged pixel value $I_{merged}$ is:
$$I_{merged} = \frac{\sum_{i=1}^{M} w_i \cdot I_i}{\sum_{i=1}^{M} w_i}$$

#### The Weighting Formula

The weight $w$ for a specific pixel or tile is determined by its similarity to the reference frame, normalized by the expected noise variance $\sigma_{noise}^2$:
$$w = \exp \left( -\frac{D(R, T)}{k \cdot \sigma_{noise}^2} \right)$$

**Explanation of the Formula**:

* **$D(R, T)$**: The "Distance" or difference (often the SAD score) between the reference and the target.
* **$\sigma_{noise}^2$**: The expected noise variance. If the difference $D$ is within this range, it’s just noise—keep the weight high.
* **$k$**: A "sensitivity" constant. A larger $k$ makes the algorithm more "forgiving" of motion, while a smaller $k$ is more aggressive at rejecting potential ghosts.
* **$\exp(-\dots)$**: This creates an exponential drop-off. If an object moved (creating a huge $D$), the weight $w$ drops toward zero almost instantly, excluding that moving object from the final average.

---

## Common Misconceptions

### "More frames always result in a better image."

**Incorrect.** There is a sharp point of diminishing returns. Moving from 1 to 4 frames gives a massive boost ($\sqrt{4}=2\times$ noise reduction). However, to double that quality again, you would need 16 frames. Eventually, the massive battery drain and processing time outweigh the tiny visual gains. Furthermore, a longer burst increases the risk of "subject motion" (like a person blinking or a car moving) that the software might not be able to perfectly repair.

### "Align and Merge is only useful for low-light photography."

**Incorrect.** While the noise reduction is most obvious in the dark, merging frames in broad daylight is a "secret weapon" for **image precision**.

In a single shot, your sensor captures data in discrete 10-bit or 12-bit integer steps. By averaging multiple frames, the math "fills in the gaps" between those steps with fractional values. This effectively increases the **bit-depth** (e.g., creating a 14-bit or 16-bit equivalent result). This provides much smoother color gradients and significantly more "headroom" for aggressive editing or tone mapping—preventing the image from "breaking" or showing ugly color banding in the sky.

### "Simple averaging is the best way to merge."

**Incorrect.** If you simply average 10 frames and a bird flies through the frame in just one of them, you will end up with a "ghost bird" (a 10% transparent artifact). A sophisticated ISP uses **Robust Weights** to realize that the pixels containing the bird don't match the "consensus" of the other frames. It then excludes those specific pixels from the average, ensuring the final image is clean and artifact-free.

---

## Summary

Align and Merge is the "brain" of the ISP. It:

* Reduces **Signal-to-Noise Ratio (SNR)** by a factor of $\sqrt{N}$.
* Uses **Tile-Based Hierarchical Search** to handle complex, non-rigid motion quickly.
* Employs **Sub-pixel Refinement** to maintain sharpness.
* Acts as a "software-defined sensor" to overcome the physical limits of small lenses.
