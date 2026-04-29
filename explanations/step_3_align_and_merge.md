# Align and Merge (Burst Photography)

A computational technique used to combine a sequence of images (called a **burst**) into a single, high-quality photograph.<br> Modern sensors, especially those in smartphones, are physically small and limited in how much light they can capture in a single shot. Burst photography overcomes these hardware constraints by using software to "stack" multiple samples of the same scene.

The process consists of two primary theoretical challenges:

**Alignment** — Calculating the precise movement between frames so that every object in the scene sits at the exact same coordinate across the entire burst.

**Merging** — Combining these aligned pixels while ignoring "outliers" (like a moving car or a walking person) to prevent visual artifacts and "ghosting."

<br>

## Why Align and Merge?

**Improving the Signal-to-Noise Ratio (SNR)** <br>
SNR is the measure of "clean" image information (Signal) versus grainy interference (Noise). In low light, noise often overwhelms the signal. By averaging $N$ independent shots, the signal remains constant while the random noise partially cancels itself out.

Mathematically, the noise is reduced by a factor of $\sqrt{N}$.<br>
For example, if you merge **9 frames**, your noise is reduced by $3\times$. This allows a small mobile sensor to simulate the "cleanliness" of a sensor physically **9 times larger** in surface area.

**Extending Dynamic Range** <br>
A single exposure often forces a choice: capture detail in the shadows (blowing out the sky) or preserve the sky (crushing the shadows).

* Zero-Shutter-Lag (All Short): Capturing many identical short, underexposed frames. Highlights are never clipped, and the merging process cleans up the noisy shadows.
* Hybrid Burst (Short + Long): Capturing short frames for highlight detail and alignment, followed by one "long" exposure to serve as a high-quality anchor for the shadows.

**"Lucky" Imaging** <br>
Handheld photography involves constant camera shake. In a burst of 10 frames, 1 or 2 are likely captured during a moment of relative stability. The pipeline identifies this sharpest frame as the **Reference Frame**, using it as the coordinate master for the rest of the burst.

<br>

## Pipeline Priority

The placement of this step is critical for the success of the alignment math and the quality of the merge.

1. Black Level Subtraction (Mandatory)
2. Lens Shading Correction (Highly recommended)
3. ⇨ **Align and Merge** ⇦
4. White Balance
5. Demosaicing

**Why After BLS and LSC?** <br>
Alignment depends on comparing pixel values across frames. If there is a "fake" brightness offset (Black Level) or dark corners (Lens Shading), the math becomes biased. For example, a tile moving from a dark corner to the bright center would appear to "change" to the computer, even if the scene stayed the same, causing alignment to fail.

**Why Before White Balance?** <br>
White Balance applies different multipliers to the R, G, and B channels. If you WB first, you change the noise characteristics and relative "weight" of the channels. Merging in **Linear RAW** ensures that the noise follows a predictable physical model, which makes the **Ghost Rejection** math significantly more accurate.

<br>

## Core Mechanisms: Alignment

**Tile-Based Strategy** <br>
We do not align the "whole image" at once because real-world motion is **non-rigid**. Parallax (objects at different depths moving at different speeds), scene motion (a swaying tree), and rolling shutter skew mean that different parts of the image shift independently. Dividing the image into a grid (e.g., $16 \times 16$ blocks) allows the ISP to "warp" the frames locally to account for these complex distortions.

**Hierarchical (Coarse-to-Fine) Search** <br>
Searching for a 100-pixel move in a high-res image is computationally impossible for a phone. We use a **Hierarchical Search**:

1. **Coarse Level:** Downsample the image to 1/16 size. A huge 100-pixel move becomes a tiny 6-pixel move, which is found instantly.
2. **Fine Level:** Use that "rough" location as a starting point on the full-resolution image, searching only a tiny surrounding area ($\pm 1$ or $\pm 2$ pixels) to find the exact match.
This provides a massive search window with minimal battery drain.

**Sub-pixel Refinement** <br>
Pixels are discrete blocks, but light and motion are continuous. If an object moves **1.4 pixels**, aligning it to the nearest whole pixel (1 or 2) introduces a "misalignment error" that makes the final merge look blurry. We fit a **Quadratic Curve** to the alignment error scores to mathematically find the "bottom of the valley"—the true sub-pixel peak where error is absolute minimum.

<br>

## Mathematical Foundation

The efficacy of the Align and Merge process is governed by the statistical properties of sensor noise and the geometric precision of the alignment.

### 1. SNR and Dynamic Range

In a digital image, any pixel value $I$ is composed of the true scene signal $S$ and random noise $n$. When we average $N$ perfectly aligned frames, we assume the signal $S$ is constant, while the noise $n$ is an independent random variable with a mean of zero and a standard deviation of $\sigma$.

**SNR Improvement**<br>
When we average $M$ frames, we assume the signal $S$ is constant (perfectly aligned), while the noise $N$ is random with a mean of zero and a standard deviation of $\sigma$.

* Merged Signal: The average of $M$ identical signals remains $S$.
* Merged Noise: According to the *Central Limit Theorem*, when adding independent random variables, their variances add. The noise in the merged result ($\sigma_{merged}$) is reduced by the square root of the number of frames:

$$\large \sigma_{merged} = \frac{1}{N} \sqrt{\sum_{i=1}^{N} \sigma^2} = \frac{\sqrt{N \cdot \sigma^2}}{N} = \frac{\sigma}{\sqrt{N}}$$

**Dynamic Range (DR) Extension**<br>
Dynamic Range is the ratio between the brightest signal the sensor can record ($S_{max}$) and the noise floor ($\sigma$). By reducing the noise floor to $\sigma_{merged}$, we widen the gap between the darkest and brightest parts of the image:

$$\large DR = 20 \log_{10} \left( \frac{S_{max}}{\sigma_{merged}} \right)$$

By merging frames, we reduce the noise floor $\sigma$ to $\sigma_{merged}$. Because the denominator becomes smaller, the total range increases. This allows us to "see" further into the dark shadows where the signal was previously buried under grain, effectively widening the gap between the darkest and brightest parts of the image.

### 2. Alignment: Integer and Sub-pixel Flow

Alignment maps a target frame $T$ onto a reference frame $R$ by finding a motion vector $(\Delta x, \Delta y)$ that minimizes the difference between them.

**Integer Alignment (SAD)**<br>
To find the best whole-pixel movement, we calculate the *Sum of Absolute Differences (SAD)* for a given tile. We test various offsets and look for the minimum value:

$$\large SAD(\Delta x, \Delta y) = \sum_{i,j \in \text{Tile}} |R(i, j) - T(i + \Delta x, j + \Delta y)|$$

Where:

* **$R(i, j)$**: Pixel value in the reference frame.
* **$T(i + \Delta x, j + \Delta y)$**: Corresponding pixel in the target frame, shifted by the test offset.

**Sub-pixel Refinement**<br>
To find motion at a precision smaller than one pixel, we treat the SAD scores of the best integer match and its 8 neighbors as a 3D surface. We fit a **2D Quadratic Function** to these points:

$$\large f(x, y) = Ax^2 + By^2 + Cxy + Dx + Ey + F$$

Where:

* **$Ax^2, By^2, Cxy$**: These terms define the curvature (the shape of the "bowl").
* **$Dx, Ey$**: These terms define the slope.
* **$F$**: The vertical offset.

By taking the partial derivatives $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$ and setting them to zero, we solve for the exact coordinates of the "bottom of the valley." This provides a sub-pixel flow vector, maintaining sharpness during the merge.

### 3. Robust Merging (Ghost Rejection)

To combine frames without creating "ghosts" of moving objects, we use a weighted average where the weight $w$ for a specific pixel is determined by its similarity to the reference frame, normalized by the expected noise variance $\sigma_{noise}^2$:

$$\large I_{merged} = \frac{\sum_{i=1}^{N} w_i \cdot I_i}{\sum_{i=1}^{N} w_i} \quad \text{where} \quad w = \exp \left( -\frac{D(R, T)}{k \cdot \sigma_{noise}^2} \right)$$

Where:

* **$D(R, T)$**: The "Distance" or difference (often the SAD score) between the reference and the target.
* **$\sigma_{noise}^2$**: The expected noise variance. If the difference $D$ is within this range, it’s just noise—keep the weight high.
* **$k$**: A "sensitivity" constant. A larger $k$ makes the algorithm more "forgiving" of motion, while a smaller $k$ is more aggressive at rejecting potential ghosts.
* **$\exp(-\dots)$**: This creates an exponential drop-off. If an object moved (creating a huge $D$), the weight $w$ drops toward zero almost instantly, excluding that moving object from the final average.

If the difference $D$ between the reference and target is within the expected noise range, the weight remains high. If an object moved, creating a huge $D$, the exponential function drops the weight toward zero, excluding that moving object from the final average.

<br>

## Technical Clarifications

**Diminishing Returns** <br>
Moving from 1 to 4 frames gives a $2\times$ noise reduction. However, doubling that quality again requires 16 frames. A longer burst also increases the probability of unrepairable subject motion.

**Daylight Precision** <br>
Burst photography isn't just for low light. By averaging frames in daylight, the math "fills in the gaps" between integer pixel steps, effectively increasing the bit-depth of the image and preventing color banding.

**Consensus Merging** <br>
Simple averaging is inferior to robust merging. A sophisticated ISP looks for a "consensus" across frames; if a pixel is an outlier, it is rejected to ensure the result is clean and artifact-free.

<br>

---

### Summary

* Increases **SNR** and **Dynamic Range** by a factor of $\sqrt{N}$.
* Uses **Hierarchical Tile-Based Search** for speed and complex motion.
* Employs **Sub-pixel Refinement** to maintain per-pixel sharpness.
* **Must** be performed in Linear RAW space before White Balance.

<br>
