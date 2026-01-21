<p>
    <h1 align="center"><i>ISP Foundry</i></h1>
    <h3 align="center">Image Signal Processing</h3>
    <h6 align="center">From scratch</h6>
</p>

**ISPFoundry** is a clear and structured walkthrough of the Image Signal Processing (ISP) pipeline — from raw sensor data to final RGB images.

This project breaks down the ISP process into distinct, understandable steps. It starts with traditional image processing algorithms (like bilinear demosaicing and white balance) and gradually explores machine learning–based replacements for individual blocks or combined stages.

## Goals

- Provide a clean, step-by-step implementation of a standard ISP pipeline
- Help students, researchers, and developers understand how raw sensor data is processed
- Explore how machine learning can replace or enhance traditional ISP blocks

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package and project management.

- **Install uv**

    On MacOS, you can install it via Homebrew:

    ```bash
    brew install uv
    ```

- **Synchronize dependencies**

    This command will create a virtual environment and install all required packages:

    ```bash
    uv sync
    ```

## Pipeline Stages

**Implemented components:**

- [ ] *Black level subtraction*
- [ ] *Lens shading correction*
- [ ] *Merging burst of images*
- [ ] *White balancing*
- [ ] *Demosaicing*
- [ ] *Color correction*
- [ ] *Local tone mapping*
- [ ] *Global tone mapping*
- [ ] *3d Look-up table (LUT)*
- [ ] *Sharpening*

**Documentation:**

- [ ] Readme for each step
- [ ] Main readme

**Planned extensions:**

- ML-based demosaicing (CNN-based)
- Joint demosaicing + color correction with neural networks
- End-to-end ML pipelines (e.g., CNN or Transformer)

## Resources

- Nice YouTube video from TED explaining how cameras in smartphones capture and process images: <https://www.youtube.com/watch?v=7dm2AsJ3-E8>
