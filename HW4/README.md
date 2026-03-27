# 📸 COMS4732 — Computer Vision 2

## Homework 4: Neural Radiance Fields

**Name:** Vibhav Kashyap  
**UNI:** vk2581  
**Instructor:** Prof. Aleksander Holynski

---

## Overview

This homework implements a **Neural Radiance Field (NeRF)** pipeline in two stages:

1. **Part 1 — 2D Neural Field:** train a coordinate-based MLP to map 2D image coordinates to RGB values
2. **Part 2 — 3D NeRF:** train a NeRF model on a multi-view lego dataset using ray sampling, positional encoding, and volume rendering

The project also includes:

- visualizations of cameras, rays, and sampled 3D points using **viser**
- training progression figures and **PSNR** plots
- a rendered **novel-view orbit GIF** of the lego scene
- a polished summary page in `index.html` and `index.pdf`

> A visual summary of the final results is available in `index.html` and `index.pdf`.

---

## Project Structure

```text
HW4/
├── data/
│   ├── animal.jpg
│   ├── food.jpg
│   └── lego_200x200.npz
├── output/
│   ├── part1/
│   │   ├── myimg_PSNR_curve.png
│   │   ├── myimg_training_progression.png
│   │   ├── test_PSNR_curve.png
│   │   ├── test_hyperparameters_affect.png
│   │   └── test_training_progression.png
│   ├── part2/
│   │   ├── viser_visualization-1.png
│   │   ├── viser_visualization-2.png
│   │   └── viser_visualization-3.png
│   └── part3/
│       ├── best_model.pt
│       ├── lego_orbit.gif
│       ├── training loss and validation psnr.png
│       └── training_progression.png
├── src/
│   ├── nerf_part1_notebook_2d.ipynb
│   ├── nerf_part2_notebook_3d.ipynb
│   ├── model.py
│   ├── dataset_3d.py
│   ├── rendering.py
│   ├── visualize_viser.py
│   ├── vis_orbit.py
│   └── __init__.py
├── requirements.txt
├── index.html
├── index.pdf
└── README.md
```

---

## Dependencies

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Pillow
- Jupyter
- viser
- tyro
- imageio

### Environment Setup (venv)

It is recommended to create a local virtual environment before installing dependencies.

```bash
# from HW4/
python3 -m venv venv

# activate on macOS / Linux
source venv/bin/activate

# Windows PowerShell
# .\venv\Scripts\Activate.ps1
```

### Installation

```bash
pip install -r requirements.txt
```

---

## How to Run the Code

This homework is split across notebooks and helper scripts.

### Part 1 — 2D Neural Field

Open and run:

```bash
jupyter notebook src/nerf_part1_notebook_2d.ipynb
```

This notebook:

- fits an MLP from image coordinates to RGB values
- studies positional encoding and hidden width
- saves training progression figures
- plots PSNR curves for the provided and custom images

### Part 2 — 3D NeRF Training

Open and run:

```bash
jupyter notebook src/nerf_part2_notebook_3d.ipynb
```

This notebook:

- loads the multi-view lego dataset from `data/lego_200x200.npz`
- converts pixels into camera rays
- samples 3D points along rays between near/far bounds
- evaluates the NeRF model on sampled 3D positions and ray directions
- performs volume rendering to reconstruct pixel colors
- trains the network and saves validation / progression outputs

### Visualize Cameras, Rays, and Samples in 3D

```bash
python -m src.visualize_viser
```

This launches a `viser` server that displays:

- training camera frustums
- sampled rays
- sampled 3D points along those rays

### Render the Orbit GIF

```bash
python -m src.vis_orbit
```

This script:

- loads the trained checkpoint from `output/part3/best_model.pt`
- renders views for the provided test camera poses
- exports the final animation to `output/part3/lego_orbit.gif`

---

## Implementation Breakdown

### Part 1 — Fitting a Neural Field to a 2D Image

The first part treats an image as a function:

\[
f(x, y) \rightarrow (r, g, b)
\]

where the network takes 2D coordinates as input and predicts the RGB value at that location.

### Key Ideas

- represent an image continuously instead of as a fixed pixel grid
- use **positional encoding** to help the network represent high-frequency details
- compare the effect of encoding frequency and model width on reconstruction quality

### Outputs Produced

- training progression on the provided test image
- training progression on a custom image
- hyperparameter comparison plot
- PSNR curves

---

### Part 2 — Fitting a Neural Radiance Field from Multi-view Images

The second part extends the neural field idea from 2D image coordinates to a full **3D radiance field**.

A NeRF models:

\[
F(\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)
\]

where:

- \(\mathbf{x}\) is a 3D location in space
- \(\mathbf{d}\) is the viewing direction
- \(\mathbf{c}\) is the RGB color
- \(\sigma\) is the volume density

### Step 1 — Dataset Loading

Implemented in `src/dataset_3d.py`.

The loader reads:

- training images
- validation images
- camera-to-world matrices for train / val / test
- focal length from the `.npz` file

It then constructs the intrinsic matrix:

\[
K =
\begin{bmatrix}
f & 0 & c_x \\
0 & f & c_y \\
0 & 0 & 1
\end{bmatrix}
\]

### Step 2 — Pixels to Rays

Also implemented in `src/dataset_3d.py`.

For each pixel:

- convert pixel coordinates into homogeneous image coordinates
- unproject them using \(K^{-1}\)
- rotate into world coordinates using the camera pose
- normalize the direction vector

This produces:

- `rays_o`: ray origins
- `rays_d`: ray directions

The `RaysData` dataset precomputes these values and pairs them with ground-truth RGB colors.

### Step 3 — Sampling Along Rays

Implemented in `src/rendering.py`.

For each ray:

- sample uniformly between the **near** and **far** bounds
- optionally perturb samples during training
- compute 3D sample positions using

\[
\mathbf{x}(t) = \mathbf{o} + t\mathbf{d}
\]

where \(\mathbf{o}\) is the ray origin and \(\mathbf{d}\) is the ray direction.

### Step 4 — NeRF Network

Implemented in `src/model.py`.

The model contains:

- a positional encoder for 3D locations
- a positional encoder for viewing directions
- an MLP trunk with a skip connection
- a **density head** producing \(\sigma\)
- an **RGB head** producing view-dependent color

The architecture follows the standard NeRF design:

- `L_xyz = 10`
- `L_dir = 4`
- hidden width of `256`

### Step 5 — Volume Rendering

Implemented in `src/rendering.py`.

For each ray, the network predicts color and density at sampled points. These are composited using the discrete volume rendering equation:

\[
\hat{C}(\mathbf{r}) = \sum_i T_i \left(1 - e^{-\sigma_i \delta_i}\right)\mathbf{c}\_i
\]

where:

- \(T_i\) is the transmittance up to sample \(i\)
- \(\sigma_i\) is the predicted density
- \(\mathbf{c}\_i\) is the predicted color

This produces the final rendered pixel color.

### Step 6 — Training and Validation

Inside the notebook, the model is trained by:

- sampling rays from the training images
- rendering predicted colors
- comparing against ground-truth RGB values
- tracking training loss and validation PSNR over time

Saved outputs include:

- rendered training progression
- loss and validation PSNR plot
- trained model checkpoint

### Step 7 — Novel View Synthesis

Implemented in `src/vis_orbit.py`.

After training, the model is evaluated on the provided test camera poses to render unseen views of the lego object. The rendered frames are combined into:

- `output/part3/lego_orbit.gif`

---

## Main Source Files

### `src/model.py`

- `PositionalEncoding`: maps low-dimensional inputs to a higher-frequency representation using sine/cosine features
- `NeRF`: predicts RGB and density from 3D positions and ray directions

### `src/dataset_3d.py`

- `load_data`: loads images, camera poses, and focal length
- `pixel_to_camera`: unprojects homogeneous pixel coordinates
- `pixels_to_rays`: converts pixels into world-space rays
- `image_to_rays` / `images_to_rays`: batch ray generation utilities
- `RaysData`: dataset class for ray/color training samples

### `src/rendering.py`

- `sample_along_rays`: samples 3D points between near/far bounds
- `batched_T_i`: computes accumulated transmittance
- `volrend`: performs discrete volume rendering
- `predict_rgbs`: helper for rendering colors from a model

### `src/visualize_viser.py`

- visualizes camera frustums, sampled rays, and 3D sample points in an interactive `viser` scene

### `src/vis_orbit.py`

- loads the trained checkpoint
- renders novel views from test camera poses
- saves the final orbit animation

---

## Outputs

All generated figures are stored under `output/`.

### `output/part1/`

- `test_training_progression.png`
- `myimg_training_progression.png`
- `test_hyperparameters_affect.png`
- `test_PSNR_curve.png`
- `myimg_PSNR_curve.png`

### `output/part2/`

- `viser_visualization-1.png`
- `viser_visualization-2.png`
- `viser_visualization-3.png`

### `output/part3/`

- `best_model.pt`
- `training_progression.png`
- `training loss and validation psnr.png`
- `lego_orbit.gif`

---

## Notes

- The 3D dataset uses camera-to-world extrinsics stored in the `.npz` archive.
- Pixel centers are sampled using `(u + 0.5, v + 0.5)` when constructing rays.
- Training-time ray samples are perturbed within each interval, while evaluation uses deterministic sampling.
- The final orbit GIF is rendered from the provided test camera trajectory.

---

## Acknowledgements

I used ChatGPT AI to help debug parts of the implementation and to get suggestions while working on the notebooks and supporting scripts.  
I also used ChatGPT to help create `index.html` and this `README.md`.
