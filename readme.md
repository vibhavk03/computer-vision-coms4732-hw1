# 📸 COMS4732 — Computer Vision 2
## Homework 2: Automatic Feature Matching Across Images

**Name:** Vibhav Kashyap  
**UNI:** vk2581  
**Instructor:** Prof. Aleksander Holynski

---

## Overview

This project implements an automatic feature matching pipeline to establish correspondences across two images and generate a panorama. The pipeline detects Harris corners, applies non-maximum suppression (NMS), extracts patch-based descriptors, performs nearest-neighbor matching with the nearest-neighbor distance ratio (NNDR) test, and uses RANSAC to estimate a robust homography for panorama stitching.

> 💡 A visual summary of all results is provided in `index.html`.

---

## Project Structure

```
CV_HW1/
├── main.ipynb              # Main Jupyter notebook (end-to-end pipeline)
├── index.html              # Results visualization (all outputs)
├── README.md               # This file
├── data/                   # Input image pairs (img1.jpg, img2.jpg)
├── outputs/                # Saved plots + panoramas
└── utils/                  # Utility functions (Harris, NMS, SSD, etc.)
```

---

## Dependencies

- Python 3.9+
- NumPy
- Matplotlib
- scikit-image
- SciPy

### Installation

```bash
pip install numpy matplotlib scikit-image scipy
```

---

## How to Run the Code

All code is contained in `main.ipynb`.

1. **Open the notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Run all cells** from top to bottom

3. Change the filename to img1 and img2 in the `data/` folder to test with different image pairs. You should also change the filename variable in the `main.ipynb` file to match the new filenames.

4. Once you run the notebook for both images sets (taj_mahal and hill), you will find the results in the `outputs/` folder. You can also open the `index.html` file to view all the results in a single page.

### What the Notebook Does

- Loads the two input images from `data/`
- Detects corners and filters them using NMS
- Extracts patch-based descriptors for each keypoint
- Matches descriptors using SSD + NNDR
- Estimates a homography using RANSAC
- Warps/blends images to create a panorama
- Saves plots and final panoramas into `outputs/`

---

## Pipeline Breakdown

### Step 1 — Harris Corner Detection
- Convert images to grayscale
- Compute Harris response map
- Extract corner candidates and discard points near borders

### Step 2 — Non-Maximum Suppression (NMS)
- Keep only local maxima in the Harris response (spatially separated peaks)
- Remove keypoints near edges to support patch extraction

### Step 3 — Feature Descriptor Extraction
- Extract a **40×40** patch around each keypoint
- Downsample to **8×8**
- Flatten and normalize (zero-mean, unit-std)

> Descriptor size:
- **Grayscale:** 8×8 = 64 dims  
- **RGB:** 8×8×3 = 192 dims

### Step 4 — Feature Matching (NNDR)
- Compute pairwise descriptor distances (SSD)
- Find 1st and 2nd nearest neighbors in image 2 for each descriptor in image 1
- Apply NNDR ratio test to keep reliable matches

### Step 5 — RANSAC Homography Estimation
- Estimate a robust homography using matched points
- Keep **inliers** (matches consistent with the homography) and reject outliers

### Step 6 — Panorama Stitching
- Warp image 2 into image 1’s coordinate frame using the homography
- Blend the overlap region (simple averaging / feathering)
- Crop black borders for a clean panorama

---

## Outputs

All outputs are saved in `outputs/` and rendered in `index.html`.

Typical outputs include:
- Raw input images
- Harris corners overlay
- NMS-filtered corners
- NNDR histogram
- Top matches visualization (green lines) + unmatched points (red dots)
- Final panorama image

---

## Notes

- Keypoint coordinates are stored in **(y, x)** format for NumPy indexing.
- Homography estimation uses point format **(x, y)**.
- NNDR threshold and RANSAC residual threshold can be tuned for match quality.

---

## Acknowledgements

I have used ChatGPT AI to debug and get some suggestions for the code I have written in the `main.ipynb` file.  
I have also used ChatGPT to create `index.html` and `README.md` files.
