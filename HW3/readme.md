# 📸 COMS4732 — Computer Vision 2
## Homework 3: Structure from Motion

**Name:** Vibhav Kashyap  
**UNI:** vk2581  
**Instructor:** Prof. Aleksander Holynski

---

## Overview

This project implements a **Structure from Motion (SfM)** pipeline on the provided image pair. The pipeline:

1) computes the camera intrinsics matrix **K**  
2) detects **SIFT** features  
3) matches features using **L2 + NNDR (Lowe ratio test)**  
4) estimates relative pose using **RANSAC + Essential matrix (8-point algorithm)**  
5) triangulates a **sparse 3D point cloud** with reprojection/depth filtering  
6) exports a `.npz` scene file for interactive visualization in **viser**

> 💡 A visual summary of all results is provided in `index.html` and can viewed from webpage.pdf as well.

---

## Project Structure

## Project Structure

```
HW3/
├── data/
│   ├── img1.jpeg
│   └── img2.jpeg
├── outputs/
│   └── <run_folder>/
│       ├── config.yml
│       ├── launch_viser.txt
│       ├── step0_original_images.png
│       ├── step2_sift_keypoints.png
│       ├── step3_correspondences_before_ransac.png
│       ├── step3_feature_matching_nndr_histogram.png
│       ├── step3_feature_matching_top5_nndr.png
│       ├── step3_feature_matching.png
│       ├── step4_correspondences_after_ransac.png
│       ├── step4_feature_matching_after_ransac.png
│       ├── step4_epipolar_lines.png
│       ├── step4_ransac_convergence.png
│       ├── step4_pose_results.txt
│       ├── step5_3d_reconstruction.png
│       ├── step5_camera_poses.png
│       ├── step5_pipeline_grid.png
│       ├── step5_pose_summary.png
│       └── step5_scene_data.npz
├── intrinsics.py
├── features.py
├── ransac.py
├── triangulation.py
├── utils.py
├── utils_visualizations.py
├── visualize_viser.py
├── main.py
├── requirements.txt
├── index.html
└── README.md
```

---

## Dependencies

- Python **3.11**
- NumPy
- SciPy
- Matplotlib
- scikit-image
- tqdm
- PyYAML  
- (and other dependencies listed in `requirements.txt`)

### Environment Setup (venv)

It is recommended to create a local Python virtual environment before installing dependencies.

```bash
# from the project root
python3.11 -m venv venv

# activate (macOS / Linux)
source venv/bin/activate

# (Windows PowerShell)
# .\venv\Scripts\Activate.ps1
```

### Installation

```bash
pip install -r requirements.txt
```

---

## How to Run the Code

All code is contained in `main.py`.

1. **Run the script:**
   ```bash
   python main.py
   ```

2. Once you run the script, you will find the results in the `outputs/` folder. You can also open the `index.html` file to view all the results in a single page. Though you will have to update the paths in `index.html` to point to the correct output folder.

---

## Pipeline Breakdown (High-Level)

### Step 1 — Camera Intrinsics
Compute the intrinsic matrix:

\[
K=\begin{bmatrix}
f_{px} & 0 & c_x \\
0 & f_{px} & c_y \\
0 & 0 & 1
\end{bmatrix}
\]

where \( f_{px} = f_{mm} \cdot \frac{W_{px}}{w_{mm}} \), and \( (c_x, c_y) \) is the image center.

### Step 2 — SIFT Feature Detection
Detect SIFT keypoints and descriptors in both images (rotation + scale invariant).  
Save a side-by-side keypoints visualization.

### Step 3 — Feature Matching (NNDR)
Match descriptors with L2 distance and apply the NNDR ratio test to keep reliable correspondences.  
Save: NNDR histogram, top-5 matches, and correspondences before RANSAC.

### Step 4 — RANSAC Pose Estimation (Essential Matrix)
Estimate the Essential matrix \(E\) using the 8-point algorithm inside RANSAC.  
Score hypotheses with squared Sampson distance; recover pose \(R, t\) via cheirality.  
Save: correspondences after RANSAC + RANSAC convergence plot.

### Step 5 — Triangulation + Filtering
Triangulate inlier correspondences to form a sparse point cloud and filter points using:
- reprojection error threshold  
- positive depth constraints (cheirality)

---

## Acknowledgements

I have used ChatGPT AI to debug and get some suggestions for the code I have written in the `main.py` and `ransac.py` file.  
I have also used ChatGPT to create `index.html` and `README.md` files.