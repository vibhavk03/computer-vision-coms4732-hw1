# 📸 COMS4732 — Computer Vision 2
## Homework 1: Color Image Alignment (Prokudin–Gorskii)

**Name:** Vibhav Kashyap  
**UNI:** vk2581  
**Instructor:** Prof. Aleksander Holynski

---

## Overview

This project implements automatic color image reconstruction for Prokudin–Gorskii glass plate photographs. Each input image consists of three vertically stacked grayscale exposures corresponding to the Blue, Green, and Red channels. The goal is to align these channels and produce a high-quality color image.

This project can also be accessed at my GitHub repository: https://github.com/vibhavk03/computer-vision-coms4732-hw1

### 📋 Project Parts

- **Part 1:** Single-scale L2 (SSD) alignment on low-resolution JPEG images
- **Part 2:** Multi-resolution (pyramid) alignment for high-resolution TIFF images  
- **Part 3:** Additional Prokudin–Gorskii images processed using the same pipeline

> 💡 A visual summary of all results is provided in `index.html`

---

## Project Structure

```
CV_HW1/
├── main.ipynb              # Main Jupyter notebook
├── index.html              # Results visualization
├── README.md               # This file
├── shifts.txt              # Alignment displacement records
├── data/                   # Input images
├── outputJPGs/             # Part 1 results
├── outputTIFs/             # Part 2 results
└── myOutputTIFs/           # Part 3 results
```

---

## Dependencies

- Python 3.9+
- NumPy
- Matplotlib
- scikit-image

### Installation

```bash
pip install numpy matplotlib scikit-image
```

---

## How to Run the Code

All code is contained in `main.ipynb`.

1. **Open the notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Run all cells** from top to bottom

3. **View results:** Open `index.html` in your browser

### What the Notebook Does

- Reads input images from the `data/` directory
- Performs channel splitting, alignment, and reconstruction
- Saves final aligned images into:
  - `outputJPGs/`
  - `outputTIFs/`
  - `myOutputTIFs/`

---

## Part 1 — Single-scale L2 Alignment

### 🎯 Single-scale L2 Alignment (JPEG Images)

Uses sum of squared differences (L2 distance) to align color channels via exhaustive search over a small displacement window. Applied to low-resolution JPEG images.

**Images Processed:**
- `cathedral.jpg`
- `monastery.jpg`
- `tobolsk.jpg`

Results are saved in `outputJPGs/`.

---

## Part 2 — Multi-resolution (Pyramid) Alignment

### 🔍 Multi-resolution (Pyramid) Alignment (TIFF Images)

High-resolution TIF images have large channel misalignments, making single-scale exhaustive search impractical.

**Pyramid Approach:**
1. Alignment is first performed at coarse resolution
2. Displacements are propagated and refined at higher resolutions
3. L2 distance on Sobel edge responses is used for robustness

This significantly reduces computation time while maintaining accuracy.

Results are saved in `outputTIFs/`.

---

## Part 3 — Additional Prokudin–Gorskii Images

### 📷 Additional Prokudin–Gorskii Images

Three additional images from the Prokudin–Gorskii collection were selected and processed using the same multi-resolution alignment pipeline.

Results are saved in `myOutputTIFs/`.

---

## Notes

- All shifts are reported in `(dy, dx)` format
- Green and Red channels are aligned relative to the Blue channel

---

## Acknowledgements

I have used ChatGPT AI to debug and get some suggestions for the code I have written in the main.ipynb file.<br>
I have also used ChatGPT to create index.html and readme.md files.
