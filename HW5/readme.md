# COMS4732 - Computer Vision 2

## Homework 5: Diffusion Models and Flow Matching

**Name:** Vibhav Kashyap  
**UNI:** vk2581  
**Instructor:** Prof. Aleksander Holynski

---

## Overview

This homework is split into two Colab notebooks:

1. **HW5A - The Power of Diffusion Models:** experiments with pretrained diffusion models, denoising, classifier-free guidance, image-to-image translation, inpainting, visual anagrams, and hybrid images.
2. **HW5B - Flow Matching from Scratch:** implements and trains UNet-based denoising and flow matching models on MNIST, including time-conditioned and class-conditioned variants.

The project also includes saved result figures under `output/`, a combined visual summary in `index.html` and the submitted report in `report.pdf`.

> This code was run on Google Colab. To run it locally, you may need to make changes to package installation cells, Google Drive mounting, `google.colab` imports, file paths, Hugging Face authentication, and GPU/device setup.

---

## Project Structure

```text
HW5/
├── CV2_HW_5A_colab.ipynb
├── CV2_HW_5B_colab.ipynb
├── data/
│   ├── cowboy.jpg
│   ├── hybrid-theory.jpg
│   └── london-eye.JPG
├── output/
│   ├── partA/
│   │   ├── 1.5_image_stack.png
│   │   ├── 1.6_image_stack.png
│   │   ├── 1.7_test_img.png
│   │   ├── 1.7_my_img_1.png
│   │   ├── 1.7_my_img_2.png
│   │   ├── 1.7.1_web_img.png
│   │   ├── 1.7.1_my_img_1.png
│   │   ├── 1.7.1_my_img_2.png
│   │   ├── 1.7.2_campanile.png
│   │   ├── 1.7.2_campanile_inpainted.png
│   │   ├── 1.7.2_road.png
│   │   ├── 1.7.2_road_inpainted.png
│   │   ├── 1.7.2_venice.png
│   │   ├── 1.7.2_venice_inpainted.png
│   │   ├── 1.7.3_campanile.png
│   │   ├── 1.7.3_my_img_1.png
│   │   ├── 1.7.3_my_img_2.png
│   │   ├── 1.8_img_1.png
│   │   ├── 1.8_img_2.png
│   │   ├── 1.9_img_1.png
│   │   ├── 1.9_img_2.png
│   │   └── 2_roaree_skull.png
│   └── partB/
│       ├── 1.2_noising_process.png
│       ├── 1.2.1_denoising_epoch_1.png
│       ├── 1.2.1_denoising_epoch_5.png
│       ├── 1.2.1_training_loss_curve.png
│       ├── 1.2.2_ood_testing.png
│       ├── 1.2.3_pure_noise_epoch_1.png
│       ├── 1.2.3_pure_noise_epoch_5.png
│       ├── 1.2.3_training_loss_pure_noise.png
│       ├── 2.2_fm_training_loss.png
│       ├── 2.3_fm_samples_epoch_1.png
│       ├── 2.3_fm_samples_epoch_5.png
│       ├── 2.3_fm_samples_epoch_10.png
│       ├── 2.5_class_fm_training_loss.png
│       ├── 2.6_class_fm_samples_epoch_1.png
│       ├── 2.6_class_fm_samples_epoch_5.png
│       └── 2.6_class_fm_samples_epoch_10.png
├── index.html
├── report.pdf
└── readme.md
```

---

## Dependencies

The notebooks install or import the main packages they need. The key dependencies are:

- Python 3.10+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- tqdm
- einops
- mediapy
- diffusers
- transformers
- accelerate
- safetensors
- sentencepiece
- bitsandbytes
- huggingface_hub

For HW5A, access to the DeepFloyd / Hugging Face model weights may require Hugging Face authentication. For both notebooks, a GPU runtime is strongly recommended.

---

## How to Run the Code

The intended way to run this homework is in Google Colab.

### Part A - Diffusion Models

Open and run:

```text
CV2_HW_5A_colab.ipynb
```

This notebook:

- loads pretrained DeepFloyd diffusion models
- implements the forward noising process
- compares classical Gaussian denoising with diffusion-based denoising
- implements one-step and iterative denoising
- samples images from noise
- uses classifier-free guidance
- performs SDEdit-style image-to-image translation
- performs inpainting with masks
- creates text-conditioned edits
- creates visual anagrams and hybrid images
- saves selected results to `output/partA/`

### Part B - Flow Matching from Scratch

Open and run:

```text
CV2_HW_5B_colab.ipynb
```

This notebook:

- builds UNet components from scratch
- trains a single-step denoising UNet on MNIST
- evaluates denoising at out-of-distribution noise levels
- studies denoising from pure noise
- implements a time-conditioned UNet for flow matching
- samples MNIST digits by integrating the learned velocity field
- implements a class-conditioned UNet
- applies classifier-free guidance for class-conditional generation
- saves selected results to `output/partB/`

### Running Locally

This code was run on Colab, so local execution may require edits before it works end-to-end. In particular:

- replace notebook cells that use `!pip install` if using a local environment
- remove or rewrite `from google.colab import drive/files` cells
- update checkpoint and output paths that assume Google Drive
- configure Hugging Face login locally for DeepFloyd model access
- ensure CUDA/MPS/CPU device selection works for your machine
- install the same package versions used in the notebook setup cells

There is no separate `requirements.txt` for HW5, so the notebook setup cells are the source of truth for package versions.

---

## Implementation Breakdown

### Part A - Diffusion Models

The first part uses a pretrained diffusion model to explore the reverse denoising process. Starting from a clean image, the notebook adds noise according to scheduler coefficients, then uses the pretrained UNet to estimate and remove noise.

Main implemented pieces:

- forward diffusion / noising process
- one-step denoising from noisy images
- iterative DDPM-style denoising with learned variance
- classifier-free guidance using conditional and unconditional noise predictions
- SDEdit-style image-to-image translation by adding noise and projecting back to the image manifold
- inpainting by preserving unmasked regions during the denoising loop
- visual anagrams by combining noise predictions under image transformations
- hybrid images by combining low-pass and high-pass components from different prompt-conditioned predictions

### Part B - Flow Matching

The second part trains generative models on MNIST from scratch. It begins with a UNet denoiser, then extends the architecture so the model can learn a time-dependent velocity field for flow matching.

Main implemented pieces:

- convolutional UNet blocks with downsampling, upsampling, flattening, and unflattening
- unconditional UNet for denoising noisy MNIST images
- time-conditioned UNet for predicting flow fields
- forward process using linear interpolation between noise and data
- reverse sampling with Euler integration
- class-conditioned UNet using class embeddings
- classifier-free guidance for class-conditioned MNIST sampling

---

## Outputs

All generated figures are stored under `output/`.

### `output/partA/`

Contains diffusion model outputs for:

- image sampling
- classifier-free guidance
- image-to-image translation
- web and hand-drawn image edits
- inpainting
- text-conditioned edits
- visual anagrams
- hybrid images
- optional Roaree image translation

### `output/partB/`

Contains flow matching and denoising outputs for:

- MNIST noising process
- denoising training progress
- denoising loss curves
- out-of-distribution denoising tests
- pure-noise denoising
- time-conditioned flow matching loss and samples
- class-conditioned flow matching loss and samples

---

## Reports and Pages

- `index.html`: combined visual summary for HW5A and HW5B
- `report.pdf`: submitted homework report

---

## Notes

- The HW5A notebook uses pretrained diffusion models and is memory intensive.
- The HW5B notebook trains neural networks on MNIST and is faster, but still benefits from a GPU.
- Some outputs are stochastic, so exact images may differ across runs unless the same random seed and environment are used.
- Some cells download images or model weights from the internet, which requires network access.

---

## Acknowledgements

I used ChatGPT AI to help debug parts of the implementation and to get suggestions while working on the notebooks.  
I also used ChatGPT to help create `index.html`, and this `README.md`.
