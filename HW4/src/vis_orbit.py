import math
import numpy as np

import math
import imageio
import numpy as np
import torch
from pathlib import Path

from src.dataset_3d import load_data, pixels_to_rays
from src.rendering import sample_along_rays, volrend
from src.model import NeRF


def look_at_origin(pos):
    # Camera looks towards the origin
    forward = -pos / np.linalg.norm(pos)  # Normalize the direction vector

    # Define up vector (assuming y-up)
    up = np.array([0, 1, 0])

    # Compute right vector using cross product
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    # Recompute up vector to ensure orthogonality
    up = np.cross(forward, right)

    # Create the camera-to-world matrix
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos

    return c2w


def rot_x(phi):
    return np.array(
        [
            [math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), math.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


# TODO: Change start position to a good position for your scene such as
# the translation vector of one of your training camera extrinsics
START_POS = np.array([1.0, 0.0, 0.0])
NUM_SAMPLES = 60

# frames = []
# for phi in np.linspace(360.0, 0.0, NUM_SAMPLES, endpoint=False):
#     c2w = look_at_origin(START_POS)
#     extrinsic = rot_x(phi / 180.0 * np.pi) @ c2w

#     # Generate view for this camera pose
#     # TODO: Add code for generating a view with your model from the current extrinsic
#     frame = ...
#     frames.append(frame)


def render_view(model, c2w, K, H, W, near, far, num_samples, device, chunk_size=8192):
    model.eval()

    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()

    c2w = c2w.to(device)
    K = K.to(device)

    # make pixel grid
    us, vs = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    uvs = torch.stack([us.reshape(-1), vs.reshape(-1)], dim=-1)  # (H*W, 2)

    rays_o, rays_d = pixels_to_rays(
        K=K,
        c2w=c2w,
        uvs=uvs,
        device=device,
    )

    rendered_chunks = []

    with torch.no_grad():
        for start in range(0, rays_o.shape[0], chunk_size):
            end = min(start + chunk_size, rays_o.shape[0])

            ro_batch = rays_o[start:end]
            rd_batch = rays_d[start:end]

            xyzs = sample_along_rays(
                r_os=ro_batch,
                r_ds=rd_batch,
                near=near,
                far=far,
                num_samples_along_ray=num_samples,
                perturb=False,
                device=device,
            )

            rgbs, sigmas = model(xyzs, rd_batch)
            pred_rgb = volrend(
                sigmas=sigmas,
                rgbs=rgbs,
                near=near,
                far=far,
                num_samples_along_ray=num_samples,
                device=device,
            )

            rendered_chunks.append(pred_rgb)

    image = torch.cat(rendered_chunks, dim=0).reshape(H, W, 3)
    image = image.clamp(0, 1).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    return image


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = "data/lego_200x200.npz"
    ckpt_path = "output/part3/best_model.pt"
    out_gif = "output/part3/lego_orbit.gif"

    near = 2.0
    far = 6.0
    num_samples = 64

    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(
        data_path=data_path
    )

    H, W = images_train.shape[1], images_train.shape[2]

    model = NeRF(L_xyz=10, L_dir=4, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    frames = []
    for i in range(c2ws_test.shape[0]):
        frame = render_view(
            model=model,
            c2w=c2ws_test[i],
            K=K,
            H=H,
            W=W,
            near=near,
            far=far,
            num_samples=num_samples,
            device=device,
            chunk_size=8192,
        )
        frames.append(frame)
        print(f"Rendered frame {i+1}/{c2ws_test.shape[0]}")

    Path("output/part3").mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=15)
    print(f"Saved GIF to {out_gif}")


if __name__ == "__main__":
    main()
