import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, warp


# -----------------------------
# 1) Build point correspondences
# -----------------------------
def matches_to_xy(coords1, coords2, matches):
    """
    coords1, coords2: 2xN arrays (ys, xs)
    matches: Kx2 array [i1, i2] where i1 indexes coords1, i2 indexes coords2
    returns:
      src (Kx2): points in img2 as (x,y)
      dst (Kx2): points in img1 as (x,y)
    """
    i1 = matches[:, 0]
    i2 = matches[:, 1]

    x1 = coords1[1, i1]; y1 = coords1[0, i1]
    x2 = coords2[1, i2]; y2 = coords2[0, i2]

    dst = np.stack([x1, y1], axis=1)  # img1
    src = np.stack([x2, y2], axis=1)  # img2
    return src, dst


# -----------------------------
# 2) Estimate homography (RANSAC)
# -----------------------------
def estimate_homography_ransac(coords1, coords2, matches, residual_threshold=3, max_trials=2000):
    src, dst = matches_to_xy(coords1, coords2, matches)

    model, inliers = ransac(
        (src, dst),
        ProjectiveTransform,
        min_samples=4,
        residual_threshold=residual_threshold,
        max_trials=max_trials
    )
    return model, inliers


# -----------------------------
# 3) Warp img2 into img1 frame on a big canvas
# -----------------------------
def make_panorama_canvas(img1, img2, H21):
    """
    H21 maps img2 -> img1 (3x3)
    returns:
      base_canvas: canvas with img1 pasted
      img2_warped: img2 warped into the same canvas
      (tx,ty): translation used
    """
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]

    tform = ProjectiveTransform(H21)

    # corners of img2 in its own coords (x,y)
    corners2 = np.array([[0, 0], [W2, 0], [W2, H2], [0, H2]], dtype=np.float64)
    warped_corners2 = tform(corners2)

    # corners of img1
    corners1 = np.array([[0, 0], [W1, 0], [W1, H1], [0, H1]], dtype=np.float64)

    all_corners = np.vstack([corners1, warped_corners2])

    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)

    tx, ty = -min_xy[0], -min_xy[1]
    out_W = int(max_xy[0] - min_xy[0])
    out_H = int(max_xy[1] - min_xy[1])

    # translation matrix
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)

    # warp img2 with composed transform: T * H21
    tform_TH = ProjectiveTransform(T @ H21)

    img2_warp = warp(
        img2,
        inverse_map=tform_TH.inverse,
        output_shape=(out_H, out_W),
        preserve_range=True
    )

    # paste img1 into base canvas
    base = np.zeros((out_H, out_W, 3), dtype=np.float32)
    base[ty:ty + H1, tx:tx + W1] = img1.astype(np.float32)

    return base, img2_warp.astype(np.float32), (tx, ty)


# -----------------------------
# 4) Simple blending (average in overlap)
# -----------------------------
def blend_average(base, warped):
    """
    base, warped: float32 HxWx3 canvases
    """
    mask1 = np.any(base > 0, axis=2)
    mask2 = np.any(warped > 0, axis=2)

    out = base.copy()

    only2 = mask2 & (~mask1)
    out[only2] = warped[only2]

    overlap = mask1 & mask2
    out[overlap] = 0.5 * base[overlap] + 0.5 * warped[overlap]

    return out


# -----------------------------
# 5) Crop black borders
# -----------------------------
def crop_nonzero(pano, tol=1e-3):
    """
    pano: float image HxWx3
    crops to bounding box where pixels are non-black
    """
    mask = np.any(pano > tol, axis=2)
    if not np.any(mask):
        return pano
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return pano[y0:y1, x0:x1]