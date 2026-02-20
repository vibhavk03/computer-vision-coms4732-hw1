import numpy as np
from scipy.ndimage import maximum_filter
from skimage.transform import resize


def nms_local_max(hrm, window_size=11, threshold_rel=0.01):
    """
    hrm: Harris response map (HxW)
    window_size: odd integer, e.g., 7/9/11/15
    threshold_rel: keep only responses > threshold_rel * max(h)
    returns coords as 2xN (ys, xs)
    """
    assert window_size % 2 == 1, "window_size must be odd"

    # local max in WxW neighborhood
    hmax = maximum_filter(hrm, size=window_size, mode="constant")

    # maxima locations + strength threshold
    is_local_max = (hrm == hmax)
    thresh = threshold_rel * np.max(hrm)
    is_strong = (hrm > thresh)

    ys, xs = np.where(is_local_max & is_strong)
    return np.vstack([ys, xs])  # 2 x N

def discard_edges(coords, Himg, Wimg, edge=20):
    """
    coords: 2xN array of (ys, xs)
    Himg, Wimg: image dimensions
    edge: discard corners within this distance from image edges
    returns filtered coords as 2xN (ys, xs)
    """
    ys, xs = coords
    m = (ys > edge) & (ys < Himg - edge) & (xs > edge) & (xs < Wimg - edge)
    return coords[:, m]


def extract_descriptors(im, coords, patch_size=40, out_size=8):
    """
    im: HxW grayscale image
    coords: 2xN (ys, xs)
    returns:
      desc: Nx64 descriptors (flattened 8x8)
    """
    # radius for cropping
    r = patch_size // 2
    ys, xs = coords
    desc_list = []

    for y, x in zip(ys, xs):
        # 1) crop 40x40 window around (y,x)
        window = im[y - r : y + r, x - r : x + r, :]

        # skip if out-of-bounds (shouldn't happen as we discarded edges)
        if window.shape != (patch_size, patch_size, 3):
            continue

        # 2) downsample to 8x8 (apply anti_aliasing)
        small = resize(window, (out_size, out_size), anti_aliasing=True)

        # 3) bias/gain normalize: zero mean, unit std
        v = small.flatten().astype(np.float32)
        v = v - np.mean(v)
        v = v / (np.std(v) + 1e-8)

        desc_list.append(v)

    return np.vstack(desc_list) if len(desc_list) > 0 else np.zeros((0, out_size*out_size*3), dtype=np.float32)