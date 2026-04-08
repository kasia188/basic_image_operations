import numpy as np
from skimage.io import imread, imsave
from skimage.filters import gaussian
from pathlib import Path

def mean_blur(img_path, step, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    blurred_list = []
    for idx, path in enumerate(img_path):
        img = imread(path)
        h, w, c = img.shape
        blurred = np.zeros_like(img, dtype=float)

        for i in range(h):
            for j in range(w):
                i_min = max(0, i - step//2)
                i_max = min(h, i + step//2 + 1)
                j_min = max(0, j - step//2)
                j_max = min(w, j + step//2 + 1)

                window = img[i_min:i_max, j_min:j_max, :]
                blurred[i, j, :] = window.mean(axis=(0,1))
  
        blurred = blurred.astype(img.dtype)
        blurred_list.append(blurred)

        out_path = out_folder / f"blurred_{idx}.png"
        imsave(out_path, blurred)

    return blurred_list

def gaussian_blur(img_path, step, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    blurred_list = []
    for idx, path in enumerate(img_path):
        img = imread(path)
        h, w, c = img.shape
        blurred = np.zeros_like(img, dtype=float)

        for i in range(h):
            for j in range(w):
                i_min = max(0, i - step//2)
                i_max = min(h, i + step//2 + 1)
                j_min = max(0, j - step//2)
                j_max = min(w, j + step//2 + 1)

                window = img[i_min:i_max, j_min:j_max, :]
                blurred = gaussian(window, sigma=2, channel_axis=-1)
  
        blurred = blurred.astype(img.dtype)
        blurred_list.append(blurred)

        out_path = out_folder / f"blurred_{idx}.png"
        imsave(out_path, blurred)

    return blurred_list