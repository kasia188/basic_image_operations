from skimage.io import imread, imsave
import numpy as np
from pathlib import Path


def binning_mean(img_paths, box_size, out_folder="binned_images"):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    binned_list = []
    for idx, path in enumerate(img_paths):
        img = imread(path)
        h, w, c = img.shape

        #slicing of img
        img = img[:h - h % box_size, :w - w % box_size, :] 
        h, w, _ = img.shape

        binned = img.reshape(h // box_size, box_size, w // box_size, box_size, c).mean(axis=(1,3))

        binned = binned.astype(img.dtype)
        binned_list.append(binned)

        out_path = out_folder / f"binned_{idx}.png"
        imsave(out_path, binned)

    return binned_list

def binning_sum(img_paths, box_size, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    binned_list = []
    for idx, path in enumerate(img_paths):
        img = imread(path)
        h, w, c = img.shape

        img = img[:h - h % box_size, :w - w % box_size, :]
        h, w, _ = img.shape

        binned = img.reshape(h // box_size, box_size, w // box_size, box_size, c).sum(axis=(1, 3))

        binned = binned.astype(img.dtype)
        binned_list.append(binned)

        out_path = out_folder / f"binned_{idx}.png"
        imsave(out_path, binned)

def binning_min(img_paths, box_size, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    binned_list = []
    for idx, path in enumerate(img_paths):
        img = imread(path)
        h, w, c = img.shape

        img = img[:h - h % box_size, :w - w % box_size, :]
        h, w, _ = img.shape

        binned = img.reshape(h // box_size, box_size, w // box_size, box_size, c).min(axis=(1, 3))

        binned = binned.astype(img.dtype)
        binned_list.append(binned)

        out_path = out_folder / f"binned_{idx}.png"
        imsave(out_path, binned)

def binning_max(img_paths, box_size, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    binned_list = []
    for idx, path in enumerate(img_paths):
        img = imread(path)
        h, w, c = img.shape

        img = img[:h - h % box_size, :w - w % box_size, :]
        h, w, _ = img.shape

        binned = img.reshape(h // box_size, box_size, w // box_size, box_size, c).max(axis=(1, 3))

        binned = binned.astype(img.dtype)
        binned_list.append(binned)

        out_path = out_folder / f"binned_{idx}.png"
        imsave(out_path, binned)

def binning_median(img_paths, box_size, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    binned_list = []
    for idx, path in enumerate(img_paths):
        img = imread(path)
        h, w, c = img.shape

        img = img[:h - h % box_size, :w - w % box_size, :]
        h, w, _ = img.shape

        binned = np.median((img.reshape(h // box_size, box_size, w // box_size, box_size, c)), axis=(1, 3))

        binned = binned.astype(img.dtype)
        binned_list.append(binned)

        out_path = out_folder / f"binned_{idx}.png"
        imsave(out_path, binned)