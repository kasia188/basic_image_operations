from skimage import io, img_as_float, color
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_local
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from pathlib import Path
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def watershed_images(config_path):
    config = load_config(config_path)

    input_folder = Path(config["INPUT_FOLDER"])
    output_folder = Path(config["OUTPUT_FOLDER_2"])
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_folder.glob("*.*"))

    for img_path in image_paths:
        img_folder = output_folder / img_path.stem
        img_folder.mkdir(parents=True, exist_ok=True)

        image_rgb = io.imread(img_path)
        image_gray = img_as_float(color.rgb2gray(image_rgb))


        background = ndi.gaussian_filter(image_gray, sigma=10)
        enhanced = image_gray - background
        binary = enhanced > 0

        thresh = threshold_local(image_gray, 35)
        binary = image_gray < thresh
        binary = ndi.binary_fill_holes(binary)

        distance = ndi.distance_transform_edt(binary)

        coords = peak_local_max(
            distance,
            footprint=np.ones((3, 3)),
            labels=binary,
            min_distance=10
        )

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True

        markers, _ = ndi.label(mask)

        labels = watershed(-distance, markers, mask=binary)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax = axes.ravel()

        ax[0].imshow(image_gray, cmap="gray")
        ax[0].set_title("Grayscale")

        ax[1].imshow(binary, cmap="gray")
        ax[1].set_title("Binary mask")

        ax[2].imshow(distance, cmap="gray")
        ax[2].set_title("Distance map")

        ax[3].imshow(labels, cmap="nipy_spectral")
        ax[3].set_title("Watershed result")

        for a in ax:
            a.axis("off")

        fig.tight_layout()

        save_path = img_folder / f"{img_path.stem}_watershed.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    watershed_images("config.yaml")