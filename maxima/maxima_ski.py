from scipy import ndimage as ndi
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import io, img_as_float
from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def peaks_skimage(config_path):
    config = load_config(config_path)

    input_folder = Path(config["INPUT_FOLDER"])
    output_folder = Path(config["OUTPUT_FOLDER_2"])
    output_folder.mkdir(parents=True, exist_ok=True)

    min_distance = config["MIN_DISTANCE"]
    filter_size = config["FILTER_SIZE"]

    image_paths = list(input_folder.glob("*.*"))

    for img_path in image_paths:
        im = img_as_float(io.imread(img_path, as_gray=True))

        image_max = ndi.maximum_filter(im, size=filter_size, mode='constant')

        coordinates = peak_local_max(im, min_distance=min_distance)

        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(im, cmap="gray")
        ax[0].axis("off")
        ax[0].set_title("Original")

        ax[1].imshow(image_max, cmap="gray")
        ax[1].axis("off")
        ax[1].set_title("Maximum filter")

        ax[2].imshow(im, cmap="gray")
        ax[2].autoscale(False)
        ax[2].plot(coordinates[:, 1], coordinates[:, 0], "r.")
        ax[2].axis("off")
        ax[2].set_title("Peak local max")

        fig.tight_layout()

        save_path = output_folder / f"{img_path.stem}_peaks.png"
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    peaks_skimage("config.yaml")