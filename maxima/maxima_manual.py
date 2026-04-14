from skimage import io, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def maxima(config_path):
    config = load_config(config_path)

    input_folder = Path(config["INPUT_FOLDER"])
    output_folder = Path(config["OUTPUT_FOLDER"])
    output_folder.mkdir(parents=True, exist_ok=True)

    print("OUTPUT:", output_folder.resolve())

    window_size = config["WINDOW_SIZE"]
    
    image_paths = list(input_folder.glob("*.*"))

    for img_path in image_paths:
        img = img_as_float(io.imread(img_path, as_gray=True))

        h, w = img.shape
        mask = np.zeros_like(img)

        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = img[i:i+window_size, j:j+window_size]
                max_val = np.max(window)
                mask[i:i+window_size, j:j+window_size] = (window == max_val)
            
        coords = np.column_stack(np.where(mask))

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.scatter(coords[:, 1], coords[:, 0], c="r", s=5)
        ax.axis("off")
        ax.set_title("Peak local max")

        save_path = output_folder / f"{img_path.stem}_result.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    maxima("config.yaml")
