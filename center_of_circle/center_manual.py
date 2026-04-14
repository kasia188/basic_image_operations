from skimage import io, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import dilation
from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def center_of_circle(config_path):
    config = load_config(config_path)

    input_folder = Path(config["INPUT_FOLDER"])
    output_folder = Path(config["OUTPUT_FOLDER"])
    output_folder.mkdir(parents=True, exist_ok=True)

    x_start = config["X_START"]
    x_end = config["X_END"]
    y_start = config["Y_START"]
    y_end = config["Y_END"]

    image_paths = list(input_folder.glob("*.*"))

    for img_path in image_paths:
        img = img_as_float(io.imread(img_path, as_gray=True))
        
        print(img.shape)
        img = img[y_start:y_end, x_start:x_end]

        sum_w = np.sum(img, axis=0)
        sum_h = np.sum(img, axis=1)

        x_center = np.argmax(sum_w)
        y_center = np.argmax(sum_h)

        print(f"Circle centre: {x_center}, {y_center}")

        fig1 = plt.figure(figsize=(12, 4))

        plt.subplot(1,2,1)
        plt.plot(sum_w)
        plt.title("Sum vertically columns")

        plt.subplot(1,2,2)
        plt.plot(sum_h)
        plt.title("Sum horizontally rows")

        save_path1 = output_folder / f"{img_path.stem}_profile.png"
        plt.savefig(save_path1)
        plt.close(fig1)


        fig2 = plt.figure()

        plt.imshow(img, cmap='gray')
        plt.scatter(x_center, y_center, c='red', s=15)
        plt.axis('off')
        plt.title("circle center")

        save_path2 = output_folder / f"{img_path.stem}_center.png"
        plt.savefig(save_path2, bbox_inches="tight")
        plt.close(fig2)


        fig3 = plt.figure()
        img_f = img.flatten()

        plt.hist(img_f, bins=256, range=(0,1))
        plt.title("Histogram")
        plt.xlabel("Pixel valuse")
        plt.ylabel("No. pixel")
        
        save_path3 = output_folder / f"{img_path.stem}_histogram.png"
        plt.savefig(save_path3, bbox_inches="tight")
        plt.close(fig3)


        fig4 = plt.figure()

        threshold = 0.59
        mask = (img >= threshold).astype(int)
        original_shape = img.shape
        mask = mask.reshape(original_shape)
        mask = dilation(mask, footprint=np.ones((3,3)))
        mask_r = mask

        plt.imshow(mask_r, cmap='gray')
        plt.axis('off')

        save_path4 = output_folder / f"{img_path.stem}_mask.png"
        plt.savefig(save_path4, bbox_inches="tight")
        plt.close(fig4)

        fig5 = plt.figure()
        y_coords, x_coords = np.where(mask_r == 1)
        x_center_mask = int(np.mean(x_coords))
        y_center_mask = int(np.mean(y_coords))

        plt.imshow(mask_r, cmap='gray')
        plt.scatter(x_center_mask, y_center_mask, color='red', s=50)
        plt.title("Centre or circle")
        plt.axis('off')

        save_path5 = output_folder / f"{img_path.stem}_center_mask.png"
        plt.savefig(save_path5, bbox_inches="tight")
        plt.close(fig5)

        

if __name__ == "__main__":
    center_of_circle("config.yaml")