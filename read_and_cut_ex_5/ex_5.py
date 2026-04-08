from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt

folder = Path("data")
img_paths = list(folder.glob("*.png"))

img = imread(img_paths[0])

h, w, c = img.shape
fragment = img[
    h//4 : 3*h//4,
    w//4 : 3*w//4
]

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("Fragment")
plt.imshow(fragment, cmap='gray')

plt.show()