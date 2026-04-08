import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from kernels import blurr_kernel, sharpened


folder = Path("data")
img_paths = list(folder.glob("*.png"))


for path in img_paths:
    img = imread(path)
    img = img[:, :, 0]
    img_blurred = blurr_kernel(img, 1/9)

    img_minus = img - img_blurred

    img_plus = img + img_minus

    #img_sharpen = sharpened(img)

    plt.subplot(1,4,1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')

    plt.subplot(1,4,2)
    plt.title("Img blurred")
    plt.imshow(img_blurred, cmap='gray')

    plt.subplot(1,4,3)
    plt.title("Minus")
    plt.imshow(img_minus, cmap='gray')

    plt.subplot(1,4,4)
    plt.title("Org + minus")
    plt.imshow(img_plus, cmap='gray')

    plt.show()