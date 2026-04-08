from pathlib import Path
from blur import mean_blur, gaussian_blur

folder = Path("data")
img = list(folder.glob("*.png"))

"""
mean_blur(img, step=3, out_folder="blurred_3")
mean_blur(img, step=5, out_folder="blurred_5")
mean_blur(img, step=10, out_folder="blurred_10")
"""

gaussian_blur(img, step=3, out_folder="blurred_g_3")
gaussian_blur(img, step=5, out_folder="blurred_g_5")
gaussian_blur(img, step=10, out_folder="blurred_g_10")