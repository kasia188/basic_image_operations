from pathlib import Path
from binning import binning_mean, binning_sum, binning_min, binning_max, binning_median

folder = Path("data")
img = list(folder.glob("*.png"))

"""""
binning_mean(img, box_size=2, out_folder="binned_images_mean2")
binning_mean(img, box_size=5, out_folder="binned_images_mean5")
binning_mean(img, box_size=10, out_folder="binned_images_mean10")
binning_sum(img, box_size=2, out_folder="binned_images_sum2")
binning_sum(img, box_size=5, out_folder="binned_images_sum5")
binning_sum(img, box_size=10, out_folder="binned_images_sum10")
"""""

box_sizes = [2, 5, 10]

for i in box_sizes:
    binning_mean(img, box_size=i, out_folder=f"binned_images_mean{i}")

for i in box_sizes:
    binning_sum(img, box_size=i, out_folder=f"binned_images_sum{i}")

for i in box_sizes:
    binning_min(img, box_size=i, out_folder=f"binned_images_min{i}")

for i in box_sizes:
    binning_max(img, box_size=i, out_folder=f"binned_images_max{i}")

for i in box_sizes:
    binning_median(img, box_size=i, out_folder=f"binned_images_median{i}")