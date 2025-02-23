# -*- coding: utf-8 -*-
import numpy as np

from skimage import io

from matplotlib import pyplot as plt

img = io.imread("dataset/Dataset_BUSI_with_GT/normal/normal (1).png", as_gray=True)

img2 = io.imread("dataset/Dataset_BUSI_with_GT/normal/normal (1)_mask.png", as_gray=True)

img3 = io.imread("dataset/Dataset_BUSI_with_GT/benign/benign (1).png", as_gray=True)

img4 = io.imread("dataset/Dataset_BUSI_with_GT/benign/benign (1)_mask.png", as_gray=True)

img5 = io.imread("dataset/Dataset_BUSI_with_GT/malignant/malignant (1).png", as_gray=True)

img6 = io.imread("dataset/Dataset_BUSI_with_GT/malignant/malignant (1)_mask.png", as_gray=True)

# Ensure mask images are correctly formatted
img2 = img2.astype(np.uint8) * 255 if img2.dtype == bool else img2
img4 = img4.astype(np.uint8) * 255 if img4.dtype == bool else img4
img6 = img6.astype(np.uint8) * 255 if img6.dtype == bool else img6

# Display images
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

axs[0, 0].imshow(img, cmap="gray")
axs[0, 0].set_title("Normal")

axs[0, 1].imshow(img2, cmap="gray")
axs[0, 1].set_title("Normal Mask")

axs[0, 2].imshow(img3, cmap="gray")
axs[0, 2].set_title("Benign")

axs[1, 0].imshow(img4, cmap="gray")
axs[1, 0].set_title("Benign Mask")

axs[1, 1].imshow(img5, cmap="gray")
axs[1, 1].set_title("Malignant")

axs[1, 2].imshow(img6, cmap="gray")
axs[1, 2].set_title("Malignant Mask")

plt.tight_layout()
plt.show()
