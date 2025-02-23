#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from skimage import io
from glob import glob
from sklearn.metrics import jaccard_score  # Alternative metric (IoU)
import matplotlib.pyplot as plt

# Function to load images and masks
def load_images_masks(image_dir):
    """Load images and masks from a directory, ensuring they are paired correctly."""
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))  # Load images
    mask_paths = sorted(glob(os.path.join(image_dir, "*_mask.png")))  # Load masks
    
    images = [io.imread(img_path, as_gray=True) for img_path in image_paths]
    masks = [io.imread(mask_path, as_gray=True) for mask_path in mask_paths]
    
    # Convert boolean masks to uint8 (0 and 255)
    masks = [(mask > 0).astype(np.uint8) * 255 for mask in masks]
    
    return images, masks

# Load datasets
normal_images, normal_masks = load_images_masks("dataset/Dataset_BUSI_with_GT/normal")
benign_images, benign_masks = load_images_masks("dataset/Dataset_BUSI_with_GT/benign")
malignant_images, malignant_masks = load_images_masks("dataset/Dataset_BUSI_with_GT/malignant")

def dice_coefficient(mask1, mask2):
    """
    Compute Dice coefficient between two binary masks.
    mask1 and mask2 should have values of 0 and 255.
    """
    mask1 = (mask1 > 0).astype(np.uint8)  # Convert to binary (0 and 1)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    
    return (2. * intersection) / union if union > 0 else 1.0  # Avoid division by zero

# Compute Dice scores for each class
normal_dice_scores = [dice_coefficient(mask, mask) for mask in normal_masks]
benign_dice_scores = [dice_coefficient(mask, mask) for mask in benign_masks]
malignant_dice_scores = [dice_coefficient(mask, mask) for mask in malignant_masks]

# Display average Dice scores per class
print(f"Average Dice Coefficient for Normal: {np.mean(normal_dice_scores):.4f}")
print(f"Average Dice Coefficient for Benign: {np.mean(benign_dice_scores):.4f}")
print(f"Average Dice Coefficient for Malignant: {np.mean(malignant_dice_scores):.4f}")

def show_sample(images, masks, title="Sample"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(images[0], cmap="gray")
    axes[0].set_title(f"{title} Image")
    
    axes[1].imshow(masks[0], cmap="gray")
    axes[1].set_title(f"{title} Mask")
    
    plt.show()

# Show samples
show_sample(normal_images, normal_masks, "Normal")
show_sample(benign_images, benign_masks, "Benign")
show_sample(malignant_images, malignant_masks, "Malignant")



