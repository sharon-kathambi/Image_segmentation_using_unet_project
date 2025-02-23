#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

# Define the image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256


def load_images_and_masks(image_folder, mask_folder):
    images = []
    masks = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            #Load and process image
            if filename.endswith(").png"):
                img = load_img(os.path.join(image_folder, filename), color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
                img = img_to_array(img) / 255.0
                
                images.append(img)
                
            if filename.endswith("_mask.png"):
                mask_path = os.path.join(mask_folder, filename)
                mask = load_img(mask_path, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
                mask = img_to_array(mask) / 255.0
                
                masks.append(mask)
            
           
    return np.array(images), np.array(masks)




def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Encoder (Contracting Path)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder (Expanding Path)
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model


# Example usage
normal_images, normal_masks = load_images_and_masks(
    "dataset/Dataset_BUSI_with_GT/normal/", 
    "dataset/Dataset_BUSI_with_GT/normal/"
)
benign_images, benign_masks = load_images_and_masks(
    "dataset/Dataset_BUSI_with_GT/benign/", 
    "dataset/Dataset_BUSI_with_GT/benign/"
)
malignant_images, malignant_masks = load_images_and_masks(
    "dataset/Dataset_BUSI_with_GT/malignant/", 
    "dataset/Dataset_BUSI_with_GT/malignant/"
)

print("Normal images shape:", normal_images.shape)
print("Normal masks shape:", normal_masks.shape)
print("Benign images shape:", benign_images.shape)
print("Benign masks shape:", benign_masks.shape)
print("Malignant images shape:", malignant_images.shape)
print("Malignant masks shape:", malignant_masks.shape)

# Compile model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



