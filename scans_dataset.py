import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_HEIGHT = 512  # Change according to your requirement
IMG_WIDTH = 512

def load_images_and_masks(folder_path):
    images = []
    masks = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") and "_mask" not in filename:  
            # Process and load the image (excluding masks)
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img) / 255.0  
            images.append(img)

            # Find corresponding mask
            mask_filename = filename.replace(".png", "_mask.png")
            mask_path = os.path.join(folder_path, mask_filename)
            
            if os.path.exists(mask_path):  # Ensure mask exists before loading
                mask = load_img(mask_path, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
                mask = img_to_array(mask) / 255.0
                masks.append(mask)
            else:
                print(f"Warning: No mask found for {filename}")
    
    return np.array(images), np.array(masks)


class SegmentationDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.images, self.masks = load_images_and_masks(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

       # Ensure correct shape for grayscale images (H, W)
        image = image.squeeze()  # Remove extra dimension if shape is (H, W, 1)
        mask = mask.squeeze()

        # Convert to uint8 (0-255) before using PIL
        image = Image.fromarray((image * 255).astype(np.uint8), mode="L")
        mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Define transformations (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image to tensor and scales to [0,1]
])

# Example usage
dataset = SegmentationDataset("dataset/Dataset_BUSI_with_GT/normal", transform=transform)
image, mask = dataset[0]  # Get first sample

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)