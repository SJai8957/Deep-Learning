#WAP to implement Image Augmentation Techniques for application of pre processing & transformation.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random
import os

def show_image(img, title="Image"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

image_path = "airplane.jpeg"  # Replace with your image path
if not os.path.exists(image_path):
    raise FileNotFoundError("Image not found at the specified path.")

image = cv2.imread(image_path)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=45, p=0.5),
    A.RandomCrop(width=224, height=224, p=1.0),
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2)
])

# Apply augmentations
augmented = transform(image=image)
augmented_image = augmented["image"]
 
# Show original and augmented images
print("Original Image:")
show_image(image)

print("Augmented Image:")
show_image(augmented_image)
