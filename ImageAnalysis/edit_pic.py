import cv2
import numpy as np

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Adjust brightness and contrast of an image.
    
    Parameters:
    image (ndarray): The input image.
    alpha (float): Contrast control (1.0-3.0). Default is 1.0.
    beta (int): Brightness control (-100 to 100). Default is 0.
    
    Returns:
    ndarray: The resulting image after brightness and contrast adjustment.
    """
    # Apply the brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# Path to the input image
image_path = 'C:\\Users\\landgrafn\\Desktop\\CA1_maxproj.tiff'

# Read the image
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Parameters for adjusting
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 0    # Brightness control (-100 to 100)

# Adjust brightness and contrast
adjusted_image = adjust_brightness_contrast(image, alpha, beta)

# Save or display the adjusted image
output_path = 'adjusted_image.png'
cv2.imwrite(output_path, adjusted_image)

# Optionally display the original and adjusted image
cv2.imshow('Original Image', image)
cv2.imshow('Adjusted')