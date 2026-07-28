#%%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def mask_image_with_circles(image_file, circles, alpha=0.5):
    """
    Apply semi-transparent white circles on an image and show it with plt.

    Parameters:
    - image_file: path to the image (jpg, png, etc.)
    - circles: list of dicts [{'center': (x, y), 'radius': r}, ...]
    - alpha: transparency of white circles (0=transparent, 1=solid)
    """
    # Read image (RGB)
    img = mpimg.imread(image_file)
    
    # If image is in float [0,1], convert to 0-255
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    overlay = img.copy()

    # Draw each circle on the overlay
    for c in circles:
        cv2.circle(overlay, c['center'], c['radius'], (255, 255, 255), -1)  # filled white

    # Blend overlay with original image
    masked_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Show the result
    plt.imshow(masked_img)
    plt.axis('off')
    plt.show()



image_file = "C:\Users\landgrafn\Desktop\vlcsnap-2025-09-15-16h49m05s273.png"

# Define circles
circles = [
    {'center': (320, 240), 'radius': 100},
    {'center': (500, 400), 'radius': 150},
]

mask_image_with_circles(image_file, circles)

