import cv2
import numpy as np

# Path to the video file
video_path = 'C:\\Users\\landgrafn\\Desktop\\2024-09-18-18-12-18_378_CA1.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Initialize variables to store the maximum projection
max_projection = None

# Iterate through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break when there are no more frames

    # Convert frame to grayscale (optional, if working with grayscale images)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize the max_projection array with the first frame
    if max_projection is None:
        max_projection = np.zeros_like(frame, dtype=np.float32)

    # Compute the maximum projection
    max_projection = np.maximum(max_projection, frame.astype(np.float32))

# After the loop, we have the maximum projection image
max_projection = max_projection.astype(np.uint8)

# Save the resulting max projection image
cv2.imwrite('max_projection_image.png', max_projection)

# Release the video capture object
cap.release()

# Optionally, display the max projection image
cv2.imshow('Max Projection', max_projection)
cv2.waitKey(0)
cv2.destroyAllWindows()
