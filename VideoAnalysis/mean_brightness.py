'''
Goes through all frames of an mp4 video and checks the mean brightness 

'''


import cv2
import numpy as np

def analyze_video_brightness(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    brightness_values = []
    
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness of the frame
        mean_brightness = np.mean(gray_frame)
        brightness_values.append(mean_brightness)
        
        # Display the frame, if needed
        #cv2.imshow('Frame', gray_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    return brightness_values

# Example usage:
video_path = 'C:\\Users\\landgrafn\\Desktop\\2024-09-18-18-35-37_379_LC_rate64.mp4'
brightness_values = analyze_video_brightness(video_path)
print("Mean brightness values for each frame:", brightness_values)