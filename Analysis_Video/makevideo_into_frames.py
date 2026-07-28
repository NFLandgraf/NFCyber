import cv2
import os

# Function to extract frames from the video
def extract_frames(video_path, output_folder):
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties (e.g., frame count, frame rate)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Extracting frames from video: {video_path}")
    print(f"Total frames: {frame_count}, Frame rate: {frame_rate} FPS")

    frame_number = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video
        
        # Save each frame as a .jpg file
        frame_filename = os.path.join(output_folder, f"{frame_number}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # Print progress
        if frame_number % 100 == 0:  # Print every 100th frame
            print(f"Extracting frame {frame_number}/{frame_count}")
        
        frame_number += 1
    
    # Release the video capture object
    cap.release()
    print(f"Frames extraction completed. {frame_number} frames saved to {output_folder}")

# Define video path and output folder
video_path = "C:\\Users\\landgrafn\\Desktop\\m90\\2024-10-31-16-52-47_CA1-m90_OF_DLC_labeled.mp4"
output_folder = "C:\\Users\\landgrafn\\Desktop\\m90\\pics"  # Folder to save frames

# Call the function
extract_frames(video_path, output_folder)
