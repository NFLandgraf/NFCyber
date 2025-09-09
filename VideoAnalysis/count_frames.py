import cv2

# Replace with the path to your video file
video_path = "C:\\Users\\landgrafn\\Desktop\\checko\\FF_Test_10Hz.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

# Release the video capture object
cap.release()