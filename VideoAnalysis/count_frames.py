#%%
import cv2

video_path = r"C:\Users\landgrafn\Downloads\timed.mp4"

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

cap.release()