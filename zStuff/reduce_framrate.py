#%%
import cv2
from pathlib import Path

# Input and output
input_path = Path(r"C:\Users\landgrafn\Desktop\AxonNice.mp4")
output_path = input_path.with_name(input_path.stem + "_30fps.mp4")

# Open video
cap = cv2.VideoCapture(str(input_path))

original_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Keep every 7th frame
frame_step = 6
new_fps = original_fps / frame_step

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, new_fps, (width, height))

frame_idx = 0
saved_frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % frame_step == 0:
        out.write(frame)
        saved_frames += 1

    frame_idx += 1

cap.release()
out.release()

print(f"Original FPS: {original_fps:.2f}")
print(f"New FPS: {new_fps:.2f}")
print(f"Original frames: {frame_idx}")
print(f"Saved frames: {saved_frames}")
print(f"Saved to: {output_path}")