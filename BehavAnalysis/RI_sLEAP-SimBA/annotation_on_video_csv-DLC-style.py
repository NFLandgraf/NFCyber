#%%
import cv2
import pandas as pd
import numpy as np

video_path = r"E:\AGG-FF\test\2026_06-08_AGG-GRABNE_RI1_01_fps.mp4"
csv_path = r"E:\AGG-FF\test\2026_06-08_AGG-GRABNE_RI1_01_fps_sLEAP_DLC.csv"
output_path = r"E:\AGG-FF\test\2026_06-08_AGG-GRABNE_RI1_01_fps_DLC.mp4"

likelihood_threshold = 0.5
point_radius = 4
font_scale = 0.4
font_thickness = 1

# Read SLEAP-style CSV with 3 header rows
df = pd.read_csv(csv_path, header=[0, 1, 2])

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Get bodypart columns, excluding frame column
bodyparts = []
for col in df.columns:
    scorer, bodypart, coord = col
    if bodypart != "frame" and coord == "x":
        bodyparts.append(bodypart)

frame_idx = 0

while frame_idx<1000:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx >= len(df):
        break

    row = df.iloc[frame_idx]

    for bodypart in bodyparts:
        try:
            x = row[("sLEAP", bodypart, "x")]
            y = row[("sLEAP", bodypart, "y")]
            likelihood = row[("sLEAP", bodypart, "likelihood")]
        except KeyError:
            continue

        if pd.isna(x) or pd.isna(y) or pd.isna(likelihood):
            continue

        if likelihood < likelihood_threshold:
            continue

        x = int(round(x))
        y = int(round(y))
        cv2.circle(frame, (x, y), point_radius, (0, 255, 0), -1)
        

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Saved annotated video to: {output_path}")