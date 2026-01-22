#%%
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# ==== USER INPUT ====
csv_path   = Path(r"D:\32_Week1_OF_final.csv")
out_video  = Path(r"D:\32_Week1_OF_final.mp4")
fps        = 30  # frames per second for the output video
point_radius = 5
# ====================

# Load CSV
df = pd.read_csv(csv_path)

# Detect bodyparts: columns like "nose_x", "nose_y"
bodyparts = sorted({
    col[:-2] for col in df.columns
    if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
})

if not bodyparts:
    raise ValueError("No bodypart_x / bodypart_y columns found!")

print("Bodyparts found:", bodyparts)

# Get coordinate ranges (for all bodyparts together)
all_x = []
all_y = []

for bp in bodyparts:
    all_x.append(df[f"{bp}_x"].values)
    all_y.append(df[f"{bp}_y"].values)

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

# Remove NaNs
mask = ~np.isnan(all_x) & ~np.isnan(all_y)
all_x = all_x[mask]
all_y = all_y[mask]

if len(all_x) == 0:
    raise ValueError("All coordinates are NaN, nothing to plot.")

min_x, max_x = all_x.min(), all_x.max()
min_y, max_y = all_y.min(), all_y.max()
print(min_x, max_x, min_y, max_y)

# Define canvas size with some margin
width  = 460
height = 460

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

n_frames = len(df)

# Colors for each bodypart (BGR)
# You can define as many as you like; they cycle if fewer than bodyparts.
colors = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 128, 255),
    (128, 255, 128),
    (255, 128, 128),
]

for t in range(n_frames):
    # White background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    for i, bp in enumerate(bodyparts):
        x = df.at[t, f"{bp}_x"]
        y = df.at[t, f"{bp}_y"]

        # Skip if NaN
        if pd.isna(x) or pd.isna(y):
            continue

        # Shift coordinates to canvas coords
        # Option A: keep same orientation as in original video
        x_canvas = int(x)
        y_canvas = int(y)


        # Option B (Cartesian y-up): flip vertically
        # y_canvas = height - 1 - (int(round(y - min_y)) + margin)

        color = colors[i % len(colors)]
        cv2.circle(frame, (x_canvas, y_canvas), point_radius, color, thickness=-1)

        # Optional: label with bodypart name
        # cv2.putText(frame, bp, (x_canvas+5, y_canvas-5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    writer.write(frame)

writer.release()
print("Saved video to:", out_video)
