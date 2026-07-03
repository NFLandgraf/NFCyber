#%%
import cv2
import pandas as pd
import numpy as np

csv_path = r"C:\Users\landgrafn\Desktop\a\2025-02-12_hTauxAPP1(3m)_RI3_m231_Test_edit_fps_sLEAP_DLC.csv"
video_path = r"C:\Users\landgrafn\Desktop\a\2025-02-12_hTauxAPP1(3m)_RI3_m231_Test_edit_fps_solomon.avi"
out_path = r"C:\Users\landgrafn\Desktop\a\RI_Example_sLEAP_SimBA.mp4"

x_frames = 100000000
p_thresh = 0.1
circle_radius = 4
line_thickness = 2

bodyparts = ["Nose", "Ear_left", "Ear_right", "Lat_left", "Lat_right", "Tail_base", "Tail_end", "Center"]

connections = [
    ("Nose", "Ear_left"),
    ("Nose", "Ear_right"),
    ("Ear_left", "Lat_left"),
    ("Ear_right", "Lat_right"),
    ("Lat_left", "Tail_base"),
    ("Lat_right", "Tail_base"),
    ("Tail_base", "Tail_end")
]

colors = {
    1: (0, 0, 255),   # red in BGR
    2: (255, 0, 0)    # blue in BGR
}

df = pd.read_csv(csv_path)
frame_col = df.columns[0]
df[frame_col] = df[frame_col].astype(int)
df = df.set_index(frame_col)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

x_frames = min(x_frames, total_frames)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

def get_point(row, bp, animal, p_thresh):
    x_col = f"{bp}_{animal}_x"
    y_col = f"{bp}_{animal}_y"
    p_col = f"{bp}_{animal}_p"

    if x_col not in row or y_col not in row or p_col not in row:
        return None

    x, y, p = row[x_col], row[y_col], row[p_col]

    if pd.isna(x) or pd.isna(y) or pd.isna(p):
        return None
    if p < p_thresh:
        return None
    if x == 0 and y == 0:
        return None

    return int(round(x)), int(round(y))

for frame_idx in range(x_frames):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in df.index:
        row = df.loc[frame_idx]

        if True:
            for animal in [1, 2]:
                color = colors[animal]
                points = {}

                for bp in bodyparts:
                    pt = get_point(row, bp, animal, p_thresh)
                    if pt is not None:
                        points[bp] = pt
                        cv2.circle(frame, pt, circle_radius, color, -1)

                for bp1, bp2 in connections:
                    if bp1 in points and bp2 in points:
                        cv2.line(frame, points[bp1], points[bp2], color, line_thickness)

        # ---- add labels (top-right corner) ----
        if True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text1 = "Resident"
            text2 = "Intruder"
            (w1, h1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
            margin = 10

            # positions (top-right aligned)
            x1 = width - w1 - margin
            y1 = margin + h1 + 15
            x2 = width - w2 - margin
            y2 = y1 + h2 + 5

            cv2.putText(frame, text1, (x1, y1), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            cv2.putText(frame, text2, (x2, y2), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

        # ---- behavior labels (top-left corner) ----
        if True:
            behaviors = [
                "Approach",
                "Investigate",
                "Following",
                "AnogenitalSniff",
                "Nose2nose",
                "Agitated",
                "Mounting",
                "Circle",
                "Chase",
                "Attack"
            ]

            behavior_colors = {
                "Approach": (0, 255, 0),
                "Investigate": (0, 255, 0),
                "Following": (0, 255, 0),
                "AnogenitalSniff": (0, 255, 0),
                "Nose2nose": (0, 255, 0),
                "Mounting": (0, 165, 255),
                "Agitated": (0, 165, 255),
                "Chase": (0, 165, 255),
                "Circle": (0, 165, 255),
                "Attack": (0, 0, 255)
            }

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            margin = 10
            line_gap = 4
            y = margin + 18
            for behavior in behaviors:
                color = (0, 0, 0)
                if behavior in row.index:
                    value = row[behavior]
                    if pd.notna(value) and int(value) == 1:
                        color = behavior_colors[behavior]
                cv2.putText(
                    frame,
                    behavior,
                    (margin, y),
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA)
                y += 18 + line_gap


    writer.write(frame)

cap.release()
writer.release()

print(f"Saved: {out_path}")

#%%
# input_csv

csv_path = r"C:\Users\landgrafn\Desktop\SimBA\Test\project_folder\csv\input_csv\2025-02-12_hTauxAPP1(3m)_RI3_m222_Test_edit_fps_sLEAP_DLC.csv"

df = pd.read_csv(csv_path, header=None, skiprows=3)

# First column is frame index; remaining columns are x/y/p triplets.
df = df.rename(columns={0: "frame"})
df["frame"] = df["frame"].astype(int)
df = df.set_index("frame")

# Assign readable column names:
# animal 1 = first 8 bodyparts x/y/p = 24 columns
# animal 2 = next 8 bodyparts x/y/p = 24 columns
new_cols = []
for animal in [1, 2]:
    for bp in bodyparts:
        new_cols += [f"{bp}_{animal}_x", f"{bp}_{animal}_y", f"{bp}_{animal}_p"]

df.columns = new_cols[:len(df.columns)]

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

x_frames = min(x_frames, total_frames)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

def get_point(row, bp, animal, p_thresh):
    x_col = f"{bp}_{animal}_x"
    y_col = f"{bp}_{animal}_y"
    p_col = f"{bp}_{animal}_p"

    if x_col not in row or y_col not in row or p_col not in row:
        return None

    x, y, p = row[x_col], row[y_col], row[p_col]

    if pd.isna(x) or pd.isna(y) or pd.isna(p):
        return None
    if p < p_thresh:
        return None
    if x == 0 and y == 0:
        return None

    return int(round(x)), int(round(y))

for frame_idx in range(x_frames):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in df.index:
        row = df.loc[frame_idx]

        for animal in [1, 2]:
            color = colors[animal]
            points = {}

            for bp in bodyparts:
                pt = get_point(row, bp, animal, p_thresh)
                if pt is not None:
                    points[bp] = pt
                    cv2.circle(frame, pt, circle_radius, color, -1)

            for bp1, bp2 in connections:
                if bp1 in points and bp2 in points:
                    cv2.line(frame, points[bp1], points[bp2], color, line_thickness)

    writer.write(frame)

cap.release()
writer.release()

print(f"Saved: {out_path}")