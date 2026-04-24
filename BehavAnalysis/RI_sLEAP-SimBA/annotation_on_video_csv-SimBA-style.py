#%%
import cv2
import pandas as pd

print('go')

csv_path = r"Y:\Neuronal Networks\SimBA\pretrained_Mouse_RI\Escape\CSDS01306.csv"
video_path = r"C:\Users\landgrafn\Desktop\SimBA\try\b\2025-02-12_hTauxAPP1(3m)_RI3_m222_Test_edit_fps_sleap.avi"
output_path = r"C:\Users\landgrafn\Desktop\SimBA\try\b\2025-02-12_hTauxAPP1(3m)_RI3_m222_Test_edit_fps_sleap_SimBA.avi"

score_threshold = 0.0
dot_radius = 4
animals = ["1", "2"]
colors = {
    "1": (0, 0, 255),   # red
    "2": (255, 0, 0)    # blue
}

bodyparts = ["Nose", "Ear_left", "Ear_right", "Center", "Tail_base", "Tail_end", "Lat_left", "Lat_right"]

connections = [
    ("Nose", "Ear_left"),
    ("Nose", "Ear_right"),
    ("Ear_left", "Lat_left"),
    ("Ear_right", "Lat_right"),
    ("Lat_left", "Tail_base"),
    ("Lat_right", "Tail_base"),
    ("Tail_base", "Tail_end"),
]

df = pd.read_csv(csv_path, header=2)
df.rename(columns={df.columns[0]: "frame_idx"}, inplace=True)


# convert to numeric
for animal in animals:
    for bp in bodyparts:
        df[f"{bp}_{animal}_x"] = pd.to_numeric(df[f"{bp}_{animal}_x"], errors="coerce")
        df[f"{bp}_{animal}_y"] = pd.to_numeric(df[f"{bp}_{animal}_y"], errors="coerce")
        df[f"{bp}_{animal}_p"] = pd.to_numeric(df[f"{bp}_{animal}_p"], errors="coerce")

df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
df = df.dropna(subset=["frame_idx"])
df["frame_idx"] = df["frame_idx"].astype(int)

frame_groups = {k: v for k, v in df.groupby("frame_idx")}

# --- video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number in frame_groups:
        row = frame_groups[frame_number].iloc[0]  # only one row per frame

        for animal in animals:
            color = colors[animal]

            # --- draw lines ---
            for bp1, bp2 in connections:
                x1 = row.get(f"{bp1}_{animal}_x")
                y1 = row.get(f"{bp1}_{animal}_y")
                s1 = row.get(f"{bp1}_{animal}_p")

                x2 = row.get(f"{bp2}_{animal}_x")
                y2 = row.get(f"{bp2}_{animal}_y")
                s2 = row.get(f"{bp2}_{animal}_p")

                if pd.isna(x1) or pd.isna(y1) or pd.isna(s1): continue
                if pd.isna(x2) or pd.isna(y2) or pd.isna(s2): continue
                if s1 < score_threshold or s2 < score_threshold: continue

                if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                    continue

                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                cv2.line(frame, pt1, pt2, color, 2)

            # --- draw points + labels ---
            for bp in bodyparts:
                x = row.get(f"{bp}_{animal}_x")
                y = row.get(f"{bp}_{animal}_y")
                score = row.get(f"{bp}_{animal}_p")

                if pd.isna(x) or pd.isna(y) or pd.isna(score): continue
                if score < score_threshold: continue
                if x == 0 and y == 0: continue

                pt = (int(round(x)), int(round(y)))

                radius = dot_radius * 2 if bp == "Center" else dot_radius
                cv2.circle(frame, pt, radius, color, -1)

                # label
                cv2.putText(
                    frame,
                    bp,
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA
                )

    writer.write(frame)
    frame_number += 1

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved annotated video to: {output_path}")