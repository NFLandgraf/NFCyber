#%%
import cv2
import pandas as pd

print('go')

csv_path = r"E:\AGG-FF\test\2026_06-08_AGG-GRABNE_RI1_01_fps_sLEAP.csv"
video_path = r"E:\AGG-FF\test\2026_06-08_AGG-GRABNE_RI1_01_fps.mp4"
output_path = r"E:\AGG-FF\test\2026_06-08_AGG-GRABNE_RI1_01_fps_sLEAP.mp4"

score_threshold = 0.0
dot_radius = 4

colors = {"Resi": (0, 0, 255), "Intr": (255, 0, 0)}
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

df = pd.read_csv(csv_path)

for bp in bodyparts:
    df[f"{bp}.x"] = pd.to_numeric(df[f"{bp}.x"], errors="coerce")
    df[f"{bp}.y"] = pd.to_numeric(df[f"{bp}.y"], errors="coerce")
    df[f"{bp}.score"] = pd.to_numeric(df[f"{bp}.score"], errors="coerce")

frame_groups = {k: v for k, v in df.groupby("frame_idx")}

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number in frame_groups:
        rows = frame_groups[frame_number]

        for _, row in rows.iterrows():
            track = row["track"]
            color = colors.get(track, (0, 255, 0))

            # draw lines first
            for bp1, bp2 in connections:
                x1, y1, s1 = row.get(f"{bp1}.x"), row.get(f"{bp1}.y"), row.get(f"{bp1}.score")
                x2, y2, s2 = row.get(f"{bp2}.x"), row.get(f"{bp2}.y"), row.get(f"{bp2}.score")

                if pd.isna(x1) or pd.isna(y1) or pd.isna(s1):
                    continue
                if pd.isna(x2) or pd.isna(y2) or pd.isna(s2):
                    continue
                if s1 < score_threshold or s2 < score_threshold:
                    continue

                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                cv2.line(frame, pt1, pt2, color, 2)

            # draw points after lines
            for bp in bodyparts:
                x = row[f"{bp}.x"]
                y = row[f"{bp}.y"]
                score = row[f"{bp}.score"]

                if pd.isna(x) or pd.isna(y) or pd.isna(score):
                    continue
                if score < score_threshold:
                    continue

                radius = dot_radius * 2 if bp == "Center" else dot_radius
                cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1)

    writer.write(frame)
    frame_number += 1

cap.release()
writer.release()
#cv2.destroyAllWindows()

print(f"Saved annotated video to: {output_path}")