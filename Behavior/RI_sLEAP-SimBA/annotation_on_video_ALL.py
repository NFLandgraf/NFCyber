#%%
import cv2
import pandas as pd
import numpy as np

'''
takes a csv from sLEAP, DLC or SimBA and draws the coordinates on the orig video
'''

csv_path = r"E:\AGG-FF\a\2026_06-10_AGG-GRABNE_RI3_86_fps_sLEAP_DLC_SimBA.csv"
video_path = r"E:\AGG-FF\a\2026_06-10_AGG-GRABNE_RI3_86_fps.mp4"
output_path = r"E:\AGG-FF\a\2026_06-10_AGG-GRABNE_RI3_86_fps_SimBA.mp4"

threshold = 0.5
dot_radius = 4
line_width = 2
max_frames = 100

bodyparts =     ["Nose", "Ear_left", "Ear_right", "Center", "Tail_base", "Tail_end", "Lat_left", "Lat_right"]
connections = [ ("Nose", "Ear_left"),
                ("Nose", "Ear_right"),
                ("Ear_left", "Lat_left"),
                ("Ear_right", "Lat_right"),
                ("Lat_left", "Tail_base"),
                ("Lat_right", "Tail_base"),
                ("Tail_base", "Tail_end")]
colors = {  "Resi": (0, 0, 255),
            "Intr": (255, 0, 0),
            "1": (0, 0, 255),
            "2": (255, 0, 0)}

def valid_point(x, y, score, threshold):
    if pd.isna(x) or pd.isna(y) or pd.isna(score):
        return False
    if score < threshold:
        return False
    if x == 0 and y == 0:
        return False
    return True

def draw_animal(frame, row, bodyparts, connections, color, threshold, dot_radius, x_fmt, y_fmt, score_fmt, label=False):

    # draw lines first
    for bp1, bp2 in connections:
        x1 = row.get(x_fmt.format(bp=bp1))
        y1 = row.get(y_fmt.format(bp=bp1))
        s1 = row.get(score_fmt.format(bp=bp1))

        x2 = row.get(x_fmt.format(bp=bp2))
        y2 = row.get(y_fmt.format(bp=bp2))
        s2 = row.get(score_fmt.format(bp=bp2))

        if not valid_point(x1, y1, s1, threshold):
            continue
        if not valid_point(x2, y2, s2, threshold):
            continue

        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))

        cv2.line(frame, pt1, pt2, color, line_width)

    # draw points after lines
    for bp in bodyparts:
        x = row.get(x_fmt.format(bp=bp))
        y = row.get(y_fmt.format(bp=bp))
        score = row.get(score_fmt.format(bp=bp))

        if not valid_point(x, y, score, threshold):
            continue

        pt = (int(round(x)), int(round(y)))
        radius = dot_radius * 2 if bp == "Center" else dot_radius

        cv2.circle(frame, pt, radius, color, -1)

        if label:
            cv2.putText(frame, bp, (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

def prepare_video_writer(video_path, output_path, codec="mp4v"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, writer


def video_from_sLEAP(csv_path, video_path, output_path):

    df = pd.read_csv(csv_path)

    for bp in bodyparts:
        df[f"{bp}.x"] = pd.to_numeric(df[f"{bp}.x"], errors="coerce")
        df[f"{bp}.y"] = pd.to_numeric(df[f"{bp}.y"], errors="coerce")
        df[f"{bp}.score"] = pd.to_numeric(df[f"{bp}.score"], errors="coerce")

    frame_groups = {k: v for k, v in df.groupby("frame_idx")}

    cap, writer = prepare_video_writer(video_path, output_path, codec="mp4v")

    frame_number = 0

    while frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in frame_groups:
            rows = frame_groups[frame_number]

            for _, row in rows.iterrows():
                track = row["track"]
                color = colors.get(track, (0, 255, 0))

                draw_animal(
                    frame=frame,
                    row=row,
                    bodyparts=bodyparts,
                    connections=connections,
                    color=color,
                    threshold=threshold,
                    dot_radius=dot_radius,
                    x_fmt="{bp}.x",
                    y_fmt="{bp}.y",
                    score_fmt="{bp}.score")

        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video to: {output_path}")

def video_from_DLC__(csv_path, video_path, output_path):

    df = pd.read_csv(csv_path, header=[0, 1, 2])
    df.columns = ["frame_idx" if col[0] == "frame" else f"{col[1]}_{col[2]}" for col in df.columns]
    animals = ["1", "2"]

    for animal in animals:
        for bp in bodyparts:
            df[f"{bp}_{animal}_x"] = pd.to_numeric(df[f"{bp}_{animal}_x"], errors="coerce")
            df[f"{bp}_{animal}_y"] = pd.to_numeric(df[f"{bp}_{animal}_y"], errors="coerce")
            df[f"{bp}_{animal}_likelihood"] = pd.to_numeric(df[f"{bp}_{animal}_likelihood"], errors="coerce")

    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    df = df.dropna(subset=["frame_idx"])
    df["frame_idx"] = df["frame_idx"].astype(int)

    frame_groups = {k: v for k, v in df.groupby("frame_idx")}

    cap, writer = prepare_video_writer(video_path, output_path, codec="mp4v")

    frame_number = 0

    while frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in frame_groups:
            row = frame_groups[frame_number].iloc[0]

            for animal in animals:
                color = colors[animal]

                draw_animal(
                    frame=frame,
                    row=row,
                    bodyparts=bodyparts,
                    connections=connections,
                    color=color,
                    threshold=threshold,
                    dot_radius=dot_radius,
                    x_fmt=f"{{bp}}_{animal}_x",
                    y_fmt=f"{{bp}}_{animal}_y",
                    score_fmt=f"{{bp}}_{animal}_likelihood")

        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video to: {output_path}")

def video_from_SimBA(csv_path, video_path, output_path):

    df = pd.read_csv(csv_path)
    df.rename(columns={df.columns[0]: "frame_idx"}, inplace=True)
    animals = ["1", "2"]

    for animal in animals:
        for bp in bodyparts:
            df[f"{bp}_{animal}_x"] = pd.to_numeric(df[f"{bp}_{animal}_x"], errors="coerce")
            df[f"{bp}_{animal}_y"] = pd.to_numeric(df[f"{bp}_{animal}_y"], errors="coerce")
            df[f"{bp}_{animal}_p"] = pd.to_numeric(df[f"{bp}_{animal}_p"], errors="coerce")

    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    df = df.dropna(subset=["frame_idx"])
    df["frame_idx"] = df["frame_idx"].astype(int)

    frame_groups = {k: v for k, v in df.groupby("frame_idx")}

    cap, writer = prepare_video_writer(video_path, output_path, codec="mp4v")

    frame_number = 0

    while frame_number < max_frames:
        print(frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in frame_groups:
            row = frame_groups[frame_number].iloc[0]

            for animal in animals:
                color = colors[animal]

                draw_animal(
                    frame=frame,
                    row=row,
                    bodyparts=bodyparts,
                    connections=connections,
                    color=color,
                    threshold=threshold,
                    dot_radius=dot_radius,
                    x_fmt=f"{{bp}}_{animal}_x",
                    y_fmt=f"{{bp}}_{animal}_y",
                    score_fmt=f"{{bp}}_{animal}_p")
                
        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video to: {output_path}")


#video_from_sLEAP(csv_path, video_path, output_path)
#video_from_DLC__(csv_path, video_path, output_path)
video_from_SimBA(csv_path, video_path, output_path)

