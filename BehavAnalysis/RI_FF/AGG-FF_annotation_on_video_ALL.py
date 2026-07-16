#%%
import cv2
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter

'''
takes a csv from sLEAP, DLC or SimBA and draws the coordinates on the orig video
'''

csv_path = r"E:\AGG-FF\_test\2026_06-08_AGG-GRABNE_RI1_01_fps_sLEAP_check_DLC_SimBA_Doric.csv"
video_path = r"E:\AGG-FF\_test\2026_06-08_AGG-GRABNE_RI1_01_fps.mp4"
output_path = r"E:\AGG-FF\_test\2026_06-08_AGG-GRABNE_RI1_01_fps_annot.mp4"

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
colors = {  "Resi": (0, 0, 255), "Intr": (255, 0, 0), "1": (0, 0, 255), "2": (255, 0, 0)}
behavs = {'Attack': ["Attack"], 
          'Threat': ["Mounting", "Chase", "Circle", "Agitated", "TailRattle"],
          'Social': ["Investigate", "Nose2nose", "Anogetnical_Sniff", "Approach", "Following"]}
behav_colors = {"Attack": (0, 0, 255), "Threat": (0, 165, 255), "Social": (0, 255, 0)}


def valid_point(x, y, score, threshold):
    if pd.isna(x) or pd.isna(y) or pd.isna(score):
        return False
    if score < threshold:
        return False
    if x == 0 and y == 0:
        return False
    return True

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

def draw_behavs(frame, row, behavs=behavs, x=20, y0=35, line_spacing=25, show_inactive=True):
    
    # draws behavior names onto the frame
    inactive_color = (180, 180, 180)
    
    i = 0
    for category, behavior_list in behavs.items():

        active_color = behav_colors[category]
    
        for behavior in behavior_list:

            value = row.get(behavior, 0)
            y = y0 + i * line_spacing
            color = active_color if value == 1 else inactive_color

            if value == 1 or show_inactive:
                cv2.putText(frame, behavior, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, behavior, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            
            i += 1

def draw_zscore(frame, df, frame_idx, fps, z_window=5):

    w=200
    h=100
    margin=20
    alpha=0.65
    z_ylim=None
    frame_h, frame_w = frame.shape[:2]

    x = frame_w - w - margin
    y = margin

    half_window_frames = int(round(z_window * fps))

    start = max(0, frame_idx - half_window_frames)
    end = min(len(df), frame_idx + half_window_frames + 1)

    z = pd.to_numeric(df["zscore"].iloc[start:end], errors="coerce").to_numpy()

    if z_ylim is None:
        z_min = np.nanmin(df["zscore"])
        z_max = np.nanmax(df["zscore"])
    else:
        z_min, z_max = z_ylim

    if z_min == z_max:
        z_min -= 1
        z_max += 1

    overlay = frame.copy()

    # semi-transparent white background
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)

    # t = 0 line in the middle
    x_mid = x + w // 2
    cv2.line(frame, (x_mid, y), (x_mid, y + h), (80, 80, 80), 1)

    # z = 0 line
    if z_min <= 0 <= z_max:
        y_zero = int(y + h - ((0 - z_min) / (z_max - z_min)) * h)
        cv2.line(frame, (x, y_zero), (x + w, y_zero), (120, 120, 120), 1)

    points = []

    for k, value in enumerate(z):
        absolute_frame = start + k
        rel_frames = absolute_frame - frame_idx

        px = int(x_mid + (rel_frames / half_window_frames) * (w / 2))

        if pd.isna(value):
            continue

        py = int(y + h - ((value - z_min) / (z_max - z_min)) * h)
        points.append((px, py))

    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(frame, p1, p2, (0, 0, 0), 1)

    current_z = pd.to_numeric(df["zscore"].iloc[frame_idx], errors="coerce")

    if not pd.isna(current_z):
        current_y = int(y + h - ((current_z - z_min) / (z_max - z_min)) * h)
        cv2.circle(frame, (x_mid, current_y), 4, (0, 0, 255), -1)


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
                
                draw_behavs(frame=frame, row=row)
                
        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video to: {output_path}")

def video_from_Doric(csv_path, video_path, output_path):
    
    df = pd.read_csv(csv_path)
    df.rename(columns={df.columns[0]: "frame_idx"}, inplace=True)
    animals = ["Resi", "Intr"]

    for animal in animals:
        for bp in bodyparts:
            df[f"{animal}_{bp}_x"] = pd.to_numeric(df[f"{animal}_{bp}_x"], errors="coerce")
            df[f"{animal}_{bp}_y"] = pd.to_numeric(df[f"{animal}_{bp}_y"], errors="coerce")

    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    df = df.dropna(subset=["frame_idx"])
    df["frame_idx"] = df["frame_idx"].astype(int)
    cap, writer = prepare_video_writer(video_path, output_path, codec="mp4v")

    frame_number = 0
    df = df.set_index("behav_idx")
    
    while frame_number < len(df):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in df.index:
            
            row = df.loc[frame_number]

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
                
            draw_behavs(frame=frame, row=row)

            draw_zscore(frame=frame, df=df, frame_idx=frame_number, fps=30)
                
        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video to: {output_path}")


#video_from_sLEAP(csv_path, video_path, output_path)
#video_from_DLC__(csv_path, video_path, output_path)
#video_from_SimBA(csv_path, video_path, output_path)
video_from_Doric(csv_path, video_path, output_path)
