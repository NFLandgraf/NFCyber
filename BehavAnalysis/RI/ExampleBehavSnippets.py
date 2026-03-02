#%%
'''
Goes through the csvs and creates a short video bit of the middle of every behavioral bout
'''
from __future__ import annotations
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
import pandas as pd

files_csv = Path(r"E:\Solomon")
files_video = Path(r"E:\Solomon")
folder_out = Path(r"E:\out")
video_ext = [".mp4", ".avi"]
behaviors_all = ['Investigate', 'Pursuit', 'nose2nose', 'Anogenital_Sniff', 'Following',
                 'Circle', 'Chase', 'Tail_Rattle', 'Mounting', 'Agitated',
                 'Attack']
x_frames = 30
fps = 30


def read_csv(csv_path, na_token='_NaN_'):
    
    # reads a 1-column CSV. Works whether there's a header or not, returns a Series of strings (one per frame).
    df = pd.read_csv(csv_path)#, header=None)
    behavs = df.iloc[:, 1].astype(str)

    behavs = behavs.fillna(na_token)
    behavs = behavs.astype(str).str.strip()

    return behavs

def check_behavior_labels(labels):
   
    # Remove potential whitespace issues
    labels_clean = labels.astype(str).str.strip()
    unique_labels = sorted(labels_clean.unique())

    not_in_list = [u for u in unique_labels if u not in behaviors_all]
    if len(not_in_list) > 1:
        print("\nWARNING: These labels are NOT in behaviors_all:")
        print(not_in_list)
   
def find_behavbouts(labels, behavior, min_len=1):
    
    # Find contiguous runs (start,end) where labels == behavior, start/end are inclusive frame indices.
    is_behav = (labels.values == behavior)
    bouts = []
    i = 0
    n = len(is_behav)
    while i < n:
        if not is_behav[i]:
            i += 1
            continue
        start = i
        while i < n and is_behav[i]:
            i += 1
        end = i - 1
        if (end - start + 1) >= min_len:
            bouts.append((start, end))

    print(f'{behavior}: {len(bouts)} bouts')
    return bouts

def find_video(csv_path, video_dir):
    stem = csv_path.stem
    stem = stem.replace('_551Behav_done', '')
    for ext in video_ext:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # fallback: find any file with same stem
    matches = [p for p in video_dir.iterdir() if p.is_file() and p.stem == stem]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No matching video found for {csv_path.name} in {video_dir}")

def get_nframes(video_path):
    """
    Try to get total number of frames; may return None if unavailable.
    """
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames",
        "-of", "json",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        info = json.loads(out)
        streams = info.get("streams", [])
        if not streams:
            return None
        s0 = streams[0]
        for key in ("nb_read_frames", "nb_frames"):
            val = s0.get(key)
            if val is None:
                continue
            try:
                return int(val)
            except Exception:
                pass
        return None
    except Exception:
        return None

def trim_with_ffmpeg(video_path, start_frame, end_frame, out_path):
    
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: conda install -c conda-forge ffmpeg")

    if end_frame < start_frame:
        raise ValueError(f"end_frame ({end_frame}) < start_frame ({start_frame})")

    # Select frames by index n; reset timestamps so the output starts at t=0
    vf = f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=N/({fps})/TB"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-an",
        "-vsync", "0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


folder_out.mkdir(parents=True, exist_ok=True)
files_csv = sorted(files_csv.glob("*.csv"))

summary_rows = []
for file_csv in files_csv:
    print(f'\n--- {file_csv.stem} ---')

    behav_labels = read_csv(file_csv)
    check_behavior_labels(behav_labels)
    file_video = find_video(file_csv, files_video)
    n_video_frames = get_nframes(file_video)

    for behav in behaviors_all:
        bouts = find_behavbouts(behav_labels, behav)

        for k, (start, end) in enumerate(bouts, start=1):
            if k >= 5:
                break

            midframe = (start + end) // 2
            win_start = max(0, midframe - x_frames)
            win_end = min(n_video_frames-1, midframe + x_frames)

            # create videos
            file_out = Path(f'{folder_out}\\{behav}_{file_csv.stem}_{midframe}.mp4')
            if shutil.which("ffmpeg") is not None:
                trim_with_ffmpeg(file_video, win_start, win_end, file_out)

#%%
'''
now create a video with 6 exemplatory videos playing simultaneousely
'''
from pathlib import Path
import subprocess
import shutil

files_video = Path(r"E:\Behavior_ExampleVideos\TailRattle")
file_out = Path(r"E:\Behavior_ExampleVideos\TailRattle.mp4")
tile_width, tile_height = 800, 470

def create_2x3_video(video_paths, out_file: Path):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: conda install -c conda-forge ffmpeg")

    video_paths = list(video_paths)
    if len(video_paths) != 6:
        raise ValueError(f"Need exactly 6 videos, found {len(video_paths)}")

    cmd = ["ffmpeg", "-y"]
    for vp in video_paths:
        cmd += ["-i", str(vp)]

    # Use explicit pixel layout (no w0/h0 variables)
    # positions (x,y): (0,0) (W,0) (0,H) (W,H) (0,2H) (W,2H)
    W, H = tile_width, tile_height
    layout = f"0_0|{W}_0|0_{H}|{W}_{H}|0_{2*H}|{W}_{2*H}"

    filter_complex = (
        f"[0:v]scale={W}:{H},setsar=1[v0];"
        f"[1:v]scale={W}:{H},setsar=1[v1];"
        f"[2:v]scale={W}:{H},setsar=1[v2];"
        f"[3:v]scale={W}:{H},setsar=1[v3];"
        f"[4:v]scale={W}:{H},setsar=1[v4];"
        f"[5:v]scale={W}:{H},setsar=1[v5];"
        f"[v0][v1][v2][v3][v4][v5]"
        f"xstack=inputs=6:layout={layout}:fill=black:shortest=1[v]"
    )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(out_file),
    ]

    # Print command for debugging
    print("Running command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

video_paths = sorted([p for p in files_video.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])
create_2x3_video(video_paths[:6], file_out)

