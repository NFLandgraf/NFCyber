#%%
import cv2
import os
from pathlib import Path
from tqdm import tqdm

path = 'D:\\'
file_format = '.mp4'

def get_data():
    # get all file names in directory into list
    files = [file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                file_format in file]
    print(f'{len(files)} files found in path directory {path}\n'
        f'{files}\n')
    
    return files
files = get_data()

#%%

def convert_mp4_to_avi_opencv(input_path, output_fps=30):
    # Derive output filename
    base, _ = os.path.splitext(input_path)
    output_path = base + "_solomon.avi"

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Saved AVI video ({frame_count} frames) to: {output_path}")
    return output_path


for file in files:
    print(file)
    input_file = path + file

    convert_mp4_to_avi_opencv(input_file, output_fps=30)
