#%%
import cv2
import ffmpeg
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path


path = r"Y:\_proj_Vani\2026-11-13_OF"
common_name = '.mp4'
file_format = '.mp4'



def get_data(path, common_name):

    # get all file names in directory into list
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]

    print(f'{len(files)} files found in path directory {path}\n'
        f'{files}\n')

    # get video properties of all videos in list
    for file in files:
        vid = cv2.VideoCapture(file)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        duration = nframes / fps

        print(f'{file}\n'
                f'dimensions: {width} x {height} px\n'
                f'nframes: {nframes}\n'
                f'fps: {fps}\n'
                f'duration: {duration} s\n')
    
    return files

files = get_data(path, common_name)
      


#%%


input_video = r"C:\Users\landgrafn\Desktop\a.mp4"
output_video = r"C:\Users\landgrafn\Desktop\a_trim.mp4"

start_sec = 10
end_sec = 20

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

current = start_frame
while current < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    current += 1

cap.release()
out.release()