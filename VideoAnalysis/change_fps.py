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

for file in files:
    print(file)
    input_file = path + file
    output_file = input_file.replace(file_format, '') + '_fps' + file_format

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_file}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, codec, 30, (width, height))

    for frame_idx in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Save every second frame for 30 FPS
        if frame_idx % 2 == 0:
            out.write(frame)
        
    cap.release()
    out.release()
