#%%
import cv2
import os
from pathlib import Path
from tqdm import tqdm

path = 'D:\\AGG\\RI_Habituation_fps_resize\\'
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
    output_file = input_file.replace(file_format, '') + '_fps_resize' + file_format

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_file}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    new_fps = 10
    new_width, new_height = width//2, height//2
    frame_skip = int(fps // new_fps)

    out = cv2.VideoWriter(output_file, codec, new_fps, (new_width, new_height))

    for frame_idx in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Save every second frame for x FPS + resize
        if frame_idx % frame_skip == 0:
            frame = cv2.resize(frame, (new_width, new_height))
            out.write(frame)
        
    cap.release()
    out.release()
