#%%
import cv2
import ffmpeg
from tqdm import tqdm
import os
import numpy as np

file_path = 'C:\\Users\\landgrafn\\Desktop\\hTau1a\\'
common_name = ''
file_format = '.mp4'


def get_data():

    # get all file names in directory into list
    files = [file for file in os.listdir(file_path) 
                if os.path.isfile(os.path.join(file_path, file)) and
                common_name in file]
    print(f'{len(files)} files found in path directory {file_path}\n'
        f'{files}\n')

    # get video properties of all videos in list
    for file in files:
        vid = cv2.VideoCapture(file_path + file)

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

files = get_data()
      




#%%

start_list = [5,5,7,6,7,8,9,6,7,6,8,6,7,6,6,6,6,4,5,5,6]
file_list = files


fps = 58.82
wanted_frames = 35292

 

# Calculate the start and end frames based on x_frames and y_frames
def trim_video(file_path, file_name, start_sec):
    file = file_path + file_name
    print(file_name)

    vid = cv2.VideoCapture(file_path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_file = file_path + file_name.replace(file_format, '') + '_trim' + file_format
    print(f"Trimming video from {start_sec}s...")
    ffmpeg.input(file, ss=start_sec).output(output_file, vframes=wanted_frames, vsync='vfr').run()
    print(f"Trimmed video saved to {output_file}")


for i, file_name in enumerate(file_list):

    print(i)
    start_sec = start_list[i]
    trim_video(file_path, file_name, start_sec)
