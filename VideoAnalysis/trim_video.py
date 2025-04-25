#%%
import cv2
import ffmpeg
from tqdm import tqdm
import os
import numpy as np

file_path = 'C:\\Users\\landgrafn\\Desktop\\ooh\\'
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



#%%

import cv2

def trim_video_cv2(file_path, frames_to_trim=128, output_file='trimmed_video_cv211.mp4'):
    # Open the video
    vid = cv2.VideoCapture(file_path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video has {n_frames} frames at {fps} fps.")

    # Set up the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    # Skip the first frames
    vid.set(cv2.CAP_PROP_POS_FRAMES, frames_to_trim)

    frame_idx = frames_to_trim
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        out.write(frame)
        frame_idx += 1

    vid.release()
    out.release()
    print(f"✅ Trimmed video saved to {output_file}")

# Example usage
trim_video_cv2(file_path+files[0], frames_to_trim=242+135)




#%%

import cv2
import numpy as np
from tqdm import tqdm

def make_video_more_transparent(input_path, output_path, alpha=0.8, background_color=(255, 255, 255)):
    """
    Makes an entire video more transparent by blending frames with a background color.
    
    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save the output video.
        alpha (float): Transparency level for the original video (0 = invisible, 1 = fully visible).
        background_color (tuple): Background color in BGR (e.g., (0, 0, 0) = black).
    """
    cap = cv2.VideoCapture(input_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Processing video: {total_frames} frames at {fps} fps")

    background = np.full((height, width, 3), background_color, dtype=np.uint8)

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        blended_frame = cv2.addWeighted(frame, alpha, background, 1 - alpha, 0)
        out.write(blended_frame)
    
    cap.release()
    out.release()
    print(f"✅ Transparent video saved to {output_path}")

# Example usage
make_video_more_transparent('C:\\Users\\landgrafn\\Desktop\\ohh\\2024-10-31-16-52-47_CA1-m90_OF_DLC_trim.mp4', 'transparent_video.mp4')


