#%%
# IMPORT

import cv2
import ffmpeg
from tqdm import tqdm
import os
import numpy as np

path = r"C:\Users\landgrafn\Desktop\2025-03-14_hTau2(6m)_RI3"
common_name = 'mp4'
file_format = '.mp4'


def get_data():

    # get all file names in directory into list
    files = [file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    print(f'{len(files)} files found in path directory {path}\n'
        f'{files}\n')

    # get video properties of all videos in list
    for file in files:
        vid = cv2.VideoCapture(path + file)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
    
    return files
files = get_data()
      



#%%####################################################################################################################
# CHANGE VIDEO PARAMETERS

# CROP
x1, y1 = 12, 10   # top left corner of future cropped image
x2, y2 = 595, 590    # bottom right corner of future cropped image

# BRIGHTNESS & CONTRAST
alpha = 3.0     # contrast: 1-unchanged, <1-lower contrast, >1-higher contrast
beta = -10        # brightness: brightness that is added/taken to/from every pixel (-255 to +255)

# TRIM
start_frame = 1
start_s = 6
stop_s = 609



def adjust_video(input_file, output_file, new_width, new_height, fps, nframes):

    def bin_frame(frame, binning_factor):
        # bin the frame horizontally and vertically by the binning_factor
        # each binning operation sums up pixel values in non-overlapping regions

        height, width = frame.shape[:2]
        new_height = height // binning_factor
        new_width = width // binning_factor

        # Resize the frame by summing over binning_factor x binning_factor regions
        binned_frame = frame[:new_height * binning_factor, :new_width * binning_factor]
        binned_frame = binned_frame.reshape(new_height, binning_factor, new_width, binning_factor, -1)
        binned_frame = binned_frame.sum(axis=(1, 3))  # Sum over binning regions

        # Normalize pixel values for an 8-bit image
        binned_frame = (binned_frame / (binning_factor**2)).astype(frame.dtype)

        return binned_frame

    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, 30, (new_width, new_height)) 

    for curr_frame in tqdm(range(nframes)):    # stop_frame, nframes or 1
    
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {curr_frame}")
            break  
        
        # BINNING
        #frame = bin_frame(frame, binning_factor=2)

        # BRIGHTNESS & CONTRAST
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # CROP
        #frame = cv2.resize(frame, (new_width, new_height))  # must be the same dimensions as in video = cv2.VideoWriter() 
        
        if curr_frame >= 1:
            video.write(frame)

    cap.release()
    video.release()
    print(f'Done! input: {nframes} frames\n')

for file in files:

    input_file = path + file
    output_file = input_file.replace(file_format, '') + '_edit' + file_format
    print(input_file)

    # original video parameters
    vid = cv2.VideoCapture(input_file)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # trim
    start_frame, stop_frame = start_s * fps, stop_s * fps     # USE THIS WHEN TRIM
    #start_frame, stop_frame = 0, nframes                      # USE THIS WHEN NOT TRIM

    #new_width, new_height = x2-x1, y2-y1                        # USE THIS WHEN     CROP     & NOT REDUCE QUALITY
    #new_width, new_height = int((x2-x1)*2/3), int((y2-y1)*2/3)  # USE THIS WHEN     CROP     &     REDUCE QUALITY
    #new_width, new_height = int(width/2), int(height/2)         # USE THIS WHEN NOT CROP     &     REDUCE QUALITY
    new_width, new_height = width, height                        # USE THIS WHEN NOT CROP     & NOT REDUCE QUALITY

    adjust_video(input_file, output_file, new_width, new_height, fps, nframes)








#%%
# TRIM VIDEO
def trim_video(file, start_s, trim=False, change_fps=True):
    input_file = path + '//' + file
    output_file = input_file.replace(file_format, '') + '_fps' + file_format

    # original video parameters
    vid = cv2.VideoCapture(input_file)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #print(nframes)

    # trim the video
    if trim:
        stream = ffmpeg.input(input_file, ss=start_s)
        stream = ffmpeg.output(stream, output_file)
        ffmpeg.run(stream, overwrite_output=True)
    
    # change the fps
    if change_fps:
        stream = ffmpeg.input(input_file)
        stream = stream.filter('select', 'not(mod(n,2))')
        stream = stream.filter('setpts', 'N/30/TB')
        stream = ffmpeg.output(stream, output_file, r=30)
        ffmpeg.run(stream, overwrite_output=True)

for file in files:
    trim_video(file, start_s=6)

