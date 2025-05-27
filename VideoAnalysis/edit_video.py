#%%
# IMPORT

import cv2
import ffmpeg
from tqdm import tqdm
import os
import numpy as np

path = 'D:\\AGG\\RI_Test_edit\\'
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
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        duration = nframes / fps

        print(f'{file}\n'
                f'dimensions: {width} x {height} px\n'
                f'nframes: {nframes}\n'
                f'fps: {fps}\n'
                f'duration: {duration} s\n')
    
    return files
files = get_data()
      



#%%####################################################################################################################
# CHANGE VIDEO PARAMETERS

# CROP
x1, y1 = 12, 10   # top left corner of future cropped image
x2, y2 = 595, 590    # bottom right corner of future cropped image

# BRIGHTNESS & CONTRAST
alpha = 1.5     # contrast: 1-unchanged, <1-lower contrast, >1-higher contrast
beta = 0        # brightness: brightness that is added/taken to/from every pixel (-255 to +255)

# TRIM
start_s = 0
stop_s = 609


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

def adjust_video(input_file, output_file, new_width, new_height, fps, nframes, start_frame, stop_frame):

    cap = cv2.VideoCapture(input_file)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height)) 

    #first_frame = input_file.replace(file_format, '') + '_firstFrame.png'

    for curr_frame in tqdm(range(nframes)):    # stop_frame, nframes or 1
    
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Failed to read frame {curr_frame}")
            break  

        
        #frame = bin_frame(frame, binning_factor=2)

        # BRIGHTNESS & CONTRAST
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # CROP
        #frame = cv2.resize(frame, (new_width, new_height))  # must be the same dimensions as in video = cv2.VideoWriter() 

        video.write(frame)

        # when you had range(1) to only check the first frame, write first frame. Delete it if >0
        # if curr_frame == start_frame:
        #     cv2.imwrite(first_frame, frame)
        # elif curr_frame == start_frame+1:
        #     os.remove(first_frame)

        

    cap.release()
    video.release()

    print(f'Done! input: {nframes} frames, output: {curr_frame+1} frames\n\n')

def main():
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

        adjust_video(input_file, output_file, new_width, new_height, fps, nframes, start_frame, stop_frame)


main()





#%%####################################################################################################################
# TRIM VIDEO
def trim_video():

    input_file = path + files[0]
    output_file = input_file.replace(file_format, '') + '_trim' + file_format

    # trim video according to frames
    output = ffmpeg.output(ffmpeg.input(input_file).trim(start_frame=0, end_frame=10), output_file)
    ffmpeg.run(output)

trim_video()

