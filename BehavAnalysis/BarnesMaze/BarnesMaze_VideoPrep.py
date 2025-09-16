#%%
from pathlib import Path
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


aq1 = [54, 59, 86, 59, 63, 53, 64, 70, 107, 61, 64, 44, 62, 71]
aq2 = [56, 53, 71, 80, 55, 73, 71, 57, 62, 54, 69, 62, 71, 60]
aq3 = [71, 52, 70, 62, 105, 54, 63, 90, 73, 62, 67, 61, 64, 56]
aq4 = [78, 65, 64, 63, 79, 68, 74, 69, 66, 65, 71, 68, 64, 68]
aq5 = [103, 99, 65, 95, 66, 66, 103, 63, 61, 79, 102, 104, 103, 69]
aq6 = [70, 64, 72, 64, 63, 69, 69, 70, 67, 76, 70, 62, 65, 76]
aq7 = [60, 67, 62, 69, 76, 68, 81, 75, 0, 65, 73, 67, 75, 83]
aq8 = [103, 0, 71, 113, 75, 75, 104, 68, 69, 78, 200, 122, 125, 72]
aq9 = [76, 82, 101, 87, 59, 75, 73, 76, 68, 72, 84, 87, 77, 70]
aq = [aq1, aq2, aq3, aq4, aq5, aq6, aq7, aq8, aq9]


common_name = 'Aq'
file_useless_string = ['.mp4']


def get_files(path, common_name):

        # get files
        files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
        
        print(f'{len(files)} files found')
        for file in files:
            print(file)
        print('\n')

        return files

def manage_filename(file):

    # managing file names
    file_name = os.path.basename(file)
    file_name_short = os.path.splitext(file_name)[0]
    for word in file_useless_string:
        file_name_short = file_name_short.replace(word, '')
    
    return file_name_short

def trim_video_cv2(file, output_file, circles, radius, alpha):
    # Open the video
    vid = cv2.VideoCapture(file)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original video has {width}x{height} frames at {fps} fps.")

    # Set up the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Precompute mask once
    #yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    #mask = (xx - center[0])**2 + (yy - center[1])**2 <= radius**2

    i = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Create masked frame (white outside the circle)
        #masked_frame = np.ones_like(frame) * 255
        #masked_frame[mask] = frame[mask]

        # several circles
        overlay = frame.copy()
        for c in circles:
            cv2.circle(overlay, c, radius, (255, 255, 255), -1)  # filled white
        masked_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Write masked frame to output
        out.write(masked_frame)

        i += 1

    vid.release()
    out.release()
    print(f"✅ Saved masked video: {output_file}, {i} frames")

# to draw the circles
radius = 29
alpha = 0.8
circles = [
    (526, 64),
    (683, 74),
    (829, 130),
    (950, 227),
    (1037, 356),
    (1082, 506),
    (1077, 664),
    (1024, 813),
    (929, 938),
    (797, 1029),
    (644, 1075),
    (482, 1070),
    (329, 1015),
    (202, 914),
    (112, 777),
    (70, 620),
    (80, 459),
    (139, 310),
    (241, 188),
    (374, 104),
]

path_in = r"D:\CA1Dopa_BarnesMaze\BarnesMaze_Mask"
path_out = "D:\\CA1Dopa_BarnesMaze\\BarnesMaze_Mask_Mask\\"
files = get_files(path_in, common_name)

for i, file in enumerate(files):
    print(file)

    file_name_short = manage_filename(file)
    file_name_short = path_out + file_name_short + '_mask.mp4'

    trim_video_cv2(file, file_name_short, circles, radius, alpha)



