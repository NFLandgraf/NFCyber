#%%
# IMPORT
import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path


def get_files(path):
    # get all file names in directory into list
    files = [file for file in Path(path).iterdir() if file.is_file() and '.mp4' in file.name]
    
    print(f'{len(files)} videos found:')
    for file in files: 
        print(file)
    
    return files

def get_xmax_ymax(file):
    # get video properties
    vid = cv2.VideoCapture(str(file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return width, height

path = 'C:\\Users\\landgrafn\\NFCyber\\YLine\\data\\'
video_file_format = '.mp4'

files = get_files(path)
width, height = get_xmax_ymax(files[0])

def adapt(x, y):
    # add dx/dy and multiplay with dz
    adapt_coords = ((x + dx) * dz, (y + dy) * dz)

    # check if poits are out of the image
    if 0 > adapt_coords[0] or adapt_coords[0] > width:
        print('point exceeds width')
    elif 0 > adapt_coords[1] or adapt_coords[1] > height:
        print('point exceeds height')

    return adapt_coords



#%%
# USER INPUT
dx = -8      # if you want to shift the whole thing horizontally 
dy = 0      # if you want to shift the whole thing vertically
dz = 1      # if you want to change the size of the whole thing
# width = maximum(x-axis), height = maximum(y-axis)

# corners
left_corner = adapt(313, 250)
middle_corner = adapt(352, 179)
right_corner = adapt(392, 248)

# left arm
left_arm_end_lefter = adapt(8, 70)
left_arm_end_righter = adapt(45, 0)

# right arm
right_arm_end_righter = adapt(700, 70)
right_arm_end_lefter = adapt(660, 0)

# middle_arm
middle_arm_end_lefter = adapt(313, 600)
middle_arm_end_righter = adapt(392, 600)



# FUNCTIONS
def print_info(file):
    # get video properties
    vid = cv2.VideoCapture(str(file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    duration = nframes / fps

    print(f'{file}\n'
        f'copy the following lines:\n\n'
        f'width = {width}\n'
        f'height = {height}\n\n'
        f'#left arm\n'
        f'left_corner = {left_corner}\n'
        f'middle_corner = {middle_corner}\n'
        f'left_arm_end_lefter = {left_arm_end_lefter}\n'
        f'left_arm_end_righter = {left_arm_end_righter}\n\n'
        f'#middle arm\n'
        f'right_corner = {right_corner}\n'
        f'middle_arm_end_righter = {middle_arm_end_righter}\n'
        f'middle_arm_end_lefter = {middle_arm_end_lefter}\n\n'
        f'#right arm\n'
        f'right_arm_end_righter = {right_arm_end_righter}\n'
        f'right_arm_end_lefter = {right_arm_end_lefter}\n\n'
        f'fps = {fps}\n'
        f'nframes = {nframes}\n'
        f'duration = {duration}\n')

def get_areas():
    # define all necessary areas
    left_arm = Polygon([left_corner, middle_corner, left_arm_end_righter, left_arm_end_lefter])
    right_arm = Polygon([right_corner, middle_corner, right_arm_end_lefter, right_arm_end_righter])
    middle_arm = Polygon([left_corner, right_corner, middle_arm_end_righter, middle_arm_end_lefter])
    center = Polygon([middle_corner, left_corner, right_corner])
    areas = [center, left_arm, middle_arm, right_arm]

    whole_area = Polygon([middle_corner, right_arm_end_lefter, right_arm_end_righter, 
                        right_corner, middle_arm_end_righter, middle_arm_end_lefter, 
                        left_corner, left_arm_end_lefter, left_arm_end_righter])
    whole_area = np.array(list(whole_area.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))

    return areas, whole_area

def write_and_draw(file, areas):

    # write the first frame
    cap = cv2.VideoCapture(str(file))
    first_frame = 'first_frame.png'

    ret, frame = cap.read()
    cv2.imwrite(first_frame, frame)

    frame = image.imread(first_frame)

    for arm in areas:
            x, y = arm.exterior.xy
            plt.plot(x, y, alpha=0)
            plt.fill_between(x, y, alpha=0.5)

    plt.imshow(frame)

file = files[0]
print_info(file)
areas, whole_area = get_areas()
write_and_draw(file, areas)



#%%HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# CROP & MASK
'''''
1. Crop the videos into new dimensions so that the arena corners limit the frame:
    x[left_arm_end_lefter:left_arm_end_righter]
    y[min:max of any y maze point]

2. Fill the area outside of the arena with white jibber

'''''

def adjust_video(input_file, output_file):

    # original stuff
    vid = cv2.VideoCapture(str(input_file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f'Origin: {width}x{height}px, {nframes} frames, {fps} fps')


    # future dimensions after cropping
    x1 = left_arm_end_lefter[0]
    y1 = min(middle_arm_end_lefter[1], middle_arm_end_righter[1], left_arm_end_righter[1], right_arm_end_lefter[1])
    x2 = right_arm_end_righter[0]
    y2 = max(middle_arm_end_lefter[1], middle_arm_end_righter[1], left_arm_end_righter[1], right_arm_end_lefter[1])
    new_width, new_height = x2-x1, y2-y1
    print(f'Wanted: {new_width}x{new_height}px, {nframes} frames, {fps} fps')
    

    # create new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_file), fourcc, fps, (new_width, new_height)) 

    for curr_frame in tqdm(range(nframes)):
        ret, frame = vid.read()
        if ret:    

            # white mask around Y maze
            if curr_frame == 0:
                white_frame = np.ones_like(frame) * 255
                mask = np.zeros_like(frame[:, :, 0])
                cv2.fillPoly(mask, [whole_area], 255)
            frame = np.where(mask[:, :, None] == 255, frame, white_frame)

            # crop frame
            frame = frame[y1:y2, x1:x2]

            video.write(frame)

        else:
            break
    
    print(f'Edited: {np.shape(frame)[1]}x{np.shape(frame)[0]}px, {curr_frame+1} frames, {fps} fps\n\n')
    vid.release()
    video.release()

def main():
    for video_file in files:
        output_video_file = video_file.with_name(video_file.stem + '_YLineMasked' + video_file_format)
        print(video_file)
        adjust_video(video_file, output_video_file)

main()

