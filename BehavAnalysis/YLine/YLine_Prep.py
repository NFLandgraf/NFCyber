#%%
# IMPORT
import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import numpy as np
from pathlib import Path


path = 'C:\\Users\\landgrafn\\Desktop\\Group2\\'

def get_files(path):
    # get all file names in directory into list
    files = [file for file in Path(path).iterdir() if file.is_file() and '.mp4' in file.name]
    
    print(f'{len(files)} videos found:')
    for file in files: 
        print(file)
    
    return files
files = get_files(path)

def get_xmax_ymax(file):
    # get video properties
    vid = cv2.VideoCapture(str(file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return width, height
width, height = get_xmax_ymax(files[0])


#%%
def adapt(x, y):
    # add dx/dy and multiplay with dz
    adapt_x = (x + dx) * dz
    adapt_y = (y + dy) * dz

    # check if points are out of the image and if so, adapt
    if adapt_x < 0:
        print(f'x_coord {adapt_x} < 0 (has been changed to 0)')
        adapt_x = 0
    elif adapt_x > width:
        print(f'x_coord ({adapt_x}px) > width ({width}px) (has been changed to {width}px)')
        adapt_x = width


    if adapt_y < 0:
        print(f'y_coord {adapt_y} < 0 (has been changed to 0)')
        adapt_y = 0
    elif adapt_y > height:
        print(f'y_coord {adapt_y} > height ({height}px) (has been changed to height={height}px)')
        adapt_y = height

    return (adapt_x, adapt_y)

# USER INPUT
dx = 0      # if you want to shift the whole thing horizontally 
dy = 0     # if you want to shift the whole thing vertically
dz = 1      # if you want to change the size of the whole thing
# use width and height as the maximum of the x- and y-axis

# corners
left_corner = adapt(308, 250)
middle_corner = adapt(345, 180)
right_corner = adapt(385, 250)

# left arm
left_arm_end_lefter = adapt(0, 80)
left_arm_end_righter = adapt(40, 0)

# right arm
right_arm_end_righter = adapt(width, 70)
right_arm_end_lefter = adapt(640, 0)

# middle_arm
middle_arm_end_lefter = adapt(300, height)
middle_arm_end_righter = adapt(390, height)


def check_areas(files):

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
        first_frame = 'YLine_fframe.png'

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
    areas, coords_no_mask = get_areas()
    write_and_draw(file, areas)

    return coords_no_mask

coords_no_mask = check_areas(files)


#%%HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# CROP & MASK
'''''
1. Crop the videos into new dimensions so that the arena corners limit the frame:
    x[left_arm_end_lefter:left_arm_end_righter]
    y[min:max of any y maze point]
2. Fill the area outside of the arena with white jibber
'''''

def mask_video(files, coords_no_mask):

    def adjust_video(input_file, output_file, coords_no_mask):

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
                    cv2.fillPoly(mask, [coords_no_mask], 255)
                frame = np.where(mask[:, :, None] == 255, frame, white_frame)

                # crop frame
                frame = frame[y1:y2, x1:x2]

                video.write(frame)

            else:
                break
        
        print(f'Edited: {np.shape(frame)[1]}x{np.shape(frame)[0]}px, {curr_frame+1} frames, {fps} fps\n\n')
        vid.release()
        video.release()

    for video_file in files:
        print(video_file)

        output_video_file = video_file.with_name(video_file.stem + '_YLineMasked.mp4')
        adjust_video(video_file, output_video_file, coords_no_mask)

mask_video(files, coords_no_mask)

