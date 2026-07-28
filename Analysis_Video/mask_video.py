#%%
# IMPORT
import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import numpy as np
from pathlib import Path


path = "D:\\CA1Dopa_Longitud_Videos\\videos_3crop\\YMaze"

def get_files(path):
    # get all file names in directory into list
    files = [file for file in Path(path).iterdir() if file.is_file() and '.mp4' in file.name]
    print(f'{len(files)} videos found:')
    for file in files: 
        print(file)
    print('\n\n')
    return files

def check_areas(left_corner, middle_corner, right_corner, left_arm_end_lefter, left_arm_end_righter, 
                right_arm_end_righter, right_arm_end_lefter, middle_arm_end_lefter, middle_arm_end_righter, 
                bottom_left, bottom_right):

    whole_area = Polygon([middle_corner, right_arm_end_lefter, bottom_right, right_arm_end_righter, 
                        right_corner, middle_arm_end_righter, middle_arm_end_lefter, 
                        left_corner, left_arm_end_lefter, bottom_left, left_arm_end_righter])
    whole_area = np.array(list(whole_area.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))

    return whole_area

def adjust_video(input_file, output_file):

    # original stuff
    vid = cv2.VideoCapture(str(input_file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f'Origin: {width}x{height}px, {nframes} frames, {fps} fps')

    # coordinates of the respective positions
    left_corner = (275, 320)
    middle_corner = (330, 420)
    right_corner = (390, 320)

    left_arm_end_lefter = (0, 480)
    left_arm_end_righter = (90, height)

    right_arm_end_righter = (width, 480)
    right_arm_end_lefter = (570, height)

    middle_arm_end_lefter = (275, 0)
    middle_arm_end_righter = (390, 0)

    bottom_left = (0, height)
    bottom_right = (width, height)

    # get area where to mask
    coords_no_mask = check_areas(left_corner, middle_corner, right_corner, left_arm_end_lefter, left_arm_end_righter, 
                                 right_arm_end_righter, right_arm_end_lefter, middle_arm_end_lefter, middle_arm_end_righter, 
                                 bottom_left, bottom_right)


    # create new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height)) 

    for curr_frame in tqdm(range(nframes)):
        ret, frame = vid.read()
        if ret:    

            # white mask around Y maze
            if curr_frame == 0:
                white_frame = np.ones_like(frame) * 255
                mask = np.zeros_like(frame[:, :, 0])
                cv2.fillPoly(mask, [coords_no_mask], 255)
            frame = np.where(mask[:, :, None] == 255, frame, white_frame)

            video.write(frame)

        else:
            break
    
    print(f'Edited: {np.shape(frame)[1]}x{np.shape(frame)[0]}px, {curr_frame+1} frames, {fps} fps\n\n')
    vid.release()
    video.release()



files = get_files(path)

for video_file in files:
    print(video_file)
    output_video_file = video_file.with_name(video_file.stem + '_mask.mp4')
    adjust_video(video_file, output_video_file)




