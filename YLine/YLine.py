#%%
# IMPORT & DEFINE
# into area: front 80% (mind. 15 non-nans including tail_base) are in arm 
# exit outof area: no bp of front 80% in current area AND any bp of front 80% is in other area than current

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import cv2
from matplotlib import image 
import os
import re


# USER INPUT
# define folder with the csv files
path = 'C:\\Users\\landgrafn\\NFCyber\\YLine\\data\\'
common_name = '.csv'

width = 968
height = 608

#left arm
left_corner = (350, 250)
middle_corner = (390, 180)
left_arm_end_lefter = (30, 80)
left_arm_end_righter = (70, 0)

#middle arm
right_corner = (430, 250)
middle_arm_end_righter = (440, 608)
middle_arm_end_lefter = (350, 608)

#right arm
right_arm_end_righter = (750, 80)
right_arm_end_lefter = (700, 0)

fps = 30
nframes = 9091
duration = 303.03333333333336



#%%
# FUNCTIONS

# preparation
def get_files(path):
    # get files
    files = [file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    print(f'{len(files)} files found\n')

    return files

def out_to_txt(file, pos_changes, alterations, total_entries, alt_index, alert):
    output_file = path + '0alteration_indices'
    with open (output_file, 'a') as f:
        f.write(f'\nfile_name: {file}'
                f'\npos_chang: {pos_changes}'
                f'\nalt_index: {alterations} / {total_entries} = {round(alt_index, 4)}'
                f'\n{alert}')

def define_areas():
    left_arm = Polygon([left_corner, middle_corner, left_arm_end_righter, left_arm_end_lefter])
    right_arm = Polygon([right_corner, middle_corner, right_arm_end_lefter, right_arm_end_righter])
    middle_arm = Polygon([left_corner, right_corner, middle_arm_end_righter, middle_arm_end_lefter])
    center = Polygon([middle_corner, left_corner, right_corner])
        
    return center, left_arm, middle_arm, right_arm

def define_bodyparts():

    bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

    bps_entry = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    
    bps_exit = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

    return bps_all, bps_entry, bps_exit

def cleaning_raw_df(csv_file):
    df = pd.read_csv(path + csv_file, header=None, low_memory=False)

    # combines the string of the two rows to create new header
    new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     # df.iloc[0] takes all values of first row
    df.columns = new_columns
    df = df.drop(labels=[0, 1, 2], axis="index")

    # adapt index
    df.set_index('bodyparts_coords', inplace=True)
    df.index.names = ['frames']

    # turn all the prior string values into floats and index into int
    df = df.astype(float)
    df.index = df.index.astype(int)
    
    # flip the video along the y axis
    # for column in df.columns:
    #     if '_y' in column:
    #         df[column] = y_pixel - df[column] 

    # whereever _likelihood <= x, change _x & _y to nan
    for bodypart in bps_all:
        filter = df[f'{bodypart}_likelihood'] <= 0.9
        df.loc[filter, f'{bodypart}_x'] = np.nan
        df.loc[filter, f'{bodypart}_y'] = np.nan
        df = df.drop(labels=f'{bodypart}_likelihood', axis="columns")
    
    return df


# position calculations
def point_in_which_area(df, bp, frame, areas, areas_int):
    global bp_not_in_any_area
    # return area in which bodpart is located
    x = df.loc[frame, f'{bp}_x']
    y = df.loc[frame, f'{bp}_y']

    if np.isnan(x):
        return None
    
    bp_point = Point(x, y)
    for i, area in enumerate(areas):
        if area.contains(bp_point):
            return areas_int[i]
        
    # if point is not nan and point is in non of the areas, return 99
    bp_not_in_any_area += 1
    return 99


def point_in_spec_area(df, bp, frame, area):
    # check if point is in area
    # consider nan as not in area
    x = df.loc[frame, f'{bp}_x']
    if np.isnan(x):
        return False

    bp_point = Point(x, df.loc[frame, f'{bp}_y'])

    return area.contains(bp_point)
        
def entry_into_arm(df, bps_list, frame, areas, areas_int):
    # returns arm of entry if front 80% are in an ARM
    # of all front 80%, none is allowed to be in another area and >= X bps need to be non-nan
    nonnan_bps_in_new_area = 15
    # entry can only happen if an exit happened before!

    global frames_with_no_annot

    # create list with all the areas in which the bodyparts are in
    areas_of_bps = []
    for bp in bps_list:
        areas_of_bps.append(point_in_which_area(df, bp, frame, areas, areas_int))
    
    # only care about the ones that DLC has identified
    if all(bp is None for bp in areas_of_bps):
        frames_with_no_annot += 1
        return False
    areas_of_bps = [i for i in areas_of_bps if i is not None]

    # if all bodyparts are in the area of the first bodypart and the first bodypart is not in center (=0)
    if len(areas_of_bps) >= nonnan_bps_in_new_area:
        first_value = areas_of_bps[0]
        if all(elem == first_value for elem in areas_of_bps) and first_value != 0:
            return first_value
    return False

def exit_outof_arm(df, bps_list, frame, curr_area, areas, areas_int):
    # exit if front 80% not in current arm but rather parts of front 80% in other area (min 1 non-nan)
    
    # if a bp is in curr_arm, it is no exit -> return False; if no bp is in curr_arm, it may be an exit ->continue
    for bp in bps_list:
        if point_in_spec_area(df, bp, frame, areas[curr_area]):
            return False
    
    # check if any bp is in other area than curr_arm
    areas_of_bps = []
    for bp in bps_list:
        areas_of_bps.append(point_in_which_area(df, bp, frame, areas, areas_int))
    
    # only care for bodyparts that DLC has identified
    areas_of_bps = [i for i in areas_of_bps if i is not None]

    # if now any bodypart is in a nother area, it is an exit
    for area in areas_of_bps:
        if area != curr_area:
            return True
    
    return False

def calc_all_pos(df, bps_entry, bps_exit, areas, areas_int):
    # when animal is leaving an arm, position is automatically set as center
    all_positions = []
    curr_area = 0

    for frame in tqdm(range(len(df.index))):
        if curr_area == 0: 
            entry = entry_into_arm(df, bps_entry, frame, areas, areas_int)
            if not entry:
                all_positions.append(0)
            elif entry:
                all_positions.append(entry)
                curr_area = entry

        elif curr_area != 0:
            exit = exit_outof_arm(df, bps_exit, frame, curr_area, areas, areas_int)
            if not exit:
                all_positions.append(curr_area)
            elif exit:
                all_positions.append(0)
                curr_area = 0
    
    return all_positions

def calc_alteration_index(pos_changes):

    total_entries = len(pos_changes) - 2
    alterations = 0

    for i in range(len(pos_changes) - 2):
        tripled = [pos_changes[i], pos_changes[i+1], pos_changes[i+2]]

        if len(tripled) == len(set(tripled)):
            alterations += 1
            
    alt_index = alterations / total_entries

    return alterations, total_entries, alt_index


# main function
def do_stuff(files):

    for file in files:
        print(f'\n{file}')
        df = cleaning_raw_df(file)
        all_positions = calc_all_pos(df, bps_entry, bps_exit, areas, areas_int)

        # only add areas that were entered after a change
        pos_changes = [pos for i, pos in enumerate(all_positions) if i == 0 or pos != all_positions[i-1]]

        # remove center (0) position
        pos_changes = [i for i in pos_changes if i != 0]

        alterations, total_entries, alt_index = calc_alteration_index(pos_changes)

        alert = '!CHECK MANUALLY!' if bp_not_in_any_area > 0 or frames_with_no_annot > 1*fps else ''
        if bp_not_in_any_area > 0:
            alert = f'!CHECK MANUALLY! {bp_not_in_any_area} bps not in area'
        elif frames_with_no_annot > 1*fps:
            alert.append(f' + {frames_with_no_annot} frames without annotations')

        out_to_txt(file, pos_changes, alterations, total_entries, alt_index, alert)
    
    print(f'\nYLine done!\n'
      f'{bp_not_in_any_area} frames where a bp was outside of defined areas\n'
      f'{frames_with_no_annot} frames where no DLC annotation in frame\n')

    return all_positions



#%%
# DO STUFF 
files = get_files(path)
frames_with_no_annot = 0
bp_not_in_any_area = 0

center, left_arm, middle_arm, right_arm = define_areas()
areas = [center, left_arm, middle_arm, right_arm]
areas_int = [0, 1, 2, 3]

bps_all, bps_entry, bps_exit = define_bodyparts()

all_positions = do_stuff(files)


#%%
# CREATE VIDEO

def write_all_frames_from_video(input_file, output_folder):
    # takes video and creates png for every frame in video
    os.makedirs(output_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(input_file)
    nframes = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, image = vidcap.read()

    print('\nwrite_all_frames_from_video')
    count = 0
    pbar = tqdm(total=int(nframes))
    while success:
        cv2.imwrite(output_folder + '\\frame%d.png' % count, image)
        success,image = vidcap.read()
        count += 1
        pbar.update(1)

def draw_and_write(input_folder, output_folder, areas, positions):
    # takes pngs and draws polygon onto it, if mouse in that position
    os.makedirs(output_folder, exist_ok=True)

    images = [file for file in os.listdir(input_folder) if file.endswith('.png')]

    # sort the list of strings based on the int in the string
    sorted_intimages = sorted([int(re.sub("[^0-9]", "", element)) for element in images])
    sorted_images = [f'frame{element}.png' for element in sorted_intimages]

    print('\ndraw_and_write')
    for i in tqdm(range(len(sorted_images))):
        
        image_path = os.path.join(input_folder, sorted_images[i])
        data = cv2.imread(image_path)

        for arm, area in enumerate(areas):
            if arm == positions[i]:
                points = np.array(area.exterior.coords, dtype=np.int32)
                cv2.polylines(data, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        
        output_path = os.path.join(output_folder, f"frame{i}.png")
        cv2.imwrite(output_path, data)

def video_from_frames(input_folder, path):
    # created video from all frames with drawn polygons
    output = os.path.join(path, f'{files[0]}_edit.mp4')

    images = [file for file in os.listdir(input_folder) if file.endswith('.png')]

    # sort the list of strings based on the int in the string
    sorted_intimages = sorted([int(re.sub("[^0-9]", "", element)) for element in images])
    sorted_images = [f'frame{element}.png' for element in sorted_intimages]

    # setting the video properties according to first image 
    frame0 = cv2.imread(os.path.join(input_folder, sorted_images[0]))
    height, width, layers = frame0.shape 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, fps, (width, height)) 

    # Appending the images to the video
    print('\nvideo_from_frames')
    for i in tqdm(range(len(sorted_images))):
        img_path = os.path.join(input_folder, sorted_images[i])
        img = cv2.imread(img_path)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def create_drawn_video(video_file, folder_framesfromvideo, folder_draw_on_frames, path, areas, positions):
    write_all_frames_from_video(video_file, folder_framesfromvideo)
    draw_and_write(folder_framesfromvideo, folder_draw_on_frames, areas, positions)
    video_from_frames(folder_draw_on_frames, path)


def do_video(all_positions):
    video_file = path + '\\513_TSPO-KO_NaiveDLC_resnet50_TopoViewMouseMar22shuffle1_600000_filtered_labeled.mp4'
    vid = cv2.VideoCapture(video_file)
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    folder_framesfromvideo = path + '\\frames_from_video'
    folder_draw_on_frames = path + '\\edited_frames'
    create_drawn_video(video_file, folder_framesfromvideo, folder_draw_on_frames, path, areas, all_positions)

    print('video done')
    

do_video(all_positions)
