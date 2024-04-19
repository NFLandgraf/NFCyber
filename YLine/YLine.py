#%%
# IMPORT & DEFINE
# entry into area: front 80% (mind. 3 non-nans, until tail_base) are in arm 
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

data_DLC_csv = 'data\\Y.csv'
fps = 100


df = pd.read_csv(data_DLC_csv, header=None)

# define arena and arms
width, height = 708, 608
top_corner = (350, 180)
left_corner = (315, 250)
right_corner = (390, 250)

left_arm = Polygon([left_corner, top_corner, (top_corner[0], 0), (0, 0), (0, left_corner[1])])
right_arm = Polygon([right_corner, top_corner, (top_corner[0], 0), (width, 0), (width, right_corner[1])])
bottom_arm = Polygon([(0, left_corner[1]), left_corner, right_corner, (width, right_corner[1]), (width, height), (0, height)])
center = Polygon([top_corner, left_corner, right_corner])

areas = [center, left_arm, right_arm, bottom_arm]
areas_int = [0, 1, 2, 3]

bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

bps_80 = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

bps_head = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint']




#%%
# FUNCTIONS

def cleaning_raw_df(df):
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

df = cleaning_raw_df(df)


def draw(frame):
    for arm in areas:
        x, y = arm.exterior.xy
        plt.plot(x, y)
        plt.fill_between(x, y, alpha=0.1)

    for bp in bps_all:
        plt.scatter(df.loc[frame, f'{bp}_x'], df.loc[frame, f'{bp}_y'], color='k')
        plt.xlim(0, width)
        plt.ylim(0, height)
    
    plt.gca().invert_yaxis()
    plt.show()



def point_in_which_area(df, bp, frame, areas, areas_int):
    # return area in which bodpart is located
    x = df.loc[frame, f'{bp}_x']

    if np.isnan(x):
        return None
    
    bp_point = Point(x, df.loc[frame, f'{bp}_y'])
    for i, area in enumerate(areas):
        if area.contains(bp_point):
            return areas_int[i]
    print('Error: Bodypart not in areas nor nan')

def point_in_spec_area(df, bp, frame, area):
    # check if point is in area
    # consider nan as not in area
    x = df.loc[frame, f'{bp}_x']
    if np.isnan(x):
        return False

    bp_point = Point(x, df.loc[frame, f'{bp}_y'])

    return area.contains(bp_point)
        

def entry_into_arm(df, bps_80, frame, areas, areas_int):
    # returns arm of entry if front 80% are in an ARM
    # of all front 80%, none is allowed to be in another area and >= 3 bps need to be non-nan
    # entry can only happen if an exit happened before!
    areas_of_bps = []
    for bp in bps_80:
        areas_of_bps.append(point_in_which_area(df, bp, frame, areas, areas_int))
    
    areas_of_bps = [i for i in areas_of_bps if i is not None]

    if len(areas_of_bps) >= 3:
        first_value = areas_of_bps[0]
        if all(elem == first_value for elem in areas_of_bps) and first_value != 0:
            return first_value
    return False

def exit_outof_arm(df, bps_80, frame, curr_area, areas, areas_int):
    # exit if front 80% not in current arm but rather parts of front 80% in other area (min 1 non-nan)
    
    # if a bp is in curr_arm, it is no exit -> return False; if no bp is in curr_arm, it may be an exit ->continue
    for bp in bps_80:
        if point_in_spec_area(df, bp, frame, areas[curr_area]):
            return False
    
    # check if any bp is in other area than curr_arm
    areas_of_bps = []
    for bp in bps_80:
        areas_of_bps.append(point_in_which_area(df, bp, frame, areas, areas_int))
    
    areas_of_bps = [i for i in areas_of_bps if i is not None]

    for area in areas_of_bps:
        if area != curr_area:
            return True
    
    return False



#%%
# DO CALCULATIONS

# when animal is leaving an arm, position is automatically set as center
def big_calc(df, bps_80, areas, areas_int):
    all_positions = []
    curr_area = 0

    for frame in tqdm(range(len(df.index))):
        if curr_area == 0: 
            entry = entry_into_arm(df, bps_80, frame, areas, areas_int)
            if not entry:
                all_positions.append(0)
            elif entry:
                all_positions.append(entry)
                curr_area = entry

        elif curr_area != 0:
            exit = exit_outof_arm(df, bps_80, frame, curr_area, areas, areas_int)
            if not exit:
                all_positions.append(curr_area)
            elif exit:
                all_positions.append(0)
                curr_area = 0
    
    return all_positions


all_positions = big_calc(df, bps_80, areas, areas_int)




#%%
# CREATE VIDEO

parent_dir = "C:\\Users\\landgrafn\\NFCyber\\YLine\\data"
video_file = parent_dir + '\\Y_DLC.mp4'
vid = cv2.VideoCapture(video_file)
nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

folder_framesfromvideo = parent_dir + '\\frames_from_video'
folder_draw_on_frames = parent_dir + '\\edited_frames'


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
                cv2.polylines(data, [points], isClosed=True, color=(0, 0, 255), thickness=3)
        
        output_path = os.path.join(output_folder, f"frame{i}.png")
        cv2.imwrite(output_path, data)

def video_from_frames(input_folder, parent_dir):
    # created video from all frames with drawn polygons
    fps = 100
    output = os.path.join(parent_dir, 'video_edit.mp4')

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

def create_drawn_video(video_file, folder_framesfromvideo, folder_draw_on_frames, parent_dir, areas, positions):
    write_all_frames_from_video(video_file, folder_framesfromvideo)
    draw_and_write(folder_framesfromvideo, folder_draw_on_frames, areas, positions)
    video_from_frames(folder_draw_on_frames, parent_dir)

#create_drawn_video(video_file, folder_framesfromvideo, folder_draw_on_frames, parent_dir, areas, all_positions)


print('Job done, happy Ylining!')

#%%


