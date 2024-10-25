#%%
# IMPORT & DEFINE
'''
We only care about the head positions.
Exit = Doesn't exist, we only score an Exit if there is an Entry
Entry= All bps_head are in a new area

'''


import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import cv2
from pathlib import Path


# USER INPUT
# folder with the csv files and for each csv file, copy the corresponding video (best case with DLC annotations) into that folder
path = 'C:\\Users\\landgrafn\\NFCyber\\BehavAnalysis\\SocialInteractionAnalysis\\data'
common_name_csv = '.csv'
create_video = False
common_video_name = '.csv'
video_file_format = '.mp4'

width = 640
height = 600
fps = 30
nframes = 18485
duration = 616.1666666666666
px_per_cm = 11.55



center_N = 157, 457
center_O = 488, 127


# radius of the cup is around 4.3cm 
radii_invest_cm = [7.3]



# FUNCTIONS
# preparation
def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files

def out_to_txt(file, time_spent, alterations, total_distance, radius_invest):
    output_file = path + '\\analysis.txt'
    with open (output_file, 'a') as f:
        f.write(f'\n\nfile_name: {file}\n'
                f'Radius_invest: {round(radius_invest/px_per_cm, 2)} cm\n'
                f'center_N = {center_N}\n'
                f'center_O = {center_O}\n'
                f'Time spent in [area_inv_N, area_inv_O, area_large_N, area_large_O]: {time_spent}\n'
                f'Alterations between area_inv: {alterations}\n'
                f'Distance travelled: {round(total_distance/px_per_cm, 2)} cm\n')

def define_areas(radius_invest, offset_invest):
    # define large areas that split the whole arena
    area_large_N = Polygon([(0, 0), (width, height), (0, height)])
    area_large_O = Polygon([(0, 0), (width, height), (width, 0)])

    # define the areas of investigation
    circle_N = Point(center_N).buffer(radius_invest)
    x_N, y_N = center_N[0]+offset_invest, center_N[1]-offset_invest
    corner_N = Polygon([(x_N, y_N), (x_N, height), (0, height), (0, y_N)])
    area_inv_N = circle_N.union(corner_N)

    circle_O = Point(center_O).buffer(radius_invest)
    x_O, y_O = center_O[0]-offset_invest, center_O[1]+offset_invest
    corner_O = Polygon([(x_O, y_O), (x_O, 0), (width, 0), (width, y_O)])
    area_inv_O = circle_O.union(corner_O)

    areas = [area_inv_N, area_inv_O, area_large_N, area_large_O]
    
    return areas

def define_bodyparts():

    bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

    bps_large = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    
    bps_head = ['nose', 'left_eye', 'right_eye', 'head_midpoint']

    return bps_all, bps_large, bps_head

def cleaning_raw_df(csv_file, bps_all):
    df = pd.read_csv(csv_file, header=None, low_memory=False)

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

    # whereever _likelihood <= x, change _x & _y to nan
    for bodypart in bps_all:
        filter = df[f'{bodypart}_likelihood'] <= 0.9
        df.loc[filter, f'{bodypart}_x'] = np.nan
        df.loc[filter, f'{bodypart}_y'] = np.nan
        df = df.drop(labels=f'{bodypart}_likelihood', axis="columns")

    # !!!!!!!DELETE FIRST FRAMES OF CSV FILE!!!!!! (mouse was placed into arena)
    # df = df.iloc[600:]
    # df.reset_index(drop=True, inplace=True)
    
    return df

# calculations
def point_in_which_area(df, bp, frame, areas, areas_int):
    
    # return area in which bodpart is located
    x = df.loc[frame, f'{bp}_x']
    y = df.loc[frame, f'{bp}_y']

    if np.isnan(x):
        return None
    
    bp_point = Point(x, y)
    for i, area in enumerate(areas):
        if area.contains(bp_point):
            # if areas_int[i] == 3:
            #     print("ssf")
            return areas_int[i]
        
    # if point is not nan and point is in non of the areas
    return None

def whole_head_where(df, bps_list, frame, areas, areas_int):
    
    # create list with all the areas in which the bodyparts are in
    areas_of_bps = []
    for bp in bps_list:
        areas_of_bps.append(point_in_which_area(df, bp, frame, areas, areas_int))
    
    # only care about the ones that DLC has identified
    if all(bp is None for bp in areas_of_bps):
        return None
    areas_of_bps = [i for i in areas_of_bps if i is not None]

    # if all bodyparts are in the area of the first bodypart
    first_value = areas_of_bps[0]
    if all(elem == first_value for elem in areas_of_bps):
        return first_value
    return None

def calc_all_pos(df, bps_head, areas, areas_int):
    
    all_positions = []
    curr_area = 99

    for frame in tqdm(range(len(df.index))):
         
        whole_head_position = whole_head_where(df, bps_head, frame, areas, areas_int)
            
        if curr_area == whole_head_position:
            all_positions.append(curr_area)
        elif whole_head_position == None:
            all_positions.append(curr_area)
        else:
            all_positions.append(whole_head_position)
            curr_area = whole_head_position
    
    return all_positions

def analyze(all_positions, df, areas_int, radius_invest):

    # how many frames in each area
    time_spent = []
    for area in areas_int:
        duration = round(all_positions.count(area) / fps, 1)
        time_spent.append(duration)

    # how many changes
    alterations = 0
    curr_area = None
    for area in all_positions:
        if area == 0 or area == 1:
            if area != curr_area:
                curr_area = area
                alterations += 1
    
    # what distance did the mouse move during test
    bp = 'tail_base'
    df = df.iloc[::10].copy()
    df['distance'] = np.sqrt((df[f'{bp}_x'].diff())**2 + (df[f'{bp}_y'].diff())**2)
    total_distance = round(df['distance'].sum(), 0)

    print(f'Radius_invest: {round(radius_invest/px_per_cm, 1)}\n'
          f'Time spent in [area_inv_N, area_inv_O, area_large_N, area_large_O]:\n'
          f'{time_spent}\n'
          f'Alterations between area_inv: {alterations}\n'
          f'Distance travelled: {total_distance}px\n')
    
    return time_spent, alterations, total_distance

def do_video(csv_file, all_positions, areas, areas_int):

    # FIND VIDEO that corresponds to csv file !!needs to have the same beginning!!
    input_video_found = False
    csv_stem = Path(csv_file).stem
    endings = ['', 'YLineMasked', '_labeled']

    for i in range(len(csv_stem), 0, -1):       # start, stop, step

        for ending in endings:
            input_video_file = Path(path) / (csv_stem[:i] + ending + video_file_format)

            if input_video_file.exists():
                input_video_found = True
                break

        if input_video_found:
            break
            
    if not input_video_found:
        print(f'No matching video file found for {csv_file}')
        return


    # CREATE VIDEO
    # original stuff
    vid = cv2.VideoCapture(str(input_video_file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    if nframes != len(all_positions):
        print(f'Number of video frames ({nframes}) are different than all_positions ({len(all_positions)}). Maybe due to cutting')
    
    # create new video
    output_video_file = input_video_file.with_name(input_video_file.stem + '_AnalyAreas' + video_file_format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_video_file), fourcc, fps, (width, height)) 

    for curr_frame in tqdm(range(nframes)):
        ret, frame = vid.read()
        if ret:    

            # in which area is mouse in this frame
            mouse_area_nb = all_positions[curr_frame]

            if mouse_area_nb != 99:
                mouse_area = areas[mouse_area_nb]
                points_of_area = np.array(mouse_area.exterior.coords, dtype=np.int32)
                
                # draw on frame
                cv2.polylines(frame, [points_of_area], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(frame, f'{mouse_area_nb}', (int(width/3), 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA)

            video.write(frame)

        else:
            break
    
    vid.release()
    video.release()


# main function
def main():

    csv_files = get_files(path, common_name_csv)

    areas_int = [0, 1, 2, 3]
    bps_all, bps_large, bps_head = define_bodyparts()

    

    for csv_file in csv_files:
        print(f'\n\n{csv_file}')

        df = cleaning_raw_df(csv_file, bps_all)


        for radius_invest in radii_invest_cm:

            # convert into px
            radius_invest = radius_invest * px_per_cm
            offset_invest = radius_invest/3
            
            # necessary definitions
            areas = define_areas(radius_invest, offset_invest)
            

            all_positions = calc_all_pos(df, bps_head, areas, areas_int)
            time_spent, alterations, total_distance = analyze(all_positions, df, areas_int, radius_invest)

            out_to_txt(csv_file, time_spent, alterations, total_distance, radius_invest)

            if create_video:
                do_video(csv_file, all_positions, areas, areas_int)

    return all_positions

all_positions = main()



#%%
print(all_positions)