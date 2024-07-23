#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter
import os
from random import randrange

     
path = 'C:\\Users\\landgrafn\\NFCyber\\WalterWhities\\Test\\'
common_name = '3m_293_editDLC'

x_pixel, y_pixel = 570, 570
arena_length = 400          # in mm
fps = 100


all_bodyparts = ['snout', 'mouth', 'paw_VL','paw_VR', 'middle', 'paw_HL', 'paw_HR','anus', 'tail_start', 'tail_middle', 'tail_end']
center_bodyparts = ['middle']
centerline_bodyparts = ['tail_start', 'anus', 'middle']
paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
colors = ['b', 'g', 'r', 'c', 'm', 'y']




#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# FUNCTIONS

# basic functions
def get_files(path):
    # get files
    files = [path+file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    print(f'{len(files)} files found\n')

    return files

def euclidian_dist(point1x, point1y, point2x, point2y):

    try:
        # calculates the euclidian distance between 2 bps in mm
        distance_mm = np.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)
    except:
        print('ERROR: euclidian_dist was not working')
    
    return distance_mm

def point2line_dist(slope, intercept, point_x, point_y):
    # returns the shortest distance between a line (given slope and intercept) and a xy point
    A, B, C = slope, -1, intercept
    distance = (abs(A*point_x + B*point_y + C) / np.sqrt(A**2 + B**2))# / px_per_mm

    return distance

def fuse_dfs(df1, df2):
    # FUSE 2 dfs by reindexing df2 to the index of df1 (important if you have nan gaps)
    df2 = df2.reindex(df1.index)
    df1 = df1.join(df2)

    return df1

def timestamps_onsets(series, min_duration_s, onset_cond=float):
    # you have a series with conditions (e.g. True/False or float/nan) in each row
    # return the start & stop frame of the [cond_onset, cond_offset]
    min_frame_nb = min_duration_s * fps


    timestamp_list = []
    start_idx = None
    prev_idx = None
    
    # get on- and offset frames between switches between True & False
    if onset_cond == True:
        for idx, val in series.items():
            if (val == onset_cond) and (start_idx == None):
                start_idx = idx
            
            elif (val != onset_cond) and (start_idx != None):
                if idx - start_idx >= min_frame_nb:
                    timestamp_list.append([start_idx, prev_idx])
                start_idx = None
            prev_idx = idx

        # for last value
        if start_idx is not None:
            if series.index[-1] - start_idx >= min_frame_nb:
                timestamp_list.append([start_idx, series.index[-1]])
    
    # get on- and offset frames between switches between np.nan & float numbers
    elif onset_cond == float:
        for idx, val in series.items():
            if not np.isnan(val) and (start_idx == None):
                start_idx = idx
            
            elif np.isnan(val) and (start_idx is not None):
                if idx - start_idx >= min_frame_nb:
                    timestamp_list.append([start_idx, prev_idx])
                start_idx = None
            prev_idx = idx

        # for last value
        if start_idx is not None:
            if series.index[-1] - start_idx >= min_frame_nb:
                timestamp_list.append([start_idx, series.index[-1]])
    
    return timestamp_list


# specific functions
def cleaning_raw_df(df, nan_threshold=0.3):
    # CLEAN UP raw data frame, nan if likelihood < x, ints&floats insead of strings
    # combines the string of the two rows to create new header
    #print('1. cleaning_raw_df')

    new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     # df.iloc[0] takes all values of first row
    df.columns = new_columns
    df = df.drop(labels=[0, 1, 2], axis="index")

    # adapt index
    df.set_index('bodyparts_coords', inplace=True)
    df.index.names = ['frames']

    # turn all the prior string values into floats and index into int
    df = df.astype(float)
    df.index = df.index.astype(int)
    
    # flip the video along the x axis
    for column in df.columns:
        if '_y' in column:
            df[column] = y_pixel - df[column] 

    # whereever _likelihood <= x, change _x & _y to nan
    for bodypart in all_bodyparts:
        filter = df[f'{bodypart}_likelihood'] <= nan_threshold
        df.loc[filter, f'{bodypart}_x'] = np.nan
        df.loc[filter, f'{bodypart}_y'] = np.nan
        df = df.drop(labels=f'{bodypart}_likelihood', axis="columns")
    
    # transform every coordinate into mm
    mm_per_px = arena_length / np.mean([x_pixel, y_pixel])
    df = df * mm_per_px
    
    return df

def animal_moving(df, min_speed=50, min_moving_duration_s=0.5):

    # for defining movement
    duration_shifted_s = 0.1
    frames_shifted = int(duration_shifted_s * fps)
    df['anus_x_shifted'] = df['anus_x'].shift(periods=frames_shifted)
    df['anus_y_shifted'] = df['anus_y'].shift(periods=frames_shifted)
    df['dist_anus_shifted'] = euclidian_dist(df['anus_x'].values, df['anus_y'].values, df['anus_x_shifted'].values, df['anus_y_shifted'].values)


    # min_speed is in mm/s, min_moving_duration is in s  !!!depends on duration_shifted_s, which should be 0.1s!!!
    #print('4. animal_moving')
    nec_dist_mm = duration_shifted_s * min_speed
    
    animalmoving = df['dist_anus_shifted'] >= nec_dist_mm
    df['animalmoving'] = animalmoving

    # get True on- and offset
    animalmoving_timestamps = timestamps_onsets(animalmoving, min_moving_duration_s, onset_cond=True)

    print(moving_segments)
    return df, animalmoving_timestamps

def paw_speed(df):

    # for calculating paw movement speed
    frames_shifted_paw_speed = 1
    for paw in ['paw_HL', 'paw_HR']:

        df[f'{paw}_x_shifted'] = df[f'{paw}_x'].shift(periods=frames_shifted_paw_speed)
        df[f'{paw}_y_shifted'] = df[f'{paw}_y'].shift(periods=frames_shifted_paw_speed)
        df[f'dist_{paw}_shifted'] = euclidian_dist(df[f'{paw}_x'].values, df[f'{paw}_y'].values, df[f'{paw}_x_shifted'].values, df[f'{paw}_y_shifted'].values)
        df[f'dist_{paw}_shifted'] = df[f'dist_{paw}_shifted'] * fps/frames_shifted_paw_speed
        df[f'{paw}_speed'] = df[f'dist_{paw}_shifted'].rolling(window=4).mean()

    return df

def paw_speed_minmax(series, nec_speed_on=150, nec_speed_off=80):

    # returns the indices of the toe-off and foot-strike of each step
    in_swing = False
    step_start = None
    all_steps = []

    for idx, val in series.items():
        if not in_swing and val >= nec_speed_on:
            step_start = idx-3
            in_swing = True
        
        elif in_swing and val <= nec_speed_off and idx-step_start >= 10:

                step_end = idx
                all_steps.append((step_start, step_end))
                in_swing = False
                
    return all_steps



def plot_bodyparts(timewindow, df, spec_bodyparts, bodyparts, title):

    paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
    
    for start, end in timewindow:
        i_color = randrange(len(colors))

        # only include rows that are in the time window
        rows2plot = df.loc[start : end]

        if 'center' in spec_bodyparts:
            # plot animal center via removing the nan values
            x = rows2plot['center_x'][~np.isnan(rows2plot['center_x'])]
            y = rows2plot['center_y'][~np.isnan(rows2plot['center_y'])]
            #plt.scatter(x, y, label=f'{start}_center_{end}', s=0.2, c=colors[i_color])
            plt.plot(x, y, label=f'{start}-{end}', c=colors[i_color])
        
        if 'static_feet' in spec_bodyparts:            
            for paw in paws:
                plt.scatter(rows2plot[f'static_{paw}_x'], rows2plot[f'static_{paw}_y'], label=f'static_{paw}', c=colors[i_color], marker='x', s=20)
                
                if 'raw_feet' in spec_bodyparts:
                        plt.scatter(rows2plot[f'{paw}_x'], rows2plot[f'{paw}_y'], label=paw, c=colors[i_color], marker='.', s=10)

        
        for part in bodyparts:
            plt.scatter(rows2plot[f'{part}_x'], rows2plot[f'{part}_y'], label=part, c=colors[i_color], marker='+', s=10)
            

        i_color += 1
    
    plt.xlim(0, arena_length)
    plt.ylim(0, arena_length)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    #plt.legend()
    plt.title(title)
    plt.show()

def plot_paws(x):
        plt.plot(range(x[0],x[1]+1), main_df.loc[x[0]:x[1], 'paw_HL_speed'], c='r')
        plt.plot(range(x[0],x[1]+1), main_df.loc[x[0]:x[1], 'paw_HR_speed'], c='b')
        
        
        for start, end in steps_HL:
            plt.hlines(0, start, end, colors='r')

        for start, end in steps_HR:
            plt.hlines(10, start, end, colors='b')


        plt.xlim(x[0], x[1])
        plt.show()


file_list = get_files(path)

for file in file_list:
    print(f'{file}')
    
    main_df = pd.read_csv(file, header=None, low_memory=False)
    main_df = cleaning_raw_df(main_df)
    main_df, moving_segments = animal_moving(main_df)
    main_df = paw_speed(main_df)
    steps_HL = paw_speed_minmax(main_df['paw_HL_speed'])
    steps_HR = paw_speed_minmax(main_df['paw_HR_speed'])


    plot_paws([600, 700])
    

    
# (list of [start, stop], df, special_bodyparts=['center', 'static_feet', 'raw_feet'], bodypart=[e.g. 'snout', 'mouth'], title of plot)
    #plot_bodyparts(moving_segments[0:1], main_df, [], ['paw_HL'], f'Moving {file}')