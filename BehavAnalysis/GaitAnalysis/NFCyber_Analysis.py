#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# IMPORT & DEFINE
# Lets keep main_df as the main DataFrame, where the index is the frame number. 
# If we calculate sth new (e.g. animal center, static_paw), there is the function fuse_dfs() to add new columns to the df. 
# Whenever we work on/manipulate the df, lets do this with a df.copy(), so the df is untouched

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter
import os

     
path = 'C:\\Users\\landgrafn\\Desktop\\feet\\'
common_name = '.csv'

x_pixel, y_pixel = 570, 570
arena_length = 400          # in mm
px_per_mm = np.mean([x_pixel, y_pixel]) / arena_length
fps = 100
duration_shifted_s = 0.1  # definition for important thing

all_bodyparts = ['snout', 'mouth', 'paw_VL','paw_VR', 'middle', 'paw_HL', 'paw_HR','anus', 'tail_start', 'tail_middle', 'tail_end']
center_bodyparts = ['middle']
paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
colors = ['b', 'g', 'r', 'c', 'm', 'y']

def get_files(path):
    # get files
    files = [path+file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    print(f'{len(files)} files found\n')

    return files

file_list = get_files(path)
print(file_list)


#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# FUNCTIONS

# basic functions
def euclidian_dist(point1x, point1y, point2x, point2y):

    # calculates the euclidian distance between 2 bps in mm
    distance_mm = (np.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)) / px_per_mm
    
    return distance_mm

def point2line_dist(slope, intercept, point_x, point_y):
    # returns the shortest distance between a line (given slope and intercept) and a xy point
    A, B, C = slope, -1, intercept
    distance = (abs(A*point_x + B*point_y + C) / np.sqrt(A**2 + B**2)) / px_per_mm

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
def cleaning_raw_df(df, nan_threshold=0.9):
    # CLEAN UP raw data frame, nan if likelihood < x, ints&floats insead of strings
    # combines the string of the two rows to create new header
    print('1. cleaning_raw_df')

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
    
    # add a gaussian filter over the df to reduce outlayers
    #df = df.rolling(window=100, win_type='gaussian', min_periods=1, center=True).mean(std=1)
    #df = df[500:750]

    return df

def add_basic_columns_to_df(df):
    # calculating relatively simple things to add to df
    print('2. add_basic_columns_to_df')

    frames_shifted = int(duration_shifted_s * fps)
    new_df = pd.DataFrame()

    center_bodyparts_x = [f'{bodypart}_x' for bodypart in center_bodyparts]
    center_bodyparts_y = [f'{bodypart}_y' for bodypart in center_bodyparts]
    
    # create time series for center that is represented by the mean of stable bodyparts
    # choose bodyparts that are always tracked and gaussian filter it
    new_df['center_x'] = df[center_bodyparts_x].mean(axis="columns")
    new_df['center_y'] = df[center_bodyparts_y].mean(axis="columns")
    new_df['center_x_avrg'] = new_df['center_x'].rolling(window=34).mean()
    new_df['center_y_avrg'] = new_df['center_y'].rolling(window=34).mean()
    new_df['center_x_smooth'] = savgol_filter(new_df['center_x'], 11, 2)    # window_length, polyorder
    new_df['center_y_smooth'] = savgol_filter(new_df['center_y'], 11, 2)
    
    # add shifted center columns to df
    new_df['center_x_shifted'] = new_df['center_x'].shift(periods=frames_shifted)
    new_df['center_y_shifted'] = new_df['center_y'].shift(periods=frames_shifted)

    # add distances beetween bps    
    new_df['dist_H_paws'] = euclidian_dist(df['paw_HL_x'].values, df['paw_HL_y'].values, df['paw_HR_x'].values, df['paw_HR_y'].values)
    new_df['dist_V_paws'] = euclidian_dist(df['paw_VL_x'].values, df['paw_VL_y'].values, df['paw_VR_x'].values, df['paw_VR_y'].values)
    new_df['dist_L_paws'] = euclidian_dist(df['paw_HL_x'].values, df['paw_HL_y'].values, df['paw_VL_x'].values, df['paw_VL_y'].values)
    new_df['dist_R_paws'] = euclidian_dist(df['paw_HR_x'].values, df['paw_HR_y'].values, df['paw_VR_x'].values, df['paw_VR_y'].values)
    
    new_df['dist_center_shifted'] = euclidian_dist(new_df['center_x'].values, new_df['center_y'].values, new_df['center_x_shifted'].values, new_df['center_y_shifted'].values)

    # takes bps that would represent an intrinsic centerline, calculates linear regression line for each row and adds columns for slope and intercept
    # takes nan if less than 2 bps without nan
    def linear_fit(row, x_headers, y_headers):
        # fits a linear regression line to the column values provided and returns the coefficients
        x = row.loc[x_headers]
        y = row.loc[y_headers]

        if np.sum(~np.isnan(x)) < 2:
            return (np.nan, np.nan)
        
        try:
            coeffs = np.polyfit(x, y, 1)
            return coeffs
        except:
            return (np.nan, np.nan)
    
    new_df[['centerline_slopes', 'centerline_intercepts']] = df.apply(linear_fit, args=(center_bodyparts_x, center_bodyparts_y), axis=1, result_type='expand')
    
    # adds distances between bps and centerline to new_df
    new_df['dist_HL2centerline'] = point2line_dist(new_df['centerline_slopes'], new_df['centerline_intercepts'], df['paw_HL_x'].values, df['paw_HL_y'].values)
    new_df['dist_HR2centerline'] = point2line_dist(new_df['centerline_slopes'], new_df['centerline_intercepts'], df['paw_HR_x'].values, df['paw_HR_y'].values)
    new_df['dist_VL2centerline'] = point2line_dist(new_df['centerline_slopes'], new_df['centerline_intercepts'], df['paw_VL_x'].values, df['paw_VL_y'].values)
    new_df['dist_VR2centerline'] = point2line_dist(new_df['centerline_slopes'], new_df['centerline_intercepts'], df['paw_VR_x'].values, df['paw_VR_y'].values)
    
    # add df_center to df
    df = fuse_dfs(df, new_df)

    return df

def static_feet(df, interval_compared_s=0.1, max_paw_movement_mm=10, min_static_duration_s=0.05):
    # get list of start- and stop-frames, where paws are static and df with only the mean paw position, rest is nan
    # interval_compared_s: because of DLC juggling, dont compare next frame but every x seconds
    # max_paw_movement_mm: between your interval, what movement is allowed to be considered static
    # min_static_duration_s: how long must the static period last to be considered static paw
    # according to just looking at csv files, these settings make sense: 0.1, 10, 0.05
    print('3. static_feet')

    frames_compared = interval_compared_s * fps
    min_frame_nb = min_static_duration_s * fps

    # define paws
    paws_coor = [f'{paw}_x' for paw in paws] + [f'{paw}_y' for paw in paws]

    # calculate the difference between the k frame and the k-x frame
    df_feet = df[paws_coor].copy()

    
    paws_diff = abs(df_feet.diff(periods=frames_compared, axis='rows'))


    for paw in paws:
        x, y = f'{paw}_x', f'{paw}_y'
        
        # check in which frames the vector magnitude is below x
        vector_lengths_mm = np.sqrt(paws_diff[x]**2 + paws_diff[y]**2) / px_per_mm
        paw_is_static = vector_lengths_mm <= max_paw_movement_mm
        static_paw_timestamps = timestamps_onsets(paw_is_static, min_frame_nb, onset_cond=True)
    
        # creating a mask to change all values outside of timestamp-rows to nan, so that only static paw values exist
        mask = np.ones(len(df_feet), dtype=bool)
        for start, end in static_paw_timestamps:
            mask[start:end] = False
        df_feet.loc[mask, [x, y]] = np.nan

        # go through timestamps and change all values in this timestamp to the mean values of the timestamp
        for start, end in static_paw_timestamps:
            for coordinate in [x, y]:
                step_mean = df_feet.loc[start:end, coordinate].mean()
                df_feet.loc[start:end, coordinate] = step_mean

    
    # add prefix to column names and add df_feet to df
    df_feet = df_feet.add_prefix('static_')
    df = fuse_dfs(df, df_feet.copy())

    return df

def animal_moving(df, min_speed=10, min_moving_duration_s=0.5):
    # min_speed is in mm/s, min_moving_duration is in s  !!!depends on duration_shifted_s, which should be 0.1s!!!
    print('4. animal_moving')
    nec_dist_mm = duration_shifted_s * min_speed
    
    animalmoving = df['dist_center_shifted'] >= nec_dist_mm
    df['animalmoving'] = animalmoving

    # get True on- and offset
    animalmoving_timestamps = timestamps_onsets(animalmoving, min_moving_duration_s, onset_cond=True)

    print(animalmoving_timestamps)
    return df, animalmoving_timestamps

def catwalk_regress(df, rvalue_threshold=0.75, duration_threshold_s=0.5, dist_threshold_mm=100):
    print('5. catwalk_regress')
    '''
    Take 2 points and create the best-fitting linear regression line, then add one point after another
    After each added point, calculate R_value, if now
        - R_value is above rvalue_threshold AND animalmoving=True for all the resp. frames, continue adding points
        - R_value is below rvalue_threshold, consider that the straight segment is over but only add it to list of straight segments if
          duration is above duration_threshold AND distance is above dist_threshold
    '''
    
    # catwalk segments are calculated upon center coordinates
    x = df['center_x']
    y = df['center_y']

    frames_threshold = duration_threshold_s * fps
    segments = []
    start_idx = 0
    end_idx = 2

    while end_idx <= len(x)-1:
        # Fit a line to the points
        slope, intercept, r_value, p_value, std_err = linregress(x[start_idx:end_idx], y[start_idx:end_idx])
        
        checko = df.loc[start_idx:end_idx, 'animalmoving'].all()
        print(f'start: {start_idx}, end: {end_idx}, r: {r_value**2}, {checko}')

        # Check if R-squared value above threshold and animal moving
        if r_value**2 >= rvalue_threshold and df.loc[start_idx:end_idx, 'animalmoving'].all():
            end_idx += 1

        else:
            segment_dist = euclidian_dist(x[start_idx], y[start_idx], x[end_idx], y[end_idx])
            print(f'STOP frames:{end_idx-start_idx}, dist:{segment_dist}')

            if end_idx - start_idx >= frames_threshold and segment_dist >= dist_threshold_mm:
                segments.append((start_idx, end_idx - 1))

                start_idx = end_idx
                end_idx += 2
            else:
                start_idx = end_idx
                end_idx += 2

    # If the last segment continues till the end of the series
    if end_idx-1 == len(x) and end_idx - start_idx >= frames_threshold:
        segments.append((start_idx, end_idx-1))

    print(segments)
    return segments

def catwalk_angle(df, min_vector_length, straight_threshold, min_catwalk_length):
    # get list of start- and stop-frames and new dataframe, where the animal is moving straight according to df_center
    
    # calculate the distance (px) between the k frame and the k-x frame
    center_diff_x = df['center_x_shifted'] - df['center_x']
    center_diff_y = df['center_y_shifted'] - df['center_y']

    # check at which frames the vector magnitude is above x
    vector_length = np.sqrt(center_diff_x**2 + center_diff_y**2)
    does_move = vector_length >= min_vector_length
    

    # # checks at which frames, the angle difference is below x
    # angles = np.arctan2(changes['pos_x'], changes['pos_y'])     # angle between the vector (0,0) to (1,0) and the vector (0,0) to (x2,y2)
    # angle_diff = abs(angles.diff(periods=1))
    # does_straight = angle_diff <= max_angle_diff



    # check that not gay
    straight_segments = find_straight_line_segments(df['center_x'], df['center_y'], straight_threshold)
    does_straight = does_move.copy()
    does_straight[:] = False
    for start, end in straight_segments:
        does_straight[start:end] = True

    # combine both True/False series
    does_catwalk = does_move & does_straight


    df['does_move'] = does_move
    df['does_straight'] = does_straight
    df['does_catwalk'] = does_catwalk
    
    # get True on- and offset
    catwalk_timestamps = timestamps_onsets(does_catwalk, min_catwalk_length, onset_cond=True)

    print(straight_segments)
    print(catwalk_timestamps)

    return df, catwalk_timestamps

def plot_bodyparts(timewindow, df, spec_bodyparts, bodyparts, title):

    for start, end in timewindow:

        # only include rows that are in the time window
        rows2plot = df.loc[start : end]

        i_color = 0
        if 'center' in spec_bodyparts:
            # plot animal center via removing the nan values
            x = rows2plot['center_x'][~np.isnan(rows2plot['center_x'])]
            y = rows2plot['center_y'][~np.isnan(rows2plot['center_y'])]
            plt.scatter(x, y, label=f'{start/fps}_center_{end/fps}', s=0.2)#, c='grey')
        
        if 'static_feet' in spec_bodyparts:
            # plot static feet
            paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
            for paw in paws:
                plt.scatter(rows2plot[f'static_{paw}_x'], rows2plot[f'static_{paw}_y'], label=f'static_{paw}', c=colors[i_color], marker='x', s=60)
                
                # plot raw feet
                if 'raw_feet' in spec_bodyparts:
                    plt.scatter(rows2plot[f'{paw}_x'], rows2plot[f'{paw}_y'], label=paw, c=colors[i_color], marker='.', s=10)
                
                i_color += 1
        
        for part in bodyparts:
            plt.scatter(rows2plot[f'{part}_x'], rows2plot[f'{part}_y'], label=part, c=colors[i_color], marker='+', s=20)
            i_color += 1

    
    plt.xlim(0, x_pixel)
    plt.ylim(0, y_pixel)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    #plt.legend()
    plt.title(title)
    plt.show()


# unused functions
def find_next_float_number(df, column, start_row, dir_higher=True):
    # returns the index of the next float number while skipping NaNs (in either direction)
    column_number = df.columns.get_loc(column)
    
    # check the next float value towards higher rows
    if dir_higher == True:
        row_number = start_row + 1
        while row_number < len(df):
            value = df.iloc[row_number, column_number]
            if not np.isnan(value):
                return row_number
            row_number += 1
        return None
    
    # check the next float value towards lower rows
    else:
        row_number = start_row - 1
        while row_number >= 0:
            value = df.iloc[row_number, column_number]
            if not np.isnan(value):
                return row_number
            row_number -= 1
        return None
def add_higherlower_float_row(df, index_list, column):
    # get index_list with >= 1 index values and add indices with column_floats above & below

    low_row = df.index.get_loc(index_list[0])
    lower_row = find_next_float_number(df, column, low_row, dir_higher=False)

    high_row = df.index.get_loc(index_list[-1])
    higher_row = find_next_float_number(df, column, high_row, dir_higher=True)
    
    index_list.append(lower_row)
    index_list.append(higher_row)

    return index_list
def staticpaw2midline_dist(df, static_window, midline):
    # get distance from static paw to midline

    # get all 'pos' float values (not nan) in the frame_window ad add index to list
    pos_in_static = df.loc[static_window[0] : static_window[1], f'{midline}_x'].copy().dropna()
    pos_in_static_idx = pos_in_static.index.tolist()


    # if there are no pos_coords in the frames_window of static_feet, add a middle frame
    if len(pos_in_static_idx) == 0:
        pos_in_static_idx.append(int(np.mean(static_window)))

    # if only 2 or less pos_coords, add one above and below until we have >= 3
    while len(pos_in_static_idx) < 3:
        pos_in_static_idx = add_higherlower_float_row(df, pos_in_static_idx, f'{midline}_x')

    print(f'relevant_pos_frames: {pos_in_static_idx}')


# collect data functions
def getdata_paw2centerline(df, timewindows):
    # collects the distances from moving paws to the intrinsic centerline in each frame and returns the means etc for each paw

    paw2line_info = ['dist_VL2centerline', 'dist_VR2centerline', 'dist_HL2centerline','dist_HR2centerline']

    paw2line_data = []
    for paw in paw2line_info:
        this_paw = pd.concat([df.loc[start : end, paw] for start, end in timewindows]).tolist()
        this_paw = [x for x in this_paw if ~np.isnan(x)]
        paw2line_data.append(this_paw)

    data_mean = [round(np.mean(paw), 3) for paw in paw2line_data]
    data_std = [round(np.std(paw), 3) for paw in paw2line_data]
    data_n = [len(paw) for paw in paw2line_data]

    return data_mean, data_std, data_n

def getdata_staticpaw2centerline(df, timewindows):
    # for each window, calculate the distance between each staticpaw and the centerline in the frame of the first staticpaw data

    static_paws = ['static_paw_VL', 'static_paw_VR', 'static_paw_HL', 'static_paw_HR']
    # one list for each paw
    staticpaw2centerline_data = [[], [], [], []]

    for i, paw in enumerate(static_paws):
        for start, end in timewindows:
            for idx in range(start+1, end+1):
                prev_value_x = df.loc[idx-1, f'{paw}_x']
                curr_value_x = df.loc[idx, f'{paw}_x']

                # with this, it only adds the first value of a set of the same values for the respective paw while skipping nans
                if curr_value_x != prev_value_x and not np.isnan(curr_value_x):

                    centerline_slope = df.loc[curr_value_x, 'centerline_slopes']
                    centerline_intercept = df.loc[curr_value_x, 'centerline_intercepts']

                    # directly calculating the distance of this static_paw to centerline
                    staticpaw2centerline = point2line_dist(centerline_slope, centerline_intercept, curr_value_x, df.loc[idx, f'{paw}_y'])
                    
                    
                    staticpaw2centerline_data[i].append(staticpaw2centerline)

    data_mean = [round(np.mean(paw), 3) for paw in staticpaw2centerline_data]
    data_std = [round(np.std(paw), 3) for paw in staticpaw2centerline_data]
    data_n = [len(paw) for paw in staticpaw2centerline_data]

    return data_mean, data_std, data_n

def output_data():
    with open(f'{path}output.txt', 'a') as f:
        f.write(f'common_name: {common_name}\n'
                
                f'total_distances_travelled: {travelled_distances}\n'

                f'paw2centerlinecat_mean [[VL, VR, HL, HR], ...]: {paw2centerlinecat_mean}\n'
                f'paw2centerlinecat_std [[VL, VR, HL, HR], ...]: {paw2centerlinecat_std}\n'
                f'paw2centerlinecat_n [[VL, VR, HL, HR], ...]: {paw2centerlinecat_n}\n'

                f'staticpaw2centerlinecat_mean [[VL, VR, HL, HR], ...]: {staticpaw2centerlinecat_mean}\n'
                f'staticpaw2centerlinecat_std [[VL, VR, HL, HR], ...]: {staticpaw2centerlinecat_std}\n'
                f'staticpaw2centerlinecat_n [[VL, VR, HL, HR], ...]: {staticpaw2centerlinecat_n}\n'
        )


                
#%%
# DO STUFF



paw2centerlinecat_mean, paw2centerlinecat_std, paw2centerlinecat_n = [], [], []
staticpaw2centerlinecat_mean, staticpaw2centerlinecat_std, staticpaw2centerlinecat_n = [], [], []
travelled_distances = []

# ideas:
# total static paw intervals during catwalk (all paws)
# 

for file in file_list:
    print(f'\n{file}')
    main_df = pd.read_csv(file, header=None, low_memory=False)
    main_df = cleaning_raw_df(main_df)
    main_df = add_basic_columns_to_df(main_df)
    main_df = static_feet(main_df)

    #main_df, moving_segments = animal_moving(main_df)
    #catwalk_segments = catwalk_regress(main_df)

#plot_bodyparts([[658,667]], main_df, ['center'], [], 'straight, fast and long Catwalks (animal center)')

#%%



plt.scatter(main_df.loc[600:700, 'center_x'], main_df.loc[600:700, 'center_y'], s=1, label='orig')
plt.scatter(main_df.loc[600:700, 'center_x_smooth'], main_df.loc[600:700, 'center_y_smooth'], s=1, label='savgol')
plt.scatter(main_df.loc[600:700, 'center_x_avrg'], main_df.loc[600:700, 'center_y_avrg'], s=1, label='avrg')


plt.legend()
plt.show()

#plot_bodyparts([[601,720]], main_df, ['center'], [], 'straight, fast and long Catwalks (animal center)')
#plot_bodyparts(catwalk_segments, main_df, ['center'], [], 'straight, fast and long Catwalks (animal center)')


#%%

# total distance travelled
travelled_distances.append(round(main_df['dist_center_shifted'].sum(), 3))

# paw2centerlinecat
mean, std, n = getdata_paw2centerline(main_df, catwalk_segments)
paw2centerlinecat_mean.append(mean)
paw2centerlinecat_std.append(std)
paw2centerlinecat_n.append(n)

# staticpaw2centerline
mean, std, n = getdata_staticpaw2centerline(main_df, catwalk_segments)
staticpaw2centerlinecat_mean.append(mean)
staticpaw2centerlinecat_std.append(std)
staticpaw2centerlinecat_n.append(n)









# PLOT
# (list of [start, stop], df, special_bodyparts=['center', 'static_feet', 'raw_feet'], bodypart=[e.g. 'snout', 'mouth'], title of plot)
plot_bodyparts(catwalk_segments, main_df, ['center'], [], 'straight, fast and long Catwalks (animal center)')
plot_bodyparts([0,200], main_df, ['center'], [], 'straight, fast and long Catwalks (animal center)')


output_data()


#%%



#%%
# STATISTICS

# def add_text_to_bars(x, txt):
#     for i in range(len(x)):
#         plt.text(i, 0, f'n={txt[i]}', ha='center', size='small')


# plt.bar(paw2line_info_short, data_mean, width=0.5)
# plt.errorbar(paw2line_info_short, data_mean, yerr=data_std, fmt="o", color="r")


# add_text_to_bars(paw2line_info_short, data_n)



# plt.xlabel('paws')
# plt.ylabel('distance [mm]')
# plt.title('dist_paw2midline')
# plt.show()




print(main_df.loc[25:30, 'animalmoving'].all())