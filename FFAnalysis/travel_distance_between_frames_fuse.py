#%% 
# IMPORT & DEFINE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

     
path = 'C:\\Users\\landgrafn\\Desktop\\ffff\\'
output_path = 'C:\\Users\\landgrafn\\Desktop\\ffff\\'
common_name = '.csv'
common_name_ff = 'LockIn'

# behav definitions
common_name_behav = 'filtered'
mm_per_px = 1

def get_files(path, common_name):
    # get files
    files = [path+file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    for file in files:
        print(file)
    print(f'{len(files)} files found\n')

    return files
files = get_files(path, common_name)

def get_distances(csv_file, mm_per_px, bps_position='mid_backend', min_distance=0.001):
    # calculates the distance travelled between frames, where 

    def cleaning_raw_df(csv_file):
        df = pd.read_csv(csv_file, header=None, low_memory=False)

        # create new column names from first two rows
        new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]
        df.columns = new_columns
        df = df.drop(labels=[0, 1, 2], axis="index")

        # set index and convert to numeric
        df.set_index('bodyparts_coords', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(int)

        bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

        # remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_likelihood'] <= 0.9
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_likelihood')

        return df
    
    def euclidian_dist(point1x, point1y, point2x, point2y):
        # calculates distance between 2 points in carthesic coordinate system
        return np.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)

    df = cleaning_raw_df(csv_file)
    x_col, y_col = f'{bps_position}_x', f'{bps_position}_y'
    
    # creates a full index range (to include missing frames), fill Series with NaNs
    full_index = np.arange(df.index.min(), df.index.max() + 1)
    dist_series = pd.Series(index=full_index, dtype=float)

    # goes though all frames/rows and calcs distance between the frames with values (not NaN)
    last_valid_idx = None
    for i in df.index:
        if not np.isnan(df.at[i, x_col]) and not np.isnan(df.at[i, y_col]):
            if last_valid_idx is not None:
                dist = euclidian_dist(df.at[last_valid_idx, x_col], df.at[last_valid_idx, y_col], df.at[i, x_col], df.at[i, y_col])
                dist_series[i] = dist if dist >= min_distance else np.nan  
            last_valid_idx = i
    dist_travelled = round(dist_series.sum(), 1)

    # goes through the frames and distributes the distance evenly across NaNs
    nan_indices = np.where(dist_series.isna())[0]
    if len(nan_indices) > 0:
        last_valid_idx = None
        for i in range(len(dist_series)):
            if not np.isnan(dist_series[i]):  
                if last_valid_idx is not None:
                    num_missing = i - last_valid_idx - 1  # Frames between valid distances
                    if num_missing > 0:
                        total_dist = dist_series[i]  # Distance to be spread
                        per_frame_dist = total_dist / (num_missing + 1)
                        for j in range(1, num_missing + 2):
                            dist_series[last_valid_idx + j] = per_frame_dist
                last_valid_idx = i
    dist_series[0] = 0.0
    dist_travelled_distr = round(dist_series.sum(), 1)
    
    if dist_travelled != dist_travelled_distr:
        print(f'Error: different distances calculated: {dist_travelled} vs {dist_travelled_distr}')
    
    # convert to mm and to DataFrame
    dist = dist * mm_per_px
    dist_df = dist_series.to_frame(name=f'{bps_position}_dist')

    # print(f'distance travelled: {round(dist_df.sum(), 1)}')
    # plt.plot(dist_df)
    # plt.show()

    frame_nb = dist_df.shape[0]
    dist_df.to_csv(f'{output_path}_distances.csv')

    return dist_df, frame_nb

def add_distances_to_maindf(file_ff, df_dist):

    # Take a column from the main_df and a df of the movements between frames
    # Now in which row here is a one in the Cam_TTL_column, add the corresponding travel distance

    main_df = pd.read_csv(file_ff, index_col="Time", low_memory=False)

    timepoint_frame = main_df.index[main_df['IO2'] == 1]
    frameTTL_nb = len(timepoint_frame)
    main_df['distances'] = np.nan
    main_df.loc[timepoint_frame, 'distances'] = df_dist.iloc[:len(timepoint_frame)].values.flatten()

    return main_df, frameTTL_nb


for f in range(int(len(files)/2)):
    
    # get the files that belong together
    file_behav = files[f*2]
    file_ff = files[f*2+1]
    print(file_behav)
    print(file_ff)


    df_dist, frame_nb = get_distances(file_behav, mm_per_px)
    main_df, frameTTL_nb = add_distances_to_maindf(file_ff, df_dist)
    
    print(f'Frame Number: {frame_nb}\n'
          f'Frame_TTL nb: {frameTTL_nb}')

    main_df.to_csv(f'{output_path}_fused.csv')


    
    
