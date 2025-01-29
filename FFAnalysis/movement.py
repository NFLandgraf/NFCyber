#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# IMPORT & DEFINE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter
import os

     
path = 'C:\\Users\\nicol\\Desktop\\FC_speed\\'
common_name = 'filtered.csv'

fps = 10

bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

bps_position = 'mid_backend'

def get_files(path):
    # get files
    files = [path+file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    print(f'{len(files)} files found\n')

    return files
files = get_files(path)

#%%

def distances2(csv_file, min_distance=0.0001):

    def cleaning_raw_df(csv_file):
        df = pd.read_csv(csv_file, header=None, low_memory=False)

        # Create new column names from first two rows
        new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]
        df.columns = new_columns
        df = df.drop(labels=[0, 1, 2], axis="index")

        # Set index and convert to numeric
        df.set_index('bodyparts_coords', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(int)

        # Remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_likelihood'] <= 0.9
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_likelihood')

        return df
    
    def euclidian_dist(point1x, point1y, point2x, point2y):
        return np.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)

    df = cleaning_raw_df(csv_file)
    x_col, y_col = f'{bps_position}_x', f'{bps_position}_y'
    
    # Create a full index range (to include missing frames)
    full_index = np.arange(df.index.min(), df.index.max() + 1)
    dist_series = pd.Series(index=full_index, dtype=float)  # Initialize full index with NaNs

    last_valid_index = None  # Track last valid (non-NaN) frame

    for i in df.index:
        if not np.isnan(df.at[i, x_col]) and not np.isnan(df.at[i, y_col]):  # If current row has valid x, y
            if last_valid_index is not None:
                dist = euclidian_dist(
                    df.at[last_valid_index, x_col], df.at[last_valid_index, y_col], 
                    df.at[i, x_col], df.at[i, y_col]
                )
                # Set to NaN if distance is below threshold
                dist_series[i] = dist if dist >= min_distance else np.nan  
            last_valid_index = i  # Update last valid frame

    # Convert to DataFrame
    dist_df = dist_series.to_frame(name=f'{bps_position}_dist')

    return dist_df


for file in files:
    dist = distances2(file)
    dist_sum = round(dist.sum(), 1)
    print(dist)
    print(dist_sum)



    
    
