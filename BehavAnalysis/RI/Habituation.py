#%%
import os
from matplotlib import pyplot as plt 
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np




path = 'D:\\Habit\\'
file_format = '.csv'

def get_data():
    # get all file names in directory into list
    files = [file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                file_format in file]
    print(f'{len(files)} files found in path directory {path}\n'
        f'{files}\n')
    
    return files
files = get_data()

bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                    'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                    'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                    'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

#%%


def clean_behav(file):
    # cleans the DLC file and returns the df
    # idea: instead of picking some 'trustworthy' points to mean e.g. head, check the relations between 'trustworthy' points, so when one is missing, it can gues more or less where it is
    bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

    df = pd.read_csv(file, header=None, low_memory=False)

    # combines the string of the two rows to create new header
    # df.iloc[0] takes all values of first row
    new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     
    df.columns = new_columns
    df = df.drop(labels=[0, 1, 2], axis="index")

    # adapt index
    df.set_index('bodyparts_coords', inplace=True)
    df.index.names = ['Frame']

    # turn all the prior string values into floats and index into int
    df = df.astype(float)
    df.index = df.index.astype(int)

    # insert NaN if likelihood is below x
    for bp in bps_all:
        filter = df[f'{bp}_likelihood'] <= 0.9     
        df.loc[filter, f'{bp}_x'] = np.nan
        df.loc[filter, f'{bp}_y'] = np.nan
        df = df.drop(labels=f'{bp}_likelihood', axis="columns")

    df = df[['tail_base_x', 'tail_base_y', 'mouse_center_x', 'mouse_center_y']]

    df = df.interpolate(method='linear', axis=0, limit_direction='both')

    df = df.rolling(window=5, min_periods=1, center=True).mean()

    df['tail_base_dist'] = np.sqrt(df['tail_base_x'].diff()**2 + df['tail_base_y'].diff()**2)
    df['mouse_center_dist'] = np.sqrt(df['mouse_center_x'].diff()**2 + df['mouse_center_y'].diff()**2)

    
    return df

def open_field(df):

    def is_point_inside(row):
        point = Point(row['mouse_center_x'], row['mouse_center_y'])
        return polygon.contains(point)

    # inner rechteck for 800x480
    polygon = Polygon([
        (180, 160),
        (640, 160),
        (180, 320),
        (640, 320)])

    df['in_polygon'] = df.apply(is_point_inside, axis=1)

    n_inside = int(df['in_polygon'].sum())
    n_outside = int(len(df) - n_inside)

    return n_inside, n_outside



for file in files:
    print(file)
    input_file = path + file

    df = clean_behav(input_file)

    total_dist = int(df['tail_base_dist'].sum())
    n_inside, n_outside = open_field(df)



    with open("D:\\Habit_Summary.txt", "a") as f:
        f.write(f'{file}\n'
                f'{total_dist}, {n_inside}, {n_outside}\n')
    
    print(f'{total_dist}, {n_inside}, {n_outside}')
    





