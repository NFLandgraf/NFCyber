'''
This program will take a DLC file and based on point(s), will calculate the distance travelled in certain time bins
'''

import numpy as np
import pandas as pd
from pathlib import Path


freq = 10
x_max, y_max = 308, 322     # video dimensions in px

path = 'C:\\Users\\landgrafn\\Desktop\\FF_FS_DLC\\'

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files

def get_behav(file):
    # cleans the DLC file and returns the df
    # idea: instead of picking some 'trustworthy' points to mean e.g. head, check the relations between 'trustworthy' points, so when one is missing, it can gues more or less where it is

    bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    bps_head = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
    bps_postbody = ['mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base']

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
    
    # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
    for c in ('_x', '_y'):
        df[f'head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
        df[f'postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

    # combine the x and y column into a single tuple column
    for bp in bps_all + ['head', 'postbody']:
        df[bp] = (list(zip(round(df[f'{bp}_x'], 1), round(y_max-df[f'{bp}_y'], 1))))
        df = df.drop(labels=f'{bp}_x', axis="columns")
        df = df.drop(labels=f'{bp}_y', axis="columns")   

    #df.to_csv('F:\\2024-10-31-16-52-47_CA1-m90_OF_DLC_clean.csv')
    df = df[['head', 'postbody']]

    return df




