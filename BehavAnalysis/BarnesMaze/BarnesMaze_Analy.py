#%%
# IMPORT & DEFINE
import pandas as pd
import numpy as np
#from shapely.geometry import Polygon, Point
from tqdm import tqdm
#import cv2
from pathlib import Path

# USER INPUT
path = r"D:\BM"
common_name_csv = '.csv'

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files

def define_areas():
    holes = {
        'A': (526, 64),
        'B': (683, 74),
        'C': (829, 130),
        'D': (950, 227),
        'E': (1037, 356),
        'F': (1082, 506),
        'G': (1077, 664),
        'H': (1024, 813),
        'I': (929, 938),
        'J': (797, 1029),
        'K': (644, 1075),
        'L': (482, 1070),
        'M': (329, 1015),
        'N': (202, 914),
        'O': (112, 777),
        'P': (70, 620),
        'Q': (80, 459),
        'R': (139, 310),
        'S': (241, 188),
        'T': (374, 104)
        }
            
    return holes

def define_bodyparts():

        bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                    'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                    'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                    'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

        bps_relevant = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                    'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 
                    'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
        
        bps_head = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                    'neck', 'left_shoulder', 'right_shoulder']

        return bps_all, bps_relevant, bps_head

def cleaning_raw_df(csv_file):
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

            # interpolate to skip nans
            df = df.interpolate(method="linear")
            
            # make one column (x,y) per bodypart
            bodyparts = sorted(set(c[:-2] for c in df.columns if c.endswith(("_x", "_y"))))
            df_tuples = pd.DataFrame(index=df.index)
            for bp in bodyparts:
                df_tuples[bp] = list(zip(df[f"{bp}_x"], df[f"{bp}_y"]))
                        
            return df_tuples

def bp_at_which_hole(df, holes, bps, radius=35):

    # if bp in below threshold from hole, add holename to new df
    df_holes = pd.DataFrame(index=df.index, columns=bps)
    for bp in df.columns:
        labels = []
        for coords in df[bp]:
            if pd.isna(coords):  # skip NaN frames
                labels.append(np.nan)
                continue

            x, y = coords
            hole_found = np.nan
            for hole_name, (hx, hy) in holes.items():
                dist = np.sqrt((x - hx)**2 + (y - hy)**2)
                if dist <= radius:
                    hole_found = hole_name
                    break  # stop at first hole
            labels.append(hole_found)
        df_holes[bp] = labels
    
    # removes nans and filles it with prior hole names
    df_holes_ffilled = df_holes.ffill()

    return df_holes_ffilled

def anim_at_which_hole(df_holes, threshold=0.5):

    # checks each row and saves the most abundant hole (threshold)
    majority_holes = []

    for _, row in df_holes.iterrows():
        counts = row.value_counts(dropna=True)  # count hole names in this row
        if len(counts) == 0:  # no bodyparts in any hole
            majority_holes.append("nothing")
            continue

        # most common hole and its fraction
        hole, count = counts.index[0], counts.iloc[0]
        frac = count / len(row)

        if frac >= threshold:
            majority_holes.append(hole)
        else:
            majority_holes.append("nothing")
        
    df_holes['majority_hole'] = majority_holes

    return df_holes, majority_holes

def right_hole(df, df_holes, corr_hole='D'):
    
    maj_hole = df_holes["majority_hole"]

    corr_frames = maj_hole.index[corr_hole].tolist()
    corr_frame = min(corr_frames)

    # how many holes visited before the correct one
    visited_holes = []
    prev = object()  # placeholder
    for item in maj_hole:
        if item != prev:
            visited_holes.append(item)
        prev = item
    wrong_holes = visited_holes.index(corr_hole)
    
    # what distance travelled before the correct hole
    col_animalposi = 'tail_base'
    coords = df.loc[:corr_frame, col_animalposi].dropna().to_numpy()
    dist_until_corr = 0.0
    for i in range(1, len(coords)):
        x1, y1 = coords[i-1]
        x2, y2 = coords[i]
        dist_until_corr += np.sqrt((x2-x1)**2 + (y2-y1)**2)

    return corr_frame, wrong_holes, dist_until_corr
    

     




csv_files = get_files(path, common_name_csv)
for file in csv_files:

    # Prep
    holes = define_areas()
    bps_all, bps_relevant, bps_head = define_bodyparts()
    df = cleaning_raw_df(file)

    # Holes visited
    df_holes = bp_at_which_hole(df, holes, bps_head)
    df_holes, maj_holes = anim_at_which_hole(df_holes)

    # Get data
    corr_frame, wrong_holes, dist_until_corr = right_hole(df, df_holes)
    
    

    print(df_holes)


