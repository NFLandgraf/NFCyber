#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import re
from collections import defaultdict
import os

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
    center = (575, 575)
            
    return holes, center

def define_bodyparts():

    bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

    bps_relevant = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 
                'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    
    bps_head = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint']

    return bps_all, bps_relevant, bps_head

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
    # bodyparts = sorted(set(c[:-2] for c in df.columns if c.endswith(("_x", "_y"))))
    # df_tuples = pd.DataFrame(index=df.index)
    # for bp in bodyparts:
    #     df_tuples[bp] = list(zip(df[f"{bp}_x"], df[f"{bp}_y"]))
                
    return df

def rainbow_tract(image, main_df, file, x_col="head_midpoint_x", y_col="head_midpoint_y", linewidth=3.0, alpha=0.95, cmap_name='rainbow', downsample=1, timestamps=None, mark_start_end=True):
    
    output_file = r"C:\Users\landgrafn\Desktop\a\Aq1_1035_traj.pdf"
    bg = plt.imread(image) if isinstance(image, str) else image

    x = main_df[x_col].to_numpy()[::downsample]
    y = main_df[y_col].to_numpy()[::downsample]

    idx = np.arange(len(x))
    t = (idx - idx.min()) / max(1, (idx.max() - idx.min()))

    # build segments (N-1 segments between consecutive points)
    pts = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 2)

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg, origin='upper')
    ax.set_axis_off()
    ax.set_xlim([0, bg.shape[1]])
    ax.set_ylim([bg.shape[0], 0])  # image coords: y downward

    lc = LineCollection(segs, colors='black', linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)

    # start/end markers (optional)
    if mark_start_end and len(pts) > 0:
        ax.scatter([x[0]], [y[0]], s=60, edgecolor='black', facecolor='purple', linewidth=1.5)
        ax.scatter([x[-1]], [y[-1]], s=60, edgecolor='black', facecolor='red', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


file = r"C:\Users\landgrafn\Desktop\a\Aq1_1035_trim_mask_mask_DLC.csv"
holes, center = define_areas()
bps_all, bps_relevant, bps_head = define_bodyparts()
main_df = cleaning_raw_df(file, bps_all)
rainbow_tract(r"C:\Users\landgrafn\Desktop\a\BM.png", main_df, file)

#%%

def draw_hole_map(image, holes, radius, radius_center, letter_thickness=2, circle_thickness=2):
    """
    image: np.ndarray (BGR, as loaded by cv2) or a filepath string
    holes: dict like {'A': (x,y), ...}
    radius: circle radius to draw around each hole (pixels)
    show_circles: whether to draw the radius circles
    """
    # Load image if a path is given
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image from path: {image}")
    else:
        img = image.copy()

    for letter, (x, y) in holes.items():
        center = (int(x), int(y))

        if letter == 'center':
            cv2.circle(img, center, int(radius_center), (0, 255, 255), circle_thickness, lineType=cv2.LINE_AA)

        if letter != 'center':

            # Optional circle showing the hole radius
            cv2.circle(img, center, int(radius), (0, 255, 255), circle_thickness, lineType=cv2.LINE_AA)

            # Draw a small dot at the center
            cv2.circle(img, center, 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)

            # Put the letter slightly offset so it doesn't overlap the center
            cv2.putText(img, letter, (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), letter_thickness, cv2.LINE_AA)

    return img

holes = define_areas()
img = cv2.imread(r"D:\BM\a.png")
hole_map = draw_hole_map(img, holes, radius=55, radius_center= 410)
cv2.imwrite(r"D:\BM\b.png", hole_map)


