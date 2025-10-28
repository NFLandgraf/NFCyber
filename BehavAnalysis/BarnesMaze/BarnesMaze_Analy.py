#%%
# IMPORT & DEFINE
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


path = r"D:\BM\\"
common_name_csv = '.csv'
px_per_mm = 0.909
fps = 25


def get_files(path, common_name):
    
    files = []
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        if os.path.isfile(full_path) and common_name in fname:
            files.append(full_path)

    print(f"{len(files)} files found\n")
    for f in files:
        print(f)
    print()
    return files

def extract_animal_id(filename):
    # Extract a 4-digit animal ID when it appears as _XXXX_ in a filename string.

    m = re.search(r'_(\d{4})_', filename)
    if m:
        return m.group(1)
    
    return None

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
    
    output_file = path + Path(file).stem + "_traj.png"
    bg = plt.imread(image) if isinstance(image, str) else image

    x = main_df[x_col].to_numpy()[::downsample]
    y = main_df[y_col].to_numpy()[::downsample]

    # time base for coloring
    if timestamps is not None:
        t = np.asarray(timestamps)[::downsample]
        t = (t - t.min()) / max(1e-12, (t.max() - t.min()))
    else:
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

    lc = LineCollection(segs, cmap=plt.get_cmap(cmap_name), array=t[:-1], linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)

    # start/end markers (optional)
    if mark_start_end and len(pts) > 0:
        ax.scatter([x[0]], [y[0]], s=60, edgecolor='black', facecolor='purple', linewidth=1.5)
        ax.scatter([x[-1]], [y[-1]], s=60, edgecolor='black', facecolor='red', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def annotate_video_with_letters(file, letters, holes, font_scale=3, thickness=6, color=(0,0,255), skip_value='nothing'):
    
    video_file = path + Path(file).stem + ".mp4"
    output_file = path + Path(file).stem + "_move.mp4"

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(letters) < total_frames:
        print(f"⚠️ Warning: letters list has fewer entries ({len(letters)}) than frames ({total_frames}). Missing frames will be left unannotated.")
    elif len(letters) > total_frames:
        letters = letters[:total_frames]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        for letter, (x, y) in holes.items():
            center = (int(x), int(y))
            if letter == 'center':
                cv2.circle(frame, center, int(410), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            if letter != 'center':
                cv2.circle(frame, center, int(55), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, letter, (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        letter = letters[frame_idx] if frame_idx < len(letters) else skip_value

        if letter != skip_value and not pd.isna(letter):
            text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2

            cv2.putText(frame, letter, (text_x, text_y), font, font_scale,
                        color, thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()


def bp_at_which_hole(main_df, holes, center, bps, radius=55, center_radius=410):

    # if bp in below threshold from hole, add holename to new df
    df_holes = pd.DataFrame(index=main_df.index, columns=bps)
    for bp in bps:
        labels = []
        x_col = f'{bp}_x'
        y_col = f'{bp}_y'

        for x, y in zip(main_df[x_col], main_df[y_col]):
            if pd.isna(x) or pd.isna(y):  # skip NaN frames
                labels.append(np.nan)
                continue

            hole_found = np.nan

            # Check center first, then each hole
            center_x, center_y = center
            dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist_center <= center_radius:
                hole_found = 'center'
                
            else:
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

def anim_at_which_hole(main_df, df_holes, threshold=0.5):

    # checks each row and saves the most abundant hole (threshold)
    maj_holes = []

    for _, row in df_holes.iterrows():
        counts = row.value_counts(dropna=True)  # count hole names in this row
        if len(counts) == 0:  # no bodyparts in any hole
            maj_holes.append("nothing")
            continue

        # most common hole and its fraction
        hole, count = counts.index[0], counts.iloc[0]
        frac = count / len(row)

        if frac >= threshold:
            maj_holes.append(hole)
        else:
            maj_holes.append("nothing")
        
    main_df['maj_hole'] = maj_holes
    maj_holes_collapse = [x for i, x in enumerate(maj_holes) if i == 0 or x != maj_holes[i-1]]

    return main_df, maj_holes, maj_holes_collapse

def right_hole(main_df, corr_hole='A'):
    
    maj_hole = main_df["maj_hole"]

    # how long until correct hole
    corr_frames = maj_hole[maj_hole == corr_hole].index.tolist()
    corr_frame = min(corr_frames)
    corr_frame_s = round(corr_frame/fps, 1)

    # how many holes visited before the correct one, without center (but revisiting hole after center counts)
    visited_holes = [key for key, _ in groupby(maj_hole)]
    visited_holes = [x for x in visited_holes if x != 'center'] 
    wrong_holes = visited_holes.index(corr_hole)
    
    # what distance travelled before the correct hole
    x_col, y_col = 'tail_base_x', 'tail_base_y'
    coords = main_df.loc[:corr_frame, [x_col, y_col]].dropna().to_numpy()
    dist_until_corr = round(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)).sum() * px_per_mm, 1)     # in mm

    return corr_frame_s, wrong_holes, dist_until_corr
    
def do_analysis(file):

    holes, center = define_areas()
    bps_all, bps_relevant, bps_head = define_bodyparts()
    main_df = cleaning_raw_df(file, bps_all)

    # Holes visited
    df_holes = bp_at_which_hole(main_df, holes, center, bps_head)
    main_df, maj_holes, maj_holes_collapse = anim_at_which_hole(main_df, df_holes)

    # statistics
    corr_frame, wrong_holes, dist_until_corr = right_hole(main_df)

    # create check
    # rainbow_tract(r"D:\BM\canvass.png", main_df, file)
    # annotate_video_with_letters(file, maj_holes, holes)

    return corr_frame, wrong_holes, dist_until_corr




csv_files = get_files(path, common_name_csv)
results_by_animal = defaultdict(list)   # animal_id -> list of (latency_s, wrong_holes, dist_mm)

for file_path in csv_files:

    print(file_path)
    animal_id = extract_animal_id(file_path)

    corr_frame, wrong_holes, dist_until_corr = do_analysis(file_path)
    metrics = (corr_frame, wrong_holes, dist_until_corr)

    results_by_animal[animal_id].append(metrics)

    print('\n')


for aid, vals in results_by_animal.items():
        print(aid, "->", vals)



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


#%%

def annotate_video_with_letters(video_path, output_path, letters, holes, font_scale=3, thickness=6, color=(0,0,255), skip_value='nothing'):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(letters) < total_frames:
        print(f"⚠️ Warning: letters list has fewer entries ({len(letters)}) than frames ({total_frames}). Missing frames will be left unannotated.")
    elif len(letters) > total_frames:
        letters = letters[:total_frames]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_idx in tqdm(range(total_frames), desc="Annotating video"):
        ret, frame = cap.read()
        if not ret:
            break

        for letter, (x, y) in holes.items():
            center = (int(x), int(y))
            if letter == 'center':
                cv2.circle(frame, center, int(410), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            if letter != 'center':
                cv2.circle(frame, center, int(55), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, letter, (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        letter = letters[frame_idx] if frame_idx < len(letters) else skip_value

        if letter != skip_value and not pd.isna(letter):
            text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2

            cv2.putText(frame, letter, (text_x, text_y), font, font_scale,
                        color, thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Saved annotated video to {output_path}")

holes = define_areas()
video_path = r"D:\BM\Aq1_1006_trim_mask_mask_DLC.mp4"
output_path = r"D:\BM\Aq1_1006_trim_mask_mask_DLC_annot.mp4"
annotate_video_with_letters(video_path, output_path, maj_holes, holes)





