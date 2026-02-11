#%%
# IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from pathlib import Path
import os
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon
from matplotlib.path import Path as MplPath   # <-- IMPORTANT (not pathlib.Path)
from scipy.stats import ttest_ind


folder_in = Path(r"D:\Analy_Post2")
df_QC = pd.read_csv(r"D:\df_QC_FINAL_pass.csv", index_col='cell_ID')

def get_files(path, common_name1, common_name2, suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name1 in p.name) and (common_name2 in p.name) and p.name.endswith(suffix)])
    #print(f"\n{len(files)} files found")
    #for f in files:
    #    print(f)
    #print('\n\n')
    return files

def clean_df(main_df, df_QC, start_row=34, maxYMOF = [15,20]):

    # do QC according to pass True or False
    cols_to_keep = []
    for col in main_df.columns:
        # only apply QC to neuron columns
        if "_C" not in col:
            cols_to_keep.append(col)
            continue

        # extract cell name 
        name = col.rsplit("_", 1)[0]

        # if the name is not in pass_df, be conservative and drop
        if name not in df_QC.index:
            print('name not in pass_df')
            continue

        # keep only if pass == True
        if df_QC.loc[name, "pass"]:
            cols_to_keep.append(col)
    
    # drop columns that didnt pass the QC
    cells_dropped = int((len(main_df.columns) - len(cols_to_keep)) /3)
    main_df = main_df[cols_to_keep]
    #print(f'{cells_dropped} cells dropped due to QC -> now {int(len(cols_to_keep)/3)} cells')

    # trim main_df according to distance travelled
    name = col.rsplit("_", 1)[0]
    main_df = main_df.iloc[start_row:]
    if 'YM' in name:
        maxdist = maxYMOF[0] * 1000
    elif 'OF' in name:
        maxdist = maxYMOF[1] * 1000
    else:
        print('Zero cells in recording')
        return None
    
    dx = main_df["head_x"].diff()
    dy = main_df["head_y"].diff()
    step_dist = np.sqrt(dx**2 + dy**2)
    cum_dist = step_dist.fillna(0).cumsum()

    # if never reached max distance, just return full df
    reached = cum_dist >= maxdist
    if not reached.any():
        return main_df
    
    cut_label = cum_dist.index[reached.argmax()]  # first index where reached
    #print(f'reached maxdist after {int(cut_label)}s / {int(max(main_df.index))}s ->trimmed')
    main_df = main_df.loc[:cut_label].copy()

    return main_df

def bin_cellspikes_prob_OF(main_df, n_bins=10, xmax=580, ymax=485):
    """
    From head position and spike probabilities, compute 'spike mass' per spatial bin.
    Instead of summing 0/1 spikes, we sum spike probabilities per bin.
    """

    # grid
    x_edges = np.linspace(0, xmax, n_bins+1)
    y_edges = np.linspace(0, ymax, n_bins+1)

    # assign each frame to a spatial bin
    main_df['x_bin'] = np.clip(np.digitize(main_df['head_x'], bins=x_edges) - 1, 0, n_bins-1)
    main_df['y_bin'] = np.clip(np.digitize(main_df['head_y'], bins=y_edges) - 1, 0, n_bins-1)
    main_df['head_bin'] = main_df['y_bin'] * n_bins + main_df['x_bin']
    main_df = main_df.drop(['x_bin', 'y_bin'], axis=1)

    # choose Cascade probability columns
    spike_prob_cells = [col for col in main_df.columns if isinstance(col, str) and col.endswith('_spikeprob')]

    place_map = {}
    for cell in spike_prob_cells:
        # sum of spike_prob in each bin = expected spike count in that bin
        bin_probs = main_df.groupby('head_bin')[cell].sum()
        place_map[cell] = bin_probs

    place_df = pd.DataFrame(place_map).fillna(0.0)   # keep as float, not int
    place_df = place_df.sort_index()

    return main_df, place_df

def bin_cellspikes_prob_YM(main_df):
    """
    From head position and spike probabilities, compute 'spike mass' per spatial bin.
    Instead of summing 0/1 spikes, we sum spike probabilities per bin.
    """
    df = main_df.copy()
    x_max, y_max = 1000, 1000

    # bin coordinates
    left_corner = (250, 280)
    middle_corner = (290, 340)
    right_corner = (330, 280)

    top_left = (0, 0)
    top_right = (x_max, 0)
    bottom_right = (x_max, y_max)
    bottom_left = (0, y_max)
    wall_bottom = (middle_corner[0], y_max)
    poly_center = np.array([left_corner, middle_corner, right_corner])
    poly_arm_left = np.array([left_corner, middle_corner, wall_bottom, bottom_left, top_left])
    poly_arm_right = np.array([right_corner, middle_corner, wall_bottom, bottom_right, top_right])
    poly_arm_middle = np.array([left_corner, right_corner, top_right, top_left])
    paths = {"center":MplPath(poly_center), "arm_left":MplPath(poly_arm_left), "arm_right":MplPath(poly_arm_right), "arm_middle":MplPath(poly_arm_middle)}
    bin_order = ["center", "arm_left", "arm_right", "arm_middle"]
    bin_ids   = {"center":0, "arm_left":1, "arm_right":2, "arm_middle":3}

    # head in which bin
    pts = df[["head_x", "head_y"]].to_numpy(dtype=float)
    head_bin = np.full(len(df), -1, dtype=int)  # -1 = not in any polygon
    for name in bin_order:
        inside = paths[name].contains_points(pts)
        head_bin[(head_bin == -1) & inside] = bin_ids[name]
    df["head_bin"] = head_bin

    # sum spike probabilities
    spike_prob_cells = [c for c in df.columns if isinstance(c, str) and c.endswith("_spikeprob")]
    place_map = {}
    for cell in spike_prob_cells:
        # sum of spike probability in each bin = expected spike count in that bin
        bin_probs = df.groupby("head_bin")[cell].sum()
        place_map[cell] = bin_probs

    place_df = pd.DataFrame(place_map).fillna(0.0).sort_index()
    return df, place_df

def identify_place_cells_prob(main_df, place_df, fps=10, n_shuffles=100, significance_level=1):
    
    # assumes place_df entries are sums of spike probabilities (expected spike counts)
    
    def spatial_information(firing_freq, occupancy):

        # occupancy: number of frames per bin
        occupancy_probability = occupancy / occupancy.sum()

        # overall mean rate
        avrg_firing_rate = np.dot(firing_freq, occupancy_probability)
        #print(avrg_firing_rate)

        if avrg_firing_rate == 0:
            return 0.0

        info = 0.0
        for r, p in zip(firing_freq, occupancy_probability):
            if r > 0 and p > 0:
                ratio = r / avrg_firing_rate
                info += p * ratio * np.log2(ratio)
        return info

    # real spatial info
    occupancy = main_df.groupby('head_bin').size() # how many frames did animal spend in each bin
    occupancy = occupancy.reindex(place_df.index, fill_value=0) # no idea
    occupancy_sec = occupancy / fps 
    occ_nonzero = occupancy_sec.replace(0, np.nan)  # replacing 0 with nan

    df_firing_freq = place_df.div(occ_nonzero, axis=0).fillna(0.0)  # what was fring frequency in each bin

    spatial_info_real = {}
    for cell in df_firing_freq.columns:
        info = spatial_information(df_firing_freq[cell].values, occupancy.values)
        spatial_info_real[cell] = info

        #if info >= 1:
            #heatmap_cells(main_df, value_col=cell, out_jpg=f"D:\\out\\heatmaps\\cells\\{round(info, 5)}.jpg")


    #return spatial_info_real

    # ---------- shuffled spatial info ----------
    # We’ll keep your occupancy-based shuffle idea but adapt it to probabilities.
    # 1) total_prob ~ expected spike count
    # 2) draw an integer spike count from a Poisson with that mean
    # 3) distribute these spikes across bins according to occupancy probability
    occupancy_probs = occupancy / occupancy.sum()
    spatial_info_shuffled = {cell: [] for cell in place_df.columns}

    for _ in range(n_shuffles):
        for cell in place_df.columns:
            total_prob = place_df[cell].sum()

            # expected number of spikes -> sample integer total spikes
            n_spikes = np.random.poisson(total_prob)

            if n_spikes == 0:
                spatial_info_shuffled[cell].append(0.0)
                continue

            # distribute spikes into bins according to occupancy
            shuffled_bins = np.random.choice(a=occupancy.index, size=n_spikes, p=occupancy_probs.values)
            shuffled_counts = (pd.Series(shuffled_bins).value_counts().reindex(occupancy.index, fill_value=0))

            shuffled_firing_rate = (shuffled_counts / occ_nonzero).fillna(0.0)
            info = spatial_information(shuffled_firing_rate.values, occupancy.values)
            spatial_info_shuffled[cell].append(info)

    # significance test
    place_cells = []
    p_values = {}

    for cell in spatial_info_real:
        real_si = spatial_info_real[cell]
        shuffled_si = np.array(spatial_info_shuffled[cell])
        p = np.mean(shuffled_si >= real_si)
        p_values[cell] = p
        if p < significance_level:
            place_cells.append(cell)

    nb_place_cells, nb_all_cells = len(place_cells), len(spatial_info_real)
    #print(f'Place cells {nb_place_cells} / {nb_all_cells} ({round((nb_place_cells/nb_all_cells)*100, 1)}%)')

    percent_place = nb_place_cells / nb_all_cells

    return spatial_info_real, spatial_info_shuffled, place_cells, p_values, percent_place

def run_animal_percent_place(anim, significance_level):
    files = get_files(folder_in, anim, 'YM')
    percents = []

    for file in files:
        main_df = pd.read_csv(file, index_col='Mastertime [s]')
        main_df = clean_df(main_df, df_QC)
        if main_df is None:
            continue

        main_df, place_df = bin_cellspikes_prob_OF(main_df)

        _, _, _, _, percent_place = identify_place_cells_prob(
            main_df,
            place_df,
            fps=10,
            n_shuffles=100,
            significance_level=significance_level
        )

        percents.append(percent_place)

    if len(percents) == 0:
        return np.nan

    # average across sessions for this animal
    return float(np.mean(percents))


group_A = ['56', '77']
group_B = ['48', '66']
significance_levels = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
results = []

for alpha in significance_levels:
    animal_values = {}

    for anim in group_A + group_B:
        val = run_animal_percent_place(anim, alpha)
        animal_values[anim] = val

    groupA_vals = [animal_values[a] for a in group_A if not np.isnan(animal_values[a])]
    groupB_vals = [animal_values[a] for a in group_B if not np.isnan(animal_values[a])]

    t_stat, p_val = ttest_ind(groupA_vals, groupB_vals, equal_var=False)

    print(
        f"alpha = {alpha:.4g} | "
        f"groupA mean = {np.mean(groupA_vals):.3f} | "
        f"groupB mean = {np.mean(groupB_vals):.3f} | "
        f"t-test p = {p_val:.4g}"
    )

    results.append({
        "significance_level": alpha,
        "groupA_mean": np.mean(groupA_vals),
        "groupB_mean": np.mean(groupB_vals),
        "ttest_p": p_val
    })


#%%


print(results)










