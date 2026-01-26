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

folder_in = Path(r"D:\10_ALL_MERGE")
df_QC = pd.read_csv(r"D:\df_QC_FINAL_pass.csv", index_col='cell_ID')

def heatmap_cells(df, value_col, out_jpg, arena_size=(580, 485), blur_sigma=4.0, cmap="hot", gamma=0.9):
    # gamma = <1 brightens, >1 darkens, blur_sigma = set 0 for no blur

    width, height = arena_size
    heat = np.zeros((height, width), dtype=np.float32)

    x = pd.to_numeric(df['head_x'], errors="coerce").to_numpy()
    y = pd.to_numeric(df['head_y'], errors="coerce").to_numpy()
    val = pd.to_numeric(df[value_col], errors="coerce").to_numpy()

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
    x = x[m]
    y = y[m]
    val = val[m]

    # map coords -> integer pixels (x->col, y->row)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)

    inb = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xi = xi[inb]
    yi = yi[inb]
    val = val[inb].astype(np.float32)

    # accumulate values into pixels
    np.add.at(heat, (yi, xi), val)

    # blur once at end (much faster than stamping per row)
    if blur_sigma and blur_sigma > 0:
        heat = gaussian_filter(heat, sigma=blur_sigma)

    # normalize for display
    if heat.max() > 0:
        heat = heat / heat.max()

    # gamma adjust (optional)
    if gamma != 1.0:
        heat = np.clip(heat, 0, 1) ** gamma

    # save EXACT pixel dimensions: (width x height)
    out_jpg = str(out_jpg)
    fig = plt.figure(figsize=(width/100, height/100), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(heat, origin="lower", cmap=cmap, interpolation="nearest")
    fig.savefig(out_jpg, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return out_jpg

def heatmap_occupancy(df, out_jpg, arena_size=(580, 485), blur_sigma=4.0, cmap="Blues", gamma=0.8):
    """
    Create an occupancy heatmap (where the animal was) and save as a JPG.

    Output image size is exactly arena_size (in pixels).
    """

    width, height = arena_size
    heat = np.zeros((height, width), dtype=np.float32)

    # extract coordinates
    x = pd.to_numeric(df['head_x'], errors="coerce").to_numpy()
    y = pd.to_numeric(df['head_y'], errors="coerce").to_numpy()

    w = 1.0
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    # map to pixel indices (x -> col, y -> row)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)

    inb = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xi = xi[inb]
    yi = yi[inb]

    # accumulate occupancy
    np.add.at(heat, (yi, xi), w)

    # optional blur
    if blur_sigma and blur_sigma > 0:
        heat = gaussian_filter(heat, sigma=blur_sigma)

    # normalize for visualization
    if heat.max() > 0:
        heat = heat / heat.max()

    # gamma correction
    if gamma != 1.0:
        heat = np.clip(heat, 0, 1) ** gamma

    # save EXACT pixel size
    out_jpg = str(out_jpg)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(heat, origin="lower", cmap=cmap, interpolation="nearest")
    fig.savefig(out_jpg, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return out_jpg


def get_files(path, common_name1, common_name2, suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name1 in p.name) and (common_name2 in p.name) and p.name.endswith(suffix)])
    print(f"\n{len(files)} files found")
    for f in files:
        print(f)
    print('\n\n')
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

def bin_cellspikes_prob_OF(main_df, n_bins=3, xmax=460, ymax=460):
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

def identify_place_cells_prob(main_df, place_df, fps=10, n_shuffles=100, significance_level=0.05):
    
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

    for _ in tqdm(range(n_shuffles)):
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
    print(f'Place cells {nb_place_cells} / {nb_all_cells} ({round((nb_place_cells/nb_all_cells)*100, 1)}%)')

    percent_place = nb_place_cells / nb_all_cells

    return spatial_info_real, spatial_info_shuffled, place_cells, p_values, percent_place


animals = ['48', '51', '56', '66', '72', '77']
place_dict = {}
for anim in animals:

    files = get_files(folder_in, anim, 'YM')

    means, medians = [], []
    
    animal_results = {}
    for i, file in enumerate(files):

        print(f'\n------- {file} -------')

        main_df = pd.read_csv(file, index_col='Mastertime [s]')
        main_df = clean_df(main_df, df_QC)
        if main_df is None:
            animal_results[file.stem] = []
            continue

        main_df, place_df = bin_cellspikes_prob_YM(main_df)
        spatial_info_real, spatial_info_shuffled, place_cells, p_values, percent_place = identify_place_cells_prob(main_df, place_df)

        means.append(np.array(list(spatial_info_real.values())).mean())
        medians.append(np.median(np.array(list(spatial_info_real.values()))))
        animal_results[file.stem] = list(spatial_info_real.values())
        place_dict[file.stem] = percent_place

        #heatmap_occupancy(main_df, f"D:\\out\\heatmaps\\anim\\{anim}_{i}.jpg")

    #print(means)
    #print(medians)
    series_dict = {fname: pd.Series(vals) for fname, vals in animal_results.items()}
    
    df_out = pd.DataFrame(series_dict)
    df_out.to_csv(rf"D:\results_{anim}_YM.csv", index=False)
place_dict = {fname: pd.Series(vals) for fname, vals in place_dict.items()}
place_out = pd.DataFrame(place_dict)
place_out.to_csv(rf"D:\results_{anim}_YM_place.csv", index=False)


#%%


d = list(spatial_info_real.values())
print(np.mean(np.array(d)))
