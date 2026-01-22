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

file = r"D:\48_Week8_OF_final.csv"

def bin_cellspikes_prob(main_df, n_bins=3, xmax=460, ymax=460):
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

    return main_df, place_df, x_edges, y_edges

def identify_place_cells_prob(main_df, place_df, fps=10, n_shuffles=10, significance_level=0.05):
    
    # assumes place_df entries are sums of spike probabilities (expected spike counts)
    
    def spatial_information(firing_freq, occupancy):

        # occupancy: number of frames per bin
        occupancy_probability = occupancy / occupancy.sum()

        # overall mean rate
        avrg_firing_rate = np.dot(firing_freq, occupancy_probability)
        print(avrg_firing_rate)

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

    return spatial_info_real

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

    return spatial_info_real, spatial_info_shuffled, place_cells, p_values






main_df = pd.read_csv(file, index_col='Mastertime [s]')

main_df, place_df, x_edges, y_edges = bin_cellspikes_prob(main_df)
spatial_info_real = identify_place_cells_prob(main_df, place_df)

#%%


d = list(spatial_info_real.values())
print(np.mean(np.array(d)))
