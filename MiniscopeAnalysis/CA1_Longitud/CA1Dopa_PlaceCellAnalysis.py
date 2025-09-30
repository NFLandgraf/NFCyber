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
#%%



# PLACE CELLS
def bin_cellspikes(main_df, n_bins=3):
    # we find out, how many spikes per spatial bin (n_bins=2 for 2x2) are there for each cell

    # creates the wanted grid
    x_edges = np.linspace(0, 660, n_bins+1)
    y_edges = np.linspace(0, 600, n_bins+1)

    # puts head coords into spatial_bins
    main_df['x_bin'] = np.clip(np.digitize(main_df['head_x'], bins=x_edges) - 1, 0, n_bins-1)
    main_df['y_bin'] = np.clip(np.digitize(main_df['head_y'], bins=y_edges) - 1, 0, n_bins-1)
    main_df['head_bin'] = main_df['y_bin'] * n_bins + main_df['x_bin']

    spike_cells = [col for col in main_df.columns if type(col) == str and col.endswith('_spike')]
    place_map = {}
    for cell in spike_cells:
        bin_counts = main_df.groupby('head_bin')[cell].sum()
        place_map[cell] = bin_counts

    place_df = pd.DataFrame(place_map).fillna(0).astype(int)
    place_df = place_df.sort_index()

    return place_df, x_edges, y_edges

def identify_place_cells(main_df, place_df, fps=10, n_shuffles=1000, significance_level=0.05):
    '''
    Computing spatial information via Skaggs formula:
    SI = ∑i ⋅ pi ⋅ ri/r ⋅ log2(ri/r)
    i   = index of spatial bin
    pi  = probability of generelly being in bin i
    ri  = firing rate in bin i
    r   = overall mean firing rate (weighted by occupancy)

    If neuron fires uniformly everywhere    -> SI is low
    If neuron fires mostly in one bin       -> SI is high
    If neuron is inactive or noisy          -> SI is 0 or unreliable
    0 random, 0.3-0.5 weak place tuning, >0.5 likely a place cell, >1.0 strong, reliable place-specific firing

    We see the observed spatial info for each cell, but is it random/by chance?
    Here, we simulate null distribution, as if the cell spike had no relationship to location.
    So, for each cell, we take the n_spikes and the bin_occupancy by the animal and randomly put the spikes
    into the available bin, just dependent on the occupancy probility of the animal. This creates our shuffled/random
    distribution, which we test our real value against. If our value is higher than e.g. the 95th percentile of
    the shuffled values, it is significant.
    '''
    def spatial_information(firing_freq, occupancy):
        
        # how likely is it that animal is in bin
        occupancy_probability = occupancy / occupancy.sum()

        # what is the overall firing rate of the cell in the whole area during the recording
        avrg_firing_rate = np.dot(firing_freq, occupancy_probability)

        # loop through each bin and executes the Skaggs formula (above)
        # the info is 'bits of information per spike', so the higher, the more spatial info the spike carries
        info = 0
        for r, p in zip(firing_freq, occupancy_probability):
            if r > 0 and p > 0:
                info += p * (r / avrg_firing_rate) * np.log2(r / avrg_firing_rate)
        return info
    
    def get_spatial_info_real(place_df, occupancy):

        # get the firing frequency of each cell per bin
        occupancy_sec = occupancy / fps
        df_firing_freq = place_df.div(occupancy_sec, axis=0)

        # stores the spatial infos (SI) value for each cell in dict
        spatial_info_real = {}
        for cell in df_firing_freq.columns:
            info = spatial_information(df_firing_freq[cell].values, occupancy.values)
            spatial_info_real[cell] = info

        return spatial_info_real

    def get_spatial_info_shuffle(place_df, occupancy):

        occupancy_probs = occupancy / occupancy.sum()
        spatial_info_shuffled = {cell: [] for cell in place_df.columns}

        for _ in tqdm(range(n_shuffles)):
            for cell in place_df.columns:
                
                # distribute all spikes of that cell weighted into the available bins
                n_spikes = int(place_df[cell].sum())
                shuffled_bins = np.random.choice(a=occupancy.index, size=n_spikes, p=occupancy_probs)

                # count how many spikes landed in each bin and calculate firing rate per bin
                shuffled_counts = pd.Series(shuffled_bins).value_counts().reindex(occupancy.index, fill_value=0)
                shuffled_firing_rate = shuffled_counts / occupancy
            
                # compute spatial info
                info = spatial_information(shuffled_firing_rate.values, occupancy.values)
                spatial_info_shuffled[cell].append(info)
        
        return spatial_info_shuffled

    def check_significance(spatial_info_real, spatial_info_shuffled):
        
        # compare real_si to shuffled_si
        place_cells = []
        p_values = {}
        for cell in spatial_info_real:
            real_si = spatial_info_real[cell]
            shuffled_si = spatial_info_shuffled[cell]

            # True=1, False=0, np.mean takes mean of True/False (returns array of booleans if a shuffled_si is bigger than real_si)
            p = np.mean(np.array(shuffled_si) >= real_si)
            p_values[cell] = p

            if p < significance_level:
                place_cells.append(cell)

        return place_cells, p_values

    occupancy = main_df.groupby('head_bin').size()
    spatial_info_real = get_spatial_info_real(place_df, occupancy)
    spatial_info_shuffled = get_spatial_info_shuffle(place_df, occupancy)
    place_cells, p_values = check_significance(spatial_info_real, spatial_info_shuffled)

    nb_place_cells, nb_all_cells = len(place_cells), len(spatial_info_real)
    print(f'Place cells {nb_place_cells} / {nb_all_cells} ({round((nb_place_cells/nb_all_cells)*100, 1)}%)')

    return spatial_info_real, spatial_info_shuffled, place_cells, p_values

place_df, x_edges, y_edges = bin_cellspikes(main_df)
spatial_info_real, spatial_info_shuffled, place_cells, p_values = identify_place_cells(main_df, place_df)

