#%%
# IMPORT and create COMBINED df
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
import itertools
import seaborn as sns
import cv2
from tqdm import tqdm

data_prefix = 'C:\\Users\\nicol\\Desktop\\m90\\2024-10-31-16-52-47_CA1-m90_OF_'

file_traces =   data_prefix + 'traces.csv'
file_DLC    =   data_prefix + 'DLC.csv'
file_TTL    =   data_prefix + 'GPIO.csv'
file_video  =   data_prefix + 'DLC_labeled.mp4'
file_events =   data_prefix + 'traces_events.csv'


def combine_data(file_traces, file_DLC, file_TTL, file_video):
    # this def collects all raw data (traces, DLC, TTL, video) and synchronies them in one big df

    def get_traces(file_traces):
        # clean the csv file and return df
        df = pd.read_csv(file_traces, low_memory=False)

        # drop rejected cells/columns
        rejected_columns = df.columns[df.isin([' rejected']).any()]
        df = df.drop(columns=rejected_columns)

        # Clean up df
        df = df.drop(0)
        df = df.astype(float)
        df = df.rename(columns = {' ': 'Time'})
        df['Time'] = df['Time'].round(1)
        df = df.set_index(['Time'])
        df.index = df.index.round(2)

        df.columns = df.columns.str.replace(' C', '')
        df.columns = df.columns.astype(int)
        df_dff = df

        # normalize it via zscore
        df_z = df.copy()
        for name, data in df_z.items():
            df_z[name] = (df_z[name] - df_z[name].mean()) / df_z[name].std()    

        #df_z = df_z[[175]]

        return df_dff, df_z

    def get_traces_events(file_events, df_z, rounding=1):

        # create df filled with zeros with df_z index
        df_raw = pd.read_csv(file_events, low_memory=False)
        df_events = pd.DataFrame(0, index=df_z.index, columns=sorted(df_raw[" Cell Name"].unique()))

        # go through event csv and add every value into df filled with zeros
        for _, row in df_raw.iterrows():
            time = round(row["Time (s)"], rounding)
            cell = row[" Cell Name"]
            if time in df_events.index:
                df_events.at[time, cell] = int(row[" Value"])
            else:
                print(time, cell)

        # rename columns accordingly
        df_events.columns = df_events.columns.str.replace('C', '').astype(int)
        df_events.columns = df_events.columns.astype(str) + '_spike'

        # digitize the df (0-nospike, 1-spike) and forget the real values
        df_events = (df_events != 0).astype(int)

        #df_events = df_events[['175_spike']]

        # merge the new columns to df_z
        assert df_z.index.equals(df_events.index), "Indices do not match!"
        df_combined = pd.concat([df_z, df_events], axis=1)

        return df_combined

    def get_behav(file_DLC, file_TTL, main_df, file_video):
        # if you filmed, you will have a video file (DLC file) and TTLs in the GPIO file
        # this def gets the DLC animal position and synchronizes it with the Inspx time

        def clean_TTLs(file_TTL, channel_name = ' GPIO-1', value_threshold=500, constant=True):
            # returns list of TTL input times

            # expects nan values to be just a string: ' nan', convert these strings into proper nans
            df = pd.read_csv(file_TTL, na_values=[' nan'], low_memory=False)
            df = df.set_index('Time (s)')

            # only cares for the rows where the channel name is in the ' Channel Name' column
            df = df[df[' Channel Name'] == channel_name]
            
            # drops rows with nan and converts everything into ints
            df = df.dropna(subset=[' Value'])
            df[' Value'] = df[' Value'].astype(int)

            # creates series with True for values > threshold
            # checks at which index/time the row's value is high AND the row before's value is low
            # for that, it makes the whole series negative (True=False) and shifts the whole series one row further (.shift)
            # so, onset wherever row is > threshold and the opposite value of the shifted (1) row is positive as well
            is_high = df[' Value'] >= value_threshold
            onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]
            offsets = df.index[~is_high & is_high.shift(1, fill_value=False)]

            high_times = list(zip(onsets, offsets)) if not constant else list(onsets)
            #print(f'{len(high_times)} total events')

            # if print_csv:
            #     with open(output, 'w') as f:
            #         wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            #         wr.writerow(high_times)

            return high_times
        def get_video_dims(file_video):
            cap = cv2.VideoCapture(file_video)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (width, height)


        # cleans the DLC file and returns the df
        # idea: instead of picking some 'trustworthy' points to mean e.g. head, check the relations between 'trustworthy' points, so when one is missing, it can gues more or less where it is
        bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                    'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                    'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                    'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
        bps_head = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
        bps_postbody = ['mid_backend2', 'mid_backend3', 'tail_base']

        df = pd.read_csv(file_DLC, header=None, low_memory=False)

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

        # mirror y and forget everything except head and postbody
        width, height = get_video_dims(file_video)
        for bp in ['head', 'postbody']:
            df[f'{bp}_y'] = height - df[f'{bp}_y']
        df_behav = df[['head_x', 'head_y', 'postbody_x', 'postbody_y']].copy()

        # as an index for behavior, you have the frame number and the INSPX master time
        TTLs_high_times = clean_TTLs(file_TTL)
        df_behav['Time'] = TTLs_high_times[:-1]
        df_behav['Time'] = df_behav['Time'].round(2)
        df_behav.set_index([df_behav.index, 'Time'], inplace=True)

        # interpolate the NaNs (according to past and future values)
        for col in ['head_x', 'head_y', 'postbody_x', 'postbody_y']:
            df_behav[col] = df_behav[col].interpolate(method='linear', limit_area='inside')
        df_behav = df_behav.round(1)

        # makes index a regular column (necessary for merging) and merge
        main_df = main_df.reset_index()
        df_behav = df_behav.reset_index()
        main_df = pd.merge_asof(main_df, df_behav, on='Time', direction='nearest')
        main_df = main_df.set_index('Time')

        # index is regular time, that is limited by [first_tracked_Bp : video_max] 
        valid_mask = main_df[['head_x', 'head_y', 'postbody_x', 'postbody_y']].notna().all(axis=1)
        first_valid_index = valid_mask.idxmax()
        video_min, video_max = min(df_behav['Time']), max(df_behav['Time'])
        main_df = main_df.loc[first_valid_index : video_max]
        main_df.index = (main_df.index - first_valid_index).round(1)

        return main_df

    df_dff, df_z = get_traces(file_traces)
    main_df = get_traces_events(file_events, df_z)
    main_df = get_behav(file_DLC, file_TTL, main_df, file_video)

    return main_df

main_df = combine_data(file_traces, file_DLC, file_TTL, file_video)



#%%
# PLACE CELLS

def bin_cellspikes(main_df, n_bins=200):
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

def identify_place_cells(main_df, place_df, fps=10, n_shuffles=1, significance_level=0.05):
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

    return spatial_info_real, spatial_info_shuffled, place_cells, p_values

place_df, x_edges, y_edges = bin_cellspikes(main_df)
spatial_info_real, spatial_info_shuffled, place_cells, p_values = identify_place_cells(main_df, place_df)



#%%
# PLOT HISTOGRAM

cell = '175_spike'
real_value = spatial_info_real[cell]
shuffled_values = spatial_info_shuffled[cell]

plt.figure(figsize=(8, 4))
plt.hist(shuffled_values, bins=30, alpha=0.7, color='gray', edgecolor='black', label='randomly assigned spikes')
plt.axvline(real_value, color='orange', linewidth=5, label=f'Real SI = {real_value:.2f}')
plt.title(f'Spatial Info Shuffle Distribution for Cell_{cell}')
plt.xlabel('Spatial Information (bits/spike)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()



#%%
# PLOT SPIKES

to_plot = [30,28,8,31]
to_plot = [1,175]

plt.figure(figsize=(8,8))
plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.5, color='grey', zorder=1)
plt.vlines(x=x_edges[1:-1], ymin=0, ymax=600, colors='k')
plt.hlines(y=y_edges[1:-1], xmin=30, xmax=630, colors='k')

for cell_id in to_plot:
    spike_rows = main_df[main_df[f'{cell_id}_spike'] != 0]
    x = spike_rows['head_x']
    y = spike_rows['head_y']
    plt.scatter(x, y, s=80, alpha=0.8, label=f'{cell_id}', zorder=2)

plt.xlim(30,630)
plt.ylim(0,600)
plt.legend()
plt.show()

#%% 
# PLOT SPIKES HEATMAP
from scipy.ndimage import gaussian_filter

to_plot = [175]

spike_heatmap = np.zeros((len(y_edges)-1, len(x_edges)-1))

# Accumulate spike counts into the heatmap
for cell_id in to_plot:
    spike_rows = main_df[main_df[f'{cell_id}_spike'] != 0]
    x = spike_rows['head_x']
    y = spike_rows['head_y']
    
    H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])  # note: y first
    spike_heatmap += H

blurred_heatmap = gaussian_filter(spike_heatmap, sigma=4)

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(blurred_heatmap, origin='lower',extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='hot',alpha=0.8)
#plt.colorbar(label='Spike count')
plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.3, color='grey', zorder=1)

# Add grid lines
#plt.vlines(x=x_edges[1:-1], ymin=y_edges[0], ymax=y_edges[-1], colors='k')
#plt.hlines(y=y_edges[1:-1], xmin=x_edges[0], xmax=x_edges[-1], colors='k')

plt.title(f"Spike heatmap for cells {to_plot}")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.xlim(x_edges[0], x_edges[-1])
plt.ylim(y_edges[0], y_edges[-1])
plt.gca().set_aspect('equal')
plt.show()



