#%%
# IMPORT and COMBINE
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

#data_prefix = 'C:\\Users\\nicol\\Desktop\\m90\\2024-10-31-16-52-47_CA1-m90_OF_'
data_prefix = 'C:\\Users\\landgrafn\\Desktop\\eyy\\CA1Dopa_Longitud_f43_2025-04-28-16-53-12_OF_Baseline_'



file_traces =   data_prefix + 'traces.csv'
file_DLC    =   data_prefix + 'DLC.csv'
file_TTL    =   data_prefix + 'GPIO.csv'
file_video  =   data_prefix + 'DLC_trim_transparent.mp4'
file_events =   data_prefix + 'traces_events.csv'
file_main =   data_prefix + 'main.csv'

#%%

def combine_data(file_traces, file_events, file_DLC, file_TTL, file_video):
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
        df_z.to_csv('zscore.csv')

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

    def get_behav(file_DLC, file_TTL, main_df, file_video, mirrorY=False):
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

            
            with open('tttls and stuff.csv', 'w') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(high_times)

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

        if mirrorY:
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

        # makes index a regular column (necessary for merging) and merge iwht main_df
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

main_df = combine_data(file_traces, file_events, file_DLC, file_TTL, file_video)


#%%
#file_main = 'C:\\Users\\landgrafn\\Desktop\\m90\\2024-10-31-16-52-47_CA1-m90_OF_'
main_df = pd.read_csv(file_main, index_col='Time')
print(main_df)


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



#%%
# PLOT
def plot_spikes(main_df, x_edges, y_edges, cells):

    # plot spikes on arena
    plt.figure(figsize=(8,8))
    plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.5, color='grey', zorder=1)
    plt.vlines(x=x_edges[1:-1], ymin=0, ymax=600, colors='k')
    plt.hlines(y=y_edges[1:-1], xmin=30, xmax=630, colors='k')

    for cell_id in cells:
        spike_rows = main_df[main_df[f'{cell_id}_spike'] != 0]
        x = spike_rows['head_x']
        y = spike_rows['head_y']
        plt.scatter(x, y, s=80, alpha=0.8, label=f'{cell_id}', zorder=2)

    plt.xlim(30,630)
    plt.ylim(0,600)
    plt.legend()
    plt.show()
def plot_spikes_heatmap(main_df, cell, i, bins=200, blur=4):

    # creates the wanted grid
    x_edges = np.linspace(0, 600, bins)
    y_edges = np.linspace(0, 600, bins)

    # plot spikes heatmap on arena
    spike_heatmap = np.zeros((len(y_edges)-1, len(x_edges)-1))

    # accumulate spike counts into the heatmap
    spike_rows = main_df[main_df[f'{cell}'] != 0]
    x = spike_rows['head_x']
    y = spike_rows['head_y']
    
    H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    spike_heatmap += H

    # blur heatmap (the higher sigma, the blur goes blurrrrr)
    blurred_heatmap = gaussian_filter(spike_heatmap, sigma=blur)

    plt.figure(figsize=(8, 8))
    plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.4, color='grey', zorder=2)
    plt.imshow(blurred_heatmap, origin='lower',extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='hot',alpha=0.8, zorder=1)

    plt.gca().set_xticks([])  # no x ticks
    plt.gca().set_yticks([])  # no y ticks
    plt.gca().set_frame_on(False)  # remove box around plot
    plt.axis('off')

    plt.title(f'Cell {cell.replace('_spike', '')}')
    plt.xlim(x_edges[0], x_edges[-1])
    plt.ylim(y_edges[0], y_edges[-1])
    
    plt.gca().set_aspect('equal')
    plt.tight_layout(pad=0)
    plt.savefig(f"{i}.png", bbox_inches='tight', pad_inches=0, dpi=300)
def plot_histogram(cell, spatial_info_real, spatial_info_shuffled, bins=30):

    # plots histogram that shows the true SI and the shuffled data
    cell = f'{cell}_spike'
    real_value = spatial_info_real[cell]
    shuffled_values = spatial_info_shuffled[cell]

    plt.figure(figsize=(8, 4))
    plt.hist(shuffled_values, bins=bins, alpha=0.7, color='gray', edgecolor='black', label='randomly assigned spikes')
    plt.axvline(real_value, color='orange', linewidth=5, label=f'Real SI = {real_value:.2f}')
    plt.title(f'Spatial Info Shuffle Distribution for Cell_{cell}')
    plt.xlabel('Spatial Information (bits/spike)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
def draw_position_on_video(df, file_video):

    head_x = df["head_x"].values
    head_y = df["head_y"].values

    cap = cv2.VideoCapture(file_video)
    video_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = int(video_nframes / video_fps)
    print(f'VIDEO duration: {video_duration}s, frames: {video_nframes}, fps: {video_fps}')

    file_output = "position_overlay.mp4"
    out = cv2.VideoWriter(file_output, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    for frame_idx in tqdm(range(600)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(head_x):
            if not np.isnan(head_x[frame_idx]):
                x = int(head_x[frame_idx])
                y = int(head_y[frame_idx])
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Green dot for position

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Video with position overlay saved as 'position_overlay_first100.mp4'")

# print every heatmap for the cells
sorted_items = sorted(spatial_info_real.items(), key=lambda item: item[1], reverse=True)
for i, (cell, value) in enumerate(sorted_items):
    plot_spikes_heatmap(main_df, cell, i, bins=200, blur=10)


#plot_spikes(main_df, x_edges, y_edges, [30,28,8,31])
#plot_spikes_heatmap(main_df, 175, sigma=4)
#plot_histogram(175, spatial_info_real, spatial_info_shuffled)


#%%
# NEURAL NETWORK
def ffn(main_df, place_cells):

    def prep_data(main_df, place_cells, min_active_cells, sampling_rate=10, bin_size=1.0):

        # downsample the df to have relyable neuronal info per timepoint
        frames_per_bin = int(bin_size * sampling_rate)
        position_cols = ['head_x', 'head_y']
        n_bins = len(main_df) // frames_per_bin

        df_binned = []
        for i in range(n_bins):
            start = i * frames_per_bin
            end = (i + 1) * frames_per_bin
            segment = main_df.iloc[start:end]

            spike_sums = segment[place_cells].sum()
            position_avg = segment[position_cols].mean()

            binned_row = pd.concat([spike_sums, position_avg])
            df_binned.append(binned_row)

        df_binned = pd.DataFrame(df_binned)
        
        # only include timepoints with >= x active cells
        active_rows = (df_binned[place_cells] > 0).sum(axis=1) >= min_active_cells
        df_binned = df_binned[active_rows]
        X = df_binned[place_cells].values.astype(np.float32)
        Y = df_binned[position_cols].values.astype(np.float32)

        return X, Y

    def create_tensors_split(X, Y, test_size=0.2):
        # split data into train/test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        # convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
        Y_test_tensor  = torch.tensor(Y_test, dtype=torch.float32)

        # create datasets
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32)

        return train_loader, test_loader
    
    def create_tensors_whole(X, Y):
        # as we just want to proove that you can predict the position by looking at the neuronal activity,
        # we don't need to split the data into train/test, just use the whole dataset
        full_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))

        # for training, the rows must be shuffled around (without breaking the X-Y pairing)
        full_loader_train = DataLoader(full_dataset, batch_size=32, shuffle=True)

        # for test, shuffle must be False, otherwise it is not the correct time series anymore
        full_loader_test = DataLoader(full_dataset, batch_size=32, shuffle=False)

        return full_loader_train, full_loader_test
    
    def create_tensors_shuffled(X, Y):

        # shuffle the data to create a negative control        
        Y_shuffled = np.random.permutation(Y)
        shuffled_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y_shuffled, dtype=torch.float32))
        shuffled_loader = DataLoader(shuffled_dataset, batch_size=32, shuffle=True)

        return shuffled_loader

    class PositionDecoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super(PositionDecoder, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),   # layer_1: input -> hidden
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),  # layer_2: hidden -> hidden
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)            # layer_3: hidden -> output (x,y)
            )

        def forward(self, x):
            return self.net(x)

    def train_model(train_loader, test_loader, model, n_epochs=1000, lr=1e-3, verbose=True):

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # log to check later
        train_losses, test_losses = [], []

        # Training loop
        for epoch in range(n_epochs):

            # train model
            model.train()
            total_train_loss = 0
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                total_train_loss  += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
        
            # test model
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for batch_X, batch_Y in test_loader:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_Y)
                    total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
        
            # this outputs the squared px, so the root would be the actual pixel difference
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:03d} | Train_RMSE: {avg_train_loss**0.5:.2f}px | Test_RMSE: {avg_test_loss**0.5:.2f}px")
        
        # plot training log
        plt.plot(np.sqrt(train_losses), label='Train_RMSE')
        plt.plot(np.sqrt(test_losses), label='Test_RMSE')
        plt.xlabel("Epoch")
        plt.ylabel("RMSE [px]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return model

    def test_model(model, test_loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                preds = model(batch_X)
                all_preds.append(preds.cpu())
                all_targets.append(batch_Y.cpu())

        # Combine all batches into arrays
        Y_pred = torch.cat(all_preds).numpy()
        Y_true = torch.cat(all_targets).numpy()

        return Y_pred, Y_true

    def plot_true_vs_pred_coords(Y_true, Y_pred, accuracy):

        true_x = Y_true[:, 0]
        pred_x = Y_pred[:, 0]
        true_y = Y_true[:, 1]
        pred_y = Y_pred[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=False)

        # ---- X position ----
        ax = axes[0]
        hb = ax.hist2d(true_x, pred_x, bins=35, cmap='viridis')
        fig.colorbar(hb[3], ax=ax)
        ax.plot([true_x.min(), true_x.max()], [true_x.min(), true_x.max()], c='r', linestyle=':', label='Perfect prediction')
        # ax.set_xlim(30, 630)
        # ax.set_ylim(0, 600)
        ax.set_xlabel('True X position')
        ax.set_ylabel('Predicted X position')
        ax.set_title(f'Decoded X: Predicted vs True (Accuracy={accuracy}%)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

        # ---- Y position ----
        ax = axes[1]
        hb = ax.hist2d(true_y, pred_y, bins=35, cmap='viridis')
        fig.colorbar(hb[3], ax=ax)
        ax.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], c='r', linestyle=':', label='Perfect prediction')
        # ax.set_xlim(30, 630)
        # ax.set_ylim(0, 600)
        ax.set_xlabel('True Y position')
        ax.set_ylabel('Predicted Y position')
        ax.set_title(f'Decoded Y: Predicted vs True (Accuracy={accuracy}%)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

        errors = np.linalg.norm(Y_true - Y_pred, axis=1)
        print(f"Mean decoding error: {errors.mean():.2f} pixels")

    def calc_accuracy(Y_true, Y_pred, threshold=75):
        # calulates how many predictions are within the threshold (in px)
        errors = np.linalg.norm(Y_true - Y_pred, axis=1)
        accuracy = round(np.mean(errors <= threshold)*100, 1)
        print(f'Accuracy: {accuracy}%')
        return accuracy


    X, Y = prep_data(main_df, place_cells, min_active_cells=0)
    full_loader_train, full_loader_test = create_tensors_whole(X, Y)
    shuffled_loader_train = create_tensors_shuffled(X, Y)
    
    model = PositionDecoder(input_dim=X.shape[1])
    trained_model = train_model(shuffled_loader_train, full_loader_test, model)
    
    Y_pred, Y_true = test_model(trained_model, full_loader_test)
    accuracy = calc_accuracy(Y_true, Y_pred)
    plot_true_vs_pred_coords(Y_true, Y_pred, accuracy)

    return Y_pred, Y_true, trained_model, accuracy

Y_pred, Y_true, trained_model, accuracy = ffn(main_df, place_cells)


#%%
# VIDEO FEEDBACK
def video_TrueandPred(Y_true, Y_pred):
    # creates video in coordinate system of the animal, showing true and predicted position

    # Assume Y_true and Y_pred are both [n_frames, 2] at 1 Hz
    n_frames = len(Y_true)
    frame_rate = 30  # output video frame rate (Hz)
    duration_sec = n_frames  # because data is 1Hz
    new_n_frames = frame_rate * duration_sec

    # Interpolate both to 30Hz
    time_original = np.linspace(0, duration_sec, n_frames)
    time_interp = np.linspace(0, duration_sec, new_n_frames)

    interp_true = interp1d(time_original, Y_true, axis=0, kind='linear')
    interp_pred = interp1d(time_original, Y_pred, axis=0, kind='linear')

    Y_true_30hz = interp_true(time_interp)
    Y_pred_30hz = interp_pred(time_interp)

    # Y_pred_30hz = np.repeat(Y_pred, frame_rate, axis=0)
    # Y_pred_30hz = np.roll(Y_pred_30hz, shift=-15, axis=0)

    # Set up video writer
    video_size = (600, 600)
    out = cv2.VideoWriter("decoded_trajectory_long.mp4", cv2.VideoWriter_fourcc(*'mp4v'),120, video_size)

    # Generate frames
    for i in tqdm(range(600)):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(30, 630)
        ax.set_ylim(0, 600)
        ax.set_aspect('equal')

        #ax.plot(Y_true_30hz[:i+1, 0], Y_true_30hz[:i+1, 1], 'k', label='True', linewidth=2, alpha=0.7)
        #ax.plot(Y_pred_30hz[:i+1, 0], Y_pred_30hz[:i+1, 1], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax.scatter(Y_pred_30hz[i, 0], Y_pred_30hz[i, 1], color='red', s=500, alpha=1, label='Predicted')
        ax.scatter(Y_true_30hz[i, 0], Y_true_30hz[i, 1], color='black', s=100, label='True')
        ax.set_title(f"Frame {i+1} / {new_n_frames}")
        ax.legend()

        # Render plot to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(cv2.resize(frame_bgr, video_size))
        plt.close(fig)

    out.release()
    print("✅ Video saved as 'decoded_trajectory_long.mp4'")

def video_DrawonOriginal(Y, file_video, Y_binsize=1, white_sec=85, fade_sec=10):

    def interpolate_array(arr, arr_binsize, video_nframes, video_fps):

        video_duration = int(video_nframes / video_fps)
        #print(f'VIDEO duration: {video_duration}s, frames: {video_nframes}, fps: {video_fps}')

        # get array properties
        arr_frames = len(arr)
        arr_fps = 1 / arr_binsize
        arr_duration = int(arr_frames / arr_binsize)
        arr_wanted_frames = int(video_fps * video_duration)
        #print(f'ARR duration: {arr_duration}s, frames: {arr_frames}, fps: {arr_fps}')
        #print(f'ARR_WANTED duration: {video_duration}s, frames: {arr_wanted_frames}, fps: {video_fps}')

        arr_orig_time = np.linspace(start=0, stop=arr_duration, num=arr_frames)
        arr_time_interp = np.linspace(start=0, stop=arr_duration, num=arr_wanted_frames)
        interpol = interp1d(arr_orig_time, arr, axis=0, kind='linear')
        arr_interpol = interpol(arr_time_interp)

        return arr_interpol
    
    cap = cv2.VideoCapture(file_video)
    video_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = 10
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Y = np.array(Y)  # shape: [n_bins, 2] at 1Hz
    Y_interpol = interpolate_array(Y, Y_binsize, video_nframes, video_fps)  # shape: [n_bins, 2] at 1Hz
    
    wanted_fps = 30
    output_path = "decoded_overlay_interpolated.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), wanted_fps, (video_width, video_height))

    white_frames = white_sec * wanted_fps
    fade_frames = fade_sec * video_fps

    pbar = tqdm(total=min(len(Y_interpol), video_nframes))
    frame_idx = 0
    while frame_idx < len(Y_interpol) and frame_idx < video_nframes:

        ret, frame = cap.read()
        if not ret:
            break
        
        # to morph the background in
        white_bg = np.full_like(frame, 255)
        if frame_idx < white_frames:
            blended_frame = white_bg.copy()
        elif white_frames <= frame_idx < white_frames + fade_frames:
            fade_progress = (frame_idx - white_frames) / fade_frames
            blended_frame = cv2.addWeighted(white_bg, 1 - fade_progress, frame, fade_progress, 0)
        else:
            blended_frame = frame

        # draw interpolated prediction (small red dot)
        x, y = Y_interpol[frame_idx]
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(blended_frame, (int(x), int(y)), 15, (0, 0, 255), -1)  # red

        out.write(blended_frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    
video_DrawonOriginal(Y_pred, file_video)
#video_TrueandPred(Y_true, Y_pred)





#%%



'''
Next steps:
- camera is always at a different spot ->you need to always crop the video the same way and then adadpt px/cm, then do everything in cm
- regarding the problem with frames that have no cells spiking: maybe we can let cells "spill" over into the next frame
  maybe 0.5 into the previous & next frame (not digital anymore but we dont care)
- get the p-value of the decoder, so we can find the best decoder
- do same for Y-Maze
- check place cell stability longitudinally
- train & evaluate the network on x non-place cells and on x place cells and compare


Problems that could cause differences:
- animals dont investigate the arena the same way
- more/less cells are recruited longitudinally (problem?)
- firing freq is different



'''

