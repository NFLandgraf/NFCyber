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
from shapely.geometry import Polygon, Point
import os
from pathlib import Path




path = r"D:\Analy_Post2"


def manage_filename(file):

    # managing file names
    file_name = os.path.basename(file)
    file_name_short = os.path.splitext(file_name)[0]
    for word in file_useless_string:
        file_name_short = file_name_short.replace(word, '')
    
    return file_name_short
def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files
files = get_files(path, 'YM')



#%%

# PLACE CELLS
def bin_cellspikes(main_df, arena_is_OF=False):
    # split the area into spatial bins and for each cell, check how many spikes happened in each bin

    def get_areas_OF(dimensions=(460,460), n_bins=3):

            width, height = dimensions
            x_edges = np.linspace(0, width, n_bins+1, dtype=int)
            y_edges = np.linspace(0, height, n_bins+1, dtype=int)

            areas_OF = {}
            for col in range(n_bins):
                for row in range(n_bins):
                    # corners of the square bin
                    x0, x1 = x_edges[col], x_edges[col+1]
                    y0, y1 = y_edges[row], y_edges[row+1]

                    poly = Polygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
                    label = f'bin_{row}_{col}'  # row j, col i
                    areas_OF[label] = poly
            
            return areas_OF

    def get_areas_YMaze(dimensions=(580,485), centerpoint=(285, 310), c_rad=100, pointing_up=True):

        width, height       = dimensions
        c_x, c_y            = centerpoint

        # center
        start_angle = 90 if pointing_up else -90
        angles = np.deg2rad([start_angle, start_angle+120, start_angle+240])
        middle_corner, left_corner, right_corner = ((c_x + c_rad*np.cos(a), c_y + c_rad*np.sin(a)) for a in angles)

        #left arm
        left_arm_end_lefter     = (0, 400)
        left_arm_end            = (0, height)
        left_arm_end_righter    = (100, height)

        #middle arm
        middle_arm_end_lefter   = (c_x-c_rad, 0)
        middle_arm_end_righter  = (c_x+c_rad, 0)

        #right arm
        right_arm_end_lefter    = (500, height)
        right_arm_end           = (width, height)
        right_arm_end_righter   = (width, 400)

        # define Polygons
        left_arm = Polygon([left_arm_end_righter, left_arm_end, left_arm_end_lefter, left_corner, middle_corner, ])
        middle_arm = Polygon([middle_arm_end_lefter, middle_arm_end_righter, right_corner, left_corner])
        right_arm = Polygon([right_arm_end_righter, right_arm_end, right_arm_end_lefter, middle_corner, right_corner])
        center = Polygon([middle_corner, left_corner, right_corner])

        areas_YM = {'L':left_arm, 'M':middle_arm, 'R':right_arm, 'C':center}

        return areas_YM
    
    areas = get_areas_OF() if arena_is_OF else get_areas_YMaze()

    # puts head coords into spatial_bins and save the bin_labels in main_df
    labels = []
    for x, y in zip(main_df["head_x"], main_df["head_y"]):
        point = Point(x, y)
        label = None
        for bin_id, poly in areas.items():
            if poly.contains(point):
                label = bin_id
                break
        labels.append(label)
    main_df["head_bin"] = labels


    # creates df with the respective bin activity per cell
    spike_cells = [col for col in main_df.columns if 'spike' in col]
    place_map = {}
    for cell in spike_cells:
        bin_counts = main_df.groupby('head_bin')[cell].sum()
        place_map[cell] = bin_counts

    place_df = pd.DataFrame(place_map).fillna(0).astype(int)
    place_df = place_df.sort_index()

    return place_df, areas

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
        # how many spatial information (SI) is cell carrying
        
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
        print('Compute random distributon for SI...')
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

        return p_values, place_cells

    occupancy = main_df.groupby('head_bin').size()
    spatial_info_real = get_spatial_info_real(place_df, occupancy)
    spatial_info_shuffled = get_spatial_info_shuffle(place_df, occupancy)
    p_values, place_cells = check_significance(spatial_info_real, spatial_info_shuffled)

    nb_place_cells, nb_all_cells = len(place_cells), len(spatial_info_real)
    print(f'Place cells {nb_place_cells} / {nb_all_cells} ({round((nb_place_cells/nb_all_cells)*100, 1)}%)\n')

    save_df = pd.DataFrame(list(spatial_info_real.items()), columns=['cell', 'value'])
    #save_df.to_csv(.csv', index=False)

    return spatial_info_real, spatial_info_shuffled, p_values, place_cells

def ffn(main_df, place_cells):

    def prep_data(main_df, place_cells, min_active_cells=0, sampling_rate=10, bin_size=1.0):

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
        pbar = tqdm(range(n_epochs), desc="Training FFN", leave=True)
        for epoch in pbar:

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


            if verbose:
                pbar.set_postfix({"Train_RMSE": f"{avg_train_loss**0.5:.2f}","Test_RMSE": f"{avg_test_loss**0.5:.2f}"})

            # this outputs the squared px, so the root would be the actual pixel difference
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                pass
                #print(f"Epoch {epoch:03d} | Train_RMSE: {avg_train_loss**0.5:.2f}px | Test_RMSE: {avg_test_loss**0.5:.2f}px")
        
        def plot_training_log():
            plt.plot(np.sqrt(train_losses), label='Train_RMSE')
            plt.plot(np.sqrt(test_losses), label='Test_RMSE')
            plt.xlabel("Epoch")
            plt.ylabel("RMSE [px]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        #plot_training_log()

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

    def calc_accuracy(Y_true, Y_pred, threshold_mm=50):

        # calulates how many predictions are within the threshold (in depends if mm or px (main_df creation))
        errors = np.linalg.norm(Y_true - Y_pred, axis=1)
        accuracy = round(np.mean(errors <= threshold_mm)*100, 1)
        print(f"Mean decoding error: {errors.mean():.2f} mm, Accuracy: {accuracy}%\n")

        return accuracy, errors


    X, Y = prep_data(main_df, place_cells)
    full_loader_train, full_loader_test = create_tensors_whole(X, Y)
    shuffled_loader_train = create_tensors_shuffled(X, Y)
    
    model = PositionDecoder(input_dim=X.shape[1])
    trained_model = train_model(full_loader_train, full_loader_test, model)
    
    Y_pred, Y_true = test_model(trained_model, full_loader_test)
    accuracy, errors = calc_accuracy(Y_true, Y_pred)
    #plot_true_vs_pred_coords(Y_true, Y_pred, accuracy)

    return Y_pred, Y_true, trained_model, accuracy, errors




for file_main in files:
    print('\n\n')
    print(file_main)

    
    main_df = pd.read_csv(file_main, index_col='Time', low_memory=False)

    place_df, areas = bin_cellspikes(main_df)
    spatial_info_real, spatial_info_shuffled, p_values, place_cells = identify_place_cells(main_df, place_df)

    n_cells = 40
    n_repeats = 10
    results_acc, results_err = [], []
    for i in range(n_repeats):
        sampled_cells = np.random.choice(place_cells, size=min(n_cells, len(place_cells)), replace=False)
        Y_pred, Y_true, model, acc, error = ffn(main_df, sampled_cells)
        results_acc.append(acc)
        results_err.append(error)


    print(file_main)

    results_acc_mean = np.mean(results_acc)
    results_acc_stdv = np.std(results_acc)
    print(results_acc)
    print(results_acc_mean)
    print(results_acc_stdv)

    results_err_mean = np.mean(results_err)
    results_err_stdv = np.std(results_err)
    #print(results_err)
    #print(results_err_mean)
    #print(results_err_stdv)

    print('\n\n')






#%%

# PLOT
def plot_spikes_polygons(main_df, cell_id, areas):
    
    plt.figure(figsize=(8,8))
    plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.5, color='grey', zorder=1)

    # Plot spike locations for this cell
    spike_rows = main_df[main_df[cell_id] != 0]
    x = spike_rows['head_x']
    y = spike_rows['head_y']
    plt.scatter(x, y, s=80, alpha=0.8, label=f'{cell_id}', zorder=2)

    # Overlay polygons
    for label, poly in areas.items():
        px, py = poly.exterior.xy
        plt.plot(px, py, linewidth=2, label=label)

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()
def plot_spikes_heatmap_polygons(main_df, cell, i, areas, arena_size=(580,485), bins=200, blur=4):
    # 2D histogram of spike locations
    spike_rows = main_df[main_df[cell] != 0]
    x = spike_rows['head_x']
    y = spike_rows['head_y']

    x_edges = np.linspace(0, arena_size[0], bins)
    y_edges = np.linspace(0, arena_size[1], bins)
    H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    blurred_heatmap = gaussian_filter(H, sigma=blur)

    plt.figure(figsize=(8,8))
    plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.4, color='grey', zorder=2)
    plt.imshow(blurred_heatmap, origin='lower',
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
               cmap='hot', alpha=0.8, zorder=1)

    # Overlay polygons
    for label, poly in areas.items():
        px, py = poly.exterior.xy
        plt.plot(px, py, linewidth=2, label=label)

    plt.axis('off')
    plt.title(f'Cell {cell.replace("_spike","")}')
    plt.gca().set_aspect('equal')
    plt.tight_layout(pad=0)
    plt.savefig(f"{i}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
def plot_histogram(cell, spatial_info_real, spatial_info_shuffled, bins=30):

    # plots histogram that shows the true SI and the shuffled data
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

    for frame_idx in tqdm(range(2000)):
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


sorted_items = sorted(spatial_info_real.items(), key=lambda item: item[1], reverse=True)
for i, (cell, value) in enumerate(sorted_items):
    if i<1:
        print(cell)
        print(value)
        plot_spikes_heatmap_polygons(main_df, cell, i, areas)
        plot_spikes_polygons(main_df, cell, areas)


