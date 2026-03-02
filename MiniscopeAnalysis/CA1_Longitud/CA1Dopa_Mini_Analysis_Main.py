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
from sklearn.preprocessing import StandardScaler
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
from matplotlib.path import Path as MplPath
from scipy.stats import linregress


folder_in = Path(r"C:\Users\landgrafn\Desktop\a\data\Master_10_merge\mKate")
folder_out = r"C:\Users\landgrafn\Desktop\a\Test"
file_QC = r"C:\Users\landgrafn\Desktop\a\QC_FINAL_pass.csv"

out_spikeprob_individ = folder_out + '\\SpikeRate_Individ.csv'
out_spikeprob_summary = folder_out + '\\SpikeRate_Summary.csv'
out_spikerate_speed   = folder_out + '\\SpikeRate-Speed.csv'
out_SI_summary = folder_out + '\\SI_Summary.csv'
out_FFN_summary = folder_out + '\\FFN_Summary.csv'

dist_travelled = []


def get_files(path, common_name1, common_name2, suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name1 in p.name) and (common_name2 in p.name) and p.name.endswith(suffix)])
    #print(f"\n{len(files)} files found")
    #for f in files:
    #    print(f)
    #print('\n\n')
    return files

def clean_df_QC(df, df_QC):

    if df is None:
        return None

    # do QC according to pass True or False
    cols_to_keep = []
    for col in df.columns:
        # only apply QC to neuron columns
        if "_C" not in col:
            cols_to_keep.append(col)
            continue

        # if the name is not in pass_df, be conservative and drop
        name = col.rsplit("_", 1)[0]
        if name not in df_QC.index:
            print('name not in pass_df')
            continue

        # keep only if pass == True
        if df_QC.loc[name, "pass"]:
            cols_to_keep.append(col)
    
    cells_dropped = int((len(df.columns) - len(cols_to_keep)) /3)
    df = df[cols_to_keep]
    #print(f'{cells_dropped} cells dropped due to QC -> now {int(len(cols_to_keep)/3)} cells')

    return df

def clean_df_distance(df, file, start_row=34, maxYM=20, maxOF=None):
    global distances

    if df is None:
        return None

    # trim main_df according to distance travelled
    name = df.columns[-2].rsplit("_", 1)[0]
    df = df.iloc[start_row:]

    if 'YM' in name:
        maxdist = maxYM * 1000
    elif 'OF' in name:
        maxdist = maxOF * 1000
    else:
        print('Zero cells in recording')
        return None
    
    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    step_dist = np.sqrt(dx**2 + dy**2)
    cum_dist = step_dist.fillna(0).cumsum()
    dist_travelled.append(cum_dist.values)

    # if never reached max distance, just return full df
    reached = cum_dist >= maxdist
    if not reached.any():
        return df
    
    cut_label = cum_dist.index[reached.argmax()]  # first index where reached
    #print(f'reached maxdist after {int(cut_label)}s / {int(max(df.index))}s ->trimmed')
    df = df.loc[:cut_label].copy()

    return df


def heatmap_occupancy(df, arena_size=(580, 485), blur_sigma=4.0, cmap="Blues", gamma=0.8):
    
    # Create an occupancy heatmap (where the animal was) and save as a JPG, Output image size is exactly arena_size (in pixels)
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
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(heat, origin="lower", cmap=cmap, interpolation="nearest")
    fig.savefig(f'{folder_out}\\Occupancy_Heatmap.png', dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def heatmap_cells(df, arena_size=(580, 485), blur_sigma=3.0, cmap="viridis", gamma=0.9, vmin=0, vmax=0.5):
    # gamma = <1 brightens, >1 darkens, blur_sigma = set 0 for no blur

    heatmap_folder = folder_out / "Heatmaps_Cells"
    heatmap_folder.mkdir(parents=True, exist_ok=True)

    width, height = arena_size
    heat = np.zeros((height, width), dtype=np.float32)
    x = pd.to_numeric(df['head_x'], errors="coerce").to_numpy()
    y = pd.to_numeric(df['head_y'], errors="coerce").to_numpy()

    cell_cols = [c for c in df.columns if "spikeprob" in c]

    for cell in cell_cols:

        val = pd.to_numeric(df[cell], errors="coerce").to_numpy()
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
        fig = plt.figure(figsize=(width/100, height/100), dpi=100, frameon=False)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(heat, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        fig.savefig(f'{heatmap_folder}\\{cell}.png', dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def heatmap_spikeprob(df, min_frames=32, max_frames=1362, max_cells=298, vmin=0, vmax=0.5, heatmap=None, trim_x_axis='Distance'):

    def reorder_rows(combined, shuffle=None):

        if shuffle:
            rng = np.random.default_rng(None)
            idx = rng.permutation(combined.shape[0])
            combined_shuffled = combined[idx]
            return combined_shuffled
        
        else:
            # sort rows by their total activity (row sum)
            row_sums = combined.sum(axis=1)
            sort_idx = np.argsort(row_sums)  # descending
            combined_sorted = combined[sort_idx]
            return combined_sorted
    
    def draw_heatmap_pdf(mat, out_path, vmin, vmax, cmap="viridis", figsize=(5,10)):
    
            plt.figure(figsize=figsize)
            im = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

            cbar = plt.colorbar(im)
            cbar.set_label("Value")

            plt.xlabel("Time (frames / samples)")
            plt.ylabel("Cells")
            plt.tight_layout()
            plt.savefig(out_path + '\\Heatmap.pdf', format='pdf')
            #plt.show()
            plt.close()

    def draw_heatmap_vec(mat, out_path, vmin, vmax, cmap="viridis", figsize=(5,10)):

        fig, ax = plt.subplots(figsize=figsize)

        # pcolormesh expects grid edges; this simple call works for regular grids
        mesh = ax.pcolormesh(mat, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Value")

        ax.set_xlabel("Time (frames / samples)")
        ax.set_ylabel("Cells")

        # match imshow-style orientation (origin="lower")
        ax.set_ylim(0, mat.shape[0])

        fig.tight_layout()
        fig.savefig(out_path+'_heatmap_vector.pdf', format="pdf", bbox_inches="tight")
        plt.close(fig)

    if heatmap is None:
        cols = [c for c in df.columns if str(c).endswith('spikeprob')]

        if trim_x_axis == 'Time':
            # trim recording to a certain max frame
            df = df.iloc[min_frames:max_frames]
            arr = df[cols].to_numpy()
            mat = arr.T

            return mat
        
        elif trim_x_axis == 'Distance':
            # trim recording to a certain distance (e.g. 15m)
            arr = df[cols].to_numpy()      # (time, cells)
            mat = arr.T                   # (cells, time)
            target_frames = 1000           # choose what you want

            n_cells, n_frames = mat.shape
            if n_frames == target_frames:
                return mat.copy()
            old_time = np.linspace(0, 1, n_frames)
            new_time = np.linspace(0, 1, target_frames)
            mat_resampled = np.empty((n_cells, target_frames))
            for i in range(n_cells):
                mat_resampled[i] = np.interp(new_time, old_time, mat[i])
            
            return mat_resampled



    else:
        # create shuffled cells of same number
        combined = np.vstack(heatmap)
        combined = reorder_rows(combined, shuffle=True)
        combined = combined[:max_cells]
        combined = reorder_rows(combined, shuffle=False)

        draw_heatmap_pdf(combined, folder_out, vmin, vmax)
        #draw_heatmap_vec(combined, folder_out, vmin, vmax)
    

def SpikeProb(df, file):
    # looks at the frequency of spike probabilities

    # identify spikeprob columns
    cell_cols = [c for c in df.columns if "spikeprob" in c]
    n_rows = len(df)


    per_cell_sum = df[cell_cols].sum(axis=0)
    mean_of_cell_sums = per_cell_sum.mean()

    # Method 1: Mean per cell
    per_cell_mean = df[cell_cols].mean(axis=0)
    mean_of_cell_means   = per_cell_mean.mean()
    median_of_cell_means = per_cell_mean.median()

    # Method 2: Sum per cell / number of rows
    per_cell_sum_div_rows = df[cell_cols].sum(axis=0) / n_rows
    mean_of_sum_div_rows   = per_cell_sum_div_rows.mean()
    median_of_sum_div_rows = per_cell_sum_div_rows.median()

    spikeprob_summary = {"file": Path(file).stem,
                        "spike_sum": mean_of_cell_sums,
                        "mean_of_cell_means": mean_of_cell_means,
                        "median_of_cell_means": median_of_cell_means,
                        "mean_of_cell_sum_div_rows": mean_of_sum_div_rows,
                        "median_of_cell_sum_div_rows": median_of_sum_div_rows,
                        "n_rows": n_rows
                        }
    
    # create one for saving immediately, one for all and one summary
    spikeprob_individ_all = []
    spikeprob_individ = []
    for cell in cell_cols:
        spikeprob_individ_all.append(   {'Cell_ID': cell,
                                    'SpikeRate': per_cell_mean[cell],
                                    #"per_cell_sum_div_rows": per_cell_sum_div_rows[cell]
                                    })
        spikeprob_individ.append(   {file.stem: per_cell_mean[cell]})

    spikeprob_individ_folder = Path(folder_out + "\\SpikeProb_Individ")
    spikeprob_individ_folder.mkdir(parents=True, exist_ok=True)
    df_spikeprob_individ = pd.DataFrame(spikeprob_individ)
    df_spikeprob_individ.to_csv(f'{spikeprob_individ_folder}\\{file.stem}.csv', index=False)

    df_spikeprob_individ_all = pd.DataFrame(spikeprob_individ_all)

    return spikeprob_summary, df_spikeprob_individ_all

def SpikeRate_Speed(df, file, fps=10, bin_size_s=10.0):

    cell_cols = [c for c in df.columns if 'spikeprob'  in c]

    # get distance & speed per frame
    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    dist.iloc[0] = 0
    df["speed_mm_s"] = dist * fps

    # time binning
    df["time_bin"] = (df.index // bin_size_s).astype(int)

    out_rows = []

    for cell in cell_cols:

        grouped_obj = df.groupby("time_bin")
        mean_speed = grouped_obj["speed_mm_s"].mean()
        spike_sum = grouped_obj[cell].sum()
        grouped = pd.DataFrame({"mean_speed_mm_s": mean_speed,
                                'cell': spike_sum}) # change 'cell' to file.stem if you want one big file for overview
        out_rows.append(grouped)

    if out_rows:
        # out = pd.concat(out_rows, ignore_index=True)
        # x = out['mean_speed_mm_s']
        # y = out['cell']
        # result = linregress(x, y)
        # print(result.slope)
        return pd.concat(out_rows, ignore_index=True)

    else:
        return None


def SI(df, file_in, fps=10, draw_cells=False, do_shuffle=False, n_shuffles=1, significance_level=0.001):
    
    # real spatial info
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


    main_df, place_df = bin_cellspikes_prob_OF(main_df)

    occupancy = df.groupby('head_bin').size() # how many frames did animal spend in each bin
    occupancy = occupancy.reindex(place_df.index, fill_value=0) # no idea
    occupancy_sec = occupancy / fps 
    occ_nonzero = occupancy_sec.replace(0, np.nan)  # replacing 0 with nan
    df_firing_freq = place_df.div(occ_nonzero, axis=0).fillna(0.0)  # what was fring frequency in each bin

    # goes through all cells and gets SI
    SI_real_cells = []
    for cell in df_firing_freq.columns:
        info = spatial_information(df_firing_freq[cell].values, occupancy.values)
        SI_real_cells.append(   {"cell_ID": cell,
                                "SI": info})

        if draw_cells:
            heatmap_cells(df, cell)

    # save each SI values of the file individually
    SI_individ_folder = Path(folder_out + "\\SI_Individ")
    SI_individ_folder.mkdir(parents=True, exist_ok=True)
    df_SI_individ = pd.DataFrame(SI_real_cells)
    df_SI_individ.to_csv(f'{SI_individ_folder}\\{file_in.stem}.csv', index=False)

    # return a summary of the SI
    SI_real_values = np.array([d["SI"] for d in SI_real_cells])
    SI_summary =    {"file": Path(file_in).stem,
                    "SI_mean": np.mean(SI_real_values),
                    "SI_median": np.median(SI_real_values),
                    "n_cells": len(SI_real_values)
                    }

    place_cells = []
    if do_shuffle:
        # ---------- shuffled spatial info ----------
        # We’ll keep your occupancy-based shuffle idea but adapt it to probabilities.
        # 1) total_prob ~ expected spike count
        # 2) draw an integer spike count from a Poisson with that mean
        # 3) distribute these spikes across bins according to occupancy probability
        occupancy_probs = occupancy / occupancy.sum()
        SI_shuffled = {cell: [] for cell in place_df.columns}

        for _ in tqdm(range(n_shuffles)):
            for cell in place_df.columns:
                total_prob = place_df[cell].sum()

                # expected number of spikes -> sample integer total spikes
                n_spikes = np.random.poisson(total_prob)
                if n_spikes == 0:
                    SI_shuffled[cell].append(0.0)
                    continue

                # distribute spikes into bins according to occupancy
                shuffled_bins = np.random.choice(a=occupancy.index, size=n_spikes, p=occupancy_probs.values)
                shuffled_counts = (pd.Series(shuffled_bins).value_counts().reindex(occupancy.index, fill_value=0))
                shuffled_firing_rate = (shuffled_counts / occ_nonzero).fillna(0.0)
                info = spatial_information(shuffled_firing_rate.values, occupancy.values)
                SI_shuffled[cell].append(info)

        # significance test
        p_values = {}
        SI_real_dict = {d["cell_ID"]: d["SI"] for d in SI_real_cells}   # quick transform into real dictionary, easier
        cell_IDs = np.array(list(SI_real_dict.keys()))
        for cell in cell_IDs:
            real_si = SI_real_dict[cell]
            shuffled_si = np.array(SI_shuffled[cell])
            p = np.mean(shuffled_si >= real_si)
            p_values[cell] = p
            if p < significance_level:
                place_cells.append(cell)

        # return a summary of the analyse
        SI_summary["place_cells_nb"] = len(place_cells)
        SI_summary["place_cells_quotient"] = len(place_cells) / len(SI_real_cells)
    
    return SI_summary, place_cells
    
    
def FFN(df, cells, file_in, n_cells=10, n_repeats_rand_cells=1, draw=False):

    def prep_data(main_df, place_cells, sampling_rate=10, bin_size_frames=10, input_mode="rate"):
        '''
        Prepare X (neurons) and Y (x,y). Works for spike probabilities.
        - input_mode="sum": X = sum(p_t) per bin (expected spike count)
        - input_mode="rate": X = sum(p_t) / bin_seconds (estimated spikes/s)
        - transform="sqrt": apply sqrt to X (variance-stabilizing)
        Returns:
        X (np.float32), Y (np.float32), scaler (fitted StandardScaler) -- scaler is None here;
        if you want to split train/test outside, fit scaler on train and transform test likewise.
        '''

        position_cols = ['head_x', 'head_y']
        n_bins = len(main_df) // bin_size_frames

        rows = []
        for i in range(n_bins):
            start = i * bin_size_frames
            end = (i + 1) * bin_size_frames
            segment = main_df.iloc[start:end]

            # sum of spike probabilities per bin -> expected spike counts in bin
            # per downsampled row, we have the animal position and the spikerate
            spike_sums = segment[place_cells].sum(axis=0).astype(float)
            if input_mode == "sum":
                spikes = spike_sums.values
            elif input_mode == "rate":
                bin_seconds = bin_size_frames / sampling_rate
                spikes = (spike_sums.values / bin_seconds)
            position_avg = segment[position_cols].mean().values.astype(float)
            rows.append(np.concatenate([spikes, position_avg]))
        df_binned = pd.DataFrame(rows, columns=list(place_cells) + position_cols)

        X = df_binned[place_cells].values.astype(np.float32)
        Y = df_binned[position_cols].values.astype(np.float32)

        return X, Y

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

    def train_model(train_loader, test_loader, model, n_epochs=10, lr=1e-3, verbose=True):

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

        if draw is False:
            return

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
        #print(f"Mean decoding error: {errors.mean():.2f} mm, Accuracy: {accuracy}%\n")

        return accuracy, errors


    results_acc, results_err = [], []

    for i in range(n_repeats_rand_cells):

        sampled_cells = np.random.choice(cells, size=n_cells, replace=False)

        # prepare X and Y
        X, Y = prep_data(df, sampled_cells)

        # split & scale
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        # create loaders
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
        test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(Y_test))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

        # model & train
        model = PositionDecoder(input_dim=X.shape[1], hidden_dim=128)
        trained_model = train_model(train_loader, test_loader, model)

        # test & accuracy
        Y_pred, Y_true = test_model(trained_model, test_loader)
        accuracy, errors = calc_accuracy(Y_true, Y_pred, threshold_mm=50)
        plot_true_vs_pred_coords(Y_true, Y_pred, accuracy)

        results_acc.append(accuracy)
        results_err.append(errors)
    
    all_err = np.concatenate(results_err)

    FFN_summary =    {"file": Path(file_in).stem,
                    "FFN_accuracy_mean": np.mean(results_acc),
                    "FFN_error_mean": float(np.mean(all_err)),
                    "n_repeats_rand_cells": n_repeats_rand_cells,
                    "n_cells": n_cells
                    }

    return FFN_summary




df_QC = pd.read_csv(file_QC, index_col='cell_ID')

large_heatmap = []
spikerate_speed = pd.DataFrame()

all_summary_spikeprob = []
df_all_individ_spikeprob = pd.DataFrame()
all_summary_SI = []
all_summary_FFN = []


animals = ['48', '51', '56', '66', '72', '77']
animals = ['_']
for anim in animals:

    files = get_files(folder_in, anim, 'YM')
    for i, file in enumerate(files):

        file = Path(file)
        print(f'\n------- {file.stem} -------')
        
        main_df = pd.read_csv(file, index_col='Mastertime [s]')
        main_df = clean_df_QC(main_df, df_QC)
        main_df = clean_df_distance(main_df, file)

        # drawing
        #heatmap_occupancy(main_df)
        #heatmap_cells(main_df)
        large_heatmap.append(heatmap_spikeprob(main_df))

        
        # Spike Probability
        if True and main_df is not None:
            spikeprob_summary, df_spikeprob_individ = SpikeProb(main_df, file)
            df_all_individ_spikeprob = pd.concat([df_all_individ_spikeprob, df_spikeprob_individ], ignore_index=True)
            all_summary_spikeprob.append(spikeprob_summary)

        # Correlation SpikeRate-AnimalSpeed
        if False and main_df is not None:
            spikespeed = SpikeRate_Speed(main_df, file)
            spikerate_speed = pd.concat([spikerate_speed, spikespeed], ignore_index=True)

        # Spatial Information
        if False and main_df is not None:
            SI_summary, place_cells = SI(main_df, place_df, file)
            all_summary_SI.append(SI_summary)

        # FFN
        if False and main_df is not None:
            cells_to_use = [c for c in main_df.columns if "spikeprob" in c] # or place_cells, whatever is needed
            FFN_summary = FFN(main_df, cells_to_use, file)
            all_summary_FFN.append(FFN_summary)
            

# save all the data
df_all_individ_spikeprob.to_csv(out_spikeprob_individ, index=False)
spikerate_speed.to_csv(out_spikerate_speed, index=False)

df_all_spikeprob_summary = pd.DataFrame(all_summary_spikeprob)
df_all_spikeprob_summary.to_csv(out_spikeprob_summary, index=False)

df_all_SI_summary = pd.DataFrame(all_summary_SI) 
df_all_SI_summary.to_csv(out_SI_summary, index=False)

#df_all_FFN_summary = pd.DataFrame(all_summary_FFN)
#df_all_FFN_summary.to_csv(out_FFN_summary, index=False)

# drawing
heatmap_spikeprob(main_df, heatmap=large_heatmap)

padded = []
max_len = max(len(x) for x in dist_travelled)
for arr in dist_travelled:
    pad_width = max_len - len(arr)
    padded_arr = np.pad(arr, (0, pad_width), constant_values=np.nan)
    padded.append(padded_arr)
df_export = pd.DataFrame(padded).T
df_export.to_csv("distances.csv", index=False)
