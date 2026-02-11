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


folder_in = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\Heatmap_YM")
df_QC = pd.read_csv(r"D:\CA1Dopa_Longitud\Calculations\df_QC_FINAL_pass.csv", index_col='cell_ID')

def heatmap_cells(df, value_col, out_jpg, arena_size=(580, 485), blur_sigma=3.0, cmap="viridis", gamma=0.9, vmin=0, vmax=0.5):
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

    ax.imshow(heat, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
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

def identify_place_cells_prob(main_df, place_df, fps=10, n_shuffles=100, significance_level=0.001):
    
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

        heatmap_cells(main_df, value_col=cell, out_jpg=f"C:\\Users\\landgrafn\\Desktop\\CA1Dopa_Paper\\Mini\\Heatmap_YM\\cells\\{round(info, 5)}.jpg")


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

def ffn(main_df, place_cells):

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


    # prepare X and Y
    X, Y = prep_data(main_df, place_cells)

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

    return Y_pred, Y_true, trained_model, accuracy, errors




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
        
        # get spatial information
        main_df, place_df = bin_cellspikes_prob_OF(main_df)
        spatial_info_real, spatial_info_shuffled, place_cells, p_values, percent_place = identify_place_cells_prob(main_df, place_df)
        means.append(np.array(list(spatial_info_real.values())).mean())
        medians.append(np.median(np.array(list(spatial_info_real.values()))))
        animal_results[file.stem] = list(spatial_info_real.values())
        place_dict[file.stem] = percent_place
'''
        # FFN
        n_cells = 10
        n_repeats = 1
        results_acc, results_err = [], []
        for i in range(n_repeats):
            sampled_cells = np.random.choice(place_cells, size=min(n_cells, len(place_cells)), replace=False)
            Y_pred, Y_true, model, acc, error = ffn(main_df, sampled_cells)
            results_acc.append(acc)
            results_err.append(error)

        results_acc_mean = np.mean(results_acc)
        results_acc_stdv = np.std(results_acc)
        results_err_mean = np.mean(results_err)
        results_err_stdv = np.std(results_err)
        #print(results_err)
        #print(results_err_mean)
        #print(results_err_stdv)
        # print(results_acc)
        # print(results_acc_mean)
        # print(results_acc_stdv)



    series_dict = {fname: pd.Series(vals) for fname, vals in animal_results.items()}
    
    df_out = pd.DataFrame(series_dict)
    df_out.to_csv(rf"D:\results_{anim}_YM.csv", index=False)

place_dict = {fname: pd.Series(vals) for fname, vals in place_dict.items()} 
place_out = pd.DataFrame(place_dict) 
place_out.to_csv(rf"D:\results_{anim}_YM_place.csv", index=False)
'''

#%%
print(place_dict)

place_dict = {fname: pd.Series(vals) for fname, vals in place_dict.items()} 
place_out = pd.DataFrame(place_dict) 
place_out.to_csv(rf"D:\results_{anim}_YM_place.csv", index=False)


