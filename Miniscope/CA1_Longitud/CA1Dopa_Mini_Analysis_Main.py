#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats, ndimage, interpolate
from shapely.geometry import Polygon
from matplotlib.ticker import ScalarFormatter
import copy


folder_in   = Path(r"E:\CA1Dopa_Miniscope\Analysis\Master_10_merge_dfon+dff")
folder_out  = Path(r"E:\CA1Dopa_Miniscope\Analysis\out")
QC_file     = Path(r"E:\CA1Dopa_Miniscope\Analysis\df_QC_FINAL_pass.csv")

plot_once = False
col_spikeprob = 'dff_spikeprob'

# Clean
def get_files(path, common_name1, common_name2, suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name1 in p.name) and (common_name2 in p.name) and p.name.endswith(suffix)])
    #print(f"\n{len(files)} files found")
    #for f in files:
    #    print(f)
    #print('\n\n')
    return files

def clean_df_QC(file, QC_file):

    df_QC   = pd.read_csv(QC_file, index_col='cell_ID')
    df = pd.read_csv(file, index_col='Mastertime [s]')

    if df is None:
        return None

    # do QC according to pass True or False
    cols_to_keep = []
    cells_orig, cells_kept, cells_dropped = 0, 0, 0
    for col in df.columns:
        # only apply QC to neuron columns
        if "_C" not in col:
            cols_to_keep.append(col)
            continue

        cells_orig += 1
        # if the name is not in pass_df, be conservative and drop
        name = col
        for suffix in ["_dff", "_dfon"]:
            name = name.split(suffix)[0]

        if name not in df_QC.index:
            print(f'{name} not in pass_df')
            cells_dropped += 1
            continue

        # keep only if pass == True
        if df_QC.loc[name, "pass"]:
            cols_to_keep.append(col)
            cells_kept += 1
            continue

        cells_dropped += 1


    df = df[cols_to_keep]

    QC_summary = {"QC_Cells_Original" : int(cells_orig / 6),
                  "QC_Cells_Dropped"  : int(cells_dropped / 6),
                  "QC_Cells_Kept"     : int(cells_kept / 6)}

    return df, QC_summary

def clean_df_distance(df, file, dict_dist_travelled, start_row=34, maxYM=20, maxOF=30):

    if df is None:
        return None

    # trim main_df according to distance travelled
    df = df.iloc[start_row:]

    # in meters
    print(file)
    if 'YM' in file:
        maxdist = maxYM * 1000
    elif 'OF' in file:
        maxdist = maxOF * 1000
    else:
        raise ValueError('Neither YM nor OF in file name')
    
    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    step_dist = np.sqrt(dx**2 + dy**2)
    cum_dist = step_dist.fillna(0).cumsum()
    dict_dist_travelled[file] = cum_dist.values
    
    # if never reached max distance, just return full df
    reached = cum_dist >= maxdist
    
    if reached.any():
        cut_label = cum_dist.index[reached.argmax()]  # first index where reached
        #print(f'reached maxdist after {int(cut_label)}s / {int(max(df.index))}s ->trimmed')
        df = df.loc[:cut_label].copy()

        distance_summary = {"Distance_travelled"   : round(cum_dist.values.max(), 2),
                            "Distance_Cutoff_dist" : round(maxdist, 2),
                            "Distance_Cutoff_time" : cut_label}

        return df, distance_summary, dict_dist_travelled
    
    else:
        distance_summary = {"Distance_travelled"   : round(cum_dist.values.max(), 2),
                            "Distance_Cutoff_dist" : round(maxdist, 2),
                            "Distance_Cutoff_time" : df.index.max()}
        return df, distance_summary, dict_dist_travelled

# Plot
def heatmap_occupancy(df, file, arena_size=(580, 485), blur_sigma=4.0, cmap="Blues", gamma=0.8):
    
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
        heat = ndimage.gaussian_filter(heat, sigma=blur_sigma)

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
    
    # save
    folder_occupancy = Path(folder_out / "Heatmaps_Occupancy")
    folder_occupancy.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{folder_occupancy}\\{file}_OccupancyHeatmap.png', dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def heatmap_cells(df, file, cell, spat_info, arena_size=(580, 485), blur_sigma=3.0, cmap="viridis", gamma=0.9, vmin=0, vmax=0.5):
    # gamma = <1 brightens, >1 darkens, blur_sigma = set 0 for no blur

    heatmap_folder = folder_out / "Heatmaps_Cells"
    heatmap_folder.mkdir(parents=True, exist_ok=True)

    width, height = arena_size
    x = pd.to_numeric(df['head_x'], errors="coerce").to_numpy()
    y = pd.to_numeric(df['head_y'], errors="coerce").to_numpy()

    cell_cols = [c for c in df.columns if col_spikeprob in c]

    for cell in cell_cols:

        heat = np.zeros((height, width), dtype=np.float32)

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
            heat = ndimage.gaussian_filter(heat, sigma=blur_sigma)

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

        # save
        folder_heatmap_cells      = Path(folder_out / "Heatmaps_Cells")
        folder_heatmap_cells_file = Path(folder_out / "Heatmaps_Cells" / file)
        folder_heatmap_cells     .mkdir(parents=True, exist_ok=True)
        folder_heatmap_cells_file.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{folder_heatmap_cells_file}\\{file}_{cell}_{round(spat_info, 3)}.png', dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def heatmap_spikeprob(df, file, min_frames=32, max_frames=1362, max_cells=298, vmin=0, vmax=0.001, heatmap=None, trim_x_axis='Distance'):

    def reorder_rows(combined, shuffle=None):

        if shuffle:
            rng = np.random.default_rng(None)
            idx = rng.permutation(combined.shape[0])
            combined_shuffled = combined[idx]
            return combined_shuffled
        
        else:
            # sort rows by their total activity (row sum)
            row_sums = combined.sum(axis=1)
            sort_idx = np.argsort(row_sums)  # ascending
            combined_sorted = combined[sort_idx]
            return combined_sorted

    def draw_heatmap_png(mat, vmin, vmax, file, cmap="viridis", figsize=(5,10)):

        plt.figure(figsize=figsize)
        im = plt.imshow(mat, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

        cbar = plt.colorbar(im)
        cbar.set_label("Value")

        plt.xlabel("Time (frames / samples)")
        plt.ylabel("Cells")
        plt.tight_layout()

        # save
        folder_heatmap_Spikeprob = Path(folder_out / "Heatmaps_Spikeprob")
        folder_heatmap_Spikeprob.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{folder_heatmap_Spikeprob}\\{file}.png')
        plt.close()
    
    def draw_heatmap_pdf(mat, vmin, vmax, file, cmap="viridis", figsize=(5,10)):
    
        plt.figure(figsize=figsize)
        im = plt.imshow(mat, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

        cbar = plt.colorbar(im)
        cbar.set_label("Value")

        plt.xlabel("Time (frames / samples)")
        plt.ylabel("Cells")
        plt.tight_layout()

        # save
        folder_heatmap_Spikeprob = Path(folder_out / "Heatmaps_Spikeprob")
        folder_heatmap_Spikeprob.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{folder_heatmap_Spikeprob}\\Summary.pdf', format='pdf')
        plt.close()

    def draw_heatmap_vec(mat, vmin, vmax, file, cmap="viridis", figsize=(5,10)):

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
        cols = [c for c in df.columns if str(c).endswith(col_spikeprob)]

        if trim_x_axis == 'Time':
            # trim recording to a certain max frame
            df = df.iloc[min_frames:max_frames]
            arr = df[cols].to_numpy()
            mat = arr.T

            draw_heatmap_png(mat, vmin, vmax, file)
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

            draw_heatmap_png(mat, vmin, vmax, file)
            return mat_resampled

    else:
        # create shuffled cells of same number
        combined = np.vstack(heatmap)
        combined = reorder_rows(combined, shuffle=True)
        combined = combined[:max_cells]
        combined = reorder_rows(combined, shuffle=False)

        draw_heatmap_pdf(combined, vmin, vmax, file)
        #draw_heatmap_vec(combined, folder_out, vmin, vmax)

def plot_groups_columns(results_df, column, y_title, title):

    groups = ['mKate', 'A53T']

    plt.figure(figsize=(3, 5))

    for gi, group in enumerate(groups):
        sub = results_df[results_df["file"].str.contains(group, na=False)][column].dropna()

        # mean ± SEM overlay
        mean = sub.mean()
        sem = sub.sem()
        plt.errorbar(gi, mean, yerr=sem, fmt="-", capsize=10)
        plt.hlines(mean, gi-0.1, gi+0.1, colors="black", linewidth=2)

        # scatter
        plt.scatter(gi + np.random.uniform(-0.08, 0.08, len(sub)), sub, s=70)

    plt.xticks(range(len(groups)), groups)
    plt.ylabel(y_title)

    # t-test
    a = results_df.loc[results_df["file"].str.contains("mKate", regex=False, na=False), column].dropna().to_numpy()
    b = results_df.loc[results_df["file"].str.contains("A53T", regex=False, na=False), column].dropna().to_numpy()
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=True)
    ax = plt.gca()
    ax.text(0.02, 0.98, f'p = {round(p_val, 3)}', transform=ax.transAxes, va="top")

    plt.title(title)
    plt.tight_layout()
    plt.show()

# Analyze
def SpikeProb(df, file):

    def save(spikeprob_individ_all, file):

        df_spikeprob_individ_all = pd.DataFrame(spikeprob_individ_all)
        spikeprob_individ_folder = Path(folder_out / "Individ_SpikeProb")
        spikeprob_individ_folder.mkdir(parents=True, exist_ok=True)
        df_spikeprob_individ_all.to_csv(f'{spikeprob_individ_folder}\\{file.stem}.csv', index=False)
        
    # looks at the frequency of spike probabilities

    # identify spikeprob columns
    cell_cols = [c for c in df.columns if col_spikeprob in c]
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

    # summary that appends to the main_summary
    spikeprob_summary = {"SpikeProb_SpikeSum"                   : mean_of_cell_sums,
                         "SpikeProb_Mean_of_cell_means"         : mean_of_cell_means,
                         "SpikeProb_Median_of_cell_means"       : median_of_cell_means,
                         "SpikeProb_Mean_of_cell_sum_div_rows"  : mean_of_sum_div_rows,
                         "SpikeProb_Median_of_cell_sum_div_rows": median_of_sum_div_rows,
                         "SpikeProb_RecLength"                     : n_rows}
    
    # save one per file with individual cell info
    spikeprob_individ_all = []
    for cell in cell_cols:
        spikeprob_individ_all.append({'Cell_ID'              : cell,
                                      'SpikeRate'            : per_cell_mean[cell],
                                      'per_cell_sum_div_rows': per_cell_sum_div_rows[cell]})
        
    save(spikeprob_individ_all, file)
    return spikeprob_summary

def SpikeRate_Speed(df, file, fps=10, bin_size_s=1.0):

    def save_individ(spikerates, file):
    
        spikerate_individ_folder = Path(folder_out / "Individ_SpikeRate-Speed")
        spikerate_individ_folder.mkdir(parents=True, exist_ok=True)
        spikerates.to_csv(f'{spikerate_individ_folder}\\{file}.csv', index=False)

    df = df.copy()
    cell_cols = [c for c in df.columns if col_spikeprob  in c]

    # get distance & speed per frame
    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    dist.iloc[0] = 0
    df["speed_mm_s"] = dist * fps

    # time binning
    df["time_bin"] = (df.index // bin_size_s).astype(int)


    grouped_obj = df.groupby("time_bin")
    mean_speed = grouped_obj["speed_mm_s"].mean()

    # each cell one column, spike summed for each time bin
    spikerates = grouped_obj[cell_cols].sum()
    # add metadata columns together
    info = pd.DataFrame({"time_bin_s": (spikerates.index + 0.5) * bin_size_s, "mean_speed_mm_s": mean_speed}, index=spikerates.index)
    spikerates = pd.concat([info, spikerates], axis=1)
    save_individ(spikerates, file)

    # get the regression between SpikeRate in the time bin and the corresponding animal speed
    x = spikerates["mean_speed_mm_s"]
    regression_results = []
    for cell in cell_cols:
        y = spikerates[cell]
        result = stats.linregress(x, y)
        regression_results.append({"cell": cell,
                                   "slope": result.slope,
                                   "r": result.rvalue,
                                   "p": result.pvalue})

    regression_results = pd.DataFrame(regression_results)
    regression_summary = {"SpikeRate-Speed_Slope_Mean": regression_results["slope"].mean(),
                          "SpikeRate-Speed_R_Mean": regression_results["r"].mean(),
                          "SpikeRate-Speed_P_Mean": regression_results["p"].mean()}

    return regression_summary

def SI(df, file, fps=10, OF_n_bins=5, YM_bin='mid', YM_head_dir=False, YM_grid_size=73.5, draw_cells=False, do_shuffle=True, n_shuffles=1000, sig_level=0.05, min_occup_s=1.0):
    
    # real spatial info
    # assumes place_df entries are sums of spike probabilities (expected spike counts)
    def bin_cellspikes_prob_OF(df, n_bins, xmax=580, ymax=485):
        global plot_once

        def get_arena():
            if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
                raise ValueError("n_bins must be a positive integer")
            if xmax <= 0 or ymax <= 0:
                raise ValueError("xmax and ymax must be larger than 0")

            bounds = (0.0, float(xmax), 0.0, float(ymax))
            arena_outline = np.array([[0, 0], [xmax, 0], [xmax, ymax], [0, ymax]], dtype=float)

            return arena_outline, bounds

        def get_position_bin(df, bounds):

            head_coords = (df[["head_x", "head_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))

            x_min, x_max, y_min, y_max = bounds

            finite_coords = np.isfinite(head_coords).all(axis=1)
            inside_arena =  (finite_coords & 
                            (head_coords[:, 0] >= x_min) &
                            (head_coords[:, 0] <= x_max) &
                            (head_coords[:, 1] >= y_min) &
                            (head_coords[:, 1] <= y_max))
                
            x_bin = np.full(len(df), -1, dtype=int)
            y_bin = np.full(len(df), -1, dtype=int)

            x_bin[inside_arena] = np.floor((head_coords[inside_arena, 0] - x_min) / (x_max - x_min) * n_bins).astype(int)
            y_bin[inside_arena] = np.floor((head_coords[inside_arena, 1] - y_min) / (y_max - y_min) * n_bins).astype(int)

            # Coordinates exactly at xmax/ymax would initially receive index n_bins
            x_bin[inside_arena] = np.clip(x_bin[inside_arena], 0, n_bins - 1)
            y_bin[inside_arena] = np.clip(y_bin[inside_arena], 0, n_bins - 1)

            position_bin = np.full(len(df), "", dtype=object)
            position_bin[inside_arena] = [f"OF_y{iy:03d}_x{ix:03d}" for ix, iy in zip(x_bin[inside_arena], y_bin[inside_arena])]

            return position_bin, x_bin, y_bin

        def plot_OF_bins(arena_outline, bounds):

            x_min, x_max, y_min, y_max = bounds
            x_edges = np.linspace(x_min, x_max, n_bins + 1)
            y_edges = np.linspace(y_min, y_max, n_bins + 1)

            fig, ax = plt.subplots(figsize=(7, 6))

            outline = plt.Polygon(arena_outline, fill=False, linewidth=2)
            ax.add_patch(outline)

            for x in x_edges:
                ax.plot([x, x], [y_min, y_max], linewidth=0.7)

            for y in y_edges:
                ax.plot([x_min, x_max], [y, y], linewidth=0.7)

            ax.set_xlim(x_min - 20, x_max + 20)
            ax.set_ylim(y_min - 20, y_max + 20)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_title(f"Open-field spatial bins: {n_bins} x {n_bins} = {n_bins**2} bins")

            plt.tight_layout()
            plt.show()

        # Preparation
        df = df.copy()

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        required_columns = ["head_x", "head_y"]
        missing_columns = [column for column in required_columns if column not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing position columns: {missing_columns}")

        arena_outline, bounds = get_arena()
        position_bin, x_bin, y_bin = get_position_bin(df, bounds)

        # Add spatial-bin information
        df["of_position_bin"] = position_bin
        df["of_x_bin"] = x_bin
        df["of_y_bin"] = y_bin
        df["head_bin"] = position_bin

        # Report invalid frames instead of clipping them into edge bins
        outside_spatial_bins = df["head_bin"] == ""

        if outside_spatial_bins.sum() > 0:
            print(f"Invalid OF frames outside spatial bins: {outside_spatial_bins.sum()}")

        # Keep only frames inside the open field
        df = df.loc[df["head_bin"] != ""].copy()

        if df.empty:
            raise ValueError("No valid open-field frames remained after bin assignment")

        # Find Cascade spike-probability columns
        spike_prob_cells = [column for column in df.columns if isinstance(column, str) and column.endswith(col_spikeprob)]

        if not spike_prob_cells:
            raise ValueError(f"No columns end with {col_spikeprob}")

        # Sum spike probability within every spatial bin
        place_df = (df.groupby("head_bin", sort=True)[spike_prob_cells].sum(min_count=1).fillna(0.0))

        # Include every possible OF bin, including bins never visited
        all_bins = [f"OF_y{iy:03d}_x{ix:03d}" for iy in range(n_bins) for ix in range(n_bins)]

        place_df = place_df.reindex(all_bins, fill_value=0.0)

        if plot_once:
            plot_OF_bins(arena_outline, bounds)
            plot_once = False

        return place_df, df

    def bin_cellspikes_prob_YM(df, binning, head_direction, grid_size):
        global plot_once

        # binning: mid/big/grid, head_direction: True/False

        def midpoint(a, b, t=0.5):
            # for getting middle points: (0.0-0.5 closer to a), (0.5-1.0 closer to b)
            return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))

        def make_path(vertices):
            
            # ensure that all polygons have the same orientation, makes boundary handling with contains_points more consistent
            vertices = np.asarray(vertices, dtype=float)
            x = vertices[:, 0]
            y = vertices[:, 1]

            signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            if signed_area < 0:
                vertices = vertices[::-1]

            return MplPath(vertices)

        def get_arena_bins():

            '''
            Returns:

            poly_big: the YM separated into 4 areas: the 3 arms + center
            poly_mid: the YM separated into 7 areas: 1 outer and 1 inner for the 3 arms and center
            arena_outline: the corners that defines the outer shape of the YM
            arm_vectors: the vector from the center to the arm_endpoint for each of the 3 arms
            endpoints: the endpoints of each arm and the center middle
            bounds: the outer points of the YM, basically the min and max values of the YM
            '''

            # ci = inner corner
            ci1 = (265, 275)
            ci2 = (295, 335)
            ci3 = (325, 275)

            # ca = outer corner
            caL1 = (0, 442)
            caL2 = (25, 480)
            caM1 = (265, 0)
            caM2 = (325, 0)
            caR1 = (560, 480)
            caR2 = (585, 440)

            # h  = halfway point along an arm
            hL1 = midpoint(ci1, caL1)
            hL2 = midpoint(ci2, caL2)
            hM1 = midpoint(ci1, caM1)
            hM2 = midpoint(ci3, caM2)
            hR1 = midpoint(ci2, caR1)
            hR2 = midpoint(ci3, caR2)

            # Center and distal endpoint of every arm
            end_C = np.mean([ci1, ci2, ci3], axis=0)
            arm_ends = {"L": np.mean([caL1, caL2], axis=0),
                        "M": np.mean([caM1, caM2], axis=0),
                        "R": np.mean([caR1, caR2], axis=0)}
            endpoints = {"C": end_C, **arm_ends}
                
            # Normalized vectors pointing from center toward the end of each arm
            arm_vectors = {}
            for arm, arm_endpoint in arm_ends.items():
                direction = np.asarray(arm_endpoint, dtype=float) - end_C
                length = np.linalg.norm(direction)
                if length <= 0:
                    raise ValueError(f"Zero-length direction vector for arm {arm}")
                arm_vectors[arm] = direction / length

            # complete arms plus center, center is inserted first so shared boundary points preferentially belong to the center
            poly_big = {"C": make_path([ci1, ci2, ci3]),
                        "L": make_path([caL1, caL2, ci2, ci1]),
                        "M": make_path([caM1, ci1, ci3, caM2]),
                        "R": make_path([ci2, caR1, caR2, ci3])}
                
            # inner and outer halves of every arm plus center
            poly_mid = {"C": make_path([ci1, ci2, ci3]),
                        "L_outer": make_path([caL1, caL2, hL2, hL1]),
                        "L_inner": make_path([ci1, hL1, hL2, ci2]),
                        "M_outer": make_path([caM1, hM1, hM2, caM2]),
                        "M_inner": make_path([hM1, ci1, ci3, hM2]),
                        "R_outer": make_path([hR1, caR1, caR2, hR2]),
                        "R_inner": make_path([ci2, hR1, hR2, ci3])}
                
            # one concave polygon describing the complete Y-maze, bounds are the smallest and largest y and y coords of that polygon
            arena_outline = make_path([caM1, caM2, ci3, caR2, caR1, ci2, caL2, caL1, ci1])
            bounds = (arena_outline.vertices[:, 0].min(),
                      arena_outline.vertices[:, 0].max(),
                      arena_outline.vertices[:, 1].min(),
                      arena_outline.vertices[:, 1].max())
            
            return poly_big, poly_mid, arena_outline, arm_vectors, endpoints, bounds

        def assign_area(points, areas):

            # assigns each point to an area in your arena and returns something like labels = ['L', 'L', 'C', ...]

            # creates an empty label for each point
            labels = np.full(len(points), "", dtype=object)
            for name, path in areas.items():

                # inside if the point is inside the area and points that dont already have a label and that are inside this polygon
                inside = path.contains_points(points)
                not_assigned = labels == ""
                labels[inside & not_assigned] = name
                
            return labels

        def get_position_bin(df, binning, poly_big, poly_mid, bounds):

            head_coords = (df[["head_x", "head_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
            
            # little sanity check that all coords are valid
            if np.isnan(head_coords).any():
                print("Warning: head coordinates contain NaNs.")

            # bin the animals head into the existing areas big or mid
            arm = assign_area(head_coords, poly_big)

            if binning == 'big':
                position_bin = arm.copy()
            elif binning == 'mid':
                position_bin = assign_area(head_coords, poly_mid)
    
            elif binning == 'grid':
                # put a rectangular grid over the entire YM and check for each frame, in which grid square is the nead in
                # grid size
                
                # get the extent of the whole YM, check if the head is in the grid and get the row & column of the respective cell
                x_min, x_max, y_min, y_max = bounds
                inside_arena = arena_outline.contains_points(head_coords)
                grid_x_float = np.floor((head_coords[:, 0] - x_min) / grid_size)
                grid_y_float = np.floor((head_coords[:, 1] - y_min) / grid_size)
    
                # check if the frame is valid due to head inside YM AND x and y a valid number. If not valid, its -1
                valid_grid = (inside_arena & np.isfinite(grid_x_float) & np.isfinite(grid_y_float))
                grid_x_index = np.where(np.isfinite(grid_x_float), grid_x_float, -1).astype(int)
                grid_y_index = np.where(np.isfinite(grid_y_float), grid_y_float, -1).astype(int)
    
                # create empty bin name for every frame and then allocate cell names to the frames
                position_bin = np.full(len(df), "", dtype=object)
                position_bin[valid_grid] = [f"C_y{iy:03d}_x{ix:03d}" for ix, iy in zip(grid_x_index[valid_grid], grid_y_index[valid_grid])]
            
            return position_bin, arm

        def get_head_direction(df, position_bin):

            direction = np.full(len(df), "", dtype=object)

            # get head and postbody coords for each frame
            head      = df[["head_x", "head_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            post_body = df[["postbody_x", "postbody_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

            # Vector pointing from posterior body toward the head
            body_direction = head - post_body
            length = np.linalg.norm(body_direction, axis=1)
            valid_direction = (np.isfinite(body_direction).all(axis=1) & np.isfinite(length) & (length > 1e-6))

            # normalizes the vectors so each has a length of 1
            body_direction_unit = np.full_like(body_direction, np.nan, dtype=float)
            body_direction_unit[valid_direction] = (body_direction[valid_direction] / length[valid_direction, None])

            # reset because arm frames without a valid orientation should remain invalid rather than silently becoming position-only bins
            final_bin = position_bin.copy()
            final_bin[:] = ""

            # for the center, you dont care about the direction, just keep the position
            center_frames = ((position_bin != "") & (arm == "C"))
            if binning == "grid":
                final_bin[center_frames] = [f"{position}_C" for position in position_bin[center_frames]]
            else:
                final_bin[center_frames] = position_bin[center_frames]


            for arm_name in ("L", "M", "R"):

                # frames where the head has a valid spatial bin, is in the current arm and direction is valid
                arm_frames = ((position_bin != "") & (arm == arm_name) & valid_direction)
                if not arm_frames.any():
                    continue

                # compares the mouse direction to the arm direction, "in" when mouse facing the center, "out" when facing the arm_end
                dot_product = (body_direction_unit[arm_frames] @ arm_vectors[arm_name])
                direction[arm_frames] = np.where(dot_product >= 0, "out", "in")

                # creates the final bin including in or out facing
                if binning == "grid":
                    final_bin[arm_frames] = [f"{position}_{arm_name}_{orientation}" for position, orientation in zip(position_bin[arm_frames], direction[arm_frames])]
                else:
                    final_bin[arm_frames] = [f"{position}_{orientation}" for position, orientation in zip(position_bin[arm_frames], direction[arm_frames])]

            return final_bin, direction

        def plot_YM_bins(arena_outline, poly_big, poly_mid, bounds, binning, grid_size):

            x_min, x_max, y_min, y_max = bounds
            fig, ax = plt.subplots(figsize=(7, 7))

            # outer Y-maze border
            outline = plt.Polygon(arena_outline.vertices, fill=False, linewidth=2)
            ax.add_patch(outline)

            if binning == "big":
                areas = poly_big
            elif binning == "mid":
                areas = poly_mid
            elif binning == "grid":
                areas = None

            if areas is not None:
                for name, path in areas.items():
                    polygon = plt.Polygon(path.vertices, fill=False, linewidth=1)
                    ax.add_patch(polygon)
                    center = path.vertices.mean(axis=0)
                    ax.text(center[0],center[1],name,ha="center",va="center")

            if binning == "grid":
                x_values = np.arange(x_min, x_max, grid_size)
                y_values = np.arange(y_min, y_max, grid_size)

                for x in x_values:
                    for y in y_values:
                        rectangle = plt.Rectangle((x, y),grid_size,grid_size,fill=False,linewidth=0.7)
                        rectangle.set_clip_path(outline)
                        ax.add_patch(rectangle)

            ax.set_xlim(x_min - 20, x_max + 20)
            ax.set_ylim(y_min - 20, y_max + 20)
            ax.set_aspect("equal")
            ax.invert_yaxis()

            plt.show()

        
        # Preparation
        df = df.copy()
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # gets the areas and points of the YM
        poly_big, poly_mid, arena_outline, arm_vectors, endpoints, bounds = get_arena_bins()
        position_bin, arm = get_position_bin(df, binning, poly_big, poly_mid, bounds)

        # calculates the head direction of the animal
        if head_direction:
            final_bin, direction = get_head_direction(df, position_bin)
        else:
            final_bin = position_bin.copy()
            direction = np.full(len(df), "", dtype=object)

        # just add the info to the df
        df["ym_position_bin"] = position_bin
        df["ym_arm"]          = arm
        df["ym_direction"]    = direction
        df["head_bin"]        = final_bin

        # check if head position is not allocated to YM bins
        outside_spatial_bins = position_bin == ""
        if outside_spatial_bins.sum() > 0:
            print(f"Invalid YM frames outside spatial bins: {outside_spatial_bins.sum()}")

        # check frames where mouse is inside valid spatial bin, in one arm but no final_bin allocation
        missing_usable_direction = ((position_bin != "") & (arm != "C") & (df["head_bin"] == ""))
        if missing_usable_direction.sum() > 0:
            print(f"Invalid YM frames without usable head direction: {missing_usable_direction.sum()}")

        # takes only valid frames of head_bin
        df = df.loc[df["head_bin"] != ""].copy()
        if df.empty:
            raise ValueError("No valid Y-maze frames remained after bin assignment")

        # lets now create the 
        spike_prob_cells = [column for column in df.columns if isinstance(column, str) and column.endswith(col_spikeprob)]
        if not spike_prob_cells:
            raise ValueError(f"No columns end with {col_spikeprob}")

        # group all frames belonging to the same head_bin and sums up the spike values during that bin
        place_df = (df.groupby("head_bin", sort=True)[spike_prob_cells].sum(min_count=1).fillna(0.0))

        if plot_once:
            plot_YM_bins(arena_outline, poly_big, poly_mid, bounds, binning, grid_size)
            plot_once = False

        return place_df, df

    def get_occupancy(df, place_df):

        # occupancy to normalize to time spent in bin
        occupancy = df.groupby('head_bin').size() # how many frames did animal spend in each bin
        occupancy = occupancy.reindex(place_df.index, fill_value=0) # aligns occupancy and firing-rate rows
        occupancy_sec = occupancy / fps 
    
        # set threshold for occupancy in bins, if not visited, ignore the bin
        valid_bins = occupancy_sec >= min_occup_s
        occupancy = occupancy[valid_bins]
        occupancy_sec = occupancy_sec[valid_bins]
        place_df = place_df.loc[valid_bins]
    
        occ_nonzero = occupancy_sec.replace(0, np.nan)  # replacing 0 with nan
        df_firing_freq = place_df.div(occ_nonzero, axis=0).fillna(0.0)  # what was fring frequency in each bin

        return place_df, df_firing_freq, occupancy, occ_nonzero

    def get_skaggs_spatial_information(firing_freq, occupancy):

        # occupancy: number of frames per bin
        occupancy_probability = occupancy / occupancy.sum()

        # overall mean rate
        avrg_firing_rate = np.dot(firing_freq, occupancy_probability)
        if avrg_firing_rate == 0:
            return 0.0

        info = 0.0
        for bin_firing_rate, bin_occ_probability in zip(firing_freq, occupancy_probability):
            if bin_firing_rate > 0 and bin_occ_probability > 0:
                ratio = bin_firing_rate / avrg_firing_rate
                info += bin_occ_probability * ratio * np.log2(ratio)

        return info

    def save(file, SI_real_cells):

        # save each SI values of the file individually
        df_SI_individ = pd.DataFrame(SI_real_cells)
        SI_individ_folder = folder_out / "Individ_SpatialInfo"
        SI_individ_folder.mkdir(parents=True, exist_ok=True)
        df_SI_individ.to_csv(f'{SI_individ_folder}\\{file}.csv', index=False)
    
        # return a summary of the SI
        SI_real_values = np.array([d["SI"] for d in SI_real_cells])
        SI_summary = {"SI_mean": np.mean(SI_real_values),
                      "SI_median": np.median(SI_real_values),
                      "SI_n_cells": len(SI_real_values)}

        return SI_summary

    def shuffle_cells(place_df, place_cells, occupancy, occ_nonzero, SI_real_cells, SI_summary, n_shuffles, sig_level):

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
                info = get_skaggs_spatial_information(shuffled_firing_rate.values, occupancy.values)
                SI_shuffled[cell].append(info)


        # significance test
        p_values = {}
        SI_real_dict = {d["cell_ID"]: d["SI"] for d in SI_real_cells}   # quick transform into real dictionary, easier
        cell_IDs = np.array(list(SI_real_dict.keys()))
        for cell in cell_IDs:

            real_si = SI_real_dict[cell]
            shuffled_si = np.array(SI_shuffled[cell])

            p = (1 + np.sum(shuffled_si >= real_si)) / (n_shuffles + 1)
            p_values[cell] = p
            if p < sig_level:
                place_cells.append(cell.replace(f'_{col_spikeprob}', ''))

        # return a summary of the analyse
        SI_summary["place_cells_nb"] = len(place_cells)
        SI_summary["place_cells_quotient"] = len(place_cells) / len(SI_real_cells)

        return place_cells, SI_summary

    if 'YM' in file:
        place_df, df = bin_cellspikes_prob_YM(df, YM_bin, YM_head_dir, YM_grid_size)
    elif 'OF' in file:
        place_df, df = bin_cellspikes_prob_OF(df, OF_n_bins)

    
    place_df, df_firing_freq, occupancy, occ_nonzero = get_occupancy(df, place_df)

    # goes through all cells and gets SI
    SI_real_cells = []
    for cell in df_firing_freq.columns:
        info = get_skaggs_spatial_information(df_firing_freq[cell].values, occupancy.values)
        SI_real_cells.append({"cell_ID": cell, "SI": info})

        if draw_cells:
            heatmap_cells(df, file, cell, info)

    # calculate by permutation what are place cells
    SI_summary = save(file, SI_real_cells)
    place_cells = []
    if do_shuffle:
        place_cells, SI_summary = shuffle_cells(place_df, place_cells, occupancy, occ_nonzero, SI_real_cells, SI_summary, n_shuffles, sig_level)

    return place_cells, SI_summary

def FFN(df, cells, file, bin_size_frames=5, n_cells=50, n_repeats_rand_cells=100, n_train_epochs=200, acc_threshold_mm=50, arena_size=(600,600)):

    # helper definitions
    def prep_data(df, cells, bin_size_frames, arena_scale, fps=10, input_mode="rate"):
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
        n_bins = len(df) // bin_size_frames

        rows = []
        for i in range(n_bins):
            start = i * bin_size_frames
            end = (i + 1) * bin_size_frames
            segment = df.iloc[start:end]

            # sum of spike probabilities per bin -> expected spike counts in bin
            # per downsampled row, we have the animal position and the spikerate
            spike_sums = segment[cells].sum(axis=0).astype(float)
            if input_mode == "sum":
                spikes = spike_sums.values
            elif input_mode == "rate":
                bin_seconds = bin_size_frames / fps
                spikes = (spike_sums.values / bin_seconds)
            position_avg = segment[position_cols].mean().values.astype(float)
            rows.append(np.concatenate([spikes, position_avg]))
        df_binned = pd.DataFrame(rows, columns=list(cells) + position_cols)

        X = df_binned[cells].values.astype(np.float32)
        Y_px = df_binned[position_cols].values.astype(np.float32)

        if np.any(Y_px < 0) or np.any(Y_px > arena_scale):
            raise ValueError(f"Coordinates are outside arena_size={tuple(arena_scale)}. "
                             f"Observed X range: {Y_px[:, 0].min():.2f}-{Y_px[:, 0].max():.2f}, "
                             f"Y range: {Y_px[:, 1].min():.2f}-{Y_px[:, 1].max():.2f}")

        # normalize for arena size
        Y_norm = (Y_px / arena_scale).astype(np.float32)

        # shuffle for a baseline
        min_shift = max(1, int(len(X) * 0.2))
        rng = np.random.default_rng(42)
        shift = rng.integers(min_shift, len(X) - min_shift)
        X_shuffled = np.roll(X, shift, axis=0)

        return X, X_shuffled, Y_norm

    def train_model(train_loader, test_loader, model, n_train_epochs, lr=1e-3, verbose=True, take_best=True):

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # log to check later
        train_losses, test_losses = [], []

        # for restoring the best model
        best_test_loss = np.inf
        best_model_state = None
        best_epoch = 0

        # Training loop
        pbar = tqdm(range(n_train_epochs), desc="Training FFN", leave=True)
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

            # check the model quality
            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

            if verbose:
                pbar.set_postfix({"Train_RMSE": f"{avg_train_loss**0.5:.2f}","Test_RMSE": f"{avg_test_loss**0.5:.2f}"})

        if take_best:
            model.load_state_dict(best_model_state)
            #print(f"Best epoch: {best_epoch} with test RMSE {np.sqrt(best_test_loss):.4f}")

        return model, train_losses, test_losses

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

    def calc_accuracy(Y_true, Y_pred, threshold_mm):

        # calulates how many predictions are within the threshold (in depends if mm or px (main_df creation))
        errors = np.linalg.norm(Y_true - Y_pred, axis=1)            # in px
        accuracy = round(np.mean(errors <= threshold_mm)*100, 4)    # in percent
        #print(f"Mean decoding error: {errors.mean():.2f} mm, Accuracy: {accuracy}%\n")

        return accuracy, errors

    class PositionDecoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super(PositionDecoder, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),   # layer_1: input -> hidden
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),  # layer_2: hidden -> hidden
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),           # layer_3: hidden -> output (x,y)
                nn.Sigmoid()                        # guarantees outputs between 0 and 1
            )

        def forward(self, x):
            return self.net(x)


    # plot
    def plot_individ_PredictionCoords(Y_true, Y_pred, accuracy, file, limit=600, title=''):

        true_x = Y_true[:, 0]
        pred_x = Y_pred[:, 0]
        true_y = Y_true[:, 1]
        pred_y = Y_pred[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=False)

        # ---- X position ----
        ax = axes[0]
        hb = ax.hist2d(true_x, pred_x, bins=30, range=[[0, limit], [0, limit]], cmap='viridis')
        fig.colorbar(hb[3], ax=ax)
        ax.plot([true_x.min(), true_x.max()], [true_x.min(), true_x.max()], c='r', linestyle=':', label='Perfect prediction')
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel('True X position')
        ax.set_ylabel('Predicted X position')
        ax.set_title(f'X: Pred vs True (Accuracy={accuracy}%)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

        # ---- Y position ----
        ax = axes[1]
        hb = ax.hist2d(true_y, pred_y, bins=30, range=[[0, limit], [0, limit]], cmap='viridis')
        fig.colorbar(hb[3], ax=ax)
        ax.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], c='r', linestyle=':', label='Perfect prediction')
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel('True Y position')
        ax.set_ylabel('Predicted Y position')
        ax.set_title(f'Y: Pred vs True (Accuracy={accuracy}%)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

        fig.suptitle(f'{title}: Predicted vs True ({file})')
        plt.tight_layout()
        plt.show()

    def plot_individ_Positions(Y_real_true, Y_real_pred, Y_shuf_pred, file):

        plt.figure(figsize=(7, 7))

        # connect each true position to its predictions
        for i in range(len(Y_real_true)):

            # true -> real prediction  AND  true -> shuffled prediction
            plt.plot([Y_real_true[i, 0], Y_real_pred[i, 0]], [Y_real_true[i, 1], Y_real_pred[i, 1]], color='green', alpha=0.3, linewidth=1)
            plt.plot([Y_real_true[i, 0], Y_shuf_pred[i, 0]], [Y_real_true[i, 1], Y_shuf_pred[i, 1]], color='red', alpha=0.3, linewidth=1)

        # points on top of lines
        plt.scatter(Y_real_true[:, 0], Y_real_true[:, 1], label="True", color='gray', zorder=3)
        plt.scatter(Y_real_pred[:, 0], Y_real_pred[:, 1], label="Real Predict", color='green', zorder=3)
        plt.scatter(Y_shuf_pred[:, 0], Y_shuf_pred[:, 1], label="Shuf Predict", color='red', zorder=3)

        plt.xlabel("X position [mm]")
        plt.ylabel("Y position [mm]")
        plt.legend()
        plt.grid(False)
        plt.axis("equal")

        plt.title(f"Positions in Arena ({file})")
        plt.tight_layout()
        plt.show()

    def plot_summary_real_vs_shuf_Accuracy(real, shuf, file, folder):

        real = np.asarray(real)
        shuf = np.asarray(shuf)
        x_real = np.zeros(len(real))
        x_shuf = np.ones(len(shuf))

        plt.figure(figsize=(6, 5))

        # connect paired repetitions
        for i in range(len(real)):
            plt.plot([0, 1], [real[i], shuf[i]], color="gray", alpha=0.5)

        # individual values
        plt.scatter(x_real, real, color="green", label="Real", zorder=2)
        plt.scatter(x_shuf, shuf, color="red", label="Shuffled", zorder=2)

        # mean + SEM
        mean_real = np.mean(real)
        mean_shuf = np.mean(shuf)
        sem_real = stats.sem(real)
        sem_shuf = stats.sem(shuf)

        plt.errorbar(0, mean_real, yerr=sem_real, fmt="o", color="black", markersize=8, capsize=5, linewidth=2, zorder=3)
        plt.errorbar(1, mean_shuf, yerr=sem_shuf, fmt="o", color="black", markersize=8, capsize=5, linewidth=2, zorder=3)

        # paired t-test
        t_stat, p_val = stats.ttest_rel(real, shuf)
        p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"

        ax = plt.gca()
        ax.text(0.02, 0.98, p_text, transform=ax.transAxes, va="top")

        plt.xticks([0, 1], ["Real", "Shuffled"])
        plt.ylabel("Accuracy [%]")
        plt.title("Real vs shuffled decoding accuracy")
        plt.tight_layout()
        plt.savefig(f'{folder}\\{file}_Accuracy.png')
        plt.show()

    def plot_summary_real_vs_shuf_Errors(errors_real, errors_shuf, file, folder):

        percentiles = np.arange(0, 101)

        # percentile curve for every repetition
        values_real = np.array([np.percentile(errors, percentiles) for errors in errors_real])
        values_shuf = np.array([np.percentile(errors, percentiles) for errors in errors_shuf])

        # mean across repetitions
        mean_real = np.mean(values_real, axis=0)
        mean_shuf = np.mean(values_shuf, axis=0)

        # SEM across repetitions
        sem_real = np.std(values_real, axis=0, ddof=1) / np.sqrt(values_real.shape[0])
        sem_shuf = np.std(values_shuf, axis=0, ddof=1) / np.sqrt(values_shuf.shape[0])

        plt.figure(figsize=(8, 5))

        # real
        plt.plot(percentiles, mean_real, color='green', label='Real mean ± SEM')
        plt.fill_between(percentiles, mean_real - sem_real, mean_real + sem_real, color='green', alpha=0.2)

        # shuffled
        plt.plot(percentiles, mean_shuf, color='red', label='Shuffle mean ± SEM')
        plt.fill_between(percentiles, mean_shuf - sem_shuf, mean_shuf + sem_shuf, color='red', alpha=0.2)

        plt.xlabel("Percentile")
        plt.ylabel("Decoding error [mm]")
        plt.title(f"Distribution of decoding errors ({file})")
        plt.grid()
        plt.yscale("log")
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.ylim(5, 1000)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder}\\{file}_Errors.png')
        plt.show()

    def plot_summary_real_vs_shuf_TrainingLoss(real_train_losses, real_test_losses, shuf_train_losses, shuf_test_losses, file, folder):
    
        # shape: repetitions x epochs
        real_train_losses = np.asarray(real_train_losses)
        real_test_losses = np.asarray(real_test_losses)
        shuf_train_losses = np.asarray(shuf_train_losses)
        shuf_test_losses = np.asarray(shuf_test_losses)

        # convert MSE -> RMSE before averaging
        real_train_rmse = np.sqrt(real_train_losses)
        real_test_rmse = np.sqrt(real_test_losses)
        shuf_train_rmse = np.sqrt(shuf_train_losses)
        shuf_test_rmse = np.sqrt(shuf_test_losses)

        # mean across repetitions
        real_train_mean = np.mean(real_train_rmse, axis=0)
        real_test_mean = np.mean(real_test_rmse, axis=0)
        shuf_train_mean = np.mean(shuf_train_rmse, axis=0)
        shuf_test_mean = np.mean(shuf_test_rmse, axis=0)

        # SEM across repetitions
        real_train_sem = np.std(real_train_rmse, axis=0, ddof=1) / np.sqrt(real_train_rmse.shape[0])
        real_test_sem = np.std(real_test_rmse, axis=0, ddof=1) / np.sqrt(real_test_rmse.shape[0])
        shuf_train_sem = np.std(shuf_train_rmse, axis=0, ddof=1) / np.sqrt(shuf_train_rmse.shape[0])
        shuf_test_sem = np.std(shuf_test_rmse, axis=0, ddof=1) / np.sqrt(shuf_test_rmse.shape[0])

        epochs = np.arange(len(real_train_mean))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        # real
        axes[0].plot(epochs, real_train_mean, label="Train RMSE mean ± SEM")
        axes[0].fill_between(epochs,real_train_mean - real_train_sem,real_train_mean + real_train_sem,alpha=0.25)
        axes[0].plot(epochs, real_test_mean, label="Test RMSEmean ± SEM")
        axes[0].fill_between(epochs,real_test_mean - real_test_sem, real_test_mean + real_test_sem, alpha=0.25)
        axes[0].set_title("Real")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMSE [norm units]")
        axes[0].legend()
        axes[0].grid(True)

        # shuffled
        axes[1].plot(epochs, shuf_train_mean, label="Train RMSE mean ± SEM")
        axes[1].fill_between(epochs, shuf_train_mean - shuf_train_sem, shuf_train_mean + shuf_train_sem, alpha=0.25)
        axes[1].plot(epochs, shuf_test_mean, label="Test RMSE mean ± SEM")
        axes[1].fill_between(epochs, shuf_test_mean - shuf_test_sem, shuf_test_mean + shuf_test_sem, alpha=0.25)
        axes[1].set_title("Shuffled")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True)

        fig.suptitle(f"Model Training Losses ({file})")
        plt.tight_layout()
        plt.savefig(f'{folder}\\{file}_TrainingLoss.png')
        plt.show()
    
    
    # main
    def run(X, Y_norm, results, condition):

        X_train = X[idx_train]
        X_test  = X[idx_test]

        Y_train = Y_norm[idx_train]
        Y_test  = Y_norm[idx_test]

        # scaler
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        # create loaders
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
        test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(Y_test))
        train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

        # model & train
        model = PositionDecoder(input_dim=X_real.shape[1], hidden_dim=128)
        model_trained, train_loss, test_loss = train_model(train_loader, test_loader, model, n_train_epochs)

        # test & accuracy
        Y_pred_norm, Y_true_norm = test_model(model_trained, test_loader)

        # Convert normalized coordinates back to pixels
        Y_pred = Y_pred_norm * arena_scale
        Y_true = Y_true_norm * arena_scale

        # calculate accuracy & error
        accuracy, errors = calc_accuracy(Y_true, Y_pred, acc_threshold_mm)

        # store everything
        results[condition]["accuracy"].append(accuracy)
        results[condition]["errors"].append(errors)
        results[condition]["train_loss"].append(train_loss)
        results[condition]["test_loss"].append(test_loss)
        results[condition]["Y_true"].append(Y_true)
        results[condition]["Y_pred"].append(Y_pred)

        return results

        
    # arena_size
    arena_scale = np.asarray(arena_size, dtype=np.float32)

    results = { "real": {"accuracy"     : [],
                         "errors"       : [],
                         "train_loss"   : [],
                         "test_loss"    : [],
                         "Y_true"       : [],
                         "Y_pred"       : []},
                "shuf": {"accuracy"     : [],
                         "errors"       : [],
                         "train_loss"   : [],
                         "test_loss"    : [],
                         "Y_true"       : [],
                         "Y_pred"       : []}}

                        
    for _ in range(n_repeats_rand_cells):

        # skip if there are not enough cells
        if n_cells > len(cells):
            return {"FFN_accuracy_mean": np.nan,
                    "FFN_error_mean": np.nan,
                    "n_repeats_rand_cells": np.nan,
                    "n_cells": np.nan}
        sampled_cells = np.random.choice(cells, size=n_cells, replace=False)


        # prepare X (rows=timebins, columns=neurons) and Y (rows=timebins, columns=x/y)
        X_real, X_shuf, Y_norm = prep_data(df, cells=sampled_cells, bin_size_frames=bin_size_frames, arena_scale=arena_scale)
        indices = np.arange(len(X_real))
        idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

        # run main
        results = run(X_real, Y_norm,results, "real")
        results = run(X_shuf, Y_norm,results, "shuf")

        if False:
            # plot individual plots
            plot_individ_Positions(       results["real"]["Y_true"][-1], results["real"]["Y_pred"][-1], results["shuf"]["Y_pred"][-1], file)
            plot_individ_PredictionCoords(results["real"]["Y_true"][-1], results["real"]["Y_pred"][-1], results["real"]["accuracy"][-1], file, title='Real')
            plot_individ_PredictionCoords(results["shuf"]["Y_true"][-1], results["shuf"]["Y_pred"][-1], results["shuf"]["accuracy"][-1], file, title='Shuf')

    if True:
        # plot summary plots
        folder_FFN = Path(folder_out / "FFN_Summary_Plots")
        folder_FFN.mkdir(parents=True, exist_ok=True)
        plot_summary_real_vs_shuf_Accuracy(    results["real"]["accuracy"], results["shuf"]["accuracy"], file, folder_FFN)
        plot_summary_real_vs_shuf_TrainingLoss(results["real"]["train_loss"], results["real"]["test_loss"], results["shuf"]["train_loss"], results["shuf"]["test_loss"], file, folder_FFN)
        plot_summary_real_vs_shuf_Errors(      results["real"]["errors"], results["shuf"]["errors"], file, folder_FFN)

    # save everything in summary
    all_error_real = np.concatenate(results["real"]["errors"])
    all_error_shuf = np.concatenate(results["shuf"]["errors"])
    FFN_summary =  {"FFN_real_accuracy_mean"    :float(np.mean  (results["real"]["accuracy"])),
                    "FFN_real_accuracy_median"  :float(np.median(results["real"]["accuracy"])),
                    "FFN_real_error_mean"       :float(np.mean  (all_error_real)),
                    "FFN_real_error_median"     :float(np.median(all_error_real)),
                    "FFN_shuf_accuracy_mean"    :float(np.mean  (results["shuf"]["accuracy"])),
                    "FFN_shuf_accuracy_median"  :float(np.median(results["shuf"]["accuracy"])),
                    "FFN_shuf_error_mean"       :float(np.mean  (all_error_shuf)),
                    "FFN_shuf_error_median"     :float(np.median(all_error_shuf)),
                    "FFN_n_repeats_rand_cells"  :n_repeats_rand_cells,
                    "FFN_n_cells"               :n_cells}

    return FFN_summary



dist_travelled = {}
large_heatmap  = []
Main_summary   = []
animals        = ['48', '51', '56', '66', '72', '77']

for anim in animals: 

    files = get_files(folder_in, anim, '')
    for file in files:
        
        file = Path(file)
        print(f'\n------- {file.stem} -------')
        Summary_ind = {"file": file.stem}

        # Clean
        main_df, QC_summary = clean_df_QC(file, QC_file)
        Summary_ind = Summary_ind | QC_summary

        # Distances
        main_df, Distance_summary, dist_travelled = clean_df_distance(main_df, file.stem, dist_travelled)
        Summary_ind = Summary_ind | Distance_summary

        # drawing
        heatmap_occupancy(main_df, file.stem)
        # heatmap_cells(main_df, file.stem)
        large_heatmap.append(heatmap_spikeprob(main_df, file.stem))

        # Spike Probability
        if True and main_df is not None:
            Spikeprob_summary = SpikeProb(main_df, file)
            Summary_ind = Summary_ind | Spikeprob_summary

        # Correlation SpikeRate-AnimalSpeed
        if True and main_df is not None:
            regression_summary = SpikeRate_Speed(main_df, file.stem)
            Summary_ind = Summary_ind | regression_summary

        # Spatial Information
        if True and main_df is not None:
            place_cells, SI_summary = SI(main_df, file.stem)
            Summary_ind = Summary_ind | SI_summary

        # FFN
        if True and main_df is not None:
            cells_to_use = [c for c in main_df.columns if col_spikeprob in c]     # or place_cells, whatever is needed
            FFN_summary  = FFN(main_df, cells_to_use, file.stem)
            Summary_ind  = Summary_ind | FFN_summary

    Main_summary.append(Summary_ind)


# save the data
Main_summary_df   = pd.DataFrame(Main_summary)
Main_summary_file = folder_out / 'Summary.csv'
Main_summary_df.to_csv(Main_summary_file, index=False)

dist_travelled_df = pd.DataFrame({key: pd.Series(value) for key, value in dist_travelled.items()})
dist_travelled_file = folder_out / 'Distance_travelled.csv'
dist_travelled_df.to_csv(dist_travelled_file, index=False)

# plot summary
heatmap_spikeprob(main_df, file, heatmap=large_heatmap)


#%%
plot_groups_columns(Main_summary_df, column='FFN_real_accuracy_mean', y_title='mean SI', title='')
#%%

def plot_groups_columns(results_df, column, y_title, title):

    groups = ['mKate', 'A53T']

    plt.figure(figsize=(3, 5))

    for gi, group in enumerate(groups):
        sub = results_df[results_df["file"].str.contains(group, na=False)][column].dropna()

        # mean ± SEM overlay
        mean = sub.mean()
        sem = sub.sem()
        plt.errorbar(gi, mean, yerr=sem, fmt="-", capsize=10)
        plt.hlines(mean, gi-0.1, gi+0.1, colors="black", linewidth=2)

        # scatter
        plt.scatter(gi + np.random.uniform(-0.08, 0.08, len(sub)), sub, s=70)

    plt.xticks(range(len(groups)), groups)
    plt.ylabel(y_title)

    # t-test
    a = results_df.loc[results_df["file"].str.contains("mKate", regex=False, na=False), column].dropna().to_numpy()
    b = results_df.loc[results_df["file"].str.contains("A53T", regex=False, na=False), column].dropna().to_numpy()
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=True)
    ax = plt.gca()
    ax.text(0.02, 0.98, f'p = {round(p_val, 3)}', transform=ax.transAxes, va="top")

    plt.title(title)
    plt.tight_layout()
    plt.show()

for c in Main_summary_df.columns:
    if c != 'file':
        plot_groups_columns(Main_summary_df, column=c, y_title='', title=c)


