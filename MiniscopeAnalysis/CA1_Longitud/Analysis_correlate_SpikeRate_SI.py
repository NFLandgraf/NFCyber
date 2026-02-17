#%%
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from matplotlib.path import Path as MplPath
import numpy as np


CSV_FOLDER = Path(r"D:\Analy_Post2")
OUT_CSV    = Path(r"D:\Analy_Post2\cells_meanSpikeprob_vs_SI_ALLFILES.csv")
fps = 10.0


def bin_cellspikes_prob_YM(main_df):
    """
    From head position and spike probabilities, compute 'spike mass' per spatial bin.
    Instead of summing 0/1 spikes, we sum spike probabilities per bin.
    """
    df = main_df.copy()
    x_max, y_max = 1000, 1000

    # YM geometry (your values)
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

    paths = {
        "center": MplPath(poly_center),
        "arm_left": MplPath(poly_arm_left),
        "arm_right": MplPath(poly_arm_right),
        "arm_middle": MplPath(poly_arm_middle),
    }
    bin_order = ["center", "arm_left", "arm_right", "arm_middle"]
    bin_ids   = {"center":0, "arm_left":1, "arm_right":2, "arm_middle":3}

    # head in which bin
    pts = df[["head_x", "head_y"]].to_numpy(dtype=float)
    head_bin = np.full(len(df), -1, dtype=int)  # -1 = not in any polygon
    for name in bin_order:
        inside = paths[name].contains_points(pts)
        head_bin[(head_bin == -1) & inside] = bin_ids[name]
    df["head_bin"] = head_bin

    # sum spike probabilities per bin
    spike_prob_cells = [c for c in df.columns if isinstance(c, str) and c.endswith("_spikeprob")]
    place_map = {}
    for cell in spike_prob_cells:
        bin_probs = df.groupby("head_bin")[cell].sum()
        place_map[cell] = bin_probs

    place_df = pd.DataFrame(place_map).fillna(0.0).sort_index()
    return df, place_df

def compute_spatial_info_only(main_df, place_df):
    """
    Computes Skaggs spatial information for each cell in place_df (no shuffles).
    This is the part you need for correlation with mean spikeprob.
    """

    def spatial_information(firing_freq, occupancy):
        occupancy_probability = occupancy / occupancy.sum()
        avrg_firing_rate = np.dot(firing_freq, occupancy_probability)

        if avrg_firing_rate == 0:
            return 0.0

        info = 0.0
        for r, p in zip(firing_freq, occupancy_probability):
            if r > 0 and p > 0:
                ratio = r / avrg_firing_rate
                info += p * ratio * np.log2(ratio)
        return float(info)

    # occupancy in frames per bin (includes head_bin=-1 if present)
    occupancy = main_df.groupby("head_bin").size()
    occupancy = occupancy.reindex(place_df.index, fill_value=0)

    occupancy_sec = occupancy / fps
    occ_nonzero = occupancy_sec.replace(0, np.nan)

    # firing frequency (expected spikes per second) per bin
    df_firing_freq = place_df.div(occ_nonzero, axis=0).fillna(0.0)

    spatial_info_real = {}
    for cell in df_firing_freq.columns:
        spatial_info_real[cell] = spatial_information(df_firing_freq[cell].values, occupancy.values)

    return spatial_info_real


csv_files = sorted(CSV_FOLDER.glob("*Zost2_YM_final_trim(15m).csv"))
print(f"Found {len(csv_files)} CSV files")
rows = []

for csv_path in tqdm(csv_files):
    df = pd.read_csv(csv_path)

    # mean spikeprob per cell (over frames)
    cell_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_spikeprob")]
    per_cell_mean = df[cell_cols].mean(axis=0)

    # compute SI from your YM binning
    df_binned, place_df = bin_cellspikes_prob_YM(df)
    si_map = compute_spatial_info_only(df_binned, place_df)

    # write one row per cell
    for cell in cell_cols:
        rows.append({
            "file": csv_path.stem,
            "cell": cell,
            "mean_spikeprob": float(per_cell_mean[cell])*10,
            "SI": float(si_map.get(cell, np.nan))
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("Rows (cells):", len(out_df))
print("Missing SI:", out_df["SI"].isna().sum())
