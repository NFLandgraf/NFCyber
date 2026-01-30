#%%
import pandas as pd
from pathlib import Path
from tqdm import tqdm

FOLDER   = Path(r"D:\your_folder")
OUT_CSV  = Path(r"D:\cells_meanSpikeprob_vs_SI_ALLFILES.csv")

# patterns you use in column names
SPIKE_TAG = "spikeprob"
SI_TAG    = "SI"   # adjust if your SI column pattern is different (e.g., "spatial_info")

csv_files = sorted(FOLDER.glob("*.csv"))
print(f"Found {len(csv_files)} files")

rows = []

for fp in tqdm(csv_files):
    df = pd.read_csv(fp)

    spike_cols = [c for c in df.columns if SPIKE_TAG in c]
    

    # build a mapping: base cell id -> spike col / si col
    # e.g. "C071_spikeprob" + "C071_SI" share base "C071"
    def base_name(col, tag):
        return col.replace(tag, "").replace("__", "_").strip("_")

    spike_map = {base_name(c, SPIKE_TAG): c for c in spike_cols}
    si_map    = {base_name(c, SI_TAG): c for c in si_cols}

    common_cells = sorted(set(spike_map).intersection(si_map))

    for cell_base in common_cells:
        spike_col = spike_map[cell_base]
        si_col    = si_map[cell_base]

        mean_spikeprob = df[spike_col].mean()
        # SI is typically constant per cell; take first non-nan
        si_val = df[si_col].dropna().iloc[0] if df[si_col].notna().any() else float("nan")

        rows.append({
            "file": fp.stem,
            "cell": cell_base,
            "mean_spikeprob": mean_spikeprob,
            "SI": si_val
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print("Rows (cells):", len(out_df))



#%%




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


    nb_place_cells, nb_all_cells = len(place_cells), len(spatial_info_real)
    print(f'Place cells {nb_place_cells} / {nb_all_cells} ({round((nb_place_cells/nb_all_cells)*100, 1)}%)')

    percent_place = nb_place_cells / nb_all_cells

    return spatial_info_real, spatial_info_shuffled, place_cells, p_values, percent_place

