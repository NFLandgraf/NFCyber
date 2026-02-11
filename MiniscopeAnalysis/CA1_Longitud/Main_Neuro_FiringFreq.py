#%%
'''
Goes through df and correlates speed of the animal with spike probabilities
'''
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


CSV_FOLDER     = Path(r"D:\10_ALL_MERGE")
OUT_CSV        = Path(r"D:\speed_vs_spikeprob_all_cells.csv")
fps            = 10.0        # frames per second
bin_size_s     = 1           # seconds per bin

def process_csv(csv_path, fps, bin_size_s):
    df = pd.read_csv(csv_path)

    # identify cell columns
    cell_cols = [c for c in df.columns if 'spikeprob'  in c]

    # ---- compute distance per frame ----
    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    dist_mm = np.sqrt(dx**2 + dy**2)
    dist_mm.iloc[0] = 0

    # ---- speed per frame ----
    df["speed_mm_s"] = dist_mm * fps

    # ---- time binning ----
    frames_per_bin = int(bin_size_s * fps)
    df["time_bin"] = (df.index // frames_per_bin).astype(int)

    out_rows = []

    for cell in cell_cols:
        grouped = df.groupby("time_bin").agg(
            mean_speed_mm_s = ("speed_mm_s", "mean"),
            spike_prob_sum  = (cell, "sum"),
        ).reset_index()

        grouped["cell"] = cell
        grouped["file"] = csv_path.stem

        out_rows.append(grouped)

        if out_rows:
            return pd.concat(out_rows, ignore_index=True)
        else:
            return False

all_results = []
csv_files = sorted(CSV_FOLDER.glob("*OF_final.csv"))
print(f"Found {len(csv_files)} CSV files")

for csv in tqdm(csv_files):
    res = process_csv(csv, fps=fps, bin_size_s=bin_size_s)
    if res is not None:
        all_results.append(res)

final_df = pd.concat(all_results, ignore_index=True)
final_df = final_df[["mean_speed_mm_s", "spike_prob_sum"]]


# plot
x = final_df['mean_speed_mm_s']
y = final_df['spike_prob_sum']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope * x + intercept

plt.figure()
plt.scatter(x, y, alpha=0.2, s=2)
plt.xlabel('mean_speed [mm/s]')
plt.ylabel('spike_prob [spikes/s]')
plt.plot(x, line, color='red', label='Regression Line')
stats_text = f"$R = {r_value:.2f}$\n$P = {p_value:.2e}$"
plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.ylim(0,5)
plt.show()

#%%
'''
Goes through df saves the mean and median spike probability
'''
import pandas as pd
from pathlib import Path
from tqdm import tqdm

CSV_FOLDER = Path(r"D:\11_ALL_MERGE_trim")
OUT_CSV    = Path(r"D:\spikeprob_summary_per_file.csv")

rows = []

csv_files = sorted(CSV_FOLDER.glob("*YM_final_trim(15m).csv"))
print(f"Found {len(csv_files)} CSV files")

for csv in tqdm(csv_files):
    df = pd.read_csv(csv)

    # identify cell columns
    cell_cols = [c for c in df.columns if "spikeprob" in c]

    n_rows = len(df)

    # --- method 1: mean per cell (pandas mean) ---
    per_cell_mean = df[cell_cols].mean(axis=0)
    mean_of_cell_means   = per_cell_mean.mean()
    median_of_cell_means = per_cell_mean.median()

    # --- method 2: sum per cell / number of rows ---
    per_cell_sum_div_rows = df[cell_cols].sum(axis=0) / n_rows

    mean_of_sum_div_rows   = per_cell_sum_div_rows.mean()
    median_of_sum_div_rows = per_cell_sum_div_rows.median()

    rows.append({
        "file": csv.stem,
        "mean_of_cell_means": mean_of_cell_means,
        "median_of_cell_means": median_of_cell_means,
        "mean_of_cell_sum_div_rows": mean_of_sum_div_rows,
        "median_of_cell_sum_div_rows": median_of_sum_div_rows,
        "n_rows": n_rows
    })

# =====================
# SAVE
# =====================
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)

print("Saved:")
print(OUT_CSV)



#%%
# saves each individual cell mean

import pandas as pd
from pathlib import Path
from tqdm import tqdm

CSV_FOLDER = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\traces")
OUT_CSV    = Path(r"D:\blub.csv")

rows = []

csv_files = sorted(CSV_FOLDER.glob("*Zost2_YM_final_trim(15m).csv"))
#csv_files = [Path(r"D:\11_ALL_MERGE_trim\77_Zost2_YM_final_trim(15m).csv")]#, Path(r"D:\11_ALL_MERGE_trim\77_Zost2_YM_final_trim(15m).csv")]
print(f"Found {len(csv_files)} CSV files")

for csv in tqdm(csv_files):
    df = pd.read_csv(csv)

    cell_cols = [c for c in df.columns if "spikeprob" in c]
    n_rows = len(df)

    # per-cell means (method 1)
    per_cell_mean = df[cell_cols].mean(axis=0)

    # per-cell sum / rows (method 2)
    per_cell_sum_div_rows = df[cell_cols].sum(axis=0) / n_rows

    for cell in cell_cols:
        rows.append({
            "file": csv.stem,
            "cell": cell,
            "mean_spikeprob": per_cell_mean[cell],
            "mean_spikeprob_sum_div_rows": per_cell_sum_div_rows[cell],
            "n_rows": n_rows
        })

# =====================
# SAVE
# =====================
df_out = pd.DataFrame(rows)
#df_out.to_csv(OUT_CSV, index=False)

print("Saved per-cell means:")
print(OUT_CSV)

#%%
df_filtered = df_out[(df_out["mean_spikeprob"] >= 0.045) & (df_out["mean_spikeprob"] <= 0.055)]
df_filtered.to_csv('eeh.csv')

print(df_filtered)
