#%%
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

CSV_FOLDER     = Path(r"D:\10_ALL_MERGE")
OUT_FOLDER     = Path(r"D:\11_ALL_MERGE_trim")
OUT_FOLDER.mkdir(exist_ok=True)

DIST_LIMIT_MM  = 15000
summary_rows = []

csv_files = sorted(CSV_FOLDER.glob("*YM_final.csv"))
print(f"Found {len(csv_files)} CSV files")

for csv in tqdm(csv_files):
    df = pd.read_csv(csv)

    # compute per-frame distance
    dx = df["head_x"].diff()
    dy = df["head_y"].diff()
    dist_mm = np.sqrt(dx**2 + dy**2)
    dist_mm.iloc[0] = 0

    # cumulative distance
    cum_dist = dist_mm.cumsum()

    # check if distance limit is ever reached
    if cum_dist.iloc[-1] > DIST_LIMIT_MM:
        
        # find cutoff frame and trim
        cutoff_idx = cum_dist[cum_dist >= DIST_LIMIT_MM].index[0]
        df_trim = df.iloc[:cutoff_idx + 1].copy()
    
    else:
        print(f"⚠ {csv.stem}: did not reach {DIST_LIMIT_MM} mm, skipping")
        df_trim = df.copy()

    # recompute distance on trimmed df (clean & explicit)
    dx_t = df_trim["head_x"].diff()
    dy_t = df_trim["head_y"].diff()
    dist_mm_t = np.sqrt(dx_t**2 + dy_t**2)
    dist_mm_t.iloc[0] = 0

    total_dist_trim = dist_mm_t.sum()
    n_rows_trim = len(df_trim)

    # save trimmed df
    out_csv = OUT_FOLDER / f"{csv.stem}_trim(15m).csv"
    df_trim.to_csv(out_csv, index=False)

    # store summary
    summary_rows.append({"file": csv.stem,"trimmed_rows": n_rows_trim,"distance_mm": total_dist_trim})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_FOLDER / "trimmed_15000mm_summary.csv", index=False)

print("Done.")

# %%
