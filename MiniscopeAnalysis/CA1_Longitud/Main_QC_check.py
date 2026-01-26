#%%
'''
check the QC parameters
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from random import shuffle

df = pd.read_csv(r"D:\df_QC_FINAL.csv", index_col='cell_ID')

x = 'area_px'
y = 'fit_raw'
low = 250
high = 1000

plt.figure()
plt.scatter(df[x], df[y], alpha=0.05)
plt.xlabel(x)
plt.ylabel(y)
#plt.xlim(low,high)
#plt.ylim(-400,0)
plt.show()

idx = list(df.index[df[x].between(low, high)])
shuffle(idx)
folder_in = r"D:\10_ALL_MERGE"
if True:
    for i, cell in enumerate(idx):
        if i > 10:
            break
        # split name
        parts = cell.split("_")
        
        # file base name (everything except the last part, which is Cxxx)
        file_base = "_".join(parts[:-1])          # e.g. 32_Zost2_YM
        cell_id = parts[-1]                        # e.g. C007

        csv_path = Path(f'{folder_in}//{file_base}_final.csv')
        
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            continue

        # column name to plot
        col_name1 = f"{file_base}_{cell_id}_dfovernoise"
        col_name2 = f"{file_base}_{cell_id}_spikeprob"
        df = pd.read_csv(csv_path)
        if col_name1 not in df.columns:
            print(f"Column not found: {col_name1}")
            continue

        plt.figure()
        plt.plot(df[col_name1].values)
        plt.plot(df[col_name2].values/5, alpha=0.8)
        plt.title(col_name1)
        plt.xlabel("frame")
        plt.ylabel("df / noise")
        plt.tight_layout()
        plt.show()

print(len(idx))


#%%
import pandas as pd

QC_file = r"D:\df_QC_FINAL.csv"
df = pd.read_csv(QC_file)
# parameters that are accepting a cell when inside boundaries
qc_params = {
    "area_px":    (50, None),
    "circul":     (0.4, None),
    "axis_ratio": (None, 5),
    "perim_px":   (20, None),
    "fit_delta":  (None, -6),
    "fit_raw":    (None, -40)}


pass_mask = pd.Series(True, index=df.index)
for col, (lo, hi) in qc_params.items():
    if lo is not None and hi is not None:
        pass_mask &= df[col].between(lo, hi)
    elif lo is not None:
        pass_mask &= df[col] >= lo
    elif hi is not None:
        pass_mask &= df[col] <= hi

df["pass"] = pass_mask
df.index = df['cell_ID']
df.to_csv(r"D:\df_QC_FINAL_pass.csv")
