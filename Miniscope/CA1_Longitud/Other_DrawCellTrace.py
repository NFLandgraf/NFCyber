#%%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


fig, ax_left = plt.subplots()

csv_file = r"E:\CA1Dopa_Miniscope\Analysis\a\66_Zost2_YM_final.csv"
df = pd.read_csv(str(csv_file), index_col='Mastertime [s]')
trace1 = pd.to_numeric(df['66_Zost2_YM_C000_dff'], errors='coerce')

csv_file = r"E:\CA1Dopa_Miniscope\Analysis\a\66_Zost2_YM_final.csv"
df = pd.read_csv(str(csv_file), index_col='Mastertime [s]')
trace2 = pd.to_numeric(df['66_Zost2_YM_C000_dff_spikeprob'], errors='coerce')
trace2.index = trace1.index

# left y-axis
ax_left.plot(trace1, linewidth=1.0, color='black')
ax_left.set_xlabel("frame")
ax_left.set_ylabel("df/noise", color='black')
#ax_left.set_ylim(-0.01, 0.07)

# right y-axis
ax_right = ax_left.twinx()
ax_right.plot(trace2, linewidth=1, color='orange')
ax_right.set_ylabel("spike probability", color='orange')
#ax_right.set_ylim(0, 3)

x = 500
plt.xlim(x,x+100)

plt.tight_layout()
plt.show()


#%%


csv_file = r"E:\CA1Dopa_Miniscope\test\Out\66_Zost2_YM_motcor_trim_TracesDff.csv"


   
# Skip the "Time(s)/Cell Status" row
df = pd.read_csv(str(csv_file), skiprows=[1], index_col='Unnamed: 0')

time_col = df.columns[0]
time = df[time_col].to_numpy(dtype=np.float32)

cell_names = list(df.columns[1:])

neurons = df.iloc[:, 1:].to_numpy(dtype=np.float32)
flipped = neurons# / 10.0
#np.save(r"D:\new\Neuro_7_CNMFe\flip.npy", flipped)


column_maxima = df.max()

max_column = column_maxima.idxmax()
max_value = column_maxima.max()

print("Column:", max_column)
print("Maximum value:", max_value)



#%%

csv_file = r"E:\CA1Dopa_Miniscope\test\Out\66_Zost2_YM_motcor_trim_TracesDff.csv"
df = pd.read_csv(str(csv_file), skiprows=[1], index_col='Unnamed: 0')
df = df[1:]

trace = pd.to_numeric(df[' C000'], errors='coerce')

plt.plot(trace, linewidth=1.0, color='black')
plt.xlim(5,150)
plt.show()


#%%
from pathlib import Path
import pandas as pd

folder = Path(r"E:\CA1Dopa_Miniscope\CA1Dopa_Miniscope_Cascade")

replacements = {
    "spikeprob": "dff_spikeprob",
    "zscore": "dff_zscore"
}

for csv_file in folder.glob("*.csv"):
    df = pd.read_csv(csv_file)

    new_columns = []
    for col in df.columns:
        new_col = str(col)

        for old, new in replacements.items():
            new_col = new_col.replace(old, new)

        new_columns.append(new_col)

    df.columns = new_columns

    df.to_csv(csv_file, index=False)

    print(f"Saved: {csv_file.name}")

#%%
import os
import re
import pandas as pd
import random

folder = r"E:\CA1Dopa_Miniscope\Analysis\Master_10_merge_dfon+dff"

for file in os.listdir(folder):


    path = os.path.join(folder, file)
    df = pd.read_csv(path)

    # find columns ending in dfon_spikeprob
    dfon_cols = [col for col in df.columns if col.endswith("dfon_spikeprob")]
    dff_cols = [col for col in df.columns if "dff" in col]
    print(dfon_cols[0])
    print(dff_cols[0])

    if len(dfon_cols) == 0:
        print(f"{file}: no dfon_spikeprob column found")

    # get C-number digit lengths from dfon columns
    dfon_lengths = []
    for col in df.columns:
        if "dfon" in col:
            match = re.search(r"_C(\d+)_dfon", col)
            if match:
                dfon_lengths.append(len(match.group(1)))

    # get C-number digit lengths from dff columns
    dff_lengths = []
    for col in df.columns:
        if "dff" in col:
            match = re.search(r"_C(\d+)_dff", col)
            if match:
                dff_lengths.append(len(match.group(1)))

    if not dfon_lengths:
        print(f"{file}: no dfon columns found")
        continue

    if not dff_lengths:
        print(f"{file}: no dff columns found")
        continue

    # check that each type has consistent formatting
    if len(set(dfon_lengths)) > 1:
        print(f"{file}: inconsistent dfon digit lengths: {set(dfon_lengths)}")
        continue

    if len(set(dff_lengths)) > 1:
        print(f"{file}: inconsistent dff digit lengths: {set(dff_lengths)}")
        continue

    n_digits_dfon = dfon_lengths[0]
    n_digits_dff = dff_lengths[0]

    
    print(f"     dfon: {n_digits_dfon}, dff: {n_digits_dff}")


    # only change dff columns if formatting differs
    if n_digits_dff != n_digits_dfon:

        print(f"  -> changing dff from {n_digits_dff} to {n_digits_dfon} digits")

        new_columns = []

        for col in df.columns:

            if "dff" in col:
                match = re.search(r"_C(\d+)(?=_dff)", col)
                if match:
                    number = int(match.group(1))
                    new_number = f"{number:0{n_digits_dfon}d}"
                    col = (col[:match.start(1)] + new_number + col[match.end(1):])
            new_columns.append(col)

        df.columns = new_columns
        df.to_csv(path, index=False)

    else:
        print("  -> digit lengths already match, nothing changed")

    dfon_cols = [col for col in df.columns if col.endswith("dfon_spikeprob")]
    dff_cols = [col for col in df.columns if "dff" in col]
    print(dfon_cols[0])
    print(dff_cols[0])

    print()
    print()

