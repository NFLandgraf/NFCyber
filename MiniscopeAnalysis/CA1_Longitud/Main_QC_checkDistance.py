#%%

import numpy as np
import pandas as pd
from pathlib import Path

folder_in = Path(r"D:\10_ALL_MERGE")

def get_files(path, common_name='OF', suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name in p.name) and p.name.endswith(suffix)])
    print(f"\n{len(files)} files found")
    for f in files:
        print(f)
    print('\n\n')
    return files


files = get_files(folder_in)

distances = []
for file in files:
    print(file)
    df = pd.read_csv(file, index_col='Mastertime [s]')
    df = df.iloc[34:]

    x = df["head_x"].values
    y = df["head_y"].values

    # differences between consecutive frames
    dx = np.diff(x)
    dy = np.diff(y)
    step_dist = np.sqrt(dx**2 + dy**2)

    dist = round(np.nansum(step_dist) /1000 ,2)
    distances.append(dist)
    print(dist)

print(distances)