#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os
import seaborn as sns



path = 'D:\\data_processed\\'


def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files
files = get_files(path, 'csv')


#%%
for file in files:
    df = pd.read_csv(file)
    df.index = df['Time']
    del df['Time']
    df = df[-2: 10]

    vmin, vmax = 0, 25
    n_rows = len(df.columns)

    # Plot each column as a heatmap in a new row
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, n_rows * 5))

    # Iterate through columns to create heatmap for each
    for idx, col in enumerate(df.columns):
        sns.heatmap(df[[col]].T, cmap='viridis', vmin=vmin, vmax=vmax, ax=axes[idx], cbar=True, xticklabels=False, yticklabels=False)
        axes[idx].set_title(f'Heatmap for {file}')

    plt.tight_layout()
    plt.savefig(f"{file}.pdf", format="pdf", bbox_inches="tight")