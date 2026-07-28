#%%

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#path = 'C:\\Users\\nicol\\Desktop\\hTau\\'
path = 'C:\\Users\\landgrafn\\Desktop\\hTau AGG\\'


def get_files(path, common_name):
    # Get files from the directory matching the common name
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    return files

def cleaning_raw_df(csv_file):
    # Clean the raw dataframe
    df = pd.read_csv(csv_file, sep=';', low_memory=False)

    df['Time'] = df['Time'].str.replace(',', '.')
    df['Time'] = df['Time'].astype('float')
    df.index = df['Time']
    del df['Time']

    df = df.notna()

    # Define behaviors
    behavs = ['AGG behavior', 'Threat behavior', 'Random behavior']
    behavior_mapping = {behavior: idx + 1 for idx, behavior in enumerate(behavs)}

    # Create the integer-based column for behavior
    df['true_behav'] = df[behavs].apply(lambda row: [behavior_mapping[col] for col in row.index[row]] if row.any() else [0], axis=1)
    df['true_behav'] = df['true_behav'].apply(lambda x: x[0] if len(x) == 1 else x)

    df = df['true_behav']

    return df

def update_array(arr, x):
    arr = arr.copy()  # Make a copy of the array to avoid modifying the original array
    n = len(arr)

    # Loop through the array to find each occurrence of 1
    for i in range(n):
        if arr[i] == 1:
            # Set the previous x elements to 1 (if within bounds)
            start = max(0, i - x)
            arr[start:i] = 1
            
            # Set the next x elements to 1 (if within bounds)
            end = min(n, i + x + 1)
            arr[i+1:end] = 1
            
    return arr

def plot(df, true_behav, ax, i):

    if i == 6:
        ax.imshow(true_behav, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.tick_params(left=False, right=False, labelleft=False, bottom=False)
    else:
        ax.imshow(true_behav, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

def heatmap(files):
    n_files = len(files)
    n_cols = 1
    n_rows = (n_files + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3))
    axes = axes.flatten()

    cmap = ListedColormap(['white', 'red', 'orange', 'green'])  # Specify colors for [0, 1, 2, 3]
    for i, file in enumerate(files[:7]):
        df = cleaning_raw_df(file)
        true_behav = df.values.reshape(1, -1)
        true_behav = np.array([update_array(true_behav[0], 10)])
        ax = axes[i] 
        plot(df, true_behav, ax, i)

        if i == 6:
            xticks_pos = np.arange(0, len(df), 300)
            xticks_labels = df.index[xticks_pos]
            xticks_labels = (xticks_labels / 60).astype('int')

            ax.set_xticks(xticks_pos)
            ax.set_xticklabels(xticks_labels)
            ax.set_xlabel('Time [min]')

    plt.show()



files = get_files(path, 'csv')
for file in files:
    df = cleaning_raw_df(file)
    print(df)
    count = (df == 3).sum()

    print(file)
    print(count)





