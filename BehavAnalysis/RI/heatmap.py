#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


path = 'C:\\Users\\nicol\\Desktop\\hTau\\'


def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    # print(f'\n{len(files)} files found')
    # for file in files:
    #     print(file)
    # print('\n')

    return files

files = get_files(path, 'csv')




def cleaning_raw_df(csv_file):
    df = pd.read_csv(csv_file, sep=';', low_memory=False)

    df['Time'] = df['Time'].str.replace(',', '.')
    df['Time'] = df['Time'].astype('float')
    df.index = df['Time']
    del df['Time']

    df = df.notna()

    behavs = ['AGG behavior', 'Threat behavior', 'Random behavior']
    behavior_mapping = {behavior: idx + 1 for idx, behavior in enumerate(behavs)}

    # Create the integer-based column
    df['true_behav'] = df[behavs].apply(lambda row: [behavior_mapping[col] for col in row.index[row]] if row.any() else [0],axis=1)
    df['true_behav'] = df['true_behav'].apply(lambda x: x[0] if len(x) == 1 else x)

    df = df['true_behav']

    return df




def plot(df, series):
    plt.imshow(true_behav, cmap=cmap, aspect='auto', interpolation='nearest')
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 


plt.figure(figsize=(8, 0.5))
cmap = ListedColormap(['white', 'red', 'yellow', 'green'])  # Specify colors for [0, 1, 2, 3]

for file in files:
    df = cleaning_raw_df(file)
    true_behav = df.values.reshape(1, -1)
    plot(df, true_behav)


gap = 300  # Interval for displaying labels
xticks_pos = np.arange(0, len(df), gap)  # Get the positions of ticks with the specified gap
xticks_labels = df.index[xticks_pos]  # Get the corresponding labels from the DataFrame index
xticks_labels = (xticks_labels / 60).astype('int')

plt.xlabel('Time [min]')
plt.yticks = []
plt.xticks(ticks=xticks_pos, labels=xticks_labels)  # Use custom tick positions and labels



plt.show()


#%%

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

path = 'C:\\Users\\nicol\\Desktop\\hTau\\'


def get_files(path, common_name):
    # Get files from the directory matching the common name
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    return files

files = get_files(path, 'csv')


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


def plot(df, true_behav, ax):
    # Plot the heatmap on the specified axes (ax)
    ax.imshow(true_behav, cmap=cmap, aspect='auto', interpolation='nearest')
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)


# Create the figure for all subplots (heatmaps)
n_files = len(files)
n_cols = 1  # Define the number of columns in the grid
n_rows = (n_files + n_cols - 1) // n_cols  # Calculate the number of rows needed to fit all files

# Create subplots with the defined rows and columns
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))  # Adjust figure size as needed
axes = axes.flatten()  # Flatten axes for easy iteration

# Define colormap for the heatmap
cmap = ListedColormap(['white', 'red', 'orange', 'green'])  # Specify colors for [0, 1, 2, 3]

# Plot heatmaps for each file
for i, file in enumerate(files):
    df = cleaning_raw_df(file)
    true_behav = df.values.reshape(1, -1)
    true_behav = np.array([update_array(true_behav[0], 10)])
    ax = axes[i]  # Get the corresponding subplot axis
    plot(df, true_behav, ax)

    # Customize x-axis for each subplot
    gap = 300  # Interval for displaying labels
    xticks_pos = np.arange(0, len(df), gap)  # Get the positions of ticks with the specified gap
    xticks_labels = df.index[xticks_pos]  # Get the corresponding labels from the DataFrame index
    xticks_labels = (xticks_labels / 60).astype('int')  # Convert to minutes

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels)
    #ax.set_xlabel('Time [min]')
    #ax.set_ylabel('Behavior')

# Remove any empty subplots
for i in range(n_files, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()  # Adjust layout for neatness
plt.show()
