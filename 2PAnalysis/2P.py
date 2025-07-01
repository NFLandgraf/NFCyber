#%%
#
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d





path = 'C:\\Users\\landgrafn\\Desktop\\Dbh-GCaMP8f_exvivo_2P_Test\\'
fps = 22.2


def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files
files = get_files(path, 'csv')

 
def get_data_csv(file):

    df = pd.read_csv(file)
    df['Slice'] = df['Slice']/fps
    df.index = df['Slice']
    del df['Slice']

    return df

df = get_data_csv(files[0])


df['zscore'] = (df['Mean'] - np.mean(df['Mean'])) / np.std(df['Mean'])
print(df)

df['smoothed'] = df['zscore'].rolling(window=6).mean()

df['smoothed'].to_csv(path + 'dff_allfiles.csv')

plt.plot(df['smoothed'])
plt.show()

