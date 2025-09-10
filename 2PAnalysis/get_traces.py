#%%
import pandas as pd
import os
from pathlib import Path

path = r"D:\2P\traces"
file_useless_string = ['CA1Dopa_2pTube_', 'alt_Ch0_prepro_MotCorr_']


def manage_filename(file):

    # managing file names
    file_name = os.path.basename(file)
    file_name_short = os.path.splitext(file_name)[0]
    for word in file_useless_string:
        file_name_short = file_name_short.replace(word, '')
    
    return file_name_short

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files

files = get_files(path, 'Spont')

#%%

base_df = pd.DataFrame()

for file in files:

    mean = pd.read_csv(file, usecols=["Mean1"])["Mean1"].reset_index(drop=True)

    file_name_short = manage_filename(file)
    base_df[file_name_short] = mean

base_df.to_csv('base_df.csv', index=False)
print(f"Done")


