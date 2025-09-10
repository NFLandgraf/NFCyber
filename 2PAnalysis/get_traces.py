#%%
import pandas as pd
import os
from pathlib import Path
#%%
path = r"D:\2P_Analysis\Traces\deconvolved"
file_useless_string = ['CA1Dopa_2pTube_', 'alt_Ch0_prepro_MotCorr_', '_Trace_deconv']


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

files = get_files(path, 'Airpuff')

#%%

correc_df = pd.DataFrame()
spikes_df = pd.DataFrame()

for file in files:

    correc = pd.read_csv(file, usecols=["spks"])["spks"].reset_index(drop=True)
    #spikes = pd.read_csv(file, usecols=["spks"])["spks"].reset_index(drop=True)

    file_name_short = manage_filename(file)
    correc_df[file_name_short] = correc
    #spikes_df[file_name_short] = spikes

correc_df.to_csv('correc_df.csv', index=False)
#spikes_df.to_csv('spikes_df.csv', index=False)
print(f"Done")




#%%

# load your CSV
df = pd.read_csv(r"D:\2P_Analysis\Traces\Airpuff_Events.csv")

# loop over columns and save each as its own CSV
for col in df.columns:
    df[[col]].to_csv(f"{col}_Events.csv", index=False)
    print(f"Saved {col}.csv")


