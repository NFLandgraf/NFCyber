#%%
import pandas as pd
import os
import matplotlib.pyplot as plt



path = 'D:\\'
file_format = '.csv'

def get_data():
    # get all file names in directory into list
    files = [file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                file_format in file]
    print(f'{len(files)} files found in path directory {path}\n'
        f'{files}\n')
    
    return files
files = get_data()

#%%

def change_csv(file):
    print(file)

    input_file = path + file
    output_file = input_file.replace(file_format, '') + '_adapt' + file_format

    df = pd.read_csv(input_file)

    # print(df.isna().sum())  # Total missing values per column
    # print('stop')
    # print((df == 0).sum())  # Any zero-values in x/y columns

    df['track'] = df['track'].replace({'Resi': 0, 'Intr': 1})

    


    # # List of body parts
    # bodyparts = ['Nose', 'Ear_left', 'Ear_right', 'Center', 'Tail_base', 'Tail_end', 'Lat_left', 'Lat_right']

    # # Create a new DataFrame with one row per frame
    # # We'll use frame_idx as index and expand all body parts for both animals
    # output_rows = []

    # # Group by frame index
    # for frame, group in df.groupby('frame_idx'):
    #     row = {'frame_idx': frame}
    #     for _, animal in group.iterrows():
    #         track = animal['track']
    #         animal_id = '1' if track.lower().startswith('resi') else '2'  # Resi=1, Intr=2
    #         for part in bodyparts:
    #             row[f'{part}_{animal_id}_x'] = animal[f'{part}.x']
    #             row[f'{part}_{animal_id}_y'] = animal[f'{part}.y']
    #             row[f'{part}_{animal_id}_p'] = animal[f'{part}.score']
    #     output_rows.append(row)

    # # Convert to DataFrame
    # reshaped_df = pd.DataFrame(output_rows)
    df.to_csv(output_file, index=False)

for file in files:
    change_csv(file)


