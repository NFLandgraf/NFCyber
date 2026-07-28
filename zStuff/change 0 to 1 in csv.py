#%%
import pandas as pd
import os
import matplotlib.pyplot as plt



path = 'D:\\targets\\'
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

for file in files:
    print(file)
    input_file = path + file
    output_file = f'D:\\j\\{file}'

    # Load the CSV file
    df = pd.read_csv(input_file, index_col=0)

    # Change the value in row index 1, column 'Attack' to 0
    df.at[1, 'Attack'] = 0

    # Save the modified CSV (overwrite or choose new name)
    df.to_csv(output_file)