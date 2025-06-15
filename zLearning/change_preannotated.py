#%%
import pandas as pd
import os
import matplotlib.pyplot as plt



path = 'D:\\annot\\'
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



cols_to_keep_DLC = ['scorer','Ear_left_1_x','Ear_left_1_y','Ear_left_1_p',
               'Ear_right_1_x','Ear_right_1_y','Ear_right_1_p',
               'Nose_1_x','Nose_1_y','Nose_1_p',
               'Center_1_x','Center_1_y','Center_1_p',
               'Lat_left_1_x','Lat_left_1_y','Lat_left_1_p',
               'Lat_right_1_x','Lat_right_1_y','Lat_right_1_p',
               'Tail_base_1_x','Tail_base_1_y','Tail_base_1_p',
               'Tail_end_1_x','Tail_end_1_y','Tail_end_1_p',

               'Ear_left_2_x','Ear_left_2_y','Ear_left_2_p',
               'Ear_right_2_x','Ear_right_2_y','Ear_right_2_p',
               'Nose_2_x','Nose_2_y','Nose_2_p',
               'Center_2_x','Center_2_y','Center_2_p',
               'Lat_left_2_x','Lat_left_2_y','Lat_left_2_p',
               'Lat_right_2_x','Lat_right_2_y','Lat_right_2_p',
               'Tail_base_2_x','Tail_base_2_y','Tail_base_2_p',
               'Tail_end_2_x','Tail_end_2_y','Tail_end_2_p']

cols_to_keep_solomon = ['scorer', 'Attack']


for file in files:
    print(file)
    input_file = path + file
    output_file_DLC = f'D:\\annot\\DLC\\{file}'
    output_file_Solomon = f'D:\\annot\\Solomon\\{file}'


    df = pd.read_csv(input_file)

    df_DLC = df[cols_to_keep_DLC]
    df_DLC.to_csv(output_file_DLC, index=False)

    df_solomon = df[cols_to_keep_solomon]
    df_solomon['Attack'] = df['Attack'].replace({1: 'Attack', 0: ''})
    df_solomon.loc[1,'Attack'] = 'Attack'
    df_solomon = df_solomon.rename(columns={'scorer': 'Time', 'Attack': 'Behaviour'})

    df_solomon.to_csv(output_file_Solomon, index=False)




