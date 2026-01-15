#%%
import os

folder_path = r"W:\_proj_CA1Dopa\CA1Dopa_Longitud\Analysis_new\Videos_4_mask"

old_text = "OF_crop"
new_text = "OF_crop_mask"

for filename in os.listdir(folder_path):

    if os.path.isfile(os.path.join(folder_path, filename)):

        if old_text in filename:
            new_filename = filename.replace(old_text, new_text)
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            os.rename(old_file, new_file)
            print(f'Renamed: "{filename}" → "{new_filename}"')