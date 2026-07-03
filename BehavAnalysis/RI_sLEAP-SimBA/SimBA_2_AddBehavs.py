#%%
'''
This takes the final files for training and adds behaviors that were missing in the annotated data.
'''

import pandas as pd
import os
import csv

path = r"C:\Users\landgrafn\Desktop\SIMBA\SimBA_Model_TailRattle\SimBA_Model_TailRattle_NicoPC\project_folder\csv\targets_inserted"

behaviors = [
    'Investigate',
    'Following',
    'nose2nose',
    'Anogenital_Sniff',
    'Approach',
    'Tail_Rattle',
    'Mounting',
    'Circle',
    'Chase',
    'Agitated',
    'Attack'
]

behaviors = ['Tail_Rattle']
count = 0
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith('.csv')]

for file in files:
    file_path = os.path.join(path, file)
    print(f'Processing: {file}')

    df = pd.read_csv(file_path)

    # Fix first column header
    if df.columns[0].startswith('Unnamed'):
        df.columns = [''] + list(df.columns[1:])

    # Add missing behaviors
    missing = [b for b in behaviors if b not in df.columns]
    for b in missing:
        df[b] = 0
        count += 1

    # --- custom writing ---
    with open(file_path, 'w', newline='') as f:

        # HEADER (all quoted)
        header = ['""'] + [f'"{col}"' for col in df.columns[1:]]
        f.write(','.join(header) + '\n')

        # DATA
        for row in df.itertuples(index=False, name=None):
            row = list(row)

            # first column quoted
            row_str = [f'"{row[0]}"']

            # rest NOT quoted
            row_str += [str(v) for v in row[1:]]

            f.write(','.join(row_str) + '\n')

    print(f'  Added: {missing}')

print('Done.')
print(count)