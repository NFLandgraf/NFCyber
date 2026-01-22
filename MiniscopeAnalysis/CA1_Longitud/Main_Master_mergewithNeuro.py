#%%
'''
This is for merging 2 csvs: 
The Master_trim that we created during the trimming step (in there is sync Mastertime, Neuro_frames, Cam_frames and DLC coords)
The traces csv that we got from Cascade (in there is framewise the spikeprob, dfovernoise and zscore for each cell)
'''
from pathlib import Path
import pandas as pd

folder_Master = Path(r"D:\new\6_Master_trim")
folder_Neuroo = Path(r"D:\new\9_Neuro_Cascade")
out_folder = Path(r"D:\new\merge")

def merge_csvs(files_Neuroo, file_master):
    
        base = file_master.stem
        base = base.replace('_master_trim', '')
        print(f'------- {base} -------')

        neuro_candidates = [b for b in files_Neuroo if base in b.name]
        
        if not neuro_candidates:
            # if there are no cells identified, just export the master file without any cell traces
            print(f"🔴🔴🔴 No match for: {file_master.name}, export master only")
            df_master = pd.read_csv(file_master)
            df_out = df_master
            df_out.to_csv(f'{out_folder}//{base}_final.csv', index=False)
            return

        elif len(neuro_candidates) > 1:
            print(f"🔴🔴🔴 Multiple matches for {file_master.name}: {[c.name for c in neuro_candidates]}")
            raise ValueError('helphelphelp')
        
        file_neuroo = neuro_candidates[0]
        df_master = pd.read_csv(file_master)
        df_neuroo = pd.read_csv(file_neuroo)

        # clean df_Neuroo
        #df_neuroo.columns = df_neuroo.columns.str.replace(f'{base}_', "", regex=False)
        df_neuroo = df_neuroo.reindex(sorted(df_neuroo.columns), axis=1)

        # Check row count
        if len(df_master) != len(df_neuroo):
            print(f"[SKIP] Row mismatch for {file_master.name} + {file_neuroo.name}: {len(df_master)} vs {len(df_neuroo)}")
            return
        
        if not df_master.index.equals(df_neuroo.index):
            print("🔴🔴🔴 different index!!!")

        # Merge by adding columns (side-by-side)
        df_out = pd.concat([df_master, df_neuroo], axis=1)

        # Optional: prevent duplicate column names by auto-renaming B duplicates
        if df_out.columns.duplicated().any():
            print('eeeh')
            # Rename duplicate columns that came from df_b by appending "_B"
            new_b_cols = []
            for c in df_neuroo.columns:
                new_b_cols.append(c if c not in df_master.columns else f"{c}_B")
            df_b_renamed = df_neuroo.copy()
            df_b_renamed.columns = new_b_cols
            df_out = pd.concat([df_master, df_b_renamed], axis=1)
        
        df_out = df_out.drop('Unnamed: 0', axis=1)
        cols = list(df_out.columns)
        cols.insert(2, cols.pop(cols.index('Cam_frames')))
        df_out = df_out[cols]

        df_out.to_csv(f'{out_folder}//{base}_final.csv', index=False)
        print(f"Master: {df_master.shape[0]} rows, {df_master.shape[1]} cols")
        print(f"Neuroo: {df_neuroo.shape[0]} rows, {df_neuroo.shape[1]} cols")
        print(f"FINALE: {df_out.shape[0]} rows, {df_out.shape[1]} cols")

files_Master = sorted([p for p in folder_Master.iterdir() if p.is_file() and p.suffix.lower() == '.csv'.lower()])
files_Neuroo = sorted([p for p in folder_Neuroo.iterdir() if p.is_file() and p.suffix.lower() == '.csv'.lower()])
print(f"Found {len(files_Master)} files in files_Master and {len(files_Neuroo)} files in files_Neuroo")

for file_master in files_Master:
    merge_csvs(files_Neuroo, file_master)
