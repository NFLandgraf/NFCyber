#%%
import pandas as pd
from pathlib import Path

folder = Path(r"E:\SimBA")
column_name = 'Attack'

results = []

for csv_file in folder.glob("*.csv"):

    print(csv_file)

    # count 0 -> 1 transitions
    df = pd.read_csv(csv_file)
    attack = df[column_name].fillna(0).astype(int)
    n_attack_bouts = ((attack == 1) & (attack.shift(fill_value=0) == 0)).sum()
    results.append({"file": csv_file.name, "n_attack_bouts": n_attack_bouts})

results_df = pd.DataFrame(results)

print(results_df)

#%%

results_df.to_csv(folder / "attack_bout_counts.csv", index=False)