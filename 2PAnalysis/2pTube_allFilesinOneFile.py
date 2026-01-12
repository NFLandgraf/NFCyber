#%%
import os
import pandas as pd

folder = "D:\\2pTube_rGRABDA_Airpuff\\Results_FullFrame"

# List all CSV files
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

dfs = []
for f in files:
    path = os.path.join(folder, f)

    # Read the one-column CSV
    df = pd.read_csv(path, header=None)  # no header in files

    # Rename column using filename without .csv
    colname = os.path.splitext(f)[0]
    df.columns = [colname]

    dfs.append(df)

# Combine all as columns
combined = pd.concat(dfs, axis=1)

# Save
combined.to_csv("combined.csv", index=False)

print("Done! Combined file saved as combined.csv")
