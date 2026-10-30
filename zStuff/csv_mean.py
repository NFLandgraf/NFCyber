#%%
import pandas as pd
import re
from pathlib import Path


file_in = Path(r"C:\Users\landgrafn\Desktop\SAA_NFL_AI_NFL_HPC.csv")

# Load CSV
df = pd.read_csv(file_in)

# 1. Baseline correction
fluorescence_cols = [c for c in df.columns if c != "Time_h"]

for col in fluorescence_cols:
    baseline = df[col].iloc[:5].mean()
    df[col] = df[col] - baseline
df.to_csv('HPC_norm.csv', index=False)


# 2. Average replicates

# Example:
# HPC_A53T_ID01_Rep1_WellE1
# HPC_A53T_ID01_Rep2_WellF1
# HPC_A53T_ID01_Rep3_WellG1
# HPC_A53T_ID01_Rep4_WellH1
#
# becomes:
# HPC_A53T_ID01

groups = {}

for col in fluorescence_cols:

    # Remove "_RepX_WellXX"
    sample_name = re.sub(r"_Rep\d+_Well[A-Za-z0-9]+$", "", col)

    groups.setdefault(sample_name, []).append(col)


# New dataframe
df_mean = pd.DataFrame()

# Keep time column if present
if "Time_h" in df.columns:
    df_mean["Time_h"] = df["Time_h"]


# Calculate mean across technical replicates
for sample_name, cols in groups.items():
    df_mean[sample_name] = df[cols].mean(axis=1)


df_mean.to_csv('HPC_norm_mean.csv', index=False)

