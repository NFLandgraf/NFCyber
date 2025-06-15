#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load DLC-style CSV
df = pd.read_csv("D:\\annot\\RI_01_6_DLC.csv")

# Drop the first column if it's 'scorer' or frame index
if 'scorer' in df.columns[0]:
    df = df.iloc[:, 1:]

# Set up body parts and colors
bodyparts = ['Nose', 'Tail_base']  # You can add more
colors = {'1': 'red', '2': 'blue'}

# Plotting
plt.figure(figsize=(8, 6))
for animal in ['1', '2']:
    for bp in bodyparts:
        x_col = f"{bp}_{animal}_x"
        y_col = f"{bp}_{animal}_y"
        if x_col in df.columns and y_col in df.columns:
            plt.plot(df[x_col], df[y_col], 'o-', label=f"{bp}_{animal}", alpha=0.7, color=colors[animal])

plt.gca().invert_yaxis()  # Match video coordinate orientation
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Body part trajectories")
plt.tight_layout()
plt.show()
