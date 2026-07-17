#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm



path = r"E:\AGG-DREADD\Habit_fps_DLC\hM4Di"
files = glob.glob(os.path.join(path, "*.csv"))


def travelled_distance(csv_path, bp= 'mouse_center', likelihood_cutoff=0.8):

    # calculates the travelled distance according to specific mouse coordinate
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # columns are: level 0 = scorer, level 1 = bodypart, level 2 = coord
    x = df.xs((bp, "x"), level=[1, 2], axis=1).iloc[:, 0]
    y = df.xs((bp, "y"), level=[1, 2], axis=1).iloc[:, 0]
    likelihood = df.xs((bp, "likelihood"), level=[1, 2], axis=1).iloc[:, 0]

    x = x.where(likelihood >= likelihood_cutoff)
    y = y.where(likelihood >= likelihood_cutoff)

    frame_distance = np.sqrt(x.diff()**2 + y.diff()**2).fillna(0)
    accumulated_distance = frame_distance.cumsum()
    total_distance = accumulated_distance.iloc[-1]

    result = pd.DataFrame({"frame_distance": frame_distance, "accumulated_distance": accumulated_distance})

    return result, total_distance


distances = []

for file in files:
    print(file)

    result, total_distance = travelled_distance(file)
    distances.append(float(total_distance))


print(distances)