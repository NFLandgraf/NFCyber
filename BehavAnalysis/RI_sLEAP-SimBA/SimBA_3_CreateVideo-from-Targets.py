#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import os

csv_path = r"Y:\Neuronal Networks\SimBA\pretrained_Mouse_RI\Defensive\RI_01_43.csv"
out_path = r"C:\Users\landgrafn\Desktop\Defensive.mp4"
fps = 30
fig_w = 8
fig_h = 8
dpi = 120
behav = 'Ground_defensive'
bodyparts = ['Nose', 'Ear_left', 'Ear_right', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Tail_end']
animals = ['1', '2']

connections = [
    ('Nose', 'Ear_left'),
    ('Nose', 'Ear_right'),
    ('Nose', 'Center'),
    ('Center', 'Lat_left'),
    ('Center', 'Lat_right'),
    ('Center', 'Tail_base'),
    ('Tail_base', 'Tail_end'),
    ('Lat_left', 'Tail_base'),
    ('Lat_right', 'Tail_base')
]

colors = {'1': 'red', '2': 'blue'}

df = pd.read_csv(csv_path)
n_behav = (pd.to_numeric(df[behav], errors='coerce') == 1).sum()
print(f'{n_behav} / {len(df)}')
print(f'{n_behav} / {len(df)}')
print(f'{n_behav} / {len(df)}')
print(f'{n_behav} / {len(df)}')

escape_frames = df.index[pd.to_numeric(df[behav], errors='coerce') == 1].tolist()
print(f'Frames: {len(escape_frames)}')
print(escape_frames[:50])

#%%
# as a last sanity check, create a video from the final targets_inserted csv with the annotated behav

if 'scorer' in df.columns:
    df = df.drop(columns=['scorer'])

needed_cols = []
for a in animals:
    for bp in bodyparts:
        needed_cols += [f'{bp}_{a}_x', f'{bp}_{a}_y']
if behav not in df.columns:
    raise ValueError(f"Column {behav} not found in csv.")

for col in needed_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

all_x = []
all_y = []
for a in animals:
    for bp in bodyparts:
        all_x.append(df[f'{bp}_{a}_x'].values)
        all_y.append(df[f'{bp}_{a}_y'].values)

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

mask = np.isfinite(all_x) & np.isfinite(all_y)
xmin, xmax = np.nanmin(all_x[mask]), np.nanmax(all_x[mask])
ymin, ymax = np.nanmin(all_y[mask]), np.nanmax(all_y[mask])

xpad = max(10, (xmax - xmin) * 0.05)
ypad = max(10, (ymax - ymin) * 0.05)

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
canvas = FigureCanvas(fig)

canvas = FigureCanvas(fig)
canvas.draw()
width, height = canvas.get_width_height()
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for i in range(len(df)):
    print(f'{i} / {len(df)}')
    ax.clear()
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Frame {i}')

    for a in animals:
        pts = {}
        for bp in bodyparts:
            x = df.at[i, f'{bp}_{a}_x']
            y = df.at[i, f'{bp}_{a}_y']
            if pd.notna(x) and pd.notna(y):
                pts[bp] = (x, y)
                ax.scatter(x, y, s=40, color=colors[a])
                ax.text(x + 2, y + 2, f'{bp}_{a}', color=colors[a], fontsize=8)

        for bp1, bp2 in connections:
            if bp1 in pts and bp2 in pts:
                x1, y1 = pts[bp1]
                x2, y2 = pts[bp2]
                ax.plot([x1, x2], [y1, y2], color=colors[a], linewidth=1.5, alpha=0.8)

    if int(df.at[i, behav]) == 1:
        ax.text(0.02, 0.98, behav, transform=ax.transAxes, ha='left', va='top', fontsize=16, color='white', bbox=dict(facecolor='red', alpha=0.8, edgecolor='none'))

    ax.grid(True, alpha=0.3)

    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(height, width, 4)
    img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    writer.write(img)

writer.release()
plt.close(fig)

print(f'Saved video to: {out_path}')