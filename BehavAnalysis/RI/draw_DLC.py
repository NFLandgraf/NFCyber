import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------- SETTINGS --------
csv_file = 'D:\\SimBA\\Rambo\\project_folder\\csv\\input_csv\\2025-02-12_hTauxAPP1(3m)_RI3_m222_Test_edit_fps.csv'
output_video = 'pose_video.mp4'
frame_width = 800
frame_height = 470
fps = 30
n_frames = 300
ioo = 0
# --------------------------

# Load DLC-style CSV with multi-index header
df = pd.read_csv(csv_file, header=[0, 1, 2])
df = df.head(n_frames)

# Parse all bodyparts from the header
all_columns = df.columns
bodypart_coords = [(bp, coord) for (bp, coord) in zip(all_columns.get_level_values(1), all_columns.get_level_values(2)) if coord == 'x']
bodyparts_unique = sorted(set(bp.rsplit('_', 1)[0] for bp, _ in bodypart_coords))
animals = sorted(set(bp.rsplit('_', 1)[1] for bp, _ in bodypart_coords))
colors = {animal: 'red' if animal == '1' else 'blue' for animal in animals}

# Define connections between body parts
connections = [
    ('Nose', 'Ear_left'),
    ('Nose', 'Ear_right'),
    ('Nose', 'Center'),
    ('Center', 'Tail_base'),
    ('Tail_base', 'Tail_end'),
    ('Ear_left', 'Lat_left'),
    ('Ear_right', 'Lat_right'),
    ('Lat_right', 'Tail_base'),
    ('Lat_left', 'Tail_base'),
]

# Create plot
fig, ax = plt.subplots()

def init():
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)  # Flip Y-axis
    return []

def update(frame_idx):
    global ioo
    ioo += 1
    print(ioo)
    ax.clear()
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)
    ax.set_title(f'Frame {frame_idx}')

    for animal_id in animals:
        for bp in bodyparts_unique:
            key = f"{bp}_{animal_id}"
            try:
                x = df[(key, 'x')].iloc[frame_idx]
                y = df[(key, 'y')].iloc[frame_idx]
                if pd.notna(x) and pd.notna(y):
                    ax.plot(x, y, 'o', color=colors[animal_id])
            except KeyError:
                continue

        for bp1, bp2 in connections:
            key1 = f"{bp1}_{animal_id}"
            key2 = f"{bp2}_{animal_id}"
            try:
                x1, y1 = df[(key1, 'x')].iloc[frame_idx], df[(key1, 'y')].iloc[frame_idx]
                x2, y2 = df[(key2, 'x')].iloc[frame_idx], df[(key2, 'y')].iloc[frame_idx]
                if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
                    ax.plot([x1, x2], [y1, y2], '-', color=colors[animal_id], linewidth=1)
            except KeyError:
                continue

    return []

# Create and save animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
ani.save(output_video, fps=fps, extra_args=['-vcodec', 'libx264'])
print(f"✅ Saved: {output_video}")