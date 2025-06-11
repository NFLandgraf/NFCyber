import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------- SETTINGS --------
csv_file = 'D:\\2025-02-12_hTauxAPP1(3m)_RI3_m222_Test_edit_fps.csv'  # Replace with your actual file path
output_video = 'pose_video.mp4'
frame_width = 800
frame_height = 470
fps = 30
n_frames = 300
ioo = 0
# --------------------------

# Load CSV with standard header (not multi-index)
df = pd.read_csv(csv_file)
df = df.head(n_frames)

# Detect body parts and animals
all_columns = df.columns
bodyparts = ['Center', 'Nose', 'Tail_base', 'Tail_end', 'Ear_right', 'Ear_left', 'Lat_right', 'Lat_left']
animals = ['1', '2']
colors = {'1': 'red', '2': 'blue'}

# Define body part connections
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
        for bp in bodyparts:
            x_col = f'{bp}_{animal_id}_x'
            y_col = f'{bp}_{animal_id}_y'
            if x_col in df.columns and y_col in df.columns:
                x = df.at[frame_idx, x_col]
                y = df.at[frame_idx, y_col]
                if pd.notna(x) and pd.notna(y):
                    ax.plot(x, y, 'o', color=colors[animal_id])

        for part1, part2 in connections:
            x1_col, y1_col = f'{part1}_{animal_id}_x', f'{part1}_{animal_id}_y'
            x2_col, y2_col = f'{part2}_{animal_id}_x', f'{part2}_{animal_id}_y'
            if all(col in df.columns for col in [x1_col, y1_col, x2_col, y2_col]):
                x1, y1 = df.at[frame_idx, x1_col], df.at[frame_idx, y1_col]
                x2, y2 = df.at[frame_idx, x2_col], df.at[frame_idx, y2_col]
                if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
                    ax.plot([x1, x2], [y1, y2], '-', color=colors[animal_id], linewidth=1)
    return []

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(df), init_func=init, blit=False)

# Save video
ani.save(output_video, fps=fps, extra_args=['-vcodec', 'libx264'])
print(f"✅ Saved video to: {output_video}")