
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === PARAMETERS ===
input_csv = "D:\\annot\\DLC\\CSDS01303.csv"   # <-- Change to your file name

output_video = "mouse_tracking_video.mp4"
fps = 30
dpi = 150
confidence_threshold = 0.5  # Optional: only show body parts above this confidence

# === BODY PARTS TO TRACK ===
bodyparts = ["Ear_left", "Ear_right", "Nose", "Center", "Lat_left", "Lat_right", "Tail_base", "Tail_end"]

# === LOAD DATA ===
df = pd.read_csv(input_csv)

# === PREPARE COORDINATES ===
animal_1 = {bp: (df[f"{bp}_1_x"], df[f"{bp}_1_y"], df[f"{bp}_1_p"]) for bp in bodyparts}
animal_2 = {bp: (df[f"{bp}_2_x"], df[f"{bp}_2_y"], df[f"{bp}_2_p"]) for bp in bodyparts}

# === PLOT SETUP ===
fig, ax = plt.subplots()
ax.set_xlim(0, 640)  # Adjust if needed
ax.set_ylim(1200, 0)  # Flip y-axis

points1 = [ax.plot([], [], 'ro')[0] for _ in bodyparts]  # Red = Animal 1
points2 = [ax.plot([], [], 'bo')[0] for _ in bodyparts]  # Blue = Animal 2

def init():
    for p in points1 + points2:
        p.set_data([], [])
    return points1 + points2

def animate(i):
    for j, bp in enumerate(bodyparts):
        # Animal 1
        x1, y1, p1 = animal_1[bp][0][i], animal_1[bp][1][i], animal_1[bp][2][i]
        if p1 > confidence_threshold:
            points1[j].set_data([x1], [y1])
        else:
            points1[j].set_data([], [])

        # Animal 2
        x2, y2, p2 = animal_2[bp][0][i], animal_2[bp][1][i], animal_2[bp][2][i]
        if p2 > confidence_threshold:
            points2[j].set_data([x2], [y2])
        else:
            points2[j].set_data([], [])
    return points1 + points2

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(df), interval=1000/fps, blit=True)

ani.save(output_video, fps=fps, dpi=dpi)
plt.close()
print(f"Saved video to {output_video}")




