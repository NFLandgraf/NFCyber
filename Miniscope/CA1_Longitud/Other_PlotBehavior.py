#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def render_traces_to_canvas(
    df,
    xy_pairs,
    xlim, ylim,
    bins=800,
):
    """
    Render all bodypart traces to a 2D canvas using 2D histograms.
    Returns float canvas with larger values where the animal spent more time.
    """
    canvas = np.zeros((bins, bins), dtype=np.float32)

    x_min, x_max = xlim
    y_min, y_max = ylim

    for _, xcol, ycol in xy_pairs:
        x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()

        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue

        # keep only points in bounds
        x = x[m]
        y = y[m]
        inb = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x = x[inb]
        y = y[inb]
        if x.size == 0:
            continue

        # 2D histogram (note: histogram2d returns [x_bins, y_bins])
        H, _, _ = np.histogram2d(
            x, y,
            bins=bins,
            range=[[x_min, x_max], [y_min, y_max]]
        )

        # transpose so canvas is [row(y), col(x)]
        canvas = canvas + H.T.astype(np.float32)

    return canvas

def save_canvas_as_jpg(canvas, out_path, gamma=0.35):
    """
    Normalize + gamma-correct so traces pop, then save as JPG.
    Everything is one color (grayscale).
    """
    img = canvas.copy()
    if img.max() > 0:
        img = img / img.max()
    img = np.clip(img, 0, 1) ** gamma  # gamma < 1 brightens sparse traces

    # Save without axes/borders
    fig = plt.figure(frameon=False)
    fig.set_size_inches(6, 6)  # change output size if you like
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, origin="lower")  # origin lower to match typical coords
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def process_many_dfs_make_traces_and_maxproj(
    in_folder,
    out_folder,
    pattern="*.csv",
    bins=800,
    xlim=None, ylim=None,
):
    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(in_folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {in_folder}")
    
 # ---------- pass 2: render each + max-projection ----------
    maxproj = np.zeros((bins, bins), dtype=np.float32)

    for f in files:
        print(f)
        df = pd.read_csv(f)
        pairs = [('head', 'head_x', 'head_y')]

        canvas = render_traces_to_canvas(df, pairs, xlim=xlim, ylim=ylim, bins=bins)

        # per-file jpg
        out_jpg = out_folder / f"{f.stem}_trace.jpg"
        save_canvas_as_jpg(canvas, out_jpg)

        # global max projection
        maxproj = np.maximum(maxproj, canvas)

    # save max projection
    out_max = out_folder / "MAXPROJ_traces.jpg"
    save_canvas_as_jpg(maxproj, out_max)

    print(f"Done.\nPer-file traces in: {out_folder}\nMax projection: {out_max}")
    print(f"Bounds used: xlim={xlim}, ylim={ylim}")

# Example usage:
process_many_dfs_make_traces_and_maxproj(
    in_folder=r"D:\YM",
    out_folder=r"D:\out",
    pattern="*.csv",
    bins=900,
    xlim=(0, 580), ylim=(0, 485)  # <-- set if you know arena bounds
)

#%%
# creates maximum image from all behav traces
from pathlib import Path
import numpy as np
import pandas as pd
import imageio.v2 as imageio

def df_points_to_canvas_exact(df, xy_pairs, x_max, y_max, radius=0):
    """
    Create canvas of shape (y_max, x_max).
    Pixel (x, y) == coordinate (x, y).
    """
    canvas = np.zeros((y_max, x_max), dtype=np.uint8)

    for _, xcol, ycol in xy_pairs:
        x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue

        # integer pixel indices
        xi = np.floor(x[m]).astype(np.int32)
        yi = np.floor(y[m]).astype(np.int32)

        # keep only valid pixels
        inb = (xi >= 0) & (xi < x_max) & (yi >= 0) & (yi < y_max)
        xi = xi[inb]
        yi = yi[inb]

        if xi.size == 0:
            continue

        if radius == 0:
            canvas[yi, xi] = 255
        else:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    xj = xi + dx
                    yj = yi + dy
                    ok = (xj >= 0) & (xj < x_max) & (yj >= 0) & (yj < y_max)
                    canvas[yj[ok], xj[ok]] = 255

    return canvas

def process_many_dfs_make_traces_and_maxproj_exact(
    in_folder,
    out_folder,
    pattern="*.csv",
    x_max=580,
    y_max=485,
    xy_pairs=None,
    radius=1,
):
    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(in_folder.glob(pattern))
    if not files:
        raise FileNotFoundError("No CSVs found")

    if xy_pairs is None:
        xy_pairs = [("head", "head_x", "head_y")]

    maxproj = np.zeros((y_max, x_max), dtype=np.uint8)

    for f in files:
        print(f"Processing {f.name}")
        df = pd.read_csv(f)

        canvas = df_points_to_canvas_exact(
            df,
            xy_pairs,
            x_max=x_max,
            y_max=y_max,
            radius=radius,
        )

        imageio.imwrite(out_folder / f"{f.stem}_trace.jpg", canvas, quality=95)
        maxproj = np.maximum(maxproj, canvas)

    imageio.imwrite(out_folder / "MAXPROJ_traces.jpg", maxproj, quality=95)

    print(f"Done.")
    print(f"Output dimensions: {x_max} x {y_max} px")

# Example
process_many_dfs_make_traces_and_maxproj_exact(
    in_folder=r"D:\YM",
    out_folder=r"D:\out",
    x_max=580,
    y_max=485,
    xy_pairs=[("head", "head_x", "head_y")],
    radius=1,
)


#%%
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# ==== USER INPUT ====
csv_path   = Path(r"D:\11_ALL_MERGE_trim\77_Zost2_YM_final_trim(15m).csv")
out_video  = Path(r"D:\77_Zost2_YM_final_trim(15m)_DLC.mp4")
fps        = 30  # frames per second for the output video
point_radius = 4
# ====================

# Load CSV
df = pd.read_csv(csv_path)

# Detect bodyparts: columns like "nose_x", "nose_y"
bodyparts = sorted({
    col[:-2] for col in df.columns
    if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
})

if not bodyparts:
    raise ValueError("No bodypart_x / bodypart_y columns found!")

print("Bodyparts found:", bodyparts)

# Get coordinate ranges (for all bodyparts together)
all_x = []
all_y = []

for bp in bodyparts:
    all_x.append(df[f"{bp}_x"].values)
    all_y.append(df[f"{bp}_y"].values)

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

# Remove NaNs
mask = ~np.isnan(all_x) & ~np.isnan(all_y)
all_x = all_x[mask]
all_y = all_y[mask]

if len(all_x) == 0:
    raise ValueError("All coordinates are NaN, nothing to plot.")

min_x, max_x = all_x.min(), all_x.max()
min_y, max_y = all_y.min(), all_y.max()

# Define canvas size with some margin
width  = 580
height = 485

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

n_frames = len(df)



for t in range(n_frames):
    # White background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    for i, bp in enumerate(bodyparts):
        x = df.at[t, f"{bp}_x"]
        y = df.at[t, f"{bp}_y"]

        # Skip if NaN
        if pd.isna(x) or pd.isna(y):
            continue

        # Shift coordinates to canvas coords
        # Option A: keep same orientation as in original video
        x_canvas = int(x)
        y_canvas = int(y)


        # Option B (Cartesian y-up): flip vertically
        # y_canvas = height - 1 - (int(round(y - min_y)) + margin)

        cv2.circle(frame, (x_canvas, y_canvas), point_radius, (0,0,0), thickness=-1)

        # Optional: label with bodypart name
        # cv2.putText(frame, bp, (x_canvas+5, y_canvas-5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    writer.write(frame)

writer.release()
print("Saved video to:", out_video)