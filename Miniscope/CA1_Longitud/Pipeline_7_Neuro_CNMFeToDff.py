#%%
import numpy as np
import re
import csv
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter
import pandas as pd
from tqdm import tqdm

'''
The output from CNMFe is in df_over_noise and for the spike probability, we need df/f
In here, we extract the df/f from the motion-coorected trace according to the CNMFe cell footprints
1. Take the folder full of the motion-corrected and trimmed .tif files and iterate through the files
2. For each file, look for footprints in another folder that came from CNMFe
3. Mask each footprint and extract the fluorescence from the motion-coorected time series
4. Compute df/f from that fluorescence trace and save those from all the foorprints in a csv (1 csv for each mot-corr .tif)
'''

folder_timeseries = Path(r"E:\CA1Dopa_Miniscope\test\Neuro_6_trim")
folder_footprints = Path(r"E:\CA1Dopa_Miniscope\test\Neuro_7_CNMFe_footprints")
output_folder     = Path(r"E:\CA1Dopa_Miniscope\test\Out")
output_folder.mkdir(parents=True, exist_ok=True)


def get_cell_id(path):
    match = re.search(r"_C(\d+)(?=\.tiff?$)", path.name, flags=re.IGNORECASE)

    if match is None:
        raise ValueError(f"No cell ID found in filename: {path.name}")

    number = int(match.group(1))
    width = max(3, len(match.group(1)))

    return f"C{number:0{width}d}"

def save_dff_csv(dff_df, output_path, frame_rate=10):

    cells = list(dff_df.columns)
    time = np.arange(len(dff_df), dtype=float) / frame_rate

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        # First row: blank first cell, then C000, C001, ...
        writer.writerow([""] + [f" {cell}" for cell in cells])

        # Second row: cell statuses
        writer.writerow(["Time(s)/Cell Status"] + [" undecided"] * len(cells))

        # Remaining rows: time and ΔF/F values
        for frame, t in enumerate(time):
            values = [f" {dff_df.iloc[frame][cell]}" for cell in cells]
            writer.writerow([f"{t:.6g}"] + values)

def get_trace_from_footprint(tif_path, footprint, footprint_threshold=0, chunk_size=1000):

    # get the footprint correctly
    footprint = tifffile.imread(footprint)
    footprint = np.asarray(footprint, dtype=np.float32)
    footprint = np.squeeze(footprint)

    # create a weighted mask that appreciates the footprint values
    mask = footprint >= footprint_threshold * footprint.max()
    weights = footprint.copy()
    weights[~mask] = 0
    weights /= weights.sum()

    if footprint.ndim != 2:
        raise ValueError(f"Footprint must be 2D, but has shape {footprint.shape}")
    if not np.isfinite(footprint).all():
        footprint = np.nan_to_num(footprint)
    if footprint.max() <= 0:
        raise ValueError("Footprint does not contain positive values.")
    if not mask.any():
            raise ValueError("No footprint pixels passed the threshold.")

    # creates the trace according to the time series
    with tifffile.TiffFile(tif_path) as tif:
        n_frames = len(tif.pages)
        first_frame = tif.pages[0].asarray()

        if first_frame.shape != footprint.shape:
            raise ValueError(
                f"Movie frame shape {first_frame.shape} does not match "
                f"footprint shape {footprint.shape}")

        trace_raw = np.empty(n_frames, dtype=np.float32)

        for start in range(0, n_frames, chunk_size):

            stop = min(start + chunk_size, n_frames)
            movie_chunk = np.stack([tif.pages[i].asarray() for i in range(start, stop)]).astype(np.float32)
            trace_raw[start:stop] = np.tensordot(movie_chunk,weights, axes=([1, 2], [0, 1]))

    # calculates df/f from the weighted trace_raw
    trace_raw = trace_raw.astype(np.float32)
    baseline_perc = percentile_filter(trace_raw, percentile=30, size=300, mode="nearest")
    trace_dff = (trace_raw - baseline_perc) / baseline_perc

    return trace_raw, trace_dff, weights


movie_files = sorted(path for path in folder_timeseries.iterdir() if path.is_file() and '.tif' in path.suffix.lower())

for movie_path in movie_files:

    movie_stem = movie_path.stem
    footprint_prefix = movie_stem
    footprint_files = [path for path in folder_footprints.iterdir() if path.is_file() and '.tif' in path.suffix.lower() and footprint_prefix.lower() in path.stem.lower()]
    footprint_files = sorted(footprint_files, key=lambda path: get_cell_id(path)[1])
    print(f"\nMovie: {movie_path.name}\nFootp: {footprint_files[0].stem}, total {len(footprint_files)}")

    traces_raw = {}
    traces_dff = {}
    for footprint_path in tqdm(footprint_files):

        # for each footprint, collect the trace
        cell_id = get_cell_id(footprint_path)
        trace_raw, trace_dff, weights = get_trace_from_footprint(movie_path, footprint_path)

        traces_raw[cell_id] = trace_raw
        traces_dff[cell_id] = trace_dff

    df_raw = pd.DataFrame(traces_raw)
    df_dff = pd.DataFrame(traces_dff)

    save_dff_csv(dff_df=df_raw, output_path=output_folder/f"{movie_stem}_TracesRaw.csv")
    save_dff_csv(dff_df=df_dff, output_path=output_folder/f"{movie_stem}_TracesDff.csv")

