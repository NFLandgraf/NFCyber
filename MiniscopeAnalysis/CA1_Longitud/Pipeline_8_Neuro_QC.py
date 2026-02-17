#%%
'''
This is the Quality Control
Uses the footprints for QC_spatial and the traces for QC_temporal
According to parameters, let the cells pass or not
needs PC with CaImAn installed
Outputs a csv with the per cell parameters and pass/nopass
'''
import os
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
from skimage.measure import label, regionprops
import re
import tifffile
import scipy.sparse as sp
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.source_extraction.cnmf.deconvolution import GetSn


# directories
folder_motcor       = Path(r"D:\Nico\CaImAn\test\motcor")
folder_traces       = Path(r"D:\Nico\CaImAn\test\traces")
folder_footprints   = Path(r"D:\Nico\CaImAn\test\footprints")
folder_out          =      r"D:\Nico\CaImAn\test\out"

# Matching patterns, given movie base name BASE:
pattern_traces       = "{base}*.csv"
pattern_footprints  = "{base}*.tif*"

framerate = 10.0
round_to = 4
replace_what = 'motcor_trim_footprints_'
replace_with = ''


# QC_spat functions
def QC_spat_load_tif(tif_path):
    
    # loads a footprint tif and returns a 2D float image normalized to [0,1] via (H,W)
    tif_path = Path(tif_path)
    img = np.asarray(io.imread(str(tif_path)))

    # normalizing the pixel values to [0,1]
    img = img.astype(np.float32)
    vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
    img = (img - vmin) / (vmax - vmin)

    return img

def QC_spat_segment_footprint(img, thr=0.25):
    
    # Converts a footprint weight map into a binary mask acc to threshold
    mask = img > float(thr)
    
    return mask

def QC_spat_largest_component(mask):
    
    # Returns (largest_component, n_components) for the binary mask (if empty, returns (None, 0))
    lab = label(mask)
    p = regionprops(lab)
    if len(p) == 0:
        return None, 0
    p_sorted = sorted(p, key=lambda p: p.area, reverse=True)

    return p_sorted[0], int(len(p))

def QC_spat_compute_shapemetrics(mask):
    
    # computes footprint shape metrics for the largest connected component.
    
    comp, n_comp = QC_spat_largest_component(mask)

    if comp is None:
        print('🔴 Warning: no components found')
        return {
            "area_px": np.nan,
            "perimeter_px": np.nan,
            "circularity": np.nan,
            "solidity": np.nan,
            "eccentricity": np.nan,
            "major_axis_length": np.nan,
            "minor_axis_length": np.nan,
            "axis_ratio": np.nan,
            "n_comps": 0,
        }

    # area and perimeter
    area = float(comp.area)
    perim = float(comp.perimeter) if comp.perimeter is not None else np.nan

    # circularity = 4πA / P²
    if np.isfinite(perim) and perim > 0:
        circularity = 4.0 * np.pi * area / (perim ** 2)
        circularity = float(np.clip(circularity, 0.0, 1.5))
    else:
        circularity = np.nan

    # ellipse-based shape metrics (already computed by regionprops)
    major = float(comp.axis_major_length) if comp.axis_major_length is not None else np.nan
    minor = float(comp.axis_minor_length) if comp.axis_minor_length is not None else np.nan
    axisratio = float(major/minor) if (np.isfinite(major) and np.isfinite(minor) and minor > 0) else np.nan

    return {
        "area_px": area,
        "perim_px": perim,
        "circul": circularity,
        "solid": float(comp.solidity) if comp.solidity is not None else np.nan,
        "eccentr": float(comp.eccentricity) if comp.eccentricity is not None else np.nan,
        "maj_axis_len": major,
        "min_axis_len": minor,
        "axis_ratio": axisratio,
        "n_comps": n_comp,
    }

def QC_spat_main(footprint_files, base):
    '''
    This gets all footprint files from a CNMFe output, it loops over each footprint and masks (what is considered a neuron)
    Then, according to this mask, it calculates shape metrics and saves everything in a df (all cells from all recordings as rows, metrics as columns)
    Use it to accept/reject cells according to their footprint
    '''

    print(f'{base} QC_spat start')
    files_n = len(footprint_files)
    print(f'found {files_n} footprints')

    rows = []
    maxproj = None

    # loop over every file and comp spatial things
    # one row per cell (index = filename stem) and columns containing metrics
    for tif_path in footprint_files:

        img = QC_spat_load_tif(tif_path)
        mask = QC_spat_segment_footprint(img)
        metrics = QC_spat_compute_shapemetrics(mask)

        row = {"cell_ID": tif_path.stem, **metrics}
        rows.append(row)

        # for creating a maxproj image from all footprints
        maxproj = mask.copy() if maxproj is None else (maxproj | mask)
    maxproj_u8 = (maxproj.astype(np.uint8) * 255)
    io.imsave(f"{folder_out}\\{base}_maxproj.jpg", maxproj_u8)

    # clean QC_spatial df
    df_QC_spat = pd.DataFrame(rows).set_index("cell_ID")
    col_order = ["file", "area_px", "circul", "solid", "eccentr", "axis_ratio", "n_comps", "maj_axis_len", "min_axis_len", "perim_px"]
    df_QC_spat = df_QC_spat[[c for c in col_order if c in df_QC_spat.columns] + [c for c in df_QC_spat.columns if c not in col_order]]
    numeric_cols = df_QC_spat.select_dtypes(include=["number"]).columns
    df_QC_spat[numeric_cols] = df_QC_spat[numeric_cols].round(round_to)
    df_QC_spat.index = df_QC_spat.index.str.replace(replace_what, replace_with, regex=False)

    print(f'{base} QC_spat done')
    return df_QC_spat

# QC_temp functions
def QC_temp_get_motcorr_movies(folder_motcor, folder_out):

    QC_temp_ensure_dir(folder_out)
    mmap_dir = QC_temp_ensure_dir(folder_out / "mmap_cache")

    # get the motcor movies
    #print('get motcor movies')
    motcor_movies = []
    motcor_movies.extend(sorted(folder_motcor.glob("*.tif")))
    motcor_movies = sorted(set(motcor_movies))
    print(f'{len(motcor_movies)} tifs found\n\n')

    return motcor_movies, mmap_dir

def QC_temp_ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p

def QC_temp_pick_single_match(folder, pattern):

    hits = sorted(folder.glob(pattern))
    if len(hits) == 0:
        print(f"No match for pattern '{pattern}' in {folder}")
        return None
    if len(hits) > 1:
        print('XXX Multiple hits')
        return hits[0]
    return hits[0]

def QC_temp_load_movie_as_mmap(movie_path, mmap_dir):
    
    mmap_dir.mkdir(parents=True, exist_ok=True)
    base = movie_path.stem

    # Look for any existing mmap for this movie
    existing = sorted(mmap_dir.glob(f"{base}*_frames_*.mmap"))

    if existing:
        mmap_path = existing[0]
    else:
        mmap_path = cm.save_memmap(
            [str(movie_path)],
            base_name=str(mmap_dir / base),
            order="F",
            border_to_0=0
        )

    # Yr.shape = (H*W, T), each column is one video frame, flattened into a vector of pixels
    Yr, dims, timepoints_motcor = cm.load_memmap(str(mmap_path))
    return str(mmap_path), Yr, dims, timepoints_motcor

def QC_temp_load_inscopix_traces_csv(traces_csv):
    """
    Inscopix traces CSV format:
      - header row: '', C000, C001, ...
      - second row: 'Time(s)/Cell Status', 'undecided', ...
      - remaining rows: time(s), trace values...
    Returns:
      C: (K, T) float32
      t: (T,) float32 time vector (seconds)
      cell_names: list like ['C000', 'C001', ...]
    """
    df = pd.read_csv(traces_csv, skiprows=[1])  # skip the "Cell Status" row

    # First column is time; sometimes it's unnamed ('' or 'Unnamed: 0')
    time_orig = df.iloc[:, 0].to_numpy(dtype=np.float32)
    cell_names = list(df.columns[1:])
    cell_names = [name.replace(' C', 'C') for name in cell_names]

    traces = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # shape [timepoints T, neurons K]
    traces = traces.T  # transpose -> shape [neurons K, timepoints T]
    return traces, time_orig, cell_names

def QC_temp_load_footprints(footprints, dims_hw):
    
    
    footprint_names = []

    # Builds A (pixels, K) from a list of footprint tifs. Uses TIFF pixel values as weights.
    H, W = dims_hw
    cols = []
    for fp in footprints:

        # gets the footprint name
        m = re.search(r"(C\d+)", fp.stem)
        footprint_names.append(m.group(1))

        img = tifffile.imread(fp)
        img = np.squeeze(img)
        if img.shape != (H, W):
            raise ValueError(f"Footprint {fp.name} shape {img.shape} != {(H, W)}")
        cols.append(img.astype(np.float32).reshape(-1, order="F"))
    footprint_arrays = sp.csc_matrix(np.stack(cols, axis=1))

    return footprint_arrays, footprint_names

def QC_temp_compute_noise(C):
    
    # calculates noise estimate via GetSn
    K, T = C.shape
    noise = np.full(K, np.nan, dtype=np.float32)

    for k in range(K):
        y = C[k].astype(np.float32)
        try:
            noise[k] = float(GetSn(y))
        except Exception:
            pass

    return noise

def QC_temp_main(movie_path, footprint_files, mmap_dir, base):
    
    print(f'{base} QC_temp start')

    # Find matching traces CSV and load them
    traces_csv = QC_temp_pick_single_match(folder_traces, pattern_traces.format(base=base))
    if traces_csv == None:
        return None
    traces, time_orig, traces_cell_names = QC_temp_load_inscopix_traces_csv(traces_csv)
    K, T_traces = traces.shape  # number of neurons (K), number of timepoints (T)
    time_10Hz = np.arange(T_traces, dtype=np.float32) / 10.0
    print(f'use {traces_csv.stem}')
    
    # load mmap for the tif movie
    mmap_path, Yr, dims, T_movie = QC_temp_load_movie_as_mmap(movie_path, mmap_dir)
    if T_movie != T_traces:
        print(f"⚠ {base}: movie frames={T_movie}, trace timepoints={T_traces}")
    motcor_height, motcor_width = int(dims[0]), int(dims[1])
    print(f'found {len(footprint_files)} footprint files')
    
    # Build spatial footprint arrays
    footprint_arrays, footprints_cell_names = QC_temp_load_footprints(footprint_files, (motcor_height, motcor_width))
    if footprint_arrays.shape[1] != K:
        raise ValueError(f"{base}: #footprints={footprint_arrays.shape[1]} but #traces={K}")

    # sanity checks
    if len(traces_cell_names) != len(footprints_cell_names):
        raise ValueError(f"QC_temp: K mismatch: traces={len(traces_cell_names)} footprints={len(footprints_cell_names)}")
    if traces_cell_names != footprints_cell_names:
        raise ValueError(f"QC_temp: NOT SAME CELL NAMES")
    assert Yr.shape[1] == T_movie
    assert traces.shape[1] == T_traces
    assert footprint_arrays.shape[0] == Yr.shape[0]
    assert footprint_arrays.shape[1] == K

    # CaImAn QC
    fit_raw, fit_delta, erfc_raw, erfc_delta, r_values, sig_samples = evaluate_components(Yr, traces, footprint_arrays, traces, None, None, framerate)
    noise = QC_temp_compute_noise(traces)

    # create summary df
    cell_ids = [fp.stem for fp in footprint_files]  # footprint filenames as IDs
    cell_ids = [s.replace(replace_what, replace_with) for s in cell_ids]
    df_QC_temp = pd.DataFrame({
        "fit_raw": np.asarray(fit_raw),
        "fit_delta": np.asarray(fit_delta),
        "r_value": np.asarray(r_values),
        "noise_est": noise},
        index=cell_ids)
    df_QC_temp.index.name = "cell_ID"
    df_QC_temp = df_QC_temp.round(round_to)

    print(f'{base} QC_temp done')
    return df_QC_temp

# pass function
def QC_pass(df):
    # parameters that are accepting a cell when inside boundaries
    qc_params = {
        "area_px":    (50, None),
        "circul":     (0.4, None),
        "axis_ratio": (None, 5),
        "perim_px":   (20, None),
        "fit_delta":  (None, -6),
        "fit_raw":    (None, -40),
        "n_comps":    (1, 1)}

    pass_mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in qc_params.items():
        if lo is not None and hi is not None:
            pass_mask &= df[col].between(lo, hi)
        elif lo is not None:
            pass_mask &= df[col] >= lo
        elif hi is not None:
            pass_mask &= df[col] <= hi

    df["pass"] = pass_mask

    return df


# store all dfs in the list to collapse them all into one big df in the end
df_all_QC_spat = []
df_all_QC_temp = []
motcor_paths, mmap_dir = QC_temp_get_motcorr_movies(folder_motcor, folder_out)

for motcor in motcor_paths:

    base = motcor.stem
    print(f'\n\n-------{base}-------')

    # Find matching footprints
    footprint_files = sorted(folder_footprints.glob(pattern_footprints.format(base=base)))
    if len(footprint_files) == 0:
        print(f"XXX No footprint tifs for base '{base}' in {folder_footprints}")
        continue
    
    # do QC_spat
    df_QC_spat = QC_spat_main(footprint_files, base)
    if df_QC_spat is None or df_QC_spat.empty:
        continue
    df_all_QC_spat.append(df_QC_spat)

    # do _QC_temp
    df_QC_temp = QC_temp_main(motcor, footprint_files, mmap_dir, base)
    if df_QC_temp is None or df_QC_temp.empty:
        continue
    df_all_QC_temp.append(df_QC_temp)


df_QC_spat_final    = pd.concat(df_all_QC_spat, axis=0)
df_QC_temp_final    = pd.concat(df_all_QC_temp, axis=0)
df_QC_final         = pd.concat([df_QC_spat_final, df_QC_temp_final], axis=1)

df_QC_final_pass = QC_pass(df_QC_final)
df_QC_final_pass.to_csv(folder_out + '\\QC_FINAL_pass.csv')

#df_QC_spat_final.to_csv(f"{folder_out}\\df_QC_spat_final.csv")
#df_QC_temp_final.to_csv(f"{folder_out}\\df_QC_temp_final.csv")
#df_QC_final.to_csv(f'{folder_out}\\df_QC_final.csv')


#%%
'''
check the QC parameters
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from random import shuffle

df = pd.read_csv(r"D:\df_QC_FINAL.csv", index_col='cell_ID')

x = 'area_px'
y = 'fit_raw'
low = 250
high = 1000


plt.figure()
plt.scatter(df[x], df[y], alpha=0.05)
plt.xlabel(x)
plt.ylabel(y)
#plt.xlim(low,high)
#plt.ylim(-400,0)
plt.show()

idx = list(df.index[df[x].between(low, high)])
shuffle(idx)
folder_in = r"D:\10_ALL_MERGE"
if True:
    for i, cell in enumerate(idx):
        if i > 10:
            break
        # split name
        parts = cell.split("_")
        
        # file base name (everything except the last part, which is Cxxx)
        file_base = "_".join(parts[:-1])          # e.g. 32_Zost2_YM
        cell_id = parts[-1]                        # e.g. C007

        csv_path = Path(f'{folder_in}//{file_base}_final.csv')
        
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            continue

        # column name to plot
        col_name1 = f"{file_base}_{cell_id}_dfovernoise"
        col_name2 = f"{file_base}_{cell_id}_spikeprob"
        df = pd.read_csv(csv_path)
        if col_name1 not in df.columns:
            print(f"Column not found: {col_name1}")
            continue

        plt.figure()
        plt.plot(df[col_name1].values)
        plt.plot(df[col_name2].values/5, alpha=0.8)
        plt.title(col_name1)
        plt.xlabel("frame")
        plt.ylabel("df / noise")
        plt.tight_layout()
        plt.show()

print(len(idx))
