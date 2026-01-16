#%%
'''
QC_SPATIAL
We have TIFs of each cell from the CNMFe, this is the first part of the QC- the SPATIAL one
'''

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.segmentation import clear_border

parent_folder = Path("D:/Neuro_7_CNMFe")
path_out = "D:\\zest_out\\"

temp_rainbow_folder = "D:\\zemp_MaskRainbow\\"
temp_mask_folder = "D:\\zemp_Mask\\"

common_name = ''
suffix = '.tif'

qc_min_area_px              = 90
qc_max_area_px              = 350
qc_max_axis_ratio           = 2.5
qc_max_eccentricity         = 1
qc_min_circularity          = 0.6
qc_min_solidity             = 0
qc_max_components           = 1


def get_files(path, common_name, suffix, print_out):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name in p.name) and p.name.endswith(suffix)])
    if print_out:
        print(f"{len(files)} tifs found")
        for f in files:
            pass
            #print(f)
        #print('\n\n')
    return files

def load_tif(tif_path):
    
    # loads a footprint tif and returns a 2D float image normalized to [0,1] via (H,W)
    tif_path = Path(tif_path)
    img = np.asarray(io.imread(str(tif_path)))

    # normalizing the pixel values to [0,1]
    img = img.astype(np.float32)
    vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
    img = (img - vmin) / (vmax - vmin)

    return img

def segment_footprint(img, thr=0.15):
    
    # Converts a footprint weight map into a binary mask acc to threshold
    mask = img > float(thr)
    
    return mask

def largest_component_props(mask):
    
    # Returns (largest_component, n_components) for the binary mask (if empty, returns (None, 0))
    lab = label(mask)
    p = regionprops(lab)
    if len(p) == 0:
        return None, 0
    p_sorted = sorted(p, key=lambda p: p.area, reverse=True)

    return p_sorted[0], int(len(p))

def compute_shape_metrics(mask):
    
    # computes footprint shape metrics for the largest connected component.
    
    comp, n_comp = largest_component_props(mask)

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
            "n_components": 0,
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
    major = float(comp.major_axis_length) if comp.major_axis_length is not None else np.nan
    minor = float(comp.minor_axis_length) if comp.minor_axis_length is not None else np.nan
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

def qc_spat_footprint(area_px, circularity, solidity, eccentricity, axis_ratio, n_components):
    
    # applies all spatial QC rules to a single footprint, returns a dict with per-rule FAIL flags and overall PASS_spatial
    FAIL_too_small_area = (not np.isfinite(area_px)) or (area_px < qc_min_area_px)
    FAIL_too_large_area = (not np.isfinite(area_px)) or (area_px > qc_max_area_px)
    FAIL_too_elongated = ((not np.isfinite(axis_ratio)) or (axis_ratio > qc_max_axis_ratio) or (not np.isfinite(eccentricity)) or (eccentricity > qc_max_eccentricity))
    FAIL_fragmented = (n_components is None) or (int(n_components) > int(qc_max_components))
    FAIL_non_compact = ((not np.isfinite(circularity)) or (circularity < qc_min_circularity) or (not np.isfinite(solidity)) or (solidity < qc_min_solidity))
    FAIL_any = (FAIL_too_small_area or FAIL_too_large_area or FAIL_too_elongated or FAIL_fragmented or FAIL_non_compact)
    PASS_spatial = not FAIL_any

    return {
        "FAIL_too_small_area": bool(FAIL_too_small_area),
        "FAIL_too_large_area": bool(FAIL_too_large_area),
        "FAIL_too_elongated": bool(FAIL_too_elongated),
        "FAIL_fragmented": bool(FAIL_fragmented),
        "FAIL_non_compact": bool(FAIL_non_compact),
        "PASS_spatial": bool(PASS_spatial)}


# Creating Images
def get_color_palette(files_n):
    rng = np.random.default_rng()
    cmap = plt.get_cmap('hsv', files_n)
    palette = (cmap(np.arange(files_n))[:, :3] * 255).astype(np.uint8)  # (N,3)
    rng.shuffle(palette)

    return palette

def save_mask(mask, tif_path):

    # create temporary folder
    all_masks_path = temp_mask_folder
    if not os.path.exists(all_masks_path):
        os.makedirs(all_masks_path)

    # save the mask as a white mask
    mask = mask.astype(np.uint8) * 255
    out = temp_mask_folder + Path(tif_path).stem + '_mask.tif'
    io.imsave(out, mask, check_contrast=False)
    
def save_mask_rainbow(mask, tif_path, i, palette):

    # create temporary folder
    all_masksrainbow_path = temp_rainbow_folder
    if not os.path.exists(all_masksrainbow_path):
        os.makedirs(all_masksrainbow_path)

    color = palette[i]
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[mask] = color.astype(np.uint8)

    out = temp_rainbow_folder + Path(tif_path).stem + '_mask_rainbow.tif'
    io.imsave(str(out), rgb, check_contrast=False)

def max_img_folder(folder, name):

    files = get_files(folder, common_name, suffix, False)
    
    # create a pixelwise maximum image across all tifs in a folder.
    max_img = None
    for tif in files:
        img = load_tif(tif)

        # creates max_img
        if max_img is None:
            max_img = img.astype(np.float32)
        else:
            max_img = np.maximum(max_img, img)

    out_path = path_out + name + '_MaxImg.tif'
    io.imsave(out_path, max_img, check_contrast=False)
    print('saved MaxImg')

def max_img_folder_rainbow(folder, name):
    
    files = get_files(folder, common_name, suffix, False)

    # Load first to determine shape
    first = io.imread(str(files[0]))
    first = np.asarray(first)
    H, W, C = first.shape

    # We will store the best brightness so far and the winning RGB pixels
    best_brightness = np.full((H, W), -np.inf, dtype=np.float32)
    out_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for tif in files:
        img = io.imread(str(tif))
        img = np.asarray(img)

        b = img[..., :3].astype(np.float32).max(axis=2)

        # where this image is brighter than the best so far, update output color
        update = b > best_brightness
        best_brightness[update] = b[update]
        out_rgb[update] = img[..., :3][update].astype(np.uint8)

    out_path = path_out + name + '_MaxImgRainbow.tif'
    io.imsave(out_path, out_rgb, check_contrast=False)
    print('saved MaxImgRainbow')



def main(path_in):

    name = path_in.name
    print('\n' + name)
    files = get_files(path_in, common_name, suffix, True)
    files_n = len(files)

    # get colors for rainbow mask
    palette = get_color_palette(files_n)

    rows = []
    for i, tif_path in enumerate(files):
        print(tif_path)
        # loop over every file and comp spatial things
        # one row per cell (index = filename stem) and columns containing metrics + per-rule FAIL flags + overall PASS/FAIL.
        img = load_tif(tif_path)
        mask = segment_footprint(img)

        # save masks for max image
        save_mask(mask, tif_path)
        save_mask_rainbow(mask, tif_path, i, palette)

        m = compute_shape_metrics(mask)
        qc_flags = qc_spat_footprint(m["area_px"], m["circul"], m["solid"], m["eccentr"], m["axis_ratio"], m["n_comps"])

        row = {"cell_id": tif_path.stem, **m, **qc_flags}
        rows.append(row)

        
    # clean and safe QC_spatial df
    df = pd.DataFrame(rows).set_index("cell_id")
    col_order = [
        "file", "area_px", "circul", "solid", "eccentr", "axis_ratio", "n_comps", "maj_axis_len", "min_axis_len", "perim_px",
        "FAIL_too_small_area", "FAIL_too_large_area", "FAIL_too_elongated", "FAIL_fragmented", "FAIL_non_compact",
        "PASS_spatial"]
    df = df[[c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]]
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    df.to_csv(f'{path_out}\\{name}_QC_spatial.csv')
    print('saved QC_spatial')

    # get maximum images of the respective folders and delete them
    max_img_folder(temp_mask_folder, name)
    max_img_folder_rainbow(temp_rainbow_folder, name)
    shutil.rmtree(temp_rainbow_folder)
    shutil.rmtree(temp_mask_folder)


for subfolder in parent_folder.iterdir():
    if subfolder.is_dir():
        #main(subfolder)
        pass

main(parent_folder)


#%%

df = pd.read_csv("D:\\zest_out\\Neuro_7_CNMFe_QC_spatial.csv")
