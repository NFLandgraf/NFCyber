#%%
import numpy as np
import tifffile as tiff
from pathlib import Path

def read_tif_flexible(path):
    """
    Robust TIFF reader. Returns a NumPy array in one of these shapes:
      - (frames, H, W) for grayscale stack
      - (frames, H, W, 3) for RGB(A) frames (alpha dropped)
      - (H, W, 3) for single RGB frame
      - (H, W) for single grayscale frame

    Also prints diagnostic info about the TIFF pages.
    """
    path = Path(path)
    with tiff.TiffFile(str(path)) as tif:
        pages = tif.pages
        n_pages = len(pages)
        print(f"Reading TIFF: {path.name}, pages:", n_pages)
        page_shapes = []
        page_dtypes = []
        arrays = []
        for idx, page in enumerate(pages):
            try:
                arr = page.asarray()
            except Exception as e:
                # fallback: try page.asarray(key=0) or page.asarray(maxworkers=1)
                try:
                    arr = page.asarray()
                except Exception as e2:
                    print(f"  page {idx}: failed to asarray(): {e2!r}; skipping")
                    arr = None
            if arr is None:
                continue
            page_shapes.append(arr.shape)
            page_dtypes.append(arr.dtype)
            arrays.append(arr)
            print(f"  page {idx}: shape={arr.shape}, dtype={arr.dtype}")

        if len(arrays) == 0:
            # Some TIFFs store the image in series instead of pages or have unusual structure.
            # Try tif.asarray() (reads whole file)
            try:
                arr = tif.asarray()
                print("  fallback tif.asarray() => shape", arr.shape, arr.dtype)
                arrays = [arr]
            except Exception as e:
                raise RuntimeError(f"Could not read any image data from {path}: {e!r}")

    # If single entry, return it (maybe HxW, HxWx3 or HxWx4)
    if len(arrays) == 1:
        arr = arrays[0]
        # If arr is (frames, H, W, C) or (frames, H, W) let caller handle
        return np.asarray(arr)

    # If multiple pages, try to stack them into frames:
    # First inspect shapes: if all pages are 2D and same HxW -> stack into (N, H, W)
    all_2d = all((a.ndim == 2) for a in arrays)
    if all_2d:
        stacked = np.stack(arrays, axis=0)
        return stacked

    # If all pages are HxWx3 (RGB) and same shape -> stack into (N, H, W, 3)
    all_rgb = all((a.ndim == 3 and a.shape[2] in (3,4)) for a in arrays)
    if all_rgb:
        # drop alpha if present
        norm = [ (a[..., :3] if a.shape[2] >= 3 else np.stack([a]*3, -1)) for a in arrays ]
        stacked = np.stack(norm, axis=0)
        return stacked

    # If shapes differ, try to coerce by converting each page to grayscale HxW (luminance) then stack
    # This is a fallback: convert each page to 2D brightness image
    coerced = []
    for a in arrays:
        if a.ndim == 2:
            coerced.append(a)
        elif a.ndim == 3:
            if a.shape[2] >= 3:
                gray = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(a.dtype)
                coerced.append(gray)
            elif a.shape[2] == 1:
                coerced.append(a[...,0])
            else:
                # unexpected channel number: average across last axis
                gray = a.mean(axis=-1)
                coerced.append(gray)
        else:
            raise RuntimeError("Unhandled page ndim: " + str(a.ndim))
    # try stack coerced
    try:
        stacked = np.stack(coerced, axis=0)
        return stacked
    except Exception as e:
        raise RuntimeError("Failed to stack TIFF pages after coercion: " + repr(e))


def ensure_rgb(img):
    """img: HxW or HxWx1 or HxWx3 -> return HxWx3 uint8"""
    if img.ndim == 2:
        img3 = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img3 = np.concatenate([img]*3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        img3 = img
    else:
        raise ValueError("Unsupported image shape for RGB conversion: %s" % (img.shape,))
    # convert to uint8 if needed
    if img3.dtype != np.uint8:
        img3 = img3.astype(np.float32)
        lo, hi = img3.min(), img3.max()
        if hi == lo:
            hi = lo + 1.0
        img3 = ((img3 - lo) / (hi - lo) * 255.0).clip(0,255).astype(np.uint8)
    return img3

def normalize_to_float01(x):
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    if hi == lo:
        if hi == 0:
            return np.zeros_like(x, dtype=np.float32)
        else:
            return np.ones_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)

def composite_frame(bg_rgb, overlay_gray, color_rgb, gamma=1.0, min_alpha=0.0):
    norm = normalize_to_float01(overlay_gray)
    alpha = np.clip(norm ** gamma, 0.0, 1.0)
    if min_alpha > 0:
        alpha = np.clip(alpha, min_alpha, 1.0)
    color_arr = np.array(color_rgb, dtype=np.float32)[None, None, :]
    fg = (color_arr * 1.0).astype(np.float32)
    bg = bg_rgb.astype(np.float32)
    alpha_3 = alpha[..., None]
    out = alpha_3 * fg + (1.0 - alpha_3) * bg
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process(background_path, overlay_path, out_path, color=(51,204,204), gamma=1.0, min_alpha=0.0):
    bg_stack = read_tif_flexible(background_path)
    ov_stack = read_tif_flexible(overlay_path)

    # Normalize to stacks: make sure first axis is frames if possible
    # If single frame grayscale: shape (H,W) -> (1,H,W)
    if bg_stack.ndim == 2:
        bg_stack = bg_stack[None, ...]
    if ov_stack.ndim == 2:
        ov_stack = ov_stack[None, ...]
    # If RGB single frame (H,W,3) -> (1,H,W,3)
    if bg_stack.ndim == 3 and bg_stack.shape[2] == 3:
        bg_stack = bg_stack[None, ...]
    if ov_stack.ndim == 3 and ov_stack.shape[2] == 3:
        ov_stack = ov_stack[None, ...]

    # If overlay stack has different number of frames than background, accept min length and warn
    n_bg = bg_stack.shape[0]
    n_ov = ov_stack.shape[0]
    n_frames = min(n_bg, n_ov)
    if n_bg != n_ov:
        print(f"Warning: background frames={n_bg}, overlay frames={n_ov}. Using first {n_frames} frames.")

    out_frames = []
    for i in range(n_frames):
        bg = bg_stack[i]
        ov = ov_stack[i]
        # bg to RGB
        if bg.ndim == 3 and bg.shape[2] == 3:
            bg_rgb = bg[:, :, :3]
        else:
            bg_rgb = ensure_rgb(bg)

        # overlay to grayscale
        if ov.ndim == 3 and ov.shape[2] == 3:
            ov_gray = (0.299 * ov[...,0] + 0.587 * ov[...,1] + 0.114 * ov[...,2]).astype(np.float32)
        elif ov.ndim == 3 and ov.shape[2] == 4:
            ov_gray = (0.299 * ov[...,0] + 0.587 * ov[...,1] + 0.114 * ov[...,2]).astype(np.float32)
        else:
            ov_gray = ov.astype(np.float32)

        out = composite_frame(bg_rgb, ov_gray, color_rgb=color, gamma=gamma, min_alpha=min_alpha)
        out_frames.append(out)

    out_arr = np.stack(out_frames, axis=0)
    # save: if single frame, write single; else write stack
    if out_arr.shape[0] == 1:
        tiff.imwrite(str(out_path), out_arr[0], photometric='rgb')
    else:
        tiff.imwrite(str(out_path), out_arr, photometric='rgb')
    print("Wrote:", out_path)


path_background = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\66_Post2_YM_frame.tif")
path_overlay = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\66_Post2_YM_acc.tif")
path_out = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\66_Post2_YM_overlay.tif")

process(path_background, path_overlay, path_out)


#%%

import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image
from matplotlib import cm


def read_tiff_background(path):
    """Read background TIFF (single frame or stack)."""
    arr = tiff.imread(path)
    arr = np.asarray(arr)

    # normalize to (frames, H, W) or (frames, H, W, 3)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # single RGB frame
        arr = arr[None, ..., :3]

    return arr

def read_png_overlay(path):
    """Read PNG overlay as single grayscale image (H, W)."""
    img = Image.open(path)

    # convert to grayscale explicitly
    img = img.convert("L")

    return np.asarray(img, dtype=np.float32)

def ensure_rgb(img):
    """HxW or HxWx3 → HxWx3 uint8"""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported background shape {img.shape}")

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi == lo:
            hi = lo + 1
        img = ((img - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)

    return img

def normalize_to_float01(x, vmin=None, vmax=None):
    x = x.astype(np.float32)

    if vmin is None:
        vmin = np.nanmin(x)
    if vmax is None:
        vmax = np.nanmax(x)

    if vmax <= vmin:
        return np.zeros_like(x)

    x = (x - vmin) / (vmax - vmin)
    return np.clip(x, 0, 1)


def composite_frame(bg_rgb, overlay_gray,
                    gamma=1.0, min_alpha=0.0,
                    vmin=None, vmax=None):

    # normalize using vmin / vmax
    norm = normalize_to_float01(overlay_gray, vmin=vmin, vmax=vmax)

    # alpha from brightness
    alpha = np.clip(norm ** gamma, 0, 1)
    if min_alpha > 0:
        alpha = np.clip(alpha, min_alpha, 1)

    # viridis color
    viridis = cm.get_cmap("viridis")
    overlay_rgb = viridis(norm)[..., :3] * 255
    overlay_rgb = overlay_rgb.astype(np.float32)

    bg = bg_rgb.astype(np.float32)

    out = alpha[..., None] * overlay_rgb + (1 - alpha[..., None]) * bg
    return out.clip(0, 255).astype(np.uint8)


def process(background_path, overlay_path, out_path, gamma=1.0, min_alpha=0.0, vmin=50, vmax=110):

    bg_stack = read_tiff_background(background_path)
    ov_gray = read_png_overlay(overlay_path)

    n_frames = bg_stack.shape[0]
    out_frames = []

    for i in range(n_frames):
        bg = bg_stack[i]
        bg_rgb = ensure_rgb(bg)

        out = composite_frame(bg_rgb, ov_gray, gamma=gamma, min_alpha=min_alpha, vmin=vmin, vmax=vmax)
        out_frames.append(out)

    out_arr = np.stack(out_frames, axis=0)

    # save
    if out_arr.shape[0] == 1:
        tiff.imwrite(out_path, out_arr[0], photometric="rgb")
    else:
        tiff.imwrite(out_path, out_arr, photometric="rgb")

    print("Wrote:", out_path)


path_background = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\56_Post2_YM_frame.tif")
path_overlay = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\56_Post2_YM_acc.png")
path_out = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\56_Post2_YM_overlay.tif")

process(path_background, path_overlay, path_out)



#%%

import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


img = tiff.imread(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\66_Post2_YM_maxproj.tiff")  # shape (H, W), any dtype
path_out = r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\66_Post2_YM_maxproj_viridis.tiff"
vmin = 0.04
vmax = 0.16

plt.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
plt.colorbar(label="Intensity")
plt.axis("off")
plt.show()

img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)

# apply viridis colormap → RGBA
viridis = cm.get_cmap("viridis")
img_rgba = viridis(img_norm)
img_rgb = (img_rgba[..., :3] * 255).astype(np.uint8)

tiff.imwrite(path_out, img_rgb, photometric="rgb")



#%%

import numpy as np
import tifffile as tiff
from pathlib import Path

# settings
folder = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\56_Post2_YM")
threshold = 0.05   # pixels <= threshold are considered black
out_path = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\check\56_Post2_YM_acc.tiff")

# get tif files
tifs = sorted(folder.glob("*.tif"))
assert len(tifs) > 0, "No TIFFs found"

# initialize summary image
summary = None

for tif_path in tifs:
    img = tiff.imread(tif_path)

    if summary is None:
        summary = np.zeros_like(img)

    # mask of non-black pixels
    mask = img > threshold

    # copy only non-black pixels (keep brightest if overlapping)
    summary[mask] = np.maximum(summary[mask], img[mask])

# save
tiff.imwrite(out_path, summary)
print("Saved:", out_path)
