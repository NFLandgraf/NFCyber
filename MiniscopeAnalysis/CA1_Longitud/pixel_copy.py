#%%
import numpy as np
import tifffile as tiff

folder = r"C:\Users\landgrafn\Desktop\a\footprints"
tif_tochange = folder + "\\72_Zost1_YM_motcor_trim_footprints_C01.tif"
tif_borrowcell = folder + "\\72_Zost1_YM_motcor_trim_footprints_C00.tif"
out_path  = folder + "\\fused.tif"
threshold = 0.005

tif1 = tiff.imread(tif_tochange)
tif2 = tiff.imread(tif_borrowcell)

# --- sanity checks ---
if tif1.shape != tif2.shape:
    raise ValueError(f"Shape mismatch: {tif1.shape} vs {tif2.shape}")

# --- fuse (only change pixels where tif2 > threshold) ---
mask = tif2 > threshold

# Option A (in-place edit of tif1)
fused = tif1.copy()
fused[mask] = tif2[mask]

# Preserve dtype (fused dtype will match tif1 unless you changed it)
tiff.imwrite(out_path, fused)


#%%

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Load image (2D or 3D stack)
img = tiff.imread(r"C:\Users\landgrafn\Desktop\a\footprints\51_Week8_YM_motcor_trim_footprints_C11.tif")

# Flatten to 1D array (all pixels)
pixels = img.ravel()

# Plot histogram
plt.figure()
plt.hist(pixels, bins=100)
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.ylim(0,10)
plt.xlim(0,0.02)
plt.title("Histogram of pixel values")
plt.show()
