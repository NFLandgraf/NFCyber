#%%
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt


img = tiff.imread(r"D:\CA1Dopa_2pTube_2025-08-05_1001_Airpuff1_alt_Ch0_prepro_MotCorr.tiff")   # shape (T, Y, X)



def max_proj(img, output_file):

    # create max projection: shape (T, Y, X)    ->  shape (Y, X)
    max_proj = np.max(img, axis=0)
    #tiff.imwrite(output_file, max_proj.astype(np.float32))

    plt.imshow(max_proj, cmap="gray")
    plt.title("TIFF Image")
    plt.axis("off")
    plt.show()

def df_f(img, output_file, baseline_range= None, baseline_mode='mean'):
    # baseline_range:   Optional custom frame range [start, end] for baseline, set to None to use all frames
    # baseline_mode:    mean or min over frames

    # creates frame range of img for calc baseline
    if baseline_range == None:
        img_range = img
    else:
        start, stop = baseline_range
        img_range = img[start:stop]

    # creates baseline image
    if baseline_mode == "mean":
        f0 = img_range.mean(axis=0)
    elif baseline_mode == 'min':
        f0 = img_range.min(axis=0)
    eps = 1e-6                                  # if any baseline px is 0, it would blow up the calc (numerical safeguard)
    den = np.maximum(f0, eps)[None, :, :]       # creates a (1, Y, X) shape of the F0 image (safe baseline image broadcast across time)
    
    # df/f calculation
    dff = (img - den) / den                     # subtracts the baseline image from every frame (T, Y, X)

    tiff.imwrite(output_file, dff.astype(np.float32))
    


df_f(img, 'ss.tiff')


