#%%
import tifffile as tiff
import cv2
import numpy as np
import os

tiff_path = r"C:\Users\landgrafn\Desktop\CA1Dopa_2pTube_2025-08-05_975_Airpuff1_alt_Ch0_prepro_MotCorr.tiff"
output_path = r"C:\Users\landgrafn\Desktop\CA1Dopa_2pTube_2025-08-05_975_Airpuff1_alt_Ch0_prepro_MotCorr.mp4"
fps = 200

data = tiff.imread(tiff_path)

if data.ndim == 2:
    data = data[np.newaxis, :, :]

if data.ndim != 3:
    raise ValueError(f"Expected TIFF with shape (frames, height, width), got {data.shape}")

frames, h, w = data.shape

if data.dtype != np.uint8:
    data = data.astype(np.float32)
    dmin = data.min()
    dmax = data.max()
    if dmax == dmin:
        data = np.zeros_like(data, dtype=np.uint8)
    else:
        data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)

if not writer.isOpened():
    raise RuntimeError("OpenCV could not open the video writer. Try another codec or file path.")

for i in range(frames):
    frame = data[i]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    writer.write(frame_bgr)

writer.release()

if not os.path.exists(output_path) or os.path.getsize(output_path) < 5000:
    raise RuntimeError("Output file was created but looks too small. Codec may have failed.")
