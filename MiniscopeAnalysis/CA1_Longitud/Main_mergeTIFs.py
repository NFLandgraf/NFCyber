#%%
# Merge data
import tifffile as tiff
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

path = "D:\\zest\\51_Zost2_OF"

master1 = path + '_master_trim.csv'
master2 = path + '2_master_trim.csv'
master_out = path + '_master_trim_combined.csv'

tif1 = path + '_motcor_trim.tif'
tif2 = path + '2_motcor_trim.tif'
tif_out = path + '_motcor_trim_combined.tif'

video1 = path + '_crop_mask_DLC_trim.mp4'
video2 = path + '2_crop_mask_DLC_trim.mp4'
video_out = path + '_crop_mask_DLC_trim_combined.mp4'


def combine_Master(master1, master2, master_out):
    df1 = pd.read_csv(master1)
    df2 = pd.read_csv(master2)

    for c in ["Mastertime [s]", "Neuro_frames_10Hz", "Cam_frames"]: 
        if c not in df1.columns or c not in df2.columns: 
            raise KeyError(f"Missing required column: {c}")

    # Compute offsets so df2 continues after df1
    # Add a step so you don't duplicate the last value.
    dt_master = df1["Mastertime [s]"].diff().median()
    d_neuro   = df1["Neuro_frames_10Hz"].diff().median()
    d_cam     = df1["Cam_frames"].diff().median()

    master_offset = df1["Mastertime [s]"].iloc[-1] + dt_master - df2["Mastertime [s]"].iloc[0]
    neuro_offset  = df1["Neuro_frames_10Hz"].iloc[-1] + d_neuro   - df2["Neuro_frames_10Hz"].iloc[0]
    cam_offset    = df1["Cam_frames"].iloc[-1]       + d_cam     - df2["Cam_frames"].iloc[0]

    df2["Mastertime [s]"]   = df2["Mastertime [s]"] + master_offset
    df2["Neuro_frames_10Hz"] = df2["Neuro_frames_10Hz"] + neuro_offset
    df2["Cam_frames"]        = df2["Cam_frames"] + cam_offset

    # Concatenate
    out = pd.concat([df1, df2], ignore_index=False)
    out = out.set_index('Mastertime [s]')
    print(f'Master: {len(out)}')
    out.to_csv(master_out)

def combine_TIFs(tif1, tif2, tif_out):
    with tiff.TiffFile(tif1) as t1, tiff.TiffFile(tif2) as t2:
        arr1 = t1.asarray()
        arr2 = t2.asarray()
    combined = np.concatenate([arr1, arr2], axis=0)
    tiff.imwrite(tif_out, combined)

    print(f'TIFFFF: {len(combined)}')

def combine_mp4(video1, video2, video_out):


    def pad_frame(frame, target_w, target_h):
        h, w = frame.shape[:2]

        pad_left   = (target_w - w) // 2
        pad_right  = target_w - w - pad_left
        pad_top    = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top

        return cv2.copyMakeBorder(
            frame,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)   # black padding
        )

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) 
    frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap1.isOpened():
        raise FileNotFoundError(video1)
    if not cap2.isOpened():
        raise FileNotFoundError(video2)

    fps = cap1.get(cv2.CAP_PROP_FPS)

    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 🎯 target size = max of both
    target_w = max(w1, w2)
    target_h = max(h1, h2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (target_w, target_h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open writer: {video_out}")

    written = 0

    with tqdm(desc="Combining mp4 with frame size padding") as pbar:
        for cap in (cap1, cap2):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = pad_frame(frame, target_w, target_h)
                out.write(frame)

                written += 1
                pbar.update(1)

    cap1.release()
    cap2.release()
    out.release()

    cap_out = cv2.VideoCapture(video_out) 
    frames_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print(f'Video1 ({frames1}), Video2 ({frames2}), VideoOut ({frames_out})')
    print(f'✅ Supposed to be {frames1+frames2}')


combine_Master(master1, master2, master_out)
combine_TIFs(tif1, tif2, tif_out)
combine_mp4(video1, video2, video_out)
