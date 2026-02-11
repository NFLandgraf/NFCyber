#%%
'''
After the IDPS pipeline (Preprocess, SpatFilter, MotCor), you can split recordings into segments (Week2_YM, Week2_OF)
This depends on a csv where you told them where each segment starts/stops
This is not needed, when you process each Neuro video individually without any timeseries
'''

import os
import re
import pandas as pd
import tifffile as tiff


CSV_PATH = "D:\\1stuff\\Ispx_TimeseriesSeparates.csv"
TIF_DIR  = "D:\\1stuff"
OUT_DIR  = "D:\\1stuff"
TIF_EXT  = ".tiff"
COMPRESSION = None  # e.g. "zlib" or None

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
if "filename" not in df.columns:
    raise ValueError("CSV must contain a 'filename' column")

boundary_cols = [c for c in df.columns if c != "filename"]

for _, row in df.iterrows():
    base = str(row["filename"]).strip()
    if base == "" or base.lower() == "nan":
        continue

    tif_path = os.path.join(TIF_DIR, base + TIF_EXT)
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"TIF not found: {tif_path}")

    # Build segments from A,A,B,B,... pairs
    segments = []
    for i in range(0, len(boundary_cols) - 1, 2):
        col_start = boundary_cols[i]
        col_end   = boundary_cols[i + 1]

        s = row[col_start]
        e = row[col_end]
        if pd.isna(s) or pd.isna(e):
            continue

        start = int(s)
        end   = int(e)

        m = re.match(r"([A-Za-z]+)", str(col_start))
        label = m.group(1) if m else f"seg{i//2+1}"

        segments.append((label, start, end))

    if not segments:
        raise ValueError(f"No segments found in CSV for '{base}'")

    # Sort by start frame (so checks are meaningful even if columns are weird)
    segments.sort(key=lambda x: x[1])

    with tiff.TiffFile(tif_path) as tf:
        n_frames = len(tf.pages)
        last_frame = n_frames - 1






        # 1) all indices valid
        for label, start, end in segments:
            if start < 0 or end < 0:
                raise ValueError(f"{base}: negative indices in {label}: ({start}, {end})")
            if start > end:
                raise ValueError(f"{base}: start > end in {label}: ({start} > {end})")
            if end >= n_frames:
                raise ValueError(
                    f"{base}: segment {label} ends at {end} but TIF last frame is {last_frame} "
                    f"(n_frames={n_frames})"
                )

        # 2) must start at 0 (covers "100% fit" from the beginning)
        if segments[0][1] != 0:
            raise ValueError(f"{base}: first segment starts at {segments[0][1]}, expected 0")

        # 3) no gaps / no overlaps: next.start must be prev.end + 1
        for (lab1, s1, e1), (lab2, s2, e2) in zip(segments[:-1], segments[1:]):
            if s2 != e1 + 1:
                raise ValueError(
                    f"{base}: segments do not tile perfectly: {lab1} ends at {e1}, "
                    f"but {lab2} starts at {s2} (expected {e1 + 1})"
                )

        # 4) last CSV end must match last frame of TIF exactly
        csv_last_end = segments[-1][2]
        if csv_last_end != last_frame:
            raise ValueError(
                f"{base}: CSV last segment ends at {csv_last_end}, but TIF last frame is {last_frame} "
                f"(n_frames={n_frames}). Frames do not 100% fit."
            )





        print(f"[INFO] {base}: n_frames={n_frames}, writing {len(segments)} segments")

        for label, start, end in segments:
            out_name = f"{base}_{label}_{start:06d}-{end:06d}.tif"
            out_path = os.path.join(OUT_DIR, out_name)

            data = tf.asarray(key=slice(start, end + 1))  # inclusive end
            tiff.imwrite(out_path, data, compression=COMPRESSION)

            print(f"  [OK] {out_name} ({data.shape[0]} frames)")
