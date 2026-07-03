#%%
import pandas as pd
import numpy as np

def recover_only_complete_missing_frames(input_csv, output_csv, score_value=0.1):
    df = pd.read_csv(input_csv)

    frame_col = "frame_idx"
    track_col = "track"

    coord_cols = [c for c in df.columns if c.endswith(".x") or c.endswith(".y")]
    score_cols = [c for c in df.columns if c.endswith(".score")] + ["instance.score"]
    score_cols = list(dict.fromkeys(score_cols))

    min_frame = int(df[frame_col].min())
    max_frame = int(df[frame_col].max())

    existing_frames = set(df[frame_col].dropna().astype(int))
    all_frames = set(range(min_frame, max_frame + 1))
    missing_frames = sorted(all_frames - existing_frames)

    new_rows = []

    for frame in missing_frames:
        prev_frames = [f for f in existing_frames if f < frame]
        next_frames = [f for f in existing_frames if f > frame]

        if not prev_frames or not next_frames:
            continue

        prev_frame = max(prev_frames)
        next_frame = min(next_frames)

        prev_rows = df[df[frame_col] == prev_frame]
        next_rows = df[df[frame_col] == next_frame]

        common_tracks = set(prev_rows[track_col]) & set(next_rows[track_col])
        if not common_tracks:
            continue

        track = sorted(common_tracks)[0]

        prev_row = prev_rows[prev_rows[track_col] == track].iloc[0].copy()
        next_row = next_rows[next_rows[track_col] == track].iloc[0]

        frac = (frame - prev_frame) / (next_frame - prev_frame)

        new_row = prev_row.copy()
        new_row[frame_col] = frame

        for col in coord_cols:
            new_row[col] = prev_row[col] + frac * (next_row[col] - prev_row[col])

        for col in score_cols:
            if col in new_row.index:
                new_row[col] = score_value

        new_rows.append(new_row)

    if new_rows:
        df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_out = df.copy()

    df_out = df_out[df.columns]
    df_out = df_out.sort_values([frame_col, track_col]).reset_index(drop=True)
    df_out.to_csv(output_csv, index=False)
    return df_out


# Example usage:
recover_only_complete_missing_frames(
    input_csv=r"C:\Users\landgrafn\Desktop\FF_RI\87\2026-05-05_FF-PFC_87_RI_edit_sLEAP - Copy.csv",
    output_csv=r"C:\Users\landgrafn\Desktop\FF_RI\87\2026-05-05_FF-PFC_87_RI_edit_sLEAP_recover.csv",
    score_value=0.1
)