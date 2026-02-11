#%%
import cv2
from pathlib import Path

def keep_every_x_frame(input_path, output_path, x_frame, start=0, limit=None):
    """
    Keep every x-th frame from input video and write output with the SAME fps.
    Result: shorter video duration (~1/x), same frame rate.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v for broad compatibility. You can also try 'avc1' if available.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    # Optional: seek to a starting frame
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    kept = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Keep every x-th frame relative to the read index
        if i % x_frame == 0:
            out.write(frame)
            kept += 1
            if limit is not None and kept >= limit:
                break

        i += 1

    out.release()
    cap.release()
    return kept

inp = r"W:\_proj_CA1Dopa\CA1Dopa_Longitud\Implants\CA1Dopa_Longitud_f48_2025-05-09-11-30-27_Implant.mp4"
out = r"W:\_proj_CA1Dopa\CA1Dopa_Longitud\Implants\CA1Dopa_Longitud_f48_2025-05-09-11-30-27_Implant_fps.mp4"
kept = keep_every_x_frame(inp, out, x_frame=10)
print(f"Done. Wrote {kept} frames to {out}")
