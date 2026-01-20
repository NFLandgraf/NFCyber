#%%
import tifffile as tiff
from pathlib import Path



def get_files(path, common_name, suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name in p.name) and p.name.endswith(suffix)])
    print(f"\n{len(files)} files found")
    for f in files:
        print(f)
    print('\n\n')
    return files


def check_tif_dims(tif_path):
    with tiff.TiffFile(tif_path) as tif:
        n_frames = len(tif.pages)
        shape = tif.pages[0].shape  # (H, W) or (H, W, C)

    print(f"{tif_path}")
    print(f"  shape  : {shape}")




files = get_files("D:\\new\\out", 'trim', suffix=".tif")
for file in files:
    check_tif_dims(file)

