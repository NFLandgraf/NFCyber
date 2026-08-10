#%%
from pathlib import Path
from neo.io import AxonIO
import pandas as pd
import numpy as np


folder_in = r"C:\Users\landgrafn\Desktop\Canada_Analysis_ordered\Recordings\a"


def get_files(path, common_name1, suffix=".abf"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name1 in p.name) and p.name.endswith(suffix)])
    print(f"\n{len(files)} files found")
    for f in files:
        print(f)
    print('\n\n')
    return files

files = get_files(folder_in, '')

for file in files:

    reader = AxonIO(filename=str(file))
    block = reader.read_block()

    all_sweeps = []

    for sweep_number, segment in enumerate(block.segments):
        signal = segment.analogsignals[0]

        values = np.asarray(signal)
        time = np.asarray(signal.times.rescale("s").magnitude).ravel()

        channels = {"sweep": np.full(len(time), sweep_number),
            "time_s": time}
            

        for ch in range(values.shape[1]):
            channels[f"channel_{ch}"] = values[:, ch]

        all_sweeps.append(pd.DataFrame(channels))

    df = pd.concat(all_sweeps, ignore_index=True)

    file_out = str(file).replace('.abf', '.csv')
    df.to_csv(file_out, index=False)

    print(f"Saved: {file_out}")

