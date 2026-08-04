# 
from pathlib import Path
import isx

'''
This code takes a folder with .gpio files in it.
It loops through .gpio files in a folder and exports them into csv
'''

import_folder = Path("/Volumes/Landgraf_BU/GPIO_1_Raw")
export_folder = Path("/Volumes/Landgraf_BU/out_GPIO")


rec_files = sorted([p for p in import_folder.iterdir() if p.suffix.lower() == ".gpio"])
print(f"Found {len(rec_files)} .isxd files in {import_folder}")

for gpio in rec_files:
    
    base = gpio.stem
    gpio = [str(gpio)]
    print(f"\n=== Processing {base} ===")

    out_gpio = export_folder / f"{base}_gpio.csv" 
    isx.export_gpio_set_to_csv(gpio, str(out_gpio), inter_isxd_file_dir='/tmp', time_ref='start')