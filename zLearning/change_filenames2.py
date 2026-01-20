#%%
# changes filename to XX if YY is in filename
import os
import re

folder_path = r"E:\Backup_Server-landgrafn_Jan2026\_proj_CA1Dopa\CA1Dopa_Longitud\AllDataRaw"

search_date = "2025-12-20"
replacement = "Zost2"

# Matches: 2025-04-25-09-14-22  (date + -HH-MM-SS)
pattern = re.compile(rf"{re.escape(search_date)}-\d{{2}}-\d{{2}}-\d{{2}}")

i = 0
for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if not os.path.isfile(old_path):
        continue

    if search_date not in filename:
        continue

    new_filename = pattern.sub(replacement, filename)

    if new_filename == filename:
        print('same name')

    new_path = os.path.join(folder_path, new_filename)

    # avoid overwriting existing files
    if os.path.exists(new_path):
        print('exists')
        base, ext = os.path.splitext(new_filename)
        i = 1
        candidate = os.path.join(folder_path, f"{base}_{i}{ext}")
        while os.path.exists(candidate):
            i += 1
            candidate = os.path.join(folder_path, f"{base}_{i}{ext}")
        new_path = candidate
        new_filename = os.path.basename(new_path)

    os.rename(old_path, new_path)
    i += 1
    print(f'Renamed: "{filename}" -> "{new_filename}"')

print(i)
