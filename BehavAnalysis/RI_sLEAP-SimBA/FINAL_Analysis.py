#%%
import os
import pandas as pd

folder = r"C:\Users\landgrafn\Desktop\SimBA_MaschineResults2\RI2"
out_csv = r"C:\Users\landgrafn\Desktop\SimBA_MaschineResults2\RI2\Summary.csv"

behaviors = [
    "Investigate",
    "Following",
    "Nose2nose",
    "AnogenitalSniff",
    "Approach",
    "Mounting",
    "Circle",
    "Chase",
    "Agitated",
    "Attack"
]

rows = []

for file in os.listdir(folder):
    if not file.lower().endswith(".csv"):
        continue

    path = os.path.join(folder, file)
    df = pd.read_csv(path)

    row = {
        "Filename": file,
        "Frame_number": len(df)
    }

    for behavior in behaviors:
        prob_col = "Probability_" + behavior

        if behavior in df.columns:
            row[behavior + "_Sum"] = df[behavior].sum()
        else:
            row[behavior + "_Sum"] = pd.NA

        if prob_col in df.columns:
            row[prob_col + "_Sum"] = df[prob_col].sum()
        else:
            row[prob_col + "_Sum"] = pd.NA

    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(out_csv, index=False)

print("Saved:", out_csv)