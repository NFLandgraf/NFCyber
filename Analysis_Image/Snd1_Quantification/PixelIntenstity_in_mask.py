#%%
import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def draw(avrg_mKate, avrg_A53T):
    
    mKate = np.asarray(avrg_mKate, dtype=float)
    A53T = np.asarray(avrg_A53T, dtype=float)

    # Unpaired comparison; Welch's test does not assume equal variance
    test = ttest_ind(mKate, A53T, equal_var=False, nan_policy="omit")
    p_value = test.pvalue

    # Scatter plot with small horizontal jitter so overlapping points remain visible
    rng = np.random.default_rng(42)
    jitter_mKate = rng.normal(0, 0.06, len(mKate))
    jitter_A53T = rng.normal(0, 0.06, len(A53T))

    fig, ax = plt.subplots(figsize=(4, 5))

    ax.scatter(np.zeros(len(mKate)) + jitter_mKate,mKate,color="black",s=45,alpha=0.8,label="mKate")
    ax.scatter(np.ones(len(A53T)) + jitter_A53T,A53T,color="crimson",s=45,alpha=0.8,label="A53T")

    # Mean ± SEM
    for x, group, color in [(0, mKate, "black"), (1, A53T, "crimson")]:
        mean = np.mean(group)
        sem = np.std(group, ddof=1) / np.sqrt(len(group))
        ax.errorbar(x, mean, yerr=sem, color=color, fmt="_", markersize=25,
                    capsize=5, linewidth=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mKate", "A53T"])
    ax.set_ylabel("Mean Snd1 intensity within mask")

    y_max = max(np.max(mKate), np.max(A53T))
    y_range = y_max - min(np.min(mKate), np.min(A53T))
    ax.text(0.5, y_max + 0.08 * y_range, f"Welch's t-test\np = {p_value:.3g}",
            ha="center")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


mask_folder = Path(r"E:\TRY\out\mask")
snd1_folder = Path(r"E:\TRY\out\Snd1")

mask_files = [path for path in mask_folder.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}]
avrg_mKate, avrg_A53T = [], []

for file_mask in mask_files:
    print(file_mask.name)
    file_snd1 = snd1_folder / file_mask.name.replace("mKate_mask", "Snd1")

    if not file_snd1.exists():
        print(f"Snd1 file not found: {file_snd1.name}")
        continue

    tiff_mask = tifffile.imread(file_mask)
    tiff_snd1 = tifffile.imread(file_snd1)

    if tiff_mask.shape != tiff_snd1.shape:
        raise ValueError(f"Different shapes: {file_mask.name}")

    values = tiff_snd1[tiff_mask > 0]
    print(np.mean(values))

    if 'A53T' in file_mask.name:
        avrg_A53T.append(float(np.mean(values)))
    else:
        avrg_mKate.append(float(np.mean(values)))







draw(avrg_mKate, avrg_A53T)

