#%%
import pandas as pd
import numpy as np

csv_path = r"E:\hTauxAPP1(3m)_Zummary.csv"

df = pd.read_csv(csv_path)
df["Genotype"] = df["ID"].str.split("_").str[0]

features = [
    "Frame_Social_Sum_MEAN",
    "Frame_Threat_Sum_MEAN",
    "Frame_Threat_Sum_MAX",
    "Frame_Attack_Sum_MEAN",
    "Frame_Attack_Sum_MAX",
    "Attack_Bouts_MEAN",
    "Attack_Bouts_MAX",
    "FAL_MEAN",
    "FAL_MIN"
    ]

#%%
from sklearn.preprocessing import StandardScaler

X = df[features].copy()

log_cols = [
    #"Frame_Social_Sum_MEAN",
    #"Frame_Threat_Sum_MEAN",
    #"Frame_Threat_Sum_MAX",
    #"Frame_Attack_Sum_MEAN",
    #"Frame_Attack_Sum_MAX",
    #"Attack_Bouts_MEAN",
    #"Attack_Bouts_MAX"
    ]

X[log_cols] = np.log1p(X[log_cols])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
plt.figure(figsize=(6,5), dpi=300)
colors = {"WT": "black", "TG": "red"}

for genotype in ["WT", "TG"]:
    sub = df[df["Genotype"] == genotype]
    plt.scatter(sub["PC1"], sub["PC2"], label=genotype, s=80, alpha=0.85, color=colors[genotype])
for _, row in df.iterrows():
    plt.text(row["PC1"], row["PC2"], row["ID"], fontsize=7, ha="center", va="bottom")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1", "PC2"])
plt.figure(figsize=(6,4), dpi=300)
loadings["PC1"].sort_values().plot(kind="barh")
plt.xlabel("PC1 loading")
plt.tight_layout()
plt.show()

#%%
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_neighbors=6, min_dist=0.3, metric="euclidean", random_state=42)

X_umap = reducer.fit_transform(X_scaled)
df["UMAP1"] = X_umap[:, 0]
df["UMAP2"] = X_umap[:, 1]
plt.figure(figsize=(6,5), dpi=300)
colors = {"WT": "gray", "TG": "green"}

for genotype in ["WT", "TG"]:
    sub = df[df["Genotype"] == genotype]
    plt.scatter(sub["UMAP1"], sub["UMAP2"], label=genotype, s=150, alpha=0.85, color=colors[genotype])
#for _, row in df.iterrows():
    #plt.text(row["UMAP1"], row["UMAP2"], row["ID"], fontsize=7, ha="center", va="bottom")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(frameon=False, loc="upper left")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(r"E:\hTauxAPP1(3m)_UMAP.pdf", format="pdf", bbox_inches="tight")
plt.show()
