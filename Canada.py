#%%
import pandas as pd
import pingouin as pg

# Read data
df = pd.read_csv("C:\\Users\\landgrafn\\Desktop\\Opto_PPR_2.5mM vs 1.5mM.csv")
df = df[df["freq"].isin([5,10,20])]

df["cell"] = df["calcium"] + "_" + df["cell"].astype(str)

# Convert to categorical variables
#df["stim"] = df["stim"].astype(str)
df["freq"] = df["freq"].astype(str)
#df["pulse"] = df["pulse"].astype(str)
df["calcium"] = df["calcium"].astype(str)

# Two-way repeated-measures ANOVA
anova = pg.mixed_anova(
    data=df,
    dv="ppr",
    within="freq",
    between="calcium",
    subject="cell"
)


# ttest = pg.ttest(
#     x=df.loc[df["stim"] == "elec", "plateau"],
#     y=df.loc[df["stim"] == "opto", "plateau"],
#     paired=False,
#     correction=True   # Welch's t-test
# )

print(anova)
