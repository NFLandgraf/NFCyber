#%%
import pandas as pd
import pingouin as pg

# Read data
df = pd.read_csv("C:\\Users\\landgrafn\\Desktop\\EGTA_Opto.csv")
df = df[df["freq"].isin([50])]

df["cell"] = df["treat"] + "_" + df["cell"].astype(str)

# Convert to categorical variables
#df["stim"] = df["stim"].astype(str)
#df["freq"] = df["freq"].astype(str)
df["pulse"] = df["pulse"].astype(str)
#df["calcium"] = df["calcium"].astype(str)
df["treat"] = df["treat"].astype(str)

# Two-way repeated-measures ANOVA
anova = pg.mixed_anova(
    data=df,
    dv="ppr",
    within="pulse",
    between="treat",
    subject="cell"
)


# ttest = pg.ttest(
#     x=df.loc[df["stim"] == "elec", "plateau"],
#     y=df.loc[df["stim"] == "opto", "plateau"],
#     paired=False,
#     correction=True   # Welch's t-test
# )

print(anova)
