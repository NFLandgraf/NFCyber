#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np




#%%
# Solomon data
def ethogramm_solomon(file):
    print('Solomon')

    df = pd.read_csv(file)

    # collect similar behaviors
    df["Attack"] = (df["Behaviour"] == "Attack").astype(int)
    df["Chase"] = df["Behaviour"].isin(["Chase"]).astype(int)
    df["Circle"] = df["Behaviour"].isin(["Circle"]).astype(int)
    df["Mounting"] = df["Behaviour"].isin(["Mounting"]).astype(int)
    df["Threat"] = df["Behaviour"].isin(["Chase", "Circle", "Mounting"]).astype(int)
    df["Anogenital_Sniff"] = df["Behaviour"].isin(["Anogenital_Sniff"]).astype(int)
    df["Investigate"] = df["Behaviour"].isin(["Investigate"]).astype(int)
    df["Pursuit"] = df["Behaviour"].isin(["Pursuit"]).astype(int)
    df["Allo"] = df["Behaviour"].isin(["Anogenital_Sniff", "Investigate", "Pursuit"]).astype(int)
    df = df.drop(['Behaviour'], axis=1)



    # downsample to 
    frames_to_bin = 30
    df = df.groupby(df.index // frames_to_bin).max().reset_index(drop=True)

    # create Ethogram
    def behavior_color(row):
        if row["Attack"] == 1:
            return "red"
        elif row["Threat"] == 1:
            return "yellow"
        elif row["Anogenital_Sniff"] == 1:
            return "blue"
        elif row["Pursuit"] == 1:
            return "green"
        elif row["Investigate"] == 1:
            return "turquoise"
        else:
            return "gray"
    
    colors = df.apply(behavior_color, axis=1)

    fig, ax = plt.subplots(figsize=(10, 2))
    for i, color in enumerate(colors):
        ax.plot([i, i + 1], [1, 1], color=color, linewidth=50)

    ax.set_xlim(0, len(df))
    ax.set_yticks([])
    ax.set_xlabel("Time (chunks of 30 frames)")
    ax.set_title("Ethogram")

    plt.tight_layout()
    plt.savefig('solomon.pdf', format="pdf", dpi=300)
    plt.show()

file = 'D:\\2025-02-12_hTauxAPP1(3m)_RI3_m259_Test_edit_fps_solomon.csv'
ethogramm_solomon(file)
 




# SimBA data
attack  = 0.2
invest  = 0.5   #türkis
ano     = 0.5   #blau
circle  = 0.5
mount   = 0.4
chase   = 0.7
pursuit = 0.3   #green

def ethogramm_simba(file):
    print('Simba')

    df = pd.read_csv(file)

    # throw away useless cols
    keep_cols = ["Probability_Attack","Attack","Probability_Investigate","Investigate","Probability_Anogenital_Sniff","Anogenital_Sniff",
                    "Probability_Pursuit","Pursuit","Probability_Circle","Circle","Probability_Mounting","Mounting","Probability_Chase","Chase"]
    keep_cols = ["Attack","Investigate","Anogenital_Sniff","Pursuit","Circle","Mounting","Chase"]
    keep_cols_prob = ["Probability_Attack","Probability_Investigate","Probability_Anogenital_Sniff","Probability_Pursuit","Probability_Circle","Probability_Mounting","Probability_Chase"]
    df = df[keep_cols_prob]

    # threshold
    df['Attack'] = (df['Probability_Attack'] > attack).astype(int)
    df['Investigate'] = (df['Probability_Investigate'] > invest).astype(int)
    df['Anogenital_Sniff'] = (df['Probability_Anogenital_Sniff'] > ano).astype(int)
    df['Circle'] = (df['Probability_Circle'] > circle).astype(int)
    df['Mounting'] = (df['Probability_Mounting'] > mount).astype(int)
    df['Chase'] = (df['Probability_Chase'] > chase).astype(int)
    df['Pursuit'] = (df['Probability_Pursuit'] > pursuit).astype(int)
    df = df[keep_cols]
    df["Threat"] = ((df["Mounting"] == 1) | (df["Circle"] == 1) | (df["Chase"] == 1)).astype(int)

    # downsample to 
    frames_to_bin = 60
    df = df.groupby(df.index // frames_to_bin).max().reset_index(drop=True)

    # create Ethogram
    def behavior_color(row):
        if row["Attack"] == 1:
            return "red"
        elif row["Threat"] == 1:
            return "yellow"
        elif row["Anogenital_Sniff"] == 1:
            return "blue"
        elif row["Pursuit"] == 1:
            return "green"
        elif row["Investigate"] == 1:
            return "turquoise"
        else:
            return "gray"
    
    colors = df.apply(behavior_color, axis=1)

    fig, ax = plt.subplots(figsize=(10, 2))
    for i, color in enumerate(colors):
        ax.plot([i, i + 1], [1, 1], color=color, linewidth=50)

    ax.set_xlim(0, len(df))
    ax.set_yticks([])
    ax.set_xlabel("Time (chunks of 30 frames)")
    ax.set_title("Ethogram")

    plt.tight_layout()
    plt.savefig("simba.pdf", format="pdf", dpi=300)
    plt.show()

file = 'D:\\2025-02-12_hTauxAPP1(3m)_RI3_m259_Test_edit_fps_done_all0.5_200ms.csv'
ethogramm_simba(file)
