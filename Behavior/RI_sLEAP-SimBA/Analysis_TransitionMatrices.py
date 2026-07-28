#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import to_rgb
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

path_files = r"E:\Transition\Test"

# if several behaviors are 1 in the same frame, the first one in this list wins
behav_prio = ["Attack", 
              "Mounting", "Circle", "Chase", "Agitated", 
              "Nose2nose", "AnogenitalSniff", "Investigate", "Following", "Approach"]

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')
    return files

def framewise_behavior_sequence(df, behav_prio, no_behavior_label='None'):
    
    # converts SimBA binary behavior columns into one behavior label per frame
    # if multiple behaviors are scored as 1 in the same frame, the behav appearing earliest in behavior_priority is chosen

    missing = [b for b in behav_prio if b not in df.columns]
    if missing:
        raise ValueError(f"These behavior columns are missing from the CSV: {missing}")

    behav_seq = []
    for _, row in df.iterrows():
        chosen_behavior = no_behavior_label
        for behavior in behav_prio:
            if row[behavior] == 1:
                chosen_behavior = behavior
                break
        behav_seq.append(chosen_behavior)

    return behav_seq

def collapse_to_bouts(behav_seq):
    
    # converts frame-wise sequence into behavioral bouts (A A A B B C C A -> A B C A)

    bouts = [behav_seq[0]]
    for behavior in behav_seq[1:]:
        if behavior != bouts[-1]:
            bouts.append(behavior)
    return bouts

def compute_transition_counts(bouts, behaviors, exclude_self_transitions=True):
    
    # computes transition count matrix from behavioral bouts
    behaviors = behaviors.copy() + ['None']

    counts = pd.DataFrame(0, index=behaviors, columns=behaviors, dtype=float)

    for current_behavior, next_behavior in zip(bouts[:-1], bouts[1:]):
        if current_behavior not in behaviors or next_behavior not in behaviors:
            continue
        if exclude_self_transitions and current_behavior == next_behavior:
            continue

        counts.loc[current_behavior, next_behavior] += 1

    return counts

def normalize_transition_matrix(counts):
    
    # row-normalizes transition counts, Each row becomes: P(next behavior | current behavior)

    row_sums = counts.sum(axis=1)
    probabilities = counts.div(row_sums.replace(0, np.nan), axis=0)

    # Rows with no outgoing transitions become 0 instead of NaN
    probabilities = probabilities.fillna(0)

    return probabilities

def check_single_animal(file, behav_prio):
    df = pd.read_csv(file)

    behav_seq = framewise_behavior_sequence(df, behav_prio)
    behav_bouts = collapse_to_bouts(behav_seq)
    
    behav_counts = compute_transition_counts(behav_bouts, behav_prio)
    for i in range(len(behav_counts)):
        behav_counts.iloc[i, i] = np.nan

    behav_probs = normalize_transition_matrix(behav_counts)
    for i in range(len(behav_probs)):
        behav_probs.iloc[i, i] = np.nan

    return behav_counts, behav_probs, behav_bouts

def average_transition_matrices(probability_matrices):

    # averages transition probability matrices across animals
    stacked = np.stack([m.values for m in probability_matrices], axis=0)
    mean_values = np.mean(stacked, axis=0)
    mean_matrix = pd.DataFrame(mean_values, index=probability_matrices[0].index, columns=probability_matrices[0].columns)

    return mean_matrix

def compare_transition_matrices(matrix_1, matrix_2, test="mannwhitney", fdr_method="fdr_bh", alpha=0.05, correct_by_row=True):
    """
    Compares transition probabilities between two groups animal-by-animal
    Returns:
    - difference_matrix: group2 mean - group1 mean
    - raw_p_matrix
    - fdr_p_matrix
    - significant_matrix
    """

    behaviors = list(matrix_1[0].index)
    raw_p_matrix = pd.DataFrame(np.nan, index=behaviors, columns=behaviors)

    # calculate group differences
    mean1 = average_transition_matrices(matrix_1)
    mean2 = average_transition_matrices(matrix_2)
    difference_matrix = mean2 - mean1
    for i in range(len(difference_matrix)):
        difference_matrix.iloc[i, i] = np.nan

    # calculate p-values of group differences
    for source in behaviors:
        for target in behaviors:

            if source == target:
                raw_p_matrix.loc[source, target] = np.nan
                continue

            values1 = np.array([m.loc[source, target] for m in matrix_1])
            values2 = np.array([m.loc[source, target] for m in matrix_2])

            if np.all(values1 == values1[0]) and np.all(values2 == values2[0]) and values1[0] == values2[0]:
                p = 1.0
            else:
                if test == "ttest":
                    _, p = ttest_ind(values1, values2, equal_var=False, nan_policy="omit")
                elif test == "mannwhitney":
                    _, p = mannwhitneyu(values1, values2, alternative="two-sided")

            raw_p_matrix.loc[source, target] = p

    fdr_p_matrix = pd.DataFrame(np.nan, index=behaviors, columns=behaviors)
    significant_matrix = pd.DataFrame(False, index=behaviors, columns=behaviors)

    if correct_by_row:
        for source in behaviors:
            pvals = raw_p_matrix.loc[source].values.astype(float)
            valid = ~np.isnan(pvals)

            if valid.sum() > 0:
                reject, pvals_fdr, _, _ = multipletests(pvals[valid], alpha=alpha, method=fdr_method)

                fdr_p_matrix.loc[source, valid] = pvals_fdr
                significant_matrix.loc[source, valid] = reject

    else:
        pvals = raw_p_matrix.values.flatten().astype(float)
        valid = ~np.isnan(pvals)

        reject, pvals_fdr, _, _ = multipletests(pvals[valid], alpha=alpha, method=fdr_method)

        fdr_flat = np.full_like(pvals, np.nan, dtype=float)
        sig_flat = np.full_like(pvals, False, dtype=bool)

        fdr_flat[valid] = pvals_fdr
        sig_flat[valid] = reject

        fdr_p_matrix.iloc[:, :] = fdr_flat.reshape(raw_p_matrix.shape)
        significant_matrix.iloc[:, :] = sig_flat.reshape(raw_p_matrix.shape)

    return difference_matrix, raw_p_matrix, fdr_p_matrix, significant_matrix


def plot_trans_heatmap(matrix, title, save=''):
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, square=True, linewidths=0.5, cbar_kws={"label": "Transition probability"})
    plt.title(title)
    plt.xlabel("Next behavior")
    plt.ylabel("Current behavior")
    plt.tight_layout()

    #plt.savefig(save, dpi=300)
    plt.show()

def plot_trans_circula(matrix, title, min_probability=0.05, save=''):

    behavior_colors = {
                        "Attack":"#d62728",
                        "Mounting": "#ff7f0e",
                        "Circle": "#ff7f0e",
                        "Chase": "#ff7f0e",
                        "Agitated": "#ff7f0e",
                        "Nose2nose": "#42cf17",
                        "AnogenitalSniff": "#42cf17",
                        "Investigate": "#42cf17",
                        "Following": "#42cf17",
                        "Approach": "#42cf17",
                        "None": "#90948e"}
    
    behaviors = list(matrix.index)
    G = nx.DiGraph()
    G.add_nodes_from(behaviors)

    for current_behavior in behaviors:
        for next_behavior in behaviors:
            probability = matrix.loc[current_behavior, next_behavior]
            if probability >= min_probability:
                G.add_edge(current_behavior, next_behavior, weight=probability)

    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.circular_layout(G)
    
    # Draw nodes first
    node_colors = [behavior_colors[b] for b in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=4500, node_color=node_colors, edgecolors="black", linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)

    # Draw edges manually so arrowheads stop before node circles
    for source, target, data in G.edges(data=True):
        probability = data["weight"]

        x1, y1 = pos[source]
        x2, y2 = pos[target]

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=30,
            linewidth=1 + probability * 8,
            color = behavior_colors[target],
            alpha=0.8,
            connectionstyle="arc3,rad=0.25",
            shrinkA=35,
            shrinkB=35)
        
        ax.add_patch(arrow)

    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()

def plot_difference_heatmap(difference_matrix, significant_matrix=None, title="Transition difference"):

    plt.figure(figsize=(10, 8))
    annotations = difference_matrix.round(2).astype(str)

    #if significant_matrix is not None:
    #    annotations = annotations.where(~significant_matrix, annotations + "*")

    sns.heatmap(
        difference_matrix,
        vmin=-0.2,
        vmax=0.2,
        annot=annotations,
        fmt="",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"color": "black","fontsize": 10},
        cbar_kws={"label": "Difference in transition probability"}
    )

    plt.title(title)
    plt.xlabel("Next behavior")
    plt.ylabel("Current behavior")
    plt.tight_layout()
    plt.show()

def plot_pvalue_heatmap(p_matrix, title="FDR-corrected p-values"):

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        p_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis_r",
        vmin=0,
        vmax=0.8,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "FDR-corrected p-value"}
    )

    plt.title(title)
    plt.xlabel("Next behavior")
    plt.ylabel("Current behavior")
    plt.tight_layout()
    plt.show()



anim_IDs = ['218', '219', '220', '221', '222', '230', '231', 
            '249', '250', '251', '252', '259', '260', '261', '262']

prob_matrices_WT, prob_matrices_TG = [], []

for anim in anim_IDs:

    # get the files that correspond to each animal
    files = get_files(path_files, anim)
    animal_name = os.path.splitext(os.path.basename(files[0]))[0]
    print(f'----- {animal_name} -----')
    
    # gets mean matrix from animal (RI1, RI2, RI3)
    prob_matrices_anim = []
    for file in files:
        counts, probs, bouts = check_single_animal(file, behav_prio)
        prob_matrices_anim.append(probs)
    prob_matrix_anim = average_transition_matrices(prob_matrices_anim)

    # adds matrix to the respective group
    if 'WT' in animal_name:
        prob_matrices_WT.append(prob_matrix_anim)
    elif 'TG' in animal_name:
        prob_matrices_TG.append(prob_matrix_anim)


# calculate differences and the respective p-values between both groups
matrix_diff, matrix_p_raw, matrix_p_fdr, matrix_sig = compare_transition_matrices(matrix_1=prob_matrices_WT, matrix_2=prob_matrices_TG)
plot_difference_heatmap(matrix_diff, title="TG - WT transition probability differences")
plot_pvalue_heatmap(matrix_p_fdr, title="FDR-corrected p-values: WT vs TG")

# visualize group means
prob_matrix_WT = average_transition_matrices(prob_matrices_WT)
prob_matrix_TG = average_transition_matrices(prob_matrices_TG)
prob_matrix_all = average_transition_matrices([prob_matrix_WT, prob_matrix_TG])
plot_trans_heatmap(prob_matrix_all, title="MEAN trans heat")
plot_trans_circula(prob_matrix_all, title="MEAN trans circ")
plot_trans_circula(prob_matrix_WT, title="WT trans circ")
plot_trans_circula(prob_matrix_TG, title="TG trans circ")
plot_trans_heatmap(prob_matrix_WT, title="WT trans heat")
plot_trans_heatmap(prob_matrix_TG, title="TG trans heat")
