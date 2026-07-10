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

path_files = r"E:\Transition\WT"


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
    behav_probabilities = normalize_transition_matrix(behav_counts)

    return behav_counts, behav_probabilities, behav_bouts

def average_transition_matrices(probability_matrices):

    # averages transition probability matrices across animals
    stacked = np.stack([m.values for m in probability_matrices], axis=0)
    mean_values = np.mean(stacked, axis=0)
    mean_matrix = pd.DataFrame(mean_values, index=probability_matrices[0].index, columns=probability_matrices[0].columns)

    return mean_matrix

def plot_trans_heatmap(matrix, title, save=''):
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, square=True, linewidths=0.5, cbar_kws={"label": "Transition probability"})
    plt.title(title)
    plt.xlabel("Next behavior")
    plt.ylabel("Current behavior")
    plt.tight_layout()

    plt.savefig(save, dpi=300)
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




# if several behaviors are 1 in the same frame, the first one in this list wins
behav_prio = ["Attack", 
              "Mounting", "Circle", "Chase", "Agitated", 
              "Nose2nose", "AnogenitalSniff", "Investigate", "Following", "Approach"]

all_prob_matrices = []
all_count_matrices = []

files = get_files(path_files, '')

for file in files:
    animal_name = os.path.splitext(os.path.basename(file))[0]
    print(f'----- {animal_name} -----')
    
    counts, probs, bouts = check_single_animal( file, behav_prio)

    all_count_matrices.append(counts)
    all_prob_matrices.append(probs)

    #plot_trans_heatmap(probs, title=f"{animal_name} trans heat")
    #plot_trans_circula(probs, title=f"{animal_name} trans graph")

    print(f"{animal_name}: {len(bouts)} behavioral bouts detected")



mean_prob_matrix = average_transition_matrices(all_prob_matrices)

title = 'Mean_WT'
plot_trans_heatmap(mean_prob_matrix, title=title, save=f'{title}_Heatmap.pdf')
plot_trans_circula(mean_prob_matrix, title=title, save=f'{title}_Circular.pdf')

