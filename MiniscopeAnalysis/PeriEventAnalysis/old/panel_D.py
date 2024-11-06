import panel_C
from permutation_export import mod_groups
import random
import matplotlib.pyplot as plt
import basic_operations as basic
import statistics


def get_three_cells(timeline, z_scores, events):
    all_z_scores, cut_timeline = panel_C.windows_collection(timeline, z_scores, events, [-1, 1])
    cell_list = []

    # take random cell in permutation_export and add to list
    for group in mod_groups:
        if len(group) > 0:
            number = random.randint(0, len(group)-1)
            cell_list.append(group[number])
        else:
            cell_list.append("empty")

    # add the whole z_score
    three_cells = []
    for spec_cell in cell_list:
        if spec_cell != "empty":
            for every_cell in all_z_scores:
                if spec_cell == every_cell[0]:
                    three_cells.append(every_cell)
                    break
        else:
            three_cells.append("empty")

    return three_cells, cut_timeline


def means(timeline, z_scores, events):
    three_cells, cut_timeline = get_three_cells(timeline, z_scores, events)
    time_points = len(three_cells[0][1])

    event_means = []
    for c, cell in enumerate(three_cells):
        if cell != "empty":
            cell_list = [cell[0], []]
            del cell[0]
            for t in range(time_points):
                mean_list = []
                for event in cell:
                    mean_list.append(event[t])
                cell_list[1].append(statistics.fmean(mean_list))
            event_means.append(cell_list)

    # randomly select x events from all the events
    amount_events = 16 if len(three_cells[0]) >= 16 else len(three_cells[0])
    elimination_three_cells = []
    for i, cell in enumerate(three_cells):
        elimination_three_cells.append([])
        random_events = random.sample(range(0, len(cell)), amount_events)

        for event in random_events:
            elimination_three_cells[i].append(three_cells[i][event])

    return elimination_three_cells, cut_timeline, event_means


def draw(timeline, z_scores, events):
    three_cells, cut_timeline, event_means = means(timeline, z_scores, events)
    colors, i = ["g", "grey", "b"], 0

    while i < len(three_cells[0]) - 1:  # smaller than number of events
        plt.plot(cut_timeline, three_cells[0][i], alpha=0.2, color=colors[2])
        i += 1
    plt.plot(cut_timeline, three_cells[0][i], alpha=0.2, label=f"cell {event_means[0][0]}", color=colors[2])
    plt.plot(cut_timeline, event_means[0][1], alpha=1, color=colors[2])

    plt.axvline(x=0, color="r")
    plt.title(f"Panel D: Random Event Responses")
    plt.xlabel("Time from Event [s]")
    plt.ylabel("Neural Activity [z-score]")
    plt.legend()

    plt.show()
    # plt.savefig("Panel_B.png", dpi=300)


def main(events):
    timeline, z_scores = basic.time_line(), basic.z_score()
    three_cells, cut_timeline, event_means = means(timeline, z_scores, events)

    return three_cells, cut_timeline, event_means
