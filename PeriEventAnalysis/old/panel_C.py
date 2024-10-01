# As an input, we choose certain times_of_interest that we think has an effect on the activity of neurons.
# According to previously calculated groups of up-, non- and down-regulated neurons (via permutation),
# we calculate the mean& SEM of these populations for every time point and use that as an output

import statistics
import basic_operations as basic
from permutation_export import mod_groups, descr
import matplotlib.pyplot as plt
from scipy.stats import sem


def windows_collection(timeline, z_scores, events, window):
    # collects all windows that are around all the wanted events
    # each event has own timeline_list, time=0 at the respective event, timeline is complete
    # [[event0_timeline],[event1_timeline],...]

    multi_event_timeline = []
    for event in events:
        tl = timeline.copy()
        for i, time in enumerate(tl):
            tl[i] = round(time - event, 1)
        multi_event_timeline.append(tl)

    # add x seconds pre-and post time=0 (event) to window list and add the list positions to WINDOW [start,end]
    all_windows, check1, check2 = [], True, True
    for event in multi_event_timeline:
        event_window = []
        for i, time in enumerate(event):
            if time >= window[0] and check1:
                event_window.append(i)
                check1 = False
            elif time > window[1] and check2:
                event_window.append(i - 1)
                check2 = False
        all_windows.append(event_window)
        check1, check2 = True, True
    # print(f"all_windows: {all_windows}")
    # print(f"event0_window : {timeline[all_windows[-1][0]], timeline[all_windows[-1][1]]}")

    # delete all values below window_start and above window_end in timeline for first event (exemplary)
    timeline_cut = multi_event_timeline[0].copy()
    del timeline_cut[all_windows[0][1] + 1:], timeline_cut[:all_windows[0][0]]

    # delete values below window_start and above window_end in z_scores for each event
    z_scores_cut, amount_cells = [], len(z_scores)
    for i, cell in enumerate(z_scores):
        z_scores_cut.append([cell[0]])
        for event in all_windows:
            zs = cell[1].copy()
            del zs[event[1] + 1:], zs[:event[0]]
            z_scores_cut[i].append(zs)

    # z_scores_cut = [[cell0, [event0_z], [event1_z], [event2_z]], [cell1, [event0_z], [event1_z], [event2_z]]]
    # timeline_cut = [window_start, window_end]
    print("Panel_C: windows_collection()")
    # print(z_scores_cut[0])
    return z_scores_cut, timeline_cut


def cell_sorting(windows_zscore):
    # take group-respective cell numbers from permutation_export and sort z_score_cells into modulation_groups
    sorted_groups = []
    for i, group in enumerate(mod_groups):
        sorted_groups.append([])
        for c_number in group:
            for cell in windows_zscore:
                if cell[0] == c_number:
                    sorted_groups[i].append(cell)
                    break

    print("Panel_C: cell_sorting()")
    # print(sorted_groups[0])
    return sorted_groups


def event_means(windows_zscore):
    # per cell, per time point, mean the z-scores of every event in window around events
    event_mean_per_cell, sorted_cells = [], cell_sorting(windows_zscore)
    time_points = len(sorted_cells[0][0][1])

    for g, group in enumerate(sorted_cells):
        group_level = []
        for cell in group:
            cell_level = []
            del cell[0]
            for i in range(time_points):
                mean_list = []
                for event in cell:
                    mean_list.append(event[i])
                cell_level.append(statistics.fmean(mean_list))
            group_level.append(cell_level)
        event_mean_per_cell.append(group_level)

    print("Panel_C: event_means()")
    # print(event_mean_per_cell[0])
    return event_mean_per_cell


def calculate_mean_and_sems(windows_zscore):
    data = event_means(windows_zscore)
    mean_data, pos_sem, neg_sem = [], [], []
    time_points = len(data[0][0])

    for g, group in enumerate(data):
        mean_data.append([]), pos_sem.append([]), neg_sem.append([])
        for time in range(time_points):
            cell_level = []
            for cell in group:
                cell_level.append(cell[time])

            mean = statistics.fmean(cell_level)
            s_error = sem(cell_level)

            mean_data[g].append(mean)
            pos_sem[g].append(mean + s_error)
            neg_sem[g].append(mean - s_error)

    print("Panel_C: calculate_mean_and_sem")
    # print(mean_data)
    return mean_data, pos_sem, neg_sem


def draw(events, window):
    timeline, z_scores = basic.time_line(), basic.z_score()
    windows_zscore, windows_timeline = windows_collection(timeline, z_scores, events, window)
    means, pos_sem, neg_sem = calculate_mean_and_sems(windows_zscore)
    colors = ["g", "grey", "b"]

    for i in range(len(means)):
        plt.plot(windows_timeline, means[i], label=f"{descr[i]} (n={len(mod_groups[i])})", color=colors[i])
        plt.fill_between(windows_timeline, neg_sem[i], pos_sem[i], alpha=0.2, color=colors[i])
    plt.axvline(x=0, color="r")
    plt.title("Panel C: Event-Alignment activity (all 7 CS post conditioning)")
    plt.xlabel("Time from Event [s]")
    plt.ylabel("Neural Activity [z-score]")
    plt.legend()

    plt.show()
    # plt.savefig("Panel_C.png", dpi=300)


def main(events, window):
    timeline, z_scores = basic.time_line(), basic.z_score()
    windows_zscore, windows_timeline = windows_collection(timeline, z_scores, events, window)
    means, pos_sem, neg_sem = calculate_mean_and_sems(windows_zscore)

    return windows_timeline, means, pos_sem, neg_sem
