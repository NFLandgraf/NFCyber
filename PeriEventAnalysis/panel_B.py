# As an input, we choose certain times_of_interest that we think has an effect on the activity of neurons.
# We want to show the mean response of the neuronal population towards the stimulus,
# so for each neuron, we mean its response to all events of that certain event type.
# Then, we calculate the mean& SEM of all neurons for every time point and use that as an output.

import statistics
import basic_operations as basic
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
    print("Panel_B: windows_collection()")
    # print(z_scores_cut[0])
    return z_scores_cut, timeline_cut


def event_means(windows_zscore):
    # per cell, per time point, mean the z-scores of every event in window around event
    event_mean_per_cell = []
    for c, cell in enumerate(windows_zscore):
        del cell[0]     # we don`t care about the cell number anymore, because everything is mean anyway
        event_mean_per_cell.append([])
        for t in range(len(cell[0])):
            times = []
            for event in cell:
                times.append(event[t])
            event_mean_per_cell[c].append(statistics.fmean(times))

    print("Panel_B: event_means()")
    # print(event_mean_per_cell[0])
    # return [[cell0_event-mean_time-point0, 1, 2, 3, 4], [cell1_event-mean_time-point0, 1, 2, 3, 4], ...]
    return event_mean_per_cell


def organize_event_means_and_sem(windows_zscore):
    event_mean = event_means(windows_zscore)
    # organize event_means() into order and return means + sem

    # organize event_means [[time-point0_cell0,1,2,...], [time-point1_cell0,1,2,...], ...]
    # each time-point has one list of all cell_event_means
    organized_event_means = []
    for t in range(len(event_mean[0])):
        organized_event_means.append([])
        for cell in event_mean:
            organized_event_means[t].append(cell[t])
    # print(organized_event_means[0])

    # calculate the time-point_means and time-point_sem each time-point
    time_point_mean, pos_sem, neg_sem = [], [], []
    for time in organized_event_means:
        mean = statistics.fmean(time)
        sem_error = sem(time)

        time_point_mean.append(mean)
        pos_sem.append(mean + sem_error)
        neg_sem.append(mean - sem_error)

    print("Panel_B: organize_event_means_and_sem()")
    # print(time_point_mean[0])
    # return time_point_mean = [mean_time-point0,1,2,3,4,...]
    # return time_point_sem = [sem_time-point0,1,2,3,4,...]
    return time_point_mean, pos_sem, neg_sem


def draw(events, window, event_type):
    timeline, z_scores = basic.time_line(), basic.z_score()
    windows_zscore, windows_timeline = windows_collection(timeline, z_scores, events, window)
    means, pos_sem, neg_sem = organize_event_means_and_sem(windows_zscore)
    colors = ["g", "grey", "b"]

    legend = f"population events-mean activity +/- sem (n={len(windows_zscore)})"
    plt.plot(windows_timeline, means, label=legend, color=colors[2])
    plt.fill_between(windows_timeline, neg_sem, pos_sem, alpha=0.2, color=colors[2])

    plt.axvline(x=0, color="r")
    plt.title(f"Panel B: Event-Aligned Population Activity\nevent type: {len(events)} {event_type}")
    plt.xlabel("Time from Event [s]")
    plt.ylabel("Neural Activity [z-score]")
    plt.legend()

    plt.show()
    # plt.savefig("Panel_B.png", dpi=300)


def main(events, window):
    timeline, z_scores = basic.time_line(), basic.z_score()
    windows_zscore, windows_timeline = windows_collection(timeline, z_scores, events, window)
    means, pos_sem, neg_sem = organize_event_means_and_sem(windows_zscore)
    n_cells = len(windows_zscore)

    return windows_timeline, means, pos_sem, neg_sem, n_cells
