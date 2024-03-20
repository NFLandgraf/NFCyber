import matplotlib.pyplot as plt
import basic_operations as basic
import statistics as stat
import numpy as np

poi = 130.5                     # what`s the point of interest [DAQ time]
stat_window = [-1, 1]           # how big should pre- and post-event window be that we compare
vis_window = [-4, 6]            # which window around event you want to see
color_borders = [-2.1, 3.1]     # what is the z_score minimum& maximum for color gradient presentation (5 in total good)
color_map = "seismic"           # "viridis", "seismic", etc.
event_type = "shock"


def window_of_interest():
    # subtract the onset of the period of interest, so that 0 is the POI onset
    # return cell list of z-scores from begin_period to end_period

    normal_timeline = basic.time_line()
    cut_timeline, cut_zscore = normal_timeline.copy(), basic.z_score().copy()

    for i, time in enumerate(cut_timeline):
        cut_timeline[i] = round(time - poi, 2)

    # find list position of window of POI and add to window list
    window, count_start, count_end = [], 0, 0
    for i, time in enumerate(cut_timeline):
        if time >= vis_window[0] and count_start == 0:
            window.append(i)
            count_start += 1
        elif time > vis_window[1] and count_end == 0:
            window.append(i-1)
            count_end += 1
    print(f"Panel_A window: [{round(normal_timeline[window[0]], 1)}s, {round(poi, 1)}, "
          f"{round(normal_timeline[window[-1]], 2)}s]")

    # delete all values below window_start and above window_end in timeline and zscore_list
    del cut_timeline[window[1]+1:]
    del cut_timeline[0:window[0]]
    for cell in cut_zscore:
        del cell[1][window[1]+1:]
        del cell[1][0:window[0]]

    return cut_zscore, cut_timeline


zscore_cut, timeline_cut = window_of_interest()


def ordered_z_scores():
    # take period pre and post of point of interest to calculate the mean zscore
    pre, post = [], []
    for i, time in enumerate(timeline_cut):
        if round(time, 1) == stat_window[0]:
            pre.append(i)
        elif round(time, 1) == 0.0:
            pre.append(i)
            post.append(i)
        elif round(time, 1) == stat_window[1]:
            post.append(i)

    # calculate means of the pre- and post- POI period
    zscore_means = [[] for _ in range(len(zscore_cut))]
    for i, cell in enumerate(zscore_cut):
        zscore_means[i].append(cell[0])
        pre_mean = stat.fmean(cell[1][pre[0]: pre[1]])
        post_mean = stat.fmean(cell[1][post[0]: post[1]])
        zscore_means[i].append(post_mean - pre_mean)

    # sort by the difference of z-scores
    zscore_means.sort(reverse=True, key=lambda row: row[1])

    # order the regular z-score list
    ordered_zscore = []
    for mean in zscore_means:
        for cell in zscore_cut:
            if mean[0] == cell[0]:
                ordered_zscore.append(cell)

    return ordered_zscore


def zero_to_one_values():
    z_scores = ordered_z_scores()

    # to translate z-scores to new 0-1 scale, subtract lower border[0] from value and multiply with scale_factor
    scale_factor = 1/(color_borders[1]-color_borders[0])
    converted_z_scores = []
    for cell in z_scores:
        new_cell = []
        for value in cell[1]:
            if value <= color_borders[0]:
                new_cell.append(0)
            elif value >= color_borders[1]:
                new_cell.append(1)
            else:
                new_cell.append((value-color_borders[0])*scale_factor)
        converted_z_scores.append(new_cell)

    # convert lists into multi-dimensional array
    converted_np_z = np.vstack((converted_z_scores[0: len(converted_z_scores)]))
    print(f"Panel_A array shape: {converted_np_z.shape}")

    return converted_np_z


def draw_multiple_cells():
    rgb_zscore = zero_to_one_values()
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [25, 1]})
    fig.suptitle(f"Panel_A: Event-Aligned Activity Heatmap\n({event_type})")

    # plot the heatmap
    axs[0].imshow(rgb_zscore, cmap=color_map, aspect="auto", extent=[vis_window[0], vis_window[1], rgb_zscore.shape[0], 0])
    axs[0].set_xlabel("Time from Event [s]")
    axs[0].set_ylabel("Neurons")
    axs[0].axvline(x=0, color="r")

    # plot the color gradient
    gradient = np.vstack((np.linspace(1, 0, 256), np.linspace(1, 0, 256)))
    axs[1].imshow(gradient.T, cmap=color_map, aspect="auto", extent=[0, 1, color_borders[0], color_borders[1]])
    axs[1].set_ylabel("z-score")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[1].xaxis.set_visible(False)

    plt.show()
    # plt.savefig(f"Panel_A Heatmap ({event_type}).pdf", dpi=300)


draw_multiple_cells()
