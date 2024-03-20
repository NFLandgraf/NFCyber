import basic_operations as basic
import statistics
from permute.core import one_sample


def peri_event_means(events, stat_window):
    # for every interesting period, get the difference in the mean before and after the time point for each cell
    event_responses, timeline, zscore = [], basic.time_line(), basic.z_score()

    # [[cell0, [event0_response, event1_response, ...]], [cell1, [event0_response, event1_response, ...]]]
    for c, cell in enumerate(zscore):
        event_responses.append([cell[0], []])
        for event in events:
            for t, time in enumerate(timeline):
                if time == event:
                    mean_pre = statistics.fmean(cell[1][t + stat_window[0][0] * 10: t + stat_window[0][1] * 10])
                    mean_post = statistics.fmean(cell[1][t + stat_window[1][0] * 10: t + stat_window[1][1] * 10])
                    event_responses[c][1].append(mean_post - mean_pre)
                    break

    print("Permutation: peri_event_means()")
    return event_responses


def peri_event_permutation(events, stat_window, p_value):
    # returns cell list with cells that significantly show pos/no/neg responses towards event type
    peri_event_mean, p_values = peri_event_means(events, stat_window), []
    total_cells = len(peri_event_mean)

    for i, cell in enumerate(peri_event_mean):
        p_values.append([cell[0]])
        p, diff_mean = one_sample(cell[1], alternative="two-sided", reps=10000, stat="mean")
        p_values[i].append([p, diff_mean])
        print(f"Permutation: {i}/{total_cells} (p={round(p, 3)}, mean={diff_mean})")

    # sort cells according to p_values and diff_mean into respective list and return that list
    pos_non_neg_responders = [[], [], []]
    for cell in p_values:
        if cell[1][0] <= p_value:
            if cell[1][1] > 0:
                pos_non_neg_responders[0].append(cell[0])
            elif cell[1][1] < 0:
                pos_non_neg_responders[2].append(cell[0])
        else:
            pos_non_neg_responders[1].append(cell[0])

    print(f"Permutation: pos: {len(pos_non_neg_responders[0])}, "
          f"non: {len(pos_non_neg_responders[1])}, "
          f"neg: {len(pos_non_neg_responders[2])}")
    return pos_non_neg_responders


def export_permutation_results(events, stat_window, p_value):
    # export the results of peri_event_permutation
    responders = peri_event_permutation(events, stat_window, p_value)

    # write python file for further usage
    with open("PeriEvent_permutation_export.py", "w") as f:
        f.write(f"# results of the peri_event_permutation, all data significant-tested\n"
                f"# responders[0] = up-modulated\n"
                f"# responders[1] = non-modulated\n"
                f"# responders[2] = down-modulated\n\n"
                f"mod_groups = {responders}\n"
                f"descr = ['up modulated', 'non modulated', 'down modulated']")
