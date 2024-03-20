import csv
import scipy.stats as stats
import matplotlib.pyplot as plt

file_traces = "traces.csv"
file_GPIO = "GPIO.csv"
printing = False


def events():
    # returns list with all DAQ times of (e.g. FC) TTL onsets with a certain length
    value_threshold = 5000  # above which value should events be considered as ON
    period_duration = 0.8   # above which period duration is this event?

    with open(file_GPIO, "r") as csvfile:
        file, val_low = csv.reader(csvfile, delimiter=","), True
        event_list = []
        for row in file:
            if row[1] == " GPIO-1":
                if val_low and int(row[2]) > value_threshold:  # from OFF to ON state
                    period_start, val_low = round(float(row[0]), 1), False
                elif not val_low and int(row[2]) < value_threshold:  # from ON to OFF state
                    period_end, val_low = round(float(row[0]), 1), True

                    if period_end - period_start > period_duration:  # CS or shock begins
                        event_list.append(period_start)

    print(f"basic_events: {event_list}")
    return event_list


def peri_events():
    # returns list with TTL lengths and the time of their onset [[0.5, [10,20,30]], [0.8, [50,60,70]], ...]
    value_threshold = 5000   # which value is definitely >value_low and <value_high
    channel_name = " GPIO-1"

    with open(file_GPIO, "r") as csvfile:
        file, val_low = csv.reader(csvfile, delimiter=","), True
        event_list = [[0, []]]
        for row in file:
            if row[1] == channel_name:
                if val_low and int(row[2]) > value_threshold:  # from OFF to ON state
                    period_start, val_low = round(float(row[0]), 1), False
                elif not val_low and int(row[2]) < value_threshold:  # from ON to OFF state
                    period_end, val_low = round(float(row[0]), 1), True

                    period_duration = round(period_end - period_start, 1)
                    existing = False
                    for period in event_list:
                        if period[0] == period_duration:
                            period[1].append(period_start)
                            existing = True
                    if not existing:
                        event_list.append([period_duration, [period_start]])
    del event_list[0]
    print(f"basic_events: {event_list}")

    if True:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        colors = ["r", "b", "g", "y"]

        # first subplot ax1 (timeline with events as vertical lines)
        t = 0
        for event_type in event_list:
            i = 0
            for event in event_type[1]:
                legend = event_type[0] if i == 0 else ""
                ax1.axvline(x=event, linewidth=event_type[0]*3, color=colors[t], label=legend)
                i += 1
            t += 1
        ax1.legend(loc="best")
        ax1.get_yaxis().set_visible(False)
        ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)

        # second subplot ax2 (bar plot with the count number of each event type)
        i = 0
        for event_type in event_list:
            a = ax2.bar(x=event_type[0], height=len(event_type[1]), color=colors[i], width=0.1)
            ax2.bar_label(a, label_type="edge")
            i += 1
        ax2.set_xlabel("time [s]"), ax2.set_ylabel("event counts")
        ax2.spines["top"].set_visible(False), ax2.spines["right"].set_visible(False)

        fig.suptitle(f"peri_event_TTLs ({channel_name})", fontsize=16)
        plt.tight_layout()
        # plt.savefig("peri_event_TTLs.png", dpi=300)
        plt.show()

    return event_list


def time_line():
    # create all timepoints of traces
    with open(file_traces, "r") as csvfile:
        file, timeline = csv.reader(csvfile, delimiter=","), []
        for row in file:
            try:
                timeline.append(round(float(row[0]), 1))
            except ValueError:  # skip the first nonsense row with this
                pass

    if printing:
        print(f"basic_timeline: {timeline}")
    return timeline


def df_over_f():
    # create cell_list in classical format of all cells and their values
    with open(file_traces, "r") as csvfile:
        # throw everything from scv file into list and create timeline list
        file, one_row, whole, df_o_f_list, a = csv.reader(csvfile, delimiter=","), [], [], [], 0
        for row in file:
            for i in range(1, len(row)):
                one_row.append(row[i])
            whole.append(one_row)
            one_row = []

    # make list in classic order (see documentation)
    for i in range(len(whole[0])):
        if whole[1][i] == " accepted":
            df_o_f_list.append([])
            df_o_f_list[a].append(whole[0][i])
            extra = []
            for time in whole:
                extra.append(time[i])
            del extra[0:2]
            df_o_f_list[a].append(extra)
            a += 1

    # turn strings in list into ints/floats
    for cell in df_o_f_list:
        cell[0] = int(cell[0].replace("C", ""))
        for i, b in enumerate(cell[1]):
            cell[1][i] = float(b)

    if printing:
        print(f"basic_df_o_f_list: {df_o_f_list}")
    return df_o_f_list


def z_score():
    # turn dF/F values into z-scores
    df_list = df_over_f()
    zscore = [[] for _ in df_list]
    for i, cell in enumerate(df_list):
        zscore[i].append(cell[0])
        zscore[i].append(list(stats.zscore(cell[1])))

    if printing:
        print(f"basic_zscore: {zscore}")
    return zscore


def frames_ttls():
    # creates list of all frame TTLs (to which DAQ time corresponds a certain frame)
    with open(file_GPIO, "r") as csvfile:
        file, high_onset_list, val_low = csv.reader(csvfile, delimiter=","), [], True
        for row in file:
            if row[1] == " GPIO-1":
                if val_low and int(row[2]) > 5000:  # from OFF to ON state
                    high_onset_list.append(float(row[0]))
                    val_low = False
                elif not val_low and int(row[2]) < 5000:    # from ON to OFF state
                    val_low = True
    return high_onset_list


frames_of_interest = [22595, 22904, 23169, 23573, 24456, 25256, 25834, 26326, 26826, 27044, 27254, 27914, 28835, 29596,
                      30749, 31153]


def return_times_of_ttl_nb():
    high_onset_list = frames_ttls()
    times_of_interest = []
    for frame in frames_of_interest:
        times_of_interest.append(high_onset_list[frame])
    print(times_of_interest)
