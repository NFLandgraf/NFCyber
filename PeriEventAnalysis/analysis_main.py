import PeriEvent_Panel_A
from permutation import export_permutation_results
from permutation_export import descr, mod_groups
from panel_B import main as main_b
from panel_C import main as main_c
from panel_D import main as main_d
import matplotlib.pyplot as plt

first_show_heatmap = False            # "True" if you want to show the Peri-Event Heatmap, "False" if not, check Panel_A
first_calculate_permutation = False  # "True" if you want to sort cells into modulation groups, "False" if already done

events = [807, 817.3, 826.2, 839.7, 869.2, 895.9, 915.2, 931.6, 948.3, 955.6, 962.6,
          984.7, 1015.4, 1040.8, 1079.4, 1092.8]      # which events you want to analyze & image [s]
stat_window = [[-1, 0], [0, 1]]     # what range before& after the event you want to create the mean of [s]
p_value = 0.05                      # modulation_groups should be sorted upon which p_value
show_window = [-4, +6]              # how many seconds before and after events you want to show z_scores
event_type = "movement initiations"               # what kind of events are you investigating


def draw():
    small_size, medium_size = 6, 8
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    colors = ["g", "grey", "b", "k"]
    fig, axs = plt.subplots(2, 2)

    # PANEL_B
    legend1 = f"mean activity +/- sem (n={panB_n_cells})"
    axs[0, 0].plot(panB_wind_timeline, panB_means, label=legend1, color=colors[3])
    axs[0, 0].fill_between(panB_wind_timeline, panB_neg_sem, panB_pos_sem, alpha=0.2, color=colors[3])
    axs[0, 0].axvline(x=0, color="r")
    axs[0, 0].set_title(f"Panel B: Event-Aligned Population Activity", fontsize=medium_size)
    axs[0, 0].set_xlabel("Time from Event [s]", fontsize=medium_size)
    axs[0, 0].set_ylabel("Neural Activity [z-score]", fontsize=medium_size)
    axs[0, 0].sharey(axs[0, 1])
    axs[0, 0].legend(fontsize=small_size)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=small_size)

    # PANEL_C
    for i in range(len(panC_means)):
        axs[0, 1].plot(panC_wind_timeline, panC_means[i], label=f"{descr[i]} (n={len(mod_groups[i])})", color=colors[i])
        axs[0, 1].fill_between(panC_wind_timeline, panC_neg_sem[i], panC_pos_sem[i], alpha=0.2, color=colors[i])
    axs[0, 1].axvline(x=0, color="r")
    axs[0, 1].set_title("Panel C: Event-Alignment Activity", fontsize=medium_size)
    axs[0, 1].set_xlabel("Time from Event [s]", fontsize=medium_size)
    axs[0, 1].set_ylabel("Neural Activity [z-score]", fontsize=medium_size)
    axs[0, 1].legend(fontsize=small_size)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=small_size)

    # PANEL_D
    del panD_three_cells[1]
    del panD_means[1]

    colors = ["g", "b", "k"]
    for i in [0, 1]:
        a = 0
        while a < len(panD_three_cells[i]) - 1:  # smaller than number of events
            axs[1, i].plot(panD_cut_timeline, panD_three_cells[i][a], alpha=0.2, color=colors[i])
            a += 1
        axs[1, i].plot(panD_cut_timeline, panD_three_cells[i][a], alpha=0.2, color=colors[i])
        axs[1, i].plot(panD_cut_timeline, panD_means[i][1], alpha=1, label=f"random cell {panD_means[i][0]} mean", color=colors[i])

        bottom, top = min(panD_means[i][1]), max(panD_means[i][1])
        bottom = bottom-0.2 if bottom >= 0 else bottom-0.2
        top = top+0.2 if top >= 0 else top+0.2
        axs[1, i].set_ylim(bottom, top)

        axs[1, i].axvline(x=0, color="r")
        axs[1, i].set_title(f"Panel D: Random Event Responses", fontsize=medium_size)
        axs[1, i].set_xlabel("Time from Event [s]", fontsize=medium_size)
        axs[1, i].set_ylabel("Neural Activity [z-score]", fontsize=medium_size)
        axs[1, i].legend(fontsize=small_size)
        axs[1, i].tick_params(axis='both', which='major', labelsize=small_size)

    fig.suptitle(f"Peri_Event Analysis ({len(events)} {event_type})", weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.056, bottom=0.071, right=0.99, top=0.908, wspace=0.13, hspace=0.292)

    plt.show()
    #plt.savefig(f"Peri_event Analysis ({len(events)} {event_type}).pdf", dpi=300)


if first_show_heatmap:
    PeriEvent_Panel_A.draw_multiple_cells()
elif first_calculate_permutation:
    export_permutation_results(events, stat_window, p_value)
else:
    # collecting the data from the different panels
    panB_wind_timeline, panB_means, panB_pos_sem, panB_neg_sem, panB_n_cells = main_b(events, show_window)
    panC_wind_timeline, panC_means, panC_pos_sem, panC_neg_sem = main_c(events, show_window)
    panD_three_cells, panD_cut_timeline, panD_means = main_d(events)

    draw()
