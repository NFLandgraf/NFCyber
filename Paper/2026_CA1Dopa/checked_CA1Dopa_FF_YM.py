#%%
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
import h5py

'''
This code uses the FF (.doric) and YM (.csv) files and looks at correlations between speed and the FF-zscore
1. From the .doric files, get Fluo, Isos, Shock-Times and Cam-Times
2. From the DLC files, get the positions and speeds
3. Merge the dfs according to the Cam-Times
4. Trim the resulting df before the first and after the last Shock-Time
5. Calculate the df/f and zscore from the trimmed time series
6. Correlate speed and zscore in bins of a certain length
7. check the zscore in different speed states
'''

# path must be the folder where both the .doric and DLC files are inside
path = r"E:\FF-DA_YM"
file_useless_strings = ['CA1Dopa_FF-DA(3m)_2024-11-19_YM+FF_']


def manage_filename(file):
    name = Path(file).stem
    for word in file_useless_strings:
        name = name.replace(word, "")
    return name

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    # print(f'\n{len(files)} files found')
    # for file in files:
    #     print(file)
    # print('\n')

    return files


def get_Doric(file_doric, io_channel_cam=2):

    # do this when you have 1 IO-TTLs, e.g. IO2-Cam_TTL

    def h5print(item, leading=''):
        # prints out the h5 data structure
        for key in item:
            if isinstance(item[key], h5py.Dataset):
                print(leading + key + ':' + str(item[key].shape))
            else:
                print(leading + key)
                h5print(item[key], leading + '  ')
    
    def get_hightimes(df_io, value_threshold=0.9):
        high = df_io['digi_Cam'].to_numpy() >= value_threshold
        time = df_io.index.to_numpy(dtype=float)

        if len(high) == 0:
            return np.empty((0, 2))

        edges = np.diff(high.astype(np.int8))
        onsets = np.flatnonzero(edges == 1) + 1
        offsets = np.flatnonzero(edges == -1) + 1

        # Exclude incomplete pulses at the recording boundaries
        if high[0]:
            offsets = offsets[1:]
        if high[-1]:
            onsets = onsets[:-1]

        return np.column_stack((time[onsets], time[offsets]))

    with h5py.File(file_doric, 'r') as f:
        #h5print(f, '')     # prints the whole structure
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        digi_Cam    = np.array(f[path + f'DigitalIO/DIO0{io_channel_cam}']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])


    # extract data
    df_signal    = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo}, index=pd.Index(np.round(sig_time, 3), name='Time'))
    df_io        = pd.DataFrame({'digi_Cam': digi_Cam}, index=pd.Index(digi_time, name='Time'))
    io_hightimes = get_hightimes(df_io)

    return df_signal, io_hightimes

def get_DLC(file_DLC, dist_bp='head', DLC_mm_per_px = 0.12, fps_Cam=30):
    # calculates the distance and speed between frames
    
    def cleaning_raw_df(file_DLC):

        df = pd.read_csv(file_DLC, header=None, low_memory=False)

        # create new column names from first two rows
        new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]
        df.columns = new_columns
        df = df.drop(labels=[0, 1, 2], axis="index")

        # set index and convert to numeric
        df.set_index('bodyparts_coords', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(int)

        bps_all         = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                            'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                            'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                            'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
        bps_head        = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
        bps_postbody    = ['mid_backend2', 'mid_backend3', 'tail_base']

        # remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_likelihood'] <= 0.9
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_likelihood')

        # interpolate to skip nans
        df = df.interpolate(method="linear")

        # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
        for c in ('_x', '_y'):
            df[f'head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
            df[f'postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

        # smoothing along time via gaussian filter
        for col in df.columns:
            df[col] = scipy.ndimage.gaussian_filter1d(df[col].values, sigma=2, mode="nearest")

        return df
    
    df = cleaning_raw_df(file_DLC)
    
    # takes boypart and calculates everything from that
    dx = np.diff(df[f'{dist_bp}_x'].to_numpy())
    dy = np.diff(df[f'{dist_bp}_y'].to_numpy())
    dist = np.sqrt(dx*dx + dy*dy)              # length = n_frames - 1
    dist = np.r_[0.0, dist]                    # prepend 0 for first frame (align length)
    dist = dist * DLC_mm_per_px                # in mm
    df['Distance'] = dist                      
    df['Speed'] = dist * fps_Cam               # in mm/s
    df["Distance_cum"] = df["Distance"].cumsum()
        
    return df

def merge_Doric_DLC(df_trace, df_DLC, io_hightimes):

    # set the correct index for df_trace
    hightimes = np.array([i[0] for i in io_hightimes])
    df_trace_crop = df_trace.copy()
    df_trace_crop.index = pd.Index(pd.to_numeric(df_trace_crop.index), name="time")
    df_trace_crop = df_trace_crop.sort_index()

    # trim hightimes or DLC
    n = min(len(hightimes), len(df_DLC))
    hightimes = hightimes[:n]
    df_DLC = df_DLC.iloc[:n].copy()
    
    # make TTL hightimes the index of DLC and merge
    df_DLC.index = hightimes
    df_DLC = df_DLC.rename_axis('time')
    window_start, window_end = float(df_DLC.index.min()), float(df_DLC.index.max())

    # crop trace to DLC window
    df_trace_crop = df_trace_crop.loc[(df_trace_crop.index >= window_start) & (df_trace_crop.index <= window_end)]
    dlc_num = df_DLC.select_dtypes(include=[np.number]).copy()
    dlc_on_trace = (dlc_num.reindex(dlc_num.index.union(df_trace_crop.index)).sort_index().interpolate(method="index").reindex(df_trace_crop.index))
    df_merged = df_trace_crop.join(dlc_on_trace, how="left")

    return df_merged

def dff(df, filename, del_values=True):

    def denoise(trace, sampling_rate):

        # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f) with zero-phase filter that changes amplitude
        b, a          = scipy.signal.butter(2, 10, btype='low', fs=sampling_rate)
        trace_preproc = scipy.signal.filtfilt(b, a, trace)

        return trace_preproc

    def detrend(trace_preproc):

        def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
                    # fits a double-exponential function to trace to correct for all possible bleaching reasons
                    '''Compute a double exponential function with constant offset.
                    t               : Time vector in seconds.
                    const           : Amplitude of the constant offset. 
                    amp_fast        : Amplitude of the fast component.  
                    amp_slow        : Amplitude of the slow component.  
                    tau_slow        : Time constant of slow component in seconds.
                    tau_multiplier  : Time constant of fast component relative to slow. '''
                    
                    tau_fast = tau_slow * tau_multiplier
                    return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)
                
        def get_parameters(time_sec, trace):
            max_sig = np.max(trace)
            initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1] # initial parmeters
            bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
            trace_params, parm_conv = scipy.optimize.curve_fit(double_exponential, time_sec, trace, p0=initial_params, 
                                                        bounds=bounds, maxfev=1000)
            trace_expfit = double_exponential(time_sec, *trace_params)

            return trace_expfit
        
        expofit = get_parameters(time_sec, trace_preproc)
        trace_preproc_detrend = trace_preproc / expofit

        return trace_preproc_detrend


    fluo_raw = df['Fluo'].to_numpy(dtype=float)
    isos_raw = df['Isos'].to_numpy(dtype=float)
    time_sec = df.index.to_numpy(dtype=float)
    sampling_rate = (len(time_sec) - 1) / (time_sec[-1] - time_sec[0])

    # preprocessing
    fluo_preproc = denoise(fluo_raw, sampling_rate)
    isos_preproc = denoise(isos_raw, sampling_rate)

    # detrend
    fluo_detrend = detrend(fluo_preproc)
    isos_detrend = detrend(isos_preproc)

    # Motion
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=isos_detrend, y=fluo_detrend)
    fluo_est_motion = slope * isos_detrend + intercept

    # Normalization
    df['dff']    = (fluo_detrend - fluo_est_motion) / fluo_est_motion * 100

    # if there are clear issues
    if del_values:
        exclusions = {  '691': (111.0, 112.5),
                        '701': (130.3, 130.5),
                        '407': (41.4, 41.6)}
        for animal, (start, stop) in exclusions.items():
            if animal in str(filename):
                df.loc[start:stop, 'dff'] = np.nan

    # Pandas excludes NaNs from these calculations
    baseline     = df['dff'].mean()
    st_dev       = df['dff'].std(ddof=1)
    df['zscore'] = ((df['dff'] - baseline) / (st_dev if st_dev > 0 else np.nan))

    return df


def calc_correlate_Speed_Signal(df, file, bin_size_sec=5.0, draw=False): 

    # takes bins of certain lengths and for each bin, takes one datapoint of mean speed and one for mean zscore
    def draw_correlation(binned, r, p, slope, intercept, file):

        r, p = round(r, 3), round(p, 3)

        plt.figure()
        plt.scatter(binned["Speed"], binned["zscore"])
        x_vals = np.linspace(binned["Speed"].min(), binned["Speed"].max(), 100)
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals)

        plt.xlim(0, 14)
        plt.ylim(-1.5, 1.5)
        plt.xlabel("Mean Speed (per bin)")
        plt.ylabel("Mean z-score (per bin)")
        plt.title(f"{file}: r={r}, p={p}")
        plt.savefig(f'{file}.pdf', format='pdf')
        plt.show()
    
    bins = np.floor(df.index / bin_size_sec) * bin_size_sec
    df["time_bin"] = bins

    binned = (df.groupby("time_bin")[["Speed", "zscore"]].mean().dropna())
    r, p                 = scipy.stats.pearsonr(binned["Speed"], binned["zscore"])
    slope, intercept, *_ = scipy.stats.linregress(binned["Speed"], binned["zscore"])

    if draw:
        draw_correlation(binned, r, p, slope, intercept, file)

    return r, p

def calc_strong_speed_changes(df, delta_thresh=30):

    # detects strong acceleration and decelleration events
    def ignore_consecutive_events(event_idxs, interframe_interval_s=0.1):
        diffs = np.diff(event_idxs)
        keep_mask = np.insert(diffs > (interframe_interval_s + 1e-6), 0, True)
        return event_idxs[keep_mask]

    df['Accel'] = df['Speed'].diff() / df.index.to_series().diff()
    accel_idx = df.index[df["Accel"] > delta_thresh].to_numpy()
    decel_idx = df.index[df["Accel"] < -delta_thresh].to_numpy()

    accel_idx = ignore_consecutive_events(accel_idx)
    decel_idx = ignore_consecutive_events(decel_idx)

    return df, accel_idx, decel_idx

def calc_speed_binning(df, rest=(0,0.7), slow=(0.7,5), move=(5,15), fast=(15,200)):

    # checks the mean zscores in rest and move bins (defined by the speed)
    rest_low, rest_high = rest
    slow_low, slow_high = slow
    move_low, move_high = move
    fast_low, fast_high = fast

    rest_mask = (df["Speed"] >= rest_low) & (df["Speed"] < rest_high)
    slow_mask = (df["Speed"] >= slow_low) & (df["Speed"] < slow_high)
    move_mask = (df["Speed"] >= move_low) & (df["Speed"] < move_high)
    fast_mask = (df["Speed"] >= fast_low) & (df["Speed"] < fast_high)
    
    rest_zmean = df['zscore'].loc[rest_mask].mean()
    slow_zmean = df['zscore'].loc[slow_mask].mean()
    move_zmean = df['zscore'].loc[move_mask].mean()
    fast_zmean = df['zscore'].loc[fast_mask].mean()

    # add them to the main_df
    df['rest_mask'] = rest_mask
    df['slow_mask'] = slow_mask
    df['move_mask'] = move_mask
    df['fast_mask'] = fast_mask

    return df, rest_zmean, slow_zmean, move_zmean, fast_zmean

def calc_speed_border_crossings(df, border=1.0, pre_min_s=0.1, post_min_s=0.5, fps=100, max_gap_fr=1, post_peak_speed=10, peak_window_s=1):

    # gets the events where cerain speeds are crossed
    def runs_from_bool(mask):
        # Find continuous True segments ("runs") in a boolean series (positional indices 0..n-1)
        m = mask.fillna(False).to_numpy(dtype=bool); n = len(m)
        if n == 0: return pd.DataFrame(columns=["start_pos","end_pos","length"])
        # pad with False so we can detect edges: False->True (start) and True->False (end)
        padded = np.r_[False, m, False]
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])                 # True starts where prev was False
        ends = np.flatnonzero(~padded[1:] & padded[:-1]) - 1              # True ends where next is False (inclusive)
        return pd.DataFrame({"start_pos":starts,"end_pos":ends,"length":ends-starts+1})

    speed = df["Speed"]; dt = 1/float(fps)
    pree_min_len = int(np.ceil(pre_min_s/dt))
    post_min_len = int(np.ceil(post_min_s/dt))
    peak_window_len = None if peak_window_s is None else int(np.ceil(peak_window_s/dt))

    pree_mask = speed < border
    post_mask = speed >= border

    # Convert masks into runs and keep only long-enough runs
    pree_runs = runs_from_bool(pree_mask)
    post_runs = runs_from_bool(post_mask)
    pree_runs = pree_runs[pree_runs["length"] >= pree_min_len].reset_index(drop=True)
    post_runs = post_runs[post_runs["length"] >= post_min_len].reset_index(drop=True)

    transitions = []
    post_ptr = 0

    # For each qualifying pre-run, find the first qualifying post-run that starts after it ends
    for i in range(len(pree_runs)):

        pree_start = int(pree_runs['start_pos'].to_numpy(dtype=int)[i])
        pree_end   = int(pree_runs['end_pos'].to_numpy(dtype=int)[i])

        # advance post pointer until post_start is after pre_end (so it is a "following" run)
        while post_ptr < len(post_runs) and int(post_runs['start_pos'].to_numpy(dtype=int)[post_ptr]) <= pree_end: 
            
            post_ptr += 1
        if post_ptr >= len(post_runs):
            break

        post_start = int(post_runs['start_pos'].to_numpy(dtype=int)[post_ptr])
        post_end = int(post_runs['end_pos'].to_numpy(dtype=int)[post_ptr])

        # gap (frames) between the runs; if you want "almost immediate crossing", keep this small
        gap = post_start - pree_end - 1
        if max_gap_fr is not None and gap > max_gap_fr:
            continue

        # Optional peak criterion: during the post period, speed must reach post_peak_speed (otherwise ignore)
        # You can check either: full post-run OR only first peak_window_s seconds of the post-run
        if post_peak_speed is not None:
            window_end = post_end if peak_window_len is None else post_end + peak_window_len
            peak = speed.iloc[post_start:window_end+1].max()
            if not np.isfinite(peak) or peak < post_peak_speed: 
                post_ptr += 1
                continue

        transitions.append({
            "pre_start": pree_start, 
            "pre_end": pree_end, 
            "post_start": post_start, 
            "post_end": post_end,
            "transit_pos": post_start, 
            "gap_fr": gap,
            "pre_duration_s": (pree_end-pree_start+1)*dt, 
            "post_duration_s": (post_end-post_start+1)*dt,
            "border": border, 
        })

        # consume this post-run so it can’t be reused for the next pre-run (same behavior as your code)
        post_ptr += 1

    df_transit = pd.DataFrame(transitions)
    if df_transit.empty: 
        return [], df_transit

    # Map positional indices back to df.index values (frame numbers or timestamps)
    df_transit["transition_index"] = df.index[df_transit["transit_pos"].to_numpy()]
    transit_idx = list(df_transit["transition_index"])
    return transit_idx, df_transit

def calc_peri_event(df, transit_idx, pre_s=2.0, post_s=5.0, fps=100, baseline=(-1,-0.1), draw=False):

    # calculate the peri-event response for the speed border crossings
    def plot_aligned_windows(t, perievents, perievents_mean):

        plt.figure()
        for i in range(perievents.shape[0]): 
            plt.plot(t, perievents[i], alpha=0.25, linewidth=1, color='gray')
        plt.plot(t, perievents_mean, linewidth=2, color='black')
        plt.axvline(0, linestyle="-", linewidth=1)
        plt.axhline(0, linestyle="-", linewidth=1)
        plt.xlabel("Time from crossing (s)")
        plt.ylabel("zscore")
        plt.tight_layout()
        plt.show()

    n = len(df)
    pre_samples = int(round(pre_s*fps))
    post_samples = int(round(post_s*fps))
    window_samples = pre_samples + post_samples + 1

    # Relative time axis
    t = (np.arange(window_samples) - pre_samples) / float(fps)

    positions = df.index.get_indexer(np.asarray(transit_idx))
    positions = positions[positions >= 0]

    zscore = df['zscore'].to_numpy()
    valid_trials = []

    for posi in positions:

        # get window start and stopp
        start = posi - pre_samples
        stopp = posi + post_samples + 1

        if start < 0 or stopp > len(df):
            continue
        
        trial = zscore[start:stopp].copy()

        # do baseline correction
        baseline_start, baseline_end = baseline
        baseline_mask = ((t >= float(baseline_start)) & (t <= float(baseline_end)))
        baseline_values = trial[baseline_mask]
        if (baseline_values.size == 0 or np.all(np.isnan(baseline_values))):
            continue
        baseline_value = np.nanmean(baseline_values)
        trial = trial - baseline_value

        valid_trials.append(trial)

    # No valid events
    if len(valid_trials) == 0:
        perievents = np.empty((0, window_samples))
        perievents_mean = pd.Series(np.nan, index=t, name="peri_event_mean")
        return t, perievents_mean

    perievents = np.vstack(valid_trials)
    perievents_mean = pd.Series(np.nanmean(perievents, axis=0), index=t, name="peri_event_mean")

    if draw:
        plot_aligned_windows(t, perievents, perievents_mean.to_numpy())

    return t, perievents_mean


def create_video(df, z_window=5, flip_x=True):
    '''
    Creates a video that shows the DLC animals and the FF zscore signal 
    '''
    bps_all =  ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    
    df = df.iloc[::10]
    time = df.index.to_numpy()
    n_frames = len(time)
    
    # get bodypart coordinates
    x_cols, y_cols = [], []
    for bp in bps_all:
        xcol, ycol = f'{bp}_x', f'{bp}_y'
        x_cols.append(xcol)
        y_cols.append(ycol)
    
    # get values
    X = df[x_cols].to_numpy(dtype=float)
    Y = df[y_cols].to_numpy(dtype=float)
    Z = df['zscore'].to_numpy(dtype=float)

    if flip_x:
        xmin = np.nanmin(X)
        xmax = np.nanmax(X)
        X = xmin + xmax - X

    # intitiate plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
    scat = ax.scatter([], [], s=10)

    # inset for signal
    inset = ax.inset_axes((0.6, 0.7, 0.35, 0.25))
    inset.plot(time, Z, linewidth=0.7, color='black')
    vline = inset.axvline(time[0], linewidth=1)
    inset.set_ylim(np.nanmin(Z), np.nanmax(Z))
    inset.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    def update(i):
        pts = np.column_stack([X[i], Y[i]])
        scat.set_offsets(pts)
        current_time = time[i]
        inset.set_xlim(current_time - z_window, current_time + z_window)
        vline.set_xdata([current_time, current_time])
        return scat, vline

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/30, blit=True)
    writer = FFMpegWriter(fps=30)
    anim.save('sicko.mp4', writer=writer)
    plt.close(fig)

def plot_ttest(results_df, column, y_title, title):

    plt.figure()

    for gi, group in enumerate(groups):
        sub = results_df[results_df["group"] == group][column].dropna()

        # mean ± SEM overlay
        mean = sub.mean()
        sem = sub.sem()
        plt.errorbar(gi, mean, yerr=sem, fmt="-", capsize=10)
        plt.hlines(mean, gi-0.1, gi+0.1, colors="black", linewidth=4)

        # scatter
        x = np.random.normal(loc=gi, scale=0.1, size=len(sub))
        plt.scatter(x, sub)

    plt.xticks(range(len(groups)), groups)
    plt.ylabel(y_title)

    # t-test
    a = results_df.loc[results_df["group"] == 'mKate', column].dropna().to_numpy()
    b = results_df.loc[results_df["group"] == 'A53T', column].dropna().to_numpy()
    t_stat, p_val = scipy.stats.ttest_ind(a, b, equal_var=False)
    ax = plt.gca()
    ax.text(0.02, 0.98, f'p = {round(p_val, 3)}', transform=ax.transAxes, va="top")

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_xy(df):
    plt.plot(df['Speed'], linewidth=1, color='purple')
    plt.plot(df['rest_mask']*-3, linewidth=1, color='green')
    plt.plot(df['move_mask']*-1, linewidth=1, color='orange')
    plt.plot(df['zscore']+15, linewidth=1.0, color='turquoise')
    #plt.vlines(idx, 0, 100, colors='red', linewidth=0.5)
    plt.xlim(70,90)
    plt.ylim(-2,20)
    plt.show()


groups  = ["mKate", "A53T", "GFP"]
results = []

df_distance_cum  = pd.DataFrame()
df_perievents    = pd.DataFrame()


files = get_files(path, '.doric')

for i, file_doric in enumerate(files):

    file_DLC = str(file_doric).replace('.doric', '_DLC.csv')
    file_short = manage_filename(file_doric)
    print(f'----- {file_short} -----')

    # extract data
    df_signal, io_hightimes = get_Doric(file_doric)
    df_DLC                  = get_DLC(file_DLC)
    main_df                 = merge_Doric_DLC(df_signal, df_DLC, io_hightimes)
    main_df                 = dff(main_df, file_short)

    # calculate stuff
    pearson_r, pearson_p                    = calc_correlate_Speed_Signal(main_df, file_short)
    main_df, accel_idx, decel_idx           = calc_strong_speed_changes(main_df)
    main_df, z_rest, z_slow, z_move, z_fast = calc_speed_binning(main_df)
    transit_idx, df_transit                 = calc_speed_border_crossings(main_df)
    t, perievents_mean                      = calc_peri_event(main_df, transit_idx)

    # add stuff to results table
    distance_cum = main_df['Distance_cum'].dropna()
    group = next((g for g in groups if g in file_short), "Other")
    results.append({'file':file_short, 
                    'group':group, 
                    'pearson_r':pearson_r, 
                    'pearson_p':pearson_p, 
                    'Dist':distance_cum.iloc[-1] - distance_cum.iloc[0], 
                    'MeanSpeed':main_df['Speed'].mean(),
                    'z_rest':z_rest, 
                    'z_slow':z_slow,
                    'z_move':z_move,
                    'z_fast':z_fast,
                    'duration_rest': main_df['rest_mask'].sum()/100,
                    'duration_slow': main_df['slow_mask'].sum()/100,
                    'duration_move': main_df['move_mask'].sum()/100,
                    'duration_fast': main_df['fast_mask'].sum()/100})
    
    df_distance_cum[file_short] = main_df["Distance_cum"].reset_index(drop=True)
    df_perievents[file_short]   = perievents_mean



results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["group"], ascending=True)

if True:
    results_df.to_csv(     path + '\\Results_df.csv')
    df_distance_cum.to_csv(path + '\\Distance_cum.csv')
    df_perievents.to_csv(  path + '\\Perievents.csv')

if True:
    plot_ttest(results_df, 'pearson_r', 'Pearson R (Speed vs zscore)', 'Correlation Speed vs Z-Score')
    plot_ttest(results_df, 'z_rest', 'mean zscore', 'Rest')
    plot_ttest(results_df, 'z_move', 'mean zscore', 'Move')
