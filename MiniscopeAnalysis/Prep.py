#%%
import pandas as pd
import numpy as np
import csv

file_TTL = 'F:\\2024-10-31-16-52-47_CA1-m90_OF_GPIO.csv'
file_DLC = 'F:\\2024-10-31-16-52-47_CA1-m90_OF_DLC.csv'
file_traces = 'F:\\2024-10-31-16-52-47_CA1-m90_OF_traces.csv'
file_events = 'F:\\2024-10-31-16-52-47_CA1-m90_OF_traces_events.csv'

def get_traces(file, round_after_comma=1):
    # clean the csv file and return df
    df = pd.read_csv(file, low_memory=False)

    # drop rejected cells/columns
    rejected_columns = df.columns[df.isin([' rejected']).any()]
    df = df.drop(columns=rejected_columns)

    # Clean up df
    df = df.drop(0)
    df = df.astype(float)
    df = df.rename(columns = {' ': 'Time'})
    df['Time'] = df['Time'].round(1)
    df = df.set_index(['Time'])
    df.index.names = ['Time/dff']
    df.index = df.index.round(round_after_comma)

    df.columns = df.columns.str.replace(' C', '')
    df.columns = df.columns.astype(int)

    return df

def dff_2_zscore(df):
    # transform df/f into zscores
    df_z = df.copy()

    # transform into zscores
    df_z = df.copy()
    for name, data in df_z.items():
        df_z[name] = (df_z[name] - df_z[name].mean()) / df_z[name].std()
    df_z.index.names = ['Time/zscore']

    return df_z

def TTL_list(file, channel_name = ' GPIO-1', value_threshold=500, constant=True, print_csv=True):
    # returns list of TTL input times

    # expects nan values to be just a string: ' nan', convert these strings into proper nans
    df = pd.read_csv(file, na_values=[' nan'], low_memory=False)
    df = df.set_index('Time (s)')

    # only cares for the rows where the channel name is in the ' Channel Name' column
    df = df[df[' Channel Name'] == channel_name]
    
    # drops rows with nan and converts everything into ints
    df = df.dropna(subset=[' Value'])
    df[' Value'] = df[' Value'].astype(int)

    # creates series with True for values > threshold
    # checks at which index/time the row's value is high AND the row before's value is low
    # for that, it makes the whole series negative (True=False) and shifts the whole series one row further (.shift)
    # so, onset wherever row is > threshold and the opposite value of the shifted (1) row is positive as well
    is_high = df[' Value'] >= value_threshold
    onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]
    offsets = df.index[~is_high & is_high.shift(1, fill_value=False)]

    high_times = list(zip(onsets, offsets)) if not constant else list(onsets)
    #print(f'{len(high_times)} total events')

    # if print_csv:
    #     with open(output, 'w') as f:
    #         wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    #         wr.writerow(high_times)

    return high_times

def get_traces_events(file):
    # takes the csv with the time of events/action potentials for each cell and returns df

    df = pd.read_csv(file, low_memory=False)
   
    # group by ' Cell Name' and collect the 'Time (s)' as lists before adding to dict
    events = df.groupby(' Cell Name')['Time (s)'].apply(list).to_dict()

    # get the maximum events that happen in a cell
    max_events = max(len(event_list) for event_list in events.values())
    
    # to have the same length, add nans to the event list of each cell 
    for key, event_list in events.items():
        events[key] = event_list + [np.nan] * (max_events - len(event_list))
        
    # make a df out of the dict and clean up the cell names
    df = pd.DataFrame(events)
    df.columns = sorted(df.columns.str.replace(' C', '').astype(int))

    return df
    
def behav(file):
    # cleans the DLC file and returns the df
    # idea: instead of picking some 'trustworthy' points to mean e.g. head, check the relations between 'trustworthy' points, so when one is missing, it can gues more or less where it is

    bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    bps_head = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
    bps_postbody = ['mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base']

    df = pd.read_csv(file, header=None, low_memory=False)

    # combines the string of the two rows to create new header
    # df.iloc[0] takes all values of first row
    new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     
    df.columns = new_columns
    df = df.drop(labels=[0, 1, 2], axis="index")

    # adapt index
    df.set_index('bodyparts_coords', inplace=True)
    df.index.names = ['Frame']

    # turn all the prior string values into floats and index into int
    df = df.astype(float)
    df.index = df.index.astype(int)

    # insert NaN if likelihood is below x
    for bp in bps_all:
        filter = df[f'{bp}_likelihood'] <= 0.9     
        df.loc[filter, f'{bp}_x'] = np.nan
        df.loc[filter, f'{bp}_y'] = np.nan
        df = df.drop(labels=f'{bp}_likelihood', axis="columns")
    
    # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
    for c in ('_x', '_y'):
        df[f'head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
        df[f'postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

    # combine the x and y column into a single tuple column
    for bp in bps_all + ['head', 'postbody']:
        df[bp] = (list(zip(round(df[f'{bp}_x'], 1), round(df[f'{bp}_y'], 1))))
        df = df.drop(labels=f'{bp}_x', axis="columns")
        df = df.drop(labels=f'{bp}_y', axis="columns")   

    #df.to_csv('F:\\2024-10-31-16-52-47_CA1-m90_OF_DLC_clean.csv')
    return df[['head', 'postbody']]


traces = get_traces(file_traces)
events = get_traces_events(file_events)
TTLs = TTL_list(file_TTL)

# as an index for behavior, you have the frame number and the INSPX master time
behavior = behav(file_DLC)
behavior['Time [TTL]'] = TTLs[:-1]
behavior.set_index([behavior.index, 'Time [TTL]'], inplace=True)


#%%





