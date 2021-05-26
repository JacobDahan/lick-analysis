import pandas as pd
import numpy as np
import os
from csv import writer

ack_token = 'ACK'
release_trial_token = 'RELEASE_TRL'
trial_released_token = 'TRL_RELEASED'
start_trial_token = 'TRL_START'
trial_param_token = 'TRLP'
trial_result_token = 'TRLR'
trial_window_open_command = 'ST_CHG2'
trial_window_open_arg = '5 7'
trial_lick_command = 'TCH'
trial_reward_token = 'EV'
trial_outcome_token = 'OUTC'
manual_reward_token = 'AAR'

# dictionary for actions
# this must match the arduino code
LEFT   = 1
LEFT2  = 4033
RIGHT  = 2
RIGHT2 = 4034
NOGO   = 3

# dictionary for outcomes
# this assumes two-alternative choice
HIT = 1
ERROR = 2
SPOIL = 3

# for boolean parameters
YES = 3
NO = 2
MD = 0 # "must-define"

def nan_pad(listr, listl):
    """Sort lists by time such that each row contains only one lick time 
    and one np.nan value.
    """
    list_len            = [len(i) for i in [listr, listl]]
    target_len          = sum(list_len)
    lista               = sorted(listr + listl)
    sorted_listr        = np.full(target_len, np.nan)
    sorted_listl        = np.full(target_len, np.nan)
    rinds               = [i for i, val in enumerate(lista) if val in listr] 
    linds               = [i for i, val in enumerate(lista) if val in listl] 
    sorted_listr[rinds] = listr
    sorted_listl[linds] = listl
    return sorted_listr, sorted_listl

def find_rwside(parsed_lines):
    """Extract reward side"""
    args   = parsed_lines[parsed_lines['command'] == trial_param_token]['argument'].to_list()
    rwside = [s for i, s in enumerate(args) if 'RWSD' in s][0]
    rwside = 'LEFT' if str(LEFT) in rwside else 'RIGHT'
    return rwside

def stim2stim(stim):
    """Turns stim code number into stim string."""
    if '100' in stim:
        stim = 'ROUGH'
    elif '150' in stim:
        stim = 'ROUGH'
    elif '199' in stim:
        stim = 'ROUGH'
    elif '50' in stim:
        stim = 'SMOOTH'
    return stim

def extract(parsed_lines, trial, mouse, date):
    """Extract data from parsed lines df"""
    rec = {}

    # Trial timings (ms)
    w_open   = int(parsed_lines[(parsed_lines["command"] == trial_window_open_command) & (parsed_lines['argument'] == trial_window_open_arg)]['time'])
    lt_right = parsed_lines[(parsed_lines["command"] == trial_lick_command) & ((parsed_lines['argument'] == str(RIGHT)) | (parsed_lines['argument'] == str(RIGHT2)))]['time'].to_list()
    lt_left  = parsed_lines[(parsed_lines["command"] == trial_lick_command) & ((parsed_lines['argument'] == str(LEFT)) | (parsed_lines['argument'] == str(LEFT2)))]['time'].to_list()
    lt_all   = parsed_lines[(parsed_lines["command"] == trial_lick_command) & ((parsed_lines['argument'] == str(RIGHT)) | (parsed_lines['argument'] == str(LEFT)) | (parsed_lines['argument'] == str(RIGHT2)) | (parsed_lines['argument'] == str(LEFT2)))]['time'].to_list()

    # Response window
    # res_right = parsed_lines[(parsed_lines["command"] == trial_window_open_command) & (parsed_lines['argument'] == '7 9')]['time'].to_list()
    # res_left  = parsed_lines[(parsed_lines["command"] == trial_window_open_command) & (parsed_lines['argument'] == '7 8')]['time'].to_list()
    # res_wrong = parsed_lines[(parsed_lines["command"] == trial_window_open_command) & (parsed_lines['argument'] == '7 14')]['time'].to_list()

    # Reward side
    # rwside   = find_rwside(parsed_lines)
    # lt_left  = lt_left + res_left + res_wrong if rwside == 'RIGHT' else lt_left + res_left
    # lt_right = lt_right + res_right + res_wrong if rwside == 'LEFT' else lt_right + res_right
    lt_right.sort()
    lt_left.sort()

    # Stimulus
    stim = parsed_lines[parsed_lines["command"] == trial_param_token]['argument'].to_list()
    idx  = [i for i, s in enumerate(stim) if 'STPPOS' in s][0]
    stim = stim[idx]
    stim = stim2stim(stim)

    # Zero-correct trial timings
    lt_right_corr = [(t-w_open) for t in lt_right]
    lt_left_corr  = [(t-w_open) for t in lt_left]   

    # Pad with NaN
    lt_right_c_pad, lt_left_c_pad = nan_pad(lt_right_corr, lt_left_corr)

    # Trial outcome
    outcome = parsed_lines[(parsed_lines["command"] == trial_result_token) & (parsed_lines['argument'] == 'OUTC 2')]['argument'].to_list()
    outcome = 1 if len(outcome) == 0 else 0

    # Laser status
    opto = parsed_lines[parsed_lines["command"] == trial_param_token]['argument'].to_list()
    idx  = [i for i, o in enumerate(opto) if 'OPTO' in o][0]
    opto = opto[idx]
    opto = ''.join(['ON' if '3' in opto else 'OFF'])

    # Fill rec
    rec['right']    = lt_right_c_pad
    rec['left']     = lt_left_c_pad
    rec['opto']     = [opto for i in lt_all]
    rec['trial']    = [trial for i in lt_all]
    rec['outcome']  = [outcome for i in lt_all]
    rec['stim']     = [stim for i in lt_all]
    rec['mouse']    = [mouse for i in lt_all]
    rec['date']     = [date for i in lt_all]
    rec['raw_time'] = [i for i in lt_all]

    return rec

def extractLT(parsed_lines, trial, mouse, date):
    """Extract data from parsed lines df.
    Used for LT analysis."""
    rec = {}

    # Trial timings (ms)
    trl_start = int(parsed_lines[(parsed_lines["command"] == trial_result_token) & (parsed_lines['argument'].str.contains(trial_outcome_token))]['time'])
    lt_right  = parsed_lines[(parsed_lines["command"] == trial_lick_command) & ((parsed_lines['argument'] == str(RIGHT)) | (parsed_lines['argument'] == str(RIGHT2)))]['time'].to_list()
    lt_left   = parsed_lines[(parsed_lines["command"] == trial_lick_command) & ((parsed_lines['argument'] == str(LEFT)) | (parsed_lines['argument'] == str(LEFT2)))]['time'].to_list()
    lt_all    = parsed_lines[(parsed_lines["command"] == trial_lick_command) & ((parsed_lines['argument'] == str(RIGHT)) | (parsed_lines['argument'] == str(LEFT)) | (parsed_lines['argument'] == str(RIGHT2)) | (parsed_lines['argument'] == str(LEFT2)))]['time'].to_list()

    lt_right.sort()
    lt_left.sort()

    # Reward side
    # Take reward side of last trial included in time window.
    # This way, nogo trial reward sides are discarded in favor of the reward side
    # on which the mouse was rewarded.
    # NOGO trials are always followed by same trial type, so any RWSD argument works.
    rwsd = parsed_lines[(parsed_lines["command"] == trial_param_token) & (parsed_lines["argument"].str.contains("RWSD"))]['argument'].to_list()[-1]
    rwsd = "ROUGH" if "1" in rwsd else "SMOOTH"

    # Zero-correct trial timings
    lt_right_corr = [(t-trl_start) for t in lt_right]
    lt_left_corr  = [(t-trl_start) for t in lt_left]   

    # Pad with NaN
    lt_right_c_pad, lt_left_c_pad = nan_pad(lt_right_corr, lt_left_corr)

    # Fill rec
    rec['right']    = lt_right_c_pad
    rec['left']     = lt_left_c_pad
    rec['trial']    = [trial for i in lt_all]
    rec['rwsd']     = [rwsd for i in lt_all]
    rec['mouse']    = [mouse for i in lt_all]
    rec['date']     = [date for i in lt_all]
    rec['raw_time'] = [i for i in lt_all]

    return rec

def parse_lines_into_df(lines):
    """Parse every line into time, command, and argument.
    
    Consider replacing this with read_logfile_into_df
    
    In trial speak, each line has the same format: the time in milliseconds,
    space, a string command, space, an optional argument. This function parses
    each line into those three components and returns as a dataframe.
    """
    # Split each line
    rec_l = []
    for line in lines:
        sp_line = line.split()
        
        # Skip anything with no strings or without a time argument first
        try:
            int(sp_line[0])
        except (IndexError, ValueError):
            continue
        
        # If longer than 3, join the 3rd to the end into a single argument
        if len(sp_line) > 3:
            sp_line = [sp_line[0], sp_line[1], ' '.join(sp_line[2:])]
        rec_l.append(sp_line)
    
    # DataFrame it and convert string times to integer
    if len(rec_l) == 0:
        raise ValueError("cannot extract any lines")
    df = pd.DataFrame(rec_l, columns=['time', 'command', 'argument'])
    df['time'] = df['time'].astype(np.int)
    return df

def read_lines_from_file(log_filename):
    """Reads all lines from file and returns as list"""
    with open(log_filename) as fi:
        lines = fi.readlines()
    return lines

def split_by_trial(lines, LT):
    """Splits lines from logfile into list of lists by trial.

    Returns: splines, a list of list of lines, each beginning with
    TRIAL START (which is when the current trial params are determined).
    """
    if len(lines) == 0:
        return [[]]
    
    # Save setup info
    trial_starts = [0]

    # Remove 2990 and no-command events
    # What are these? Would be more efficient to run in one loop for all events
    lines = [line for line in lines if (len(line.split())>1 and line.split()[0] != '2990')]

    # FA and normal files
    if not LT:
        # Find start indices
        trial_s      = [i for i, x in enumerate(lines) if x.split()[1] == start_trial_token]
        # Find end indices
        trial_ends   = [i for i, x in enumerate(lines) if x.split()[1] == trial_released_token]
        # Include zero index
        trial_starts = trial_starts + trial_s
        # Find nogo trial indices (lines with OUTC 3)
        nogos        = set([x for i, x in enumerate(lines) if (x.split()[1] == trial_result_token and trial_outcome_token in x.split()[2] and str(NOGO) in x.split()[3])])
    # LT only
    else:
        # Find start indices (start defined as OUTC of previous trial [n-1])
        # Water rewards are given ~500ms prior to OUTC printing
        # Measuring from OUTC discards all immediate post-reward (non-learning)
        # type licks and appreciates quick (pre-trial-change) behavioral changes
        trial_starts = [i for i, x in enumerate(lines) if (x.split()[1] == trial_result_token and trial_outcome_token in x.split()[2])]
        # Find end indices (end defined as EV reward of current trial [n])
        # Ignore manual water rewards (demarcated EV AAR_R/L)
        trial_ends   = [i for i, x in enumerate(lines) if (x.split()[1] == trial_reward_token and 'AA' not in x.split()[2])]        
        # Find nogo trial indices (lines with OUTC 3)
        nogos        = [i for i, x in enumerate(lines) if (x.split()[1] == trial_result_token and trial_outcome_token in x.split()[2] and str(NOGO) in x.split()[3])]
        # Simply replace all NOGO trial entries (OUTC 3) with DBG entries
        # Result will be an outlier insignifanct when using median measurements
        # Transforms output from EV --> OUTC 3 (nogo) --> OUTC 1 (correct) to 
        # EV --> OUTC 1 (and can remove consecutive OUTC 3 entries)
        # This works because EV is not an output in NOGO trials
        # Sort to restore trial order (sets are unordered iterables in Python)
        for nogo in nogos:
            curr_line          = lines[nogo]
            replacement_line   = curr_line.split()[0] + ' DBG\n'
            lines[nogo]        = replacement_line
        trial_starts_corrected = list(set(trial_starts).symmetric_difference(nogos))
        trial_starts           = trial_starts_corrected
        trial_starts.sort()
        # Remove first trial start (OUTC) (always right trial)
        trial_starts = trial_starts[1:]
        # Remove first two trial ends (EV) (always right trial)
        trial_ends   = trial_ends[2:]
        # Remove last trial start (OUTC) (unmatched; no next EV)
        trial_starts = trial_starts[:-1]

    # If unfinished last trial
    if len(trial_starts) > len(trial_ends):
        last_chunk = lines[trial_starts[-1]:]
        # If outcome recorded, add outcome as trial end
        if len(list(filter(lambda x: trial_outcome_token in x, last_chunk))) > 0:
            outc = list(filter(lambda x: trial_outcome_token in x, last_chunk))[-1]
            trial_ends.append(lines.index(outc))
        # If no outcome, remove last trial
        else:
            trial_starts = trial_starts[:(len(trial_starts)-1)]

    # Find commands designating manual water rewards
    man_rewards = set([x for i, x in enumerate(lines) if (x.split()[1] == trial_reward_token and manual_reward_token in x.split()[2])])

    # Now iterate over trial_starts and append the chunks
    splines = []

    # Initiate nogo counters
    on_nogos  = 0
    off_nogos = 0

    for i in range(len(trial_starts)):
        # check if manual reward in trial
        # if manual reward found in trial lines, do not append trial
        spline = lines[trial_starts[i]:trial_ends[i]+1] if not man_rewards.intersection(set(lines[trial_starts[i]:trial_ends[i]+1])) else None
        # if nogo trial, do not append trial (already discarded LT nogos)
        if not LT and spline is not None:
            parsed_lines = parse_lines_into_df(spline)
            spline       = spline if not nogos.intersection(set(lines[trial_starts[i]:trial_ends[i]+1])) else None
            if spline is None:
                # record laser status for nogo trials only if random trial
                is_rand = parsed_lines[(parsed_lines['command'] == trial_param_token) & (parsed_lines['argument'].str.contains('ISRND'))]['argument'].item()
                is_rand = True if str(YES) in is_rand else False
                if is_rand:
                    opto = parsed_lines[parsed_lines["command"] == trial_param_token]['argument'].to_list()
                    idx  = [i for i, o in enumerate(opto) if 'OPTO' in o][0]
                    opto = opto[idx]
                    opto = 'ON' if '3' in opto else 'OFF'
                    if opto is 'OFF':
                        off_nogos += 1
                    else:
                        on_nogos += 1
        if spline is not None: 
            splines.append(spline)

    # return lines split by trial and number of nogo trials by laser status
    return splines, on_nogos, off_nogos

def make_trials_info_from_splines(lines_split_by_trial,date,mouse,LT,FA,opto,on_nogos,off_nogos):
    """Parse out the parameters and outcomes from the lines in the logfile
    
    For each trial, the following parameters are extracted:
        trial_start : time in seconds at which TRL_START was issued
        trial_released: time in seconds at which trial was released
        All parameters listed for each trial.

    The first entry in lines_split_by_trial is taken to be info about setup,
    not about the first trial.
    
    Behavior depends on the length of lines_split_by_trial:
        0:  it is empty, so None is returned
        1:  only setup info, so None is returned
        >1: the first entry is ignored, and subsequent entries become
            rows in the resulting DataFrame.

    If a DataFrame is returned, then the columns in always_insert are
    always inserted, even if they weren't present. They will be inserted
    with np.nan, so the dtype should be numerical, not stringy.
    
    The main use-case is the response columns which are missing during the
    first trial but which most code assumes exists.

    If LT is True, commands are slightly different.
    """
    if len(lines_split_by_trial) < 1:
        return None

    rec_l = []
    trial = 0

    if not LT:
        for spline in lines_split_by_trial[1:]:
            # Record trial number
            trial += 1
            
            # Parse into time, cmd, argument with helper function
            parsed_lines = parse_lines_into_df(spline)
            
            is_rand = parsed_lines[(parsed_lines['command'] == trial_param_token) & (parsed_lines['argument'].str.contains('ISRND'))]['argument'].item()
            is_rand = True if str(YES) in is_rand else False
            
            # Discard random trials
            if not is_rand and not FA:
                continue

            cols = ['mouse','date','trial','left','right','outcome','stim','opto','raw_time']
            rec  = extract(parsed_lines, trial, mouse, date)
            rec  = pd.DataFrame.from_records(rec, columns=cols)

            # Remove electrical noise 
            for idx, row in rec.iterrows():
                if idx == 0:
                    prev_time = row.raw_time
                    continue
                time = row.raw_time
                ILI  = time - prev_time
                if ILI <= 75:
                    rec.drop(idx, inplace=True)
                    continue
                prev_time = time

            # Append results
            rec_l.append(rec)
    
    # If LT
    else:

        # Remove empty final trial
        # Only occurs if behavior stopped between OUTC and EV outputs
        lines_split_by_trial = list(filter(None,lines_split_by_trial))

        for spline in lines_split_by_trial:


            # Record trial number
            trial += 1
            
            # Parse into time, cmd, argument with helper function
            parsed_lines = parse_lines_into_df(spline)

            cols = ['mouse','date','trial','left','right','rwsd','raw_time']
            rec  = extractLT(parsed_lines, trial, mouse, date)
            rec  = pd.DataFrame.from_records(rec, columns=cols)

            # Remove electrical noise
            for idx, row in rec.iterrows():
                if idx == 0:
                    prev_time = row.raw_time
                    continue
                time = row.raw_time
                ILI  = time - prev_time
                if ILI <= 75:
                    rec.drop(idx, inplace=True)
                    continue
                prev_time = time

            # Append results
            rec_l.append(rec)
    # DataFrame
    trials_info = pd.concat(rec_l, ignore_index=True)

    if np.isfinite(opto):
        # total trial count
        trials      = len(trials_info.trial.unique())
        # combine left and right licks
        tempDf      = pd.DataFrame({'left':trials_info.left, 'right':trials_info.right})  
        licksList   = tempDf.stack().groupby(level=0).sum()
        trials_info = trials_info.assign(licks=licksList)
        # split by laser stats
        onDf        = trials_info.groupby('opto').get_group('ON')
        offDf       = trials_info.groupby('opto').get_group('OFF')
        # total trial count by laser status
        on_trials   = len(onDf.trial.unique())
        off_trials  = len(offDf.trial.unique())
        # only consider reward window
        onDf        = onDf[onDf.licks >= 0]
        offDf       = offDf[offDf.licks >= 0]
        # first licks in reward window
        first_licks_on  = onDf.sort_values(by=['raw_time'],ascending=[True]).groupby('trial').nth(0)
        first_licks_off = offDf.sort_values(by=['raw_time'],ascending=[True]).groupby('trial').nth(0)
        # discard trials where first lick is after opto threshold
        discard_trials     = []
        on_post_n_trials   = first_licks_on[(first_licks_on.licks >= opto * 1000) & (first_licks_on.licks < 45 * 1000)].index
        off_post_n_trials  = first_licks_off[(first_licks_off.licks >= opto * 1000) & (first_licks_off.licks < 45 * 1000)].index
        on_post_45_trials  = first_licks_on[first_licks_on.licks >= 45 * 1000].index
        off_post_45_trials = first_licks_off[first_licks_off.licks >= 45 * 1000].index
        discard_trials.extend(on_post_n_trials)
        discard_trials.extend(off_post_n_trials)
        discard_trials.extend(on_post_45_trials)
        discard_trials.extend(off_post_45_trials)
        # drop discard trials and record performance
        # only if trials to discard
        if len(discard_trials) > 0:
            discard_rows          = trials_info.trial.isin(discard_trials).astype(int)
            discard_outcomes      = trials_info[discard_rows == True].groupby(['opto','trial']).nth(0)
            on_discard_correct    = 0 if 'ON' not in discard_outcomes.index else discard_outcomes.outcome.ON.sum()
            on_discard_incorrect  = 0 if 'ON' not in discard_outcomes.index else discard_outcomes.outcome.ON.isin([0]).sum() 
            off_discard_correct   = 0 if 'OFF' not in discard_outcomes.index else discard_outcomes.outcome.OFF.sum() 
            off_discard_incorrect = 0 if 'OFF' not in discard_outcomes.index else discard_outcomes.outcome.OFF.isin([0]).sum() 
            discard_outcome_list  = [mouse,date,on_discard_correct,on_discard_incorrect,off_discard_correct,off_discard_incorrect]
            trials_info           = trials_info[discard_rows == False]
        else:
            discard_outcome_list  = [mouse,date,0,0,0,0]
        # store trial type data as list
        trial_type_list = [mouse,date,trials,on_trials,off_trials,len(on_post_n_trials),len(off_post_n_trials),len(on_post_45_trials)+on_nogos,len(off_post_45_trials)+off_nogos]
        return trials_info, trial_type_list, discard_outcome_list
    
    return trials_info, None, None

def read(log_filename, date, mouse, LT=False, FA=False, opto=np.inf):
    """Reads all lines from file and returns as list    
    LT: lick train session (bool)
    FA: forced alternation session (bool)
    opto: time threshold within which first lick must occur (for opto sessions, use 3; else inf)"""

    # Read
    logfile_lines = read_lines_from_file(log_filename)
    
    # Spline
    lines_split_by_trial, on_nogos, off_nogos = split_by_trial(logfile_lines, LT)

    # Make matrix
    tm, trial_type_list, discard_outcome_list = make_trials_info_from_splines(lines_split_by_trial, date, mouse, LT, FA, opto, on_nogos, off_nogos)

    trial_out_csv = os.path.join(os.getcwd(),'trial_types.csv')

    with open(trial_out_csv, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add headers if file is empty
        if not os.path.getsize(trial_out_csv):
            csv_writer.writerow(['mouse','date','trials','on_trials','off_trials','on_post_n_trials','off_post_n_trials','on_post_45_trials','off_post_45_trials'])
        # Add contents of list as last row in the csv file
        if trial_type_list is not None:
            csv_writer.writerow(trial_type_list)

    # If running multiple behaviors, this over-writes
    discard_out_csv = os.path.join(os.getcwd(),'discard_outcomes.csv')

    with open(discard_out_csv, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add headers if file is empty
        if not os.path.getsize(discard_out_csv):
            csv_writer.writerow(['mouse','date','on_correct','on_incorrect','off_correct','off_incorrect'])
        # Add contents of list as last row in the csv file
        if discard_outcome_list is not None:
            csv_writer.writerow(discard_outcome_list)

    return tm