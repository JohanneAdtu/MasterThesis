import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import entropy
import re
import matplotlib.dates as mdates

#%% Functions

def load_npy_hypnogram(npy_file, epoch_sec=30):
    """
    Load a hypnogram from .npy file, returning DataFrame with stage codes and time index.
    If the array is probabilistic, assigns each epoch by maximum probability.
    """
    arr = np.load(npy_file)
    # If probabilities, use argmax
    if arr.ndim == 2 and arr.shape[1] == 4:
        stage_codes = np.argmax(arr, axis=1)
    else:
        stage_codes = arr
    # Extract reference date from file name for time index
    m = re.search(r"_night_(\d{4}-\d{2}-\d{2})", npy_file)
    if not m:
        raise ValueError("Could not parse date from hypnogram filename.")
    date_str = m.group(1)
    start_time = pd.Timestamp(f"{date_str} 21:00:00")
    times = [start_time + pd.Timedelta(seconds=epoch_sec * i) for i in range(len(stage_codes))]
    hyp_df = pd.DataFrame({"stage_code": stage_codes, "stage": [stage_names[c] for c in stage_codes]}, index=pd.to_datetime(times))
    return hyp_df

def load_h5_acc(filepath: str):
    """Load a single night from HDF5 into a pandas DataFrame.
    """
    with h5py.File(filepath, "r") as h5:
        acc = h5["data"]["accelerometry"][:]  # shape (3, N)
        fs = h5["data"]["accelerometry"].attrs["sample_frequency"]
        start_time = pd.to_datetime(h5.attrs["start_time"])
    acc_df = pd.DataFrame(acc.T, columns=["x", "y", "z"])
    time_index = pd.date_range(start=start_time, periods=acc_df.shape[0], freq=pd.to_timedelta(1/fs, unit="s"))
    acc_df.index = time_index
    return acc_df, fs

def night_exclusion_criteria(
    acc_df, 
    fs, 
    sleep_vector, 
    tst_min_hours=2, 
    tst_max_hours=12, 
    nonwear_thr_hours=3, 
    nonwear_window_start=21,
    nonwear_window_end=9,
    block_minutes=30, 
    std_thr_g=  0.02, 
    range_thr_g= 0.1):
    """
    Apply exclusion criteria: total sleep time (TST) bounds and non-wear time limits.
    Returns (exclude_flag, details_dict)
    """
    
    # --- 1. Calculate TST (total sleep time) in hours ---
    tst_sample_count = np.sum(sleep_vector)
    tst_hours = (tst_sample_count / fs) / 3600

    # --- 2. Detect nonwear in moving windows ---
    block_samples = int(block_minutes * 60 * fs)
    n_blocks = len(acc_df) // block_samples if block_samples > 0 else 0
    nonwear_mask = pd.Series(False, index=acc_df.index)
    for i in range(n_blocks):
        start_idx = i * block_samples
        end_idx = start_idx + block_samples
        block_df = acc_df.iloc[start_idx:end_idx]
        if block_df.empty:
            continue
        # Mask as nonwear if at least 2 axes are both low in std and range
        std_vals = block_df.std()
        range_vals = block_df.max() - block_df.min()
        std_count = np.sum(std_vals < std_thr_g)
        range_count = np.sum(range_vals < range_thr_g)
        if (std_count >= 2) or (range_count >= 2):
            nonwear_mask.iloc[start_idx:end_idx] = True

    # --- 3. Only consider nonwear in "main night" analysis window ---
    hours = acc_df.index.hour
    if nonwear_window_start <= nonwear_window_end:
        window_mask = (hours >= nonwear_window_start) & (hours < nonwear_window_end)
    else: # Window crosses midnight
        window_mask = (hours >= nonwear_window_start) | (hours < nonwear_window_end)
        
    nonwear_in_window_mask = nonwear_mask & window_mask
    nonwear_sample_count = np.sum(nonwear_in_window_mask)
    nonwear_hours_in_window = (nonwear_sample_count / fs) / 3600
    
    
    # --- 4. Exclude if low/high sleep or excess nonwear ---
    exclude = False
    reasons = []
    if tst_hours > tst_max_hours:
        exclude = True
        reasons.append(f"TST too long ({tst_hours:.2f}h > {tst_max_hours}h)")
    if tst_hours < tst_min_hours:
        exclude = True
        reasons.append(f"TST too short ({tst_hours:.2f}h < {tst_min_hours}h)")
    if nonwear_hours_in_window >= (nonwear_thr_hours - 1e-2):
        exclude = True
        reasons.append(f"Non-wear in window ≥ {nonwear_thr_hours}h ({nonwear_hours_in_window:.2f}h)")
    
    details = {"exclude": exclude, "tst_hours": tst_hours, "nonwear_hours_in_window": nonwear_hours_in_window, "reasons": "; ".join(reasons) if exclude else "OK"}
    
    return exclude, details

def compute_sleep_structure_features(hyp_df, epoch_sec=30):
    """
    Extract a range of sleep structure metrics from a hypnogram DataFrame.
    """
    stages = hyp_df['stage_code'].values
    total_epochs = len(stages)
    total_minutes = total_epochs * epoch_sec / 60
    tib_minutes = total_minutes

    # Sleep mask: N3, N1/N2, REM 
    sleep_mask = np.isin(stages, [0, 1, 2])
    sleep_indices = np.where(sleep_mask)[0]

    # Total Sleep Time (TST) in minutes
    tst_minutes = np.sum(sleep_mask) * epoch_sec / 60

    # Wake After Sleep Onset (WASO): minutes spent awake after sleep onset, before final awakening
    if len(sleep_indices) > 0:
        onset = sleep_indices[0]
        final_sleep = sleep_indices[-1]
        waso_mask = (stages[onset:final_sleep+1] == 3)
        waso_minutes = np.sum(waso_mask) * epoch_sec / 60
    else:
        waso_minutes = np.nan

    # Sleep effeciency: percent of night spent asleep
    sleep_efficiency = tst_minutes / tib_minutes if tib_minutes > 0 else np.nan

    # Stage proportions: percent of TST spent in each stage
    stage_counts = {s: np.sum(stages == s) for s in [0, 1, 2, 3]}
    percent_n3 = (stage_counts[0] * epoch_sec / 60) / tst_minutes * 100 if tst_minutes > 0 else np.nan
    percent_n1n2 = (stage_counts[1] * epoch_sec / 60) / tst_minutes * 100 if tst_minutes > 0 else np.nan
    percent_rem = (stage_counts[2] * epoch_sec / 60) / tst_minutes * 100 if tst_minutes > 0 else np.nan

    # Number of awakenings: transitions from sleep to wake
    awakenings = np.sum((stages[1:] == 3) & (np.isin(stages[:-1], [0, 1, 2])))

    # Number of stage transitions 
    transitions = np.diff(stages)
    num_transitions = np.sum(transitions != 0)

    # Longest continuous sleep bout (in minutes)
    if len(sleep_indices) == 0:
        longest_sleep_bout_min = np.nan
    else:
        split_points = np.where(np.diff(sleep_indices) != 1)[0] + 1
        sleep_bouts = np.split(sleep_indices, split_points)
        sleep_bout_lengths = [len(b) for b in sleep_bouts]
        longest_sleep_bout_min = max(sleep_bout_lengths) * epoch_sec / 60 if len(sleep_bout_lengths) > 0 else np.nan

    # REM latency: minutes from first sleep epoch to first REM after that
    if len(sleep_indices) > 0:
        sleep_start = sleep_indices[0]
        post_onset_stages = stages[sleep_start:]
        rem_indices_post_onset = np.where(post_onset_stages == 2)[0]
        rem_latency_minutes = (rem_indices_post_onset[0] * epoch_sec / 60) if len(rem_indices_post_onset) else np.nan
    else:
        rem_latency_minutes = np.nan

    # NREM/REM bout statistics
    nrem_mask = np.isin(stages, [0, 1])
    nrem_indices = np.where(nrem_mask)[0]
    rem_indices = np.where(stages == 2)[0]

    def episode_lengths(indices):
        if len(indices) == 0:
            return np.nan, np.nan, 0
        split_pts = np.where(np.diff(indices) != 1)[0] + 1
        bouts = np.split(indices, split_pts)
        lengths = [len(b) for b in bouts if len(b) > 0]
        max_len = max(lengths) * epoch_sec / 60 if lengths else np.nan
        mean_len = np.mean(lengths) * epoch_sec / 60 if lengths else np.nan
        return max_len, mean_len, len(lengths)

    nrem_max_bout, nrem_mean_bout, nrem_num_bouts = episode_lengths(nrem_indices)
    rem_max_bout, rem_mean_bout, rem_num_bouts = episode_lengths(rem_indices)

    # Sleep fragmentation index: transitions per hour of sleep
    fragmentation_index = num_transitions / (tst_minutes / 60) if tst_minutes > 0 else np.nan

    feat_dict = dict(
        TST_minutes=tst_minutes,
        WASO_minutes=waso_minutes,
        SleepEfficiency=sleep_efficiency,
        Percent_N3=percent_n3,
        Percent_N1N2=percent_n1n2,
        Percent_REM=percent_rem,
        Awakenings=awakenings,
        NumStageTransitions=num_transitions,
        LongestSleepBout_minutes=longest_sleep_bout_min,
        REMLatency_minutes=rem_latency_minutes,
        NREM_MaxBout_minutes=nrem_max_bout,
        NREM_MeanBout_minutes=nrem_mean_bout,
        NREM_NumBouts=nrem_num_bouts,
        REM_MaxBout_minutes=rem_max_bout,
        REM_MeanBout_minutes=rem_mean_bout,
        REM_NumBouts=rem_num_bouts,
        FragmentationIndex=fragmentation_index,
    )
    return feat_dict


#%% Main code

#data_folder = r"C:\Users\johan\Desktop\data_preprocessed\iRBD"
data_folder = r"C:\Users\johan\Desktop\data_preprocessed\Controls"
#hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\iRBD"
hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\Controls"
summary_folder = r"C:\Users\johan\Desktop\ML_features_sleep_stages"
os.makedirs(summary_folder, exist_ok=True)
epoch_sec = 30  # hypnogram epoch duration
stage_names = {0: "N3", 1: "N1/N2", 2: "REM", 3: "Wake"}

all_night_files = sorted(glob.glob(os.path.join(data_folder, "*_night_*.h5")))
person_prefixes = sorted(set(os.path.basename(f).split("_night_")[0] for f in all_night_files))

feature_rows = []

# Loop through each participant and night
for person_prefix in person_prefixes:
    file_pattern = os.path.join(data_folder, f"{person_prefix}_night_*.h5")
    night_files = sorted(glob.glob(file_pattern))
    print(f"\nProcessing {person_prefix} ({len(night_files)} nights)")
    for i, filepath in enumerate(night_files):
        # Load accelerometer data
        base_name = os.path.basename(filepath)
        night_df, fs = load_h5_acc(filepath)
        if night_df is None or night_df.shape[0] == 0 or fs is None:
            print(f"Skipping empty file: {base_name}")
            continue

        # Load hypnogram data
        night_core_m = re.search(r'(night_\d{4}-\d{2}-\d{2})', base_name)
        if not night_core_m:
            print(f"Could not parse night core from {base_name}")
            continue
        night_core = night_core_m.group(1)
        hyp_file = os.path.join(hypnogram_folder, f"{person_prefix}_{night_core}.npy")
        if not os.path.exists(hyp_file):
            print(f"  No hypnogram found for {base_name}")
            continue
        hyp_df = load_npy_hypnogram(hyp_file, epoch_sec=epoch_sec)
        hyp_df_aligned = hyp_df.reindex(night_df.index, method='nearest')

        # Non-wear detection
        sleep_vector_night = hyp_df_aligned['stage_code'].isin([0, 1, 2]).astype(int)
        exclude, details = night_exclusion_criteria(night_df, fs, sleep_vector_night)
        if exclude:
            print(f"Excluding {base_name}: {details['reasons']}")
            continue

        # Detect overlapping epochs
        acc_mask = night_df['x'].notna() & night_df['y'].notna() & night_df['z'].notna()
        valid_times = night_df.index[acc_mask]
        epoch_starts = hyp_df.index
        epoch_ends = epoch_starts + pd.Timedelta(seconds=epoch_sec)
        epoch_intervals = pd.IntervalIndex.from_arrays(epoch_starts, epoch_ends, closed='left')
        assigned_epochs = epoch_intervals.get_indexer(valid_times)
        n_epochs = len(hyp_df)
        acti_counts = np.bincount(assigned_epochs[assigned_epochs >= 0], minlength=n_epochs)
        overlap_mask = acti_counts > 0
        hyp_df_overlap = hyp_df.iloc[np.where(overlap_mask)[0]]

        # Assign sleep epoch label to each actigraphy sample where overlap exists, used for masking
        sleep_codes = [0, 1, 2]
        sleepvec_epochs = hyp_df_overlap['stage_code'].isin(sleep_codes).astype(int)
        overlap_intervals = pd.IntervalIndex.from_arrays(
            hyp_df_overlap.index, hyp_df_overlap.index + pd.Timedelta(seconds=epoch_sec), closed='left' )
        assigned_overlap = overlap_intervals.get_indexer(night_df.index)
        sleepvec_on_acti = np.zeros(len(night_df), dtype=int)
        valid = assigned_overlap >= 0
        sleepvec_on_acti[valid] = sleepvec_epochs.values[assigned_overlap[valid]]

        # Green mask for plotting: patient asleep and in the overlap region
        asleep_overlap_mask = sleepvec_on_acti == 1

        # Awakenings for plotting
        wake_ix = np.where((sleepvec_on_acti[1:] == 0) & (sleepvec_on_acti[:-1] == 1))[0] + 1

        if hyp_df_overlap.shape[0] < 5:
            print(f"Too few overlapping epochs for {base_name}: Skipping.")
            continue

        # Extract sleep structure features for overlapping epochs
        feats = compute_sleep_structure_features(hyp_df_overlap, epoch_sec=epoch_sec)
        row = {
            "person_prefix": person_prefix,
            "night_file": base_name,
            "night_date": str(night_df.index.min().date() if len(night_df.index) > 0 else ""),
            "n_overlap_epochs": hyp_df_overlap.shape[0],
            "minutes_overlap": hyp_df_overlap.shape[0] * epoch_sec / 60,
            "minutes_total_hypnogram": hyp_df.shape[0] * epoch_sec / 60
        }
        row.update(feats)
        feature_rows.append(row)

    
        # Find values for plotting:
        stages = hyp_df['stage_code'].values
        times = hyp_df.index
        sleep_mask = np.isin(stages, [0,1,2])
        sleep_indices = np.where(sleep_mask)[0]  
        # Find REM onset
        if len(sleep_indices) > 0:
            post_onset_stages = stages[sleep_indices[0]:]
            rem_indices_post_onset = np.where(post_onset_stages == 2)[0]
            rem_onset_time = times[sleep_indices[0] + rem_indices_post_onset[0]] if len(rem_indices_post_onset) > 0 else None
        else:
            rem_onset_time = None
        # Find longest sleep bout indices
        if len(sleep_indices) == 0:
            longest_bout_start = longest_bout_end = None
        else:
            split_points = np.where(np.diff(sleep_indices) != 1)[0] + 1
            sleep_bouts = np.split(sleep_indices, split_points)
            sleep_bout_lengths = [len(b) for b in sleep_bouts]
            ib_long = np.argmax(sleep_bout_lengths)
            longest_bout = sleep_bouts[ib_long]
            longest_bout_start = times[longest_bout[0]]
            longest_bout_end = times[longest_bout[-1]]
        # Find awakening times 
        awakenings_idx = np.where((stages[1:] == 3) & (np.isin(stages[:-1], [0,1,2])))[0] + 1
        awakenings_times = times[awakenings_idx]
        
        # Plotting (optional)
        fig, (ax_acc, ax_hyp, ax_nremrem) = plt.subplots(3, 1, figsize=(11, 10), sharex=True, height_ratios=[1,1,1])
        # 1. Accelerometry
        ax_acc.plot(night_df.index, night_df['x'], label='x', alpha=0.7)
        ax_acc.plot(night_df.index, night_df['y'], label='y', alpha=0.7)
        ax_acc.plot(night_df.index, night_df['z'], label='z', alpha=0.7)
        min_acc = night_df[['x','y','z']].min().min()
        max_acc = night_df[['x','y','z']].max().max()
        ax_acc.fill_between(night_df.index, min_acc, max_acc,  where=sleep_vector_night == 1, color='green', alpha=0.16, label='Sleep window (TST)')
        ax_acc.set_title(f"Accelerometer data & all_sleep inference: {base_name}")
        ax_acc.set_ylabel("Acceleration [g]")
        ax_acc.legend(loc='upper left')
        
        # 2. Hypnogram with all structure markers
        ax_hyp.step(times, stages, where='post', color='k', lw=2)
        ax_hyp.set_ylabel("Sleep stage")
        ax_hyp.set_yticks([0, 1, 2, 3])
        ax_hyp.set_yticklabels(["N3", "N1/N2", "REM", "Wake"])
        ax_hyp.set_title("Hypnogram")
        ax_hyp.fill_between(times, -0.5, 3.5, where=sleep_mask, color='green', alpha=0.13, label='Sleep window (TST)')
        # REM onset (crimsom dashed)
        if rem_onset_time is not None:
            ax_hyp.axvline(rem_onset_time, color='crimson', linestyle='--', lw=2, label='REM onset')
        # Longest sleep bout (blue span)
        if (longest_bout_start is not None) and (longest_bout_end is not None):
            ax_hyp.axvspan(longest_bout_start, longest_bout_end, color='dodgerblue', alpha=0.20, label='Longest sleep bout')
        # Awakenings (red downward triangles)
        ax_hyp.scatter(awakenings_times, np.repeat(3, len(awakenings_times)), color='red', s=45, zorder=10, marker='v', label='Awakening')
        # Sleep structure feature box (upper right)
        info = (
            f"TST: {feats['TST_minutes']:.1f} min\n"
            f"WASO: {feats['WASO_minutes']:.1f} min\n"
            f"REM latency: {feats['REMLatency_minutes']:.1f} min\n"
            f"Longest sleep bout: {feats['LongestSleepBout_minutes']:.1f} min\n"
            f"SE: {feats['SleepEfficiency']*100:.2f}%\n"
            f"% N3: {feats['Percent_N3']:.2f}%\n"
            f"% N1/N2: {feats['Percent_N1N2']:.2f}%\n"
            f"% REM: {feats['Percent_REM']:.2f}%\n"
            f"Awakenings: {feats['Awakenings']}\n"
            f"Stage transitions: {feats['NumStageTransitions']}\n"
            f"Fragmentation index: {feats['FragmentationIndex']:.2f}"
        )
        ax_hyp.text( 0.98, 0.97, info, ha='right', va='top', transform=ax_hyp.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85)  )
        ax_hyp.legend(loc='upper left')

        # 3. NREM/REM colored regions, with summary stats
        ax_nremrem.step(times, stages, where='post', color='k', lw=2)
        ax_nremrem.set_ylabel("Sleep stage")
        ax_nremrem.set_yticks([0, 1, 2, 3])
        ax_nremrem.set_yticklabels(["N3", "N1/N2", "REM", "Wake"])
        ax_nremrem.set_title("Hypnogram (NREM/REM)")
        nrem_mask = np.isin(stages, [0, 1])
        rem_mask = stages == 2
        ax_nremrem.fill_between(times, -0.5, 3.5, where=nrem_mask, color='dodgerblue', alpha=0.22, label='NREM')
        ax_nremrem.fill_between(times, -0.5, 3.5, where=rem_mask, color='orange', alpha=0.22, label='REM')
        ax_nremrem.legend(loc='upper left')
        # Box with metrics for NREM and REM
        box_text = (
            f"NREM bouts: \n"
            f"  Number: {feats['NREM_NumBouts']:.0f}\n"
            f"  Mean: {feats['NREM_MeanBout_minutes']:.1f} min\n"
            f"  Max: {feats['NREM_MaxBout_minutes']:.1f} min\n\n"
            f"REM bouts: \n"
            f"  Number: {feats['REM_NumBouts']:.0f}\n"
            f"  Mean: {feats['REM_MeanBout_minutes']:.1f} min\n"
            f"  Max: {feats['REM_MaxBout_minutes']:.1f} min"        )
        ax_nremrem.text( 0.98, 0.97, box_text, ha='right', va='top', transform=ax_nremrem.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85)  )
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
        
        
# Output summary features for all valid nights
df = pd.DataFrame(feature_rows)
#csv_path = os.path.join(summary_folder, "iRBD_sleep_structure.csv")
csv_path = os.path.join(summary_folder, "Controls_sleep_structure.csv")
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path} with {len(df)} rows.")
