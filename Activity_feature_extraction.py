import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
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

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply low-pass Butterworth filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def extract_epoch_features(acc_df, sleep_vector, fs, epoch_sec, lower_thr, upper_thr):
    """
    Extract movement (activity) features from accelerometer data per epoch.
    - activity_count: sum(abs(diff(magnitude)))
    - activity_index: 1 if activity_count > lower_thr else 0
    - twitch: 1 if lower_thr < activity_count < upper_thr and neighbors == lower_thr else 0
    """
    
    # Handle nan's
    acc_df_filled = acc_df.ffill().bfill()
    
    # Select only sleep periods
    sleep_df = acc_df_filled[sleep_vector == 1]
    
    # Epoch segmentation: 1, 15, and 30s
    epoch_len = int(epoch_sec * fs)
    n_epochs = len(sleep_df) // epoch_len
    
    # Use magnitude for activity count calculation
    mag = np.sqrt(sleep_df['x']**2 + sleep_df['y']**2 + sleep_df['z']**2).values
    mag_centered = mag - np.median(mag)
    
    # Filter    
    mag_filtered = butter_lowpass_filter(mag_centered, cutoff=2.0, fs=fs)
    
    epoch_times = []
    activity_counts = []
    activity_indices = []
    twitch_flags = []
    
    # For each epoch, compute activity count and index
    for i in range(n_epochs):
        start = i * epoch_len
        end = start + epoch_len
        if end > len(mag_filtered):
            continue  # skip incomplete/partial epoch
        epoch_mag = mag_filtered[start:end]
        epoch_times.append(sleep_df.index[start])
        
        # Activity count
        act_count = np.sum(np.abs(np.diff(epoch_mag)))
        activity_counts.append(act_count)
        
        # Activity index (needs to be above threshold, not 0)
        activity_indices.append(1 if act_count > lower_thr else 0)   
        
        twitch_flags.append(0)  # fill, assign later

    # Twitch detection (must be after all activity_counts are known)
    for i in range(1, n_epochs-1):
        if (activity_counts[i] > lower_thr) and (activity_counts[i] < upper_thr):
            if (activity_counts[i-1] <= lower_thr) and (activity_counts[i+1] <= lower_thr):
                twitch_flags[i] = 1       
    
    # Summary statistics
    mean_activity = np.mean(activity_counts) if len(activity_counts) > 0 else np.nan
    activity_index_percent = np.mean(activity_indices) * 100 if len(activity_indices) > 0 else np.nan
    n_hours = (len(activity_counts) * epoch_sec) / 3600 if len(activity_counts) > 0 else np.nan
    twitch_per_hour = np.sum(twitch_flags) / n_hours if n_hours > 0 else np.nan

    return (epoch_times, activity_counts, activity_indices, np.array(twitch_flags), mean_activity, activity_index_percent, twitch_per_hour)

#%% Main code

# Folder config
#data_folder = r"C:\Users\johan\Desktop\data_preprocessed\iRBD"
data_folder = r"C:\Users\johan\Desktop\data_preprocessed\Controls"
#hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\iRBD"
hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\Controls"
summary_folder = r"C:\Users\johan\Desktop\ML_features_sleep_stages"
os.makedirs(summary_folder, exist_ok=True)

stage_names = {0: "N3", 1: "N1/N2", 2: "REM", 3: "Wake"}
sleep_stage_groups = {
    "all_sleep": [0, 1, 2],
    "nrem": [0, 1],
    "rem": [2],
    "sleep_block": None  # handled below
}

# Prepare storage: one list (table) per stage group
stage_rows = {stage: [] for stage in sleep_stage_groups.keys()}

min_stage_minutes = 5

# Set epoch lenth, upper and lower threshold
epoch_lengths = [1, 15, 30]
lower_thr = [0.015, 0.15, 0.25]
upper_thr = [1.015, 1.15, 1.25]

# Find all night files
all_night_files = sorted(glob.glob(os.path.join(data_folder, "*_night_*.h5")))
person_prefixes = sorted(set(os.path.basename(f).split("_night_")[0] for f in all_night_files))

for person_prefix in person_prefixes:
    file_pattern = os.path.join(data_folder, f"{person_prefix}_night_*.h5")
    night_files = sorted(glob.glob(file_pattern))
    if not night_files:
        print(f"No night files for {person_prefix}")
        continue
    print(f"\nProcessing {person_prefix} ({len(night_files)} nights)")

    for filepath in night_files:
        # Load accelerometer data
        base_name = os.path.basename(filepath)
        night_df, fs = load_h5_acc(filepath)
        if night_df is None or night_df.shape[0] == 0:
            print(f"Skipping empty file: {base_name}")
            continue
        if fs is None:
            print(f"Warning: sample frequency missing for {base_name}; skipping")
            continue

        # Find hypnogram and align to acc_df
        night_core = re.search(r'(night_\d{4}-\d{2}-\d{2})', base_name).group(1)
        hyp_file = os.path.join(hypnogram_folder, f"{person_prefix}_{night_core}.npy")
        if not os.path.exists(hyp_file):
            print(f"  No hypnogram found for {base_name}")
            continue
        hyp_df = load_npy_hypnogram(hyp_file, epoch_sec=30)
        hyp_df_aligned = hyp_df.reindex(night_df.index, method='nearest')

        # Non-wear detection
        sleep_vector_night = hyp_df_aligned['stage_code'].isin([0, 1, 2]).astype(int)
        exclude, details = night_exclusion_criteria(night_df, fs, sleep_vector_night)
        if exclude:
            print(f"Excluding {base_name}: {details['reasons']}")
            continue

        # Plot (optional)
        fig, ax_top = plt.subplots(1, 1, figsize=(10, 3))
        ax_top.plot(night_df.index, night_df['x'], label='x', alpha=0.7)
        ax_top.plot(night_df.index, night_df['y'], label='y', alpha=0.7)
        ax_top.plot(night_df.index, night_df['z'], label='z', alpha=0.7)
        ax_top.set_title(f"Accelerometer data: {base_name}")
        ax_top.set_ylabel("Acceleration [g]")
        ax_top.legend(loc='upper right')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()
        
        # For sleep_block: contiguous from first to last sleep time
        sleep_codes = [0, 1, 2]
        sleep_indices = np.where(hyp_df_aligned['stage_code'].isin(sleep_codes))[0]
        sleep_block_vector = np.zeros_like(hyp_df_aligned['stage_code'])
        if len(sleep_indices) > 0:
            block_start = sleep_indices[0]
            block_end = sleep_indices[-1]
            sleep_block_vector[block_start:block_end + 1] = 1

        # Extract features per sleep window
        for stage_grp, stage_codes in sleep_stage_groups.items():
            if stage_grp == "sleep_block":
                stage_mask = pd.Series(sleep_block_vector, index=night_df.index).astype(bool)
            else:
                stage_mask = hyp_df_aligned['stage_code'].isin(stage_codes)
            acc_mask = night_df['x'].notna() & night_df['y'].notna() & night_df['z'].notna()
            window_overlap_mask = stage_mask & acc_mask
            sleep_vector_for_window = window_overlap_mask.astype(int)
            minutes_of_data = sleep_vector_for_window.sum() / fs / 60

            row = {
                "person_prefix": person_prefix,
                "night_file": base_name,
                "night_date": str(night_df.index.min().date() if len(night_df.index) > 0 else ""),
                "stage_group": stage_grp,
                "minutes_of_data": minutes_of_data,
            }
            
            # Handle too little/zero data for this window
            if sleep_vector_for_window.sum() == 0 or minutes_of_data < min_stage_minutes:
                row['status'] = 'skipped'
                for ep in epoch_lengths:
                    row[f'mean_activity_{ep}s'] = np.nan
                    row[f'activity_index_percent_{ep}s'] = np.nan
                    row[f'twitch_per_hour_{ep}s'] = np.nan
                    #row[f'n_epochs_{ep}s'] = 0
                stage_rows[stage_grp].append(row)
                continue

            # Feature extraction for this window
            feat_dict = {}
            for i, ep in enumerate(epoch_lengths):
                feat_dict[ep] = extract_epoch_features(night_df, sleep_vector_for_window, fs,
                                                      epoch_sec=ep, lower_thr=lower_thr[i], upper_thr=upper_thr[i])
            row['status'] = 'ok'
            for ep in epoch_lengths:
                (epoch_times, activity_counts, activity_indices, twitch_flags,
                 mean_activity, activity_index_percent, twitch_per_hour) = feat_dict[ep]
                row[f'mean_activity_{ep}s'] = mean_activity
                row[f'activity_index_percent_{ep}s'] = activity_index_percent
                row[f'twitch_per_hour_{ep}s'] = twitch_per_hour
                #row[f'n_epochs_{ep}s'] = len(activity_counts)
            stage_rows[stage_grp].append(row)

            # Plot only for sleep_block (optional)
            if stage_grp == "sleep_block":
            
                fig, (ax_top, ax_hyp, ax1, ax2, ax3) = plt.subplots(5, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]}                )
            
                # 1. Top plot: x, y, z and window
                ax_top.plot(night_df.index, night_df['x'], label='x')
                ax_top.plot(night_df.index, night_df['y'], label='y')
                ax_top.plot(night_df.index, night_df['z'], label='z')
                min_acc = night_df[['x','y','z']].min().min()
                max_acc = night_df[['x','y','z']].max().max()
                ax_top.fill_between(night_df.index, min_acc, max_acc, where=sleep_vector_for_window == 1, color='green', alpha=0.2, label=f"{stage_grp} window"                )
                ax_top.set_title(f"Accelerometer data & {stage_grp} inference: {base_name}")
                ax_top.set_ylabel("Acceleration [g]")
                ax_top.legend(loc='upper right')
            
                # 2. Hypnogram aligned as step plot
                ax_hyp.step(hyp_df_aligned.index, hyp_df_aligned['stage_code'], where='post', color='k')
                ax_hyp.set_ylabel("Sleep stage")
                ax_hyp.set_yticks([0, 1, 2, 3])
                ax_hyp.set_yticklabels(["N3", "N1/N2", "REM", "Wake"])
                ax_hyp.set_title("Hypnogram")
            
                # 3. 1 sec epoch plot
                epoch_times_1, activity_counts_1, _, twitch_flags_1, mean_activity_1, activity_index_percent_1, twitch_per_hour_1 = feat_dict[1]
                ax1.plot(epoch_times_1, activity_counts_1, color='tab:blue', label='Activity count (1s)', linewidth=1.5)
                series_1 = pd.Series(activity_counts_1, index=pd.to_datetime(epoch_times_1)).dropna()
                mean_1 = series_1.mean()
                std_1 = series_1.std()
                y_min_1 = 0
                y_max_1 = mean_1 + 5*std_1
                twitch_mask_1 = np.array(twitch_flags_1) == 1
                twitch_times_1 = np.array(epoch_times_1)[twitch_mask_1]
                for i, st in enumerate(twitch_times_1):
                    st_num = mdates.date2num(pd.to_datetime(st))
                    if i == 0:
                        ax1.vlines(st_num, y_min_1, y_max_1, color='red', linewidth=2, linestyle='-', label='Twitch')
                    else:
                        ax1.vlines(st_num, y_min_1, y_max_1, color='red', linewidth=2, linestyle='-')
            
                textstr1 = (
                    f'Mean activity: {mean_activity_1:.3f}\n'
                    f'Activity index (%): {activity_index_percent_1:.1f}\n'
                    f'Twitch per hour: {twitch_per_hour_1:.2f}')
                ax1.text(0.99, 0.95, textstr1, transform=ax1.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax1.set_ylabel("Activity count (1s)")
                ax1.legend(loc='upper left')
                ax1.set_title("Activity count (1s)")
                ax1.grid(True, axis='y')
            
                # 4. 15 sec epoch plot
                epoch_times_15, activity_counts_15, _, twitch_flags_15, mean_activity_15, activity_index_percent_15, twitch_per_hour_15 = feat_dict[15]
                ax2.plot(epoch_times_15, activity_counts_15, color='tab:blue', label='Activity count (15s)', linewidth=1.5)
                series_15 = pd.Series(activity_counts_15, index=pd.to_datetime(epoch_times_15)).dropna()
                mean_15 = series_15.mean()
                std_15 = series_15.std()
                y_min_15 = 0
                y_max_15 = mean_15 + 2*std_15
                twitch_mask_15 = np.array(twitch_flags_15) == 1
                twitch_times_15 = np.array(epoch_times_15)[twitch_mask_15]
                for i, st in enumerate(twitch_times_15):
                    st_num = mdates.date2num(pd.to_datetime(st))
                    if i == 0:
                        ax2.vlines(st_num, y_min_15, y_max_15, color='red', linewidth=2, linestyle='-', label='Twitch')
                    else:
                        ax2.vlines(st_num, y_min_15, y_max_15, color='red', linewidth=2, linestyle='-')
            
                textstr15 = (
                    f'Mean activity: {mean_activity_15:.3f}\n'
                    f'Activity index (%): {activity_index_percent_15:.1f}\n'
                    f'Twitch per hour: {twitch_per_hour_15:.2f}')
                ax2.text(0.99, 0.95, textstr15, transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax2.set_ylabel("Activity count (15s)")
                ax2.legend(loc='upper left')
                ax2.set_title("Activity count (15s)")
                ax2.grid(True, axis='y')
            
                # 5. 30 sec epoch plot
                epoch_times_30, activity_counts_30, _, twitch_flags_30, mean_activity_30, activity_index_percent_30, twitch_per_hour_30 = feat_dict[30]
                ax3.plot(epoch_times_30, activity_counts_30, color='tab:blue', label='Activity count (30s)', linewidth=1.5)
                series_30 = pd.Series(activity_counts_30, index=pd.to_datetime(epoch_times_30)).dropna()
                mean_30 = series_30.mean()
                std_30 = series_30.std()
                y_min_30 = 0
                y_max_30 = mean_30 + 1*std_30
                twitch_mask_30 = np.array(twitch_flags_30) == 1
                twitch_times_30 = np.array(epoch_times_30)[twitch_mask_30]
                for i, st in enumerate(twitch_times_30):
                    st_num = mdates.date2num(pd.to_datetime(st))
                    if i == 0:
                        ax3.vlines(st_num, y_min_30, y_max_30, color='red', linewidth=2, linestyle='-', label='Twitch')
                    else:
                        ax3.vlines(st_num, y_min_30, y_max_30, color='red', linewidth=2, linestyle='-')
            
                textstr30 = (
                    f'Mean activity: {mean_activity_30:.3f}\n'
                    f'Activity index (%): {activity_index_percent_30:.1f}\n'
                    f'Twitch per hour: {twitch_per_hour_30:.2f}')
                ax3.text(0.99, 0.95, textstr30, transform=ax3.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax3.set_ylabel("Activity count (30s)")
                ax3.legend(loc='upper left')
                ax3.grid(True, axis='y')
                ax3.set_title("Activity count (30s)")
                ax3.set_xlabel("Time")
            
                plt.tight_layout()
                plt.show()            

#%%
# --- Save one CSV per stage group ---
for stage_grp in sleep_stage_groups.keys():
    df_stage = pd.DataFrame(stage_rows[stage_grp])
    #csv_path = os.path.join(summary_folder, f"iRBD_{stage_grp}_activity.csv")
    csv_path = os.path.join(summary_folder, f"Controls_{stage_grp}_activity.csv")
    df_stage.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} with {len(df_stage)} rows.")
    
