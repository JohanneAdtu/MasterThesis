import os
import glob
import re
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#%% Functions

def load_h5_acc(filepath: str):
    """
    Load a single night's accelerometer data from HDF5 into a pandas DataFrame.

    Returns the DataFrame (with x, y, z columns, indexed by time) and sample rate.
    """
    with h5py.File(filepath, "r") as h5:
        acc = h5["data"]["accelerometry"][:]  # shape (3, N)
        fs = h5["data"]["accelerometry"].attrs["sample_frequency"]
        start_time = pd.to_datetime(h5.attrs["start_time"])
    acc_df = pd.DataFrame(acc.T, columns=["x", "y", "z"])
    time_index = pd.date_range(start=start_time, periods=acc_df.shape[0], freq=pd.to_timedelta(1/fs, unit="s"))
    acc_df.index = time_index
    return acc_df, fs

stage_names = {0: "N3", 1: "N1/N2", 2: "REM", 3: "Wake"}

def load_npy_hypnogram(npy_file, epoch_sec=30):
    """
    Load a hypnogram (.npy) file and return DataFrame with stage_code, stage, and time index.
    """
    arr = np.load(npy_file)
    if arr.ndim == 2 and arr.shape[1] == 4:
        stage_codes = np.argmax(arr, axis=1)
    else:
        stage_codes = arr
    m = re.search(r"_night_(\d{4}-\d{2}-\d{2})", npy_file)
    if not m:
        raise ValueError("Could not parse date from hypnogram filename.")
    date_str = m.group(1)
    start_time = pd.Timestamp(f"{date_str} 21:00:00")
    times = [start_time + pd.Timedelta(seconds=epoch_sec * i) for i in range(len(stage_codes))]
    hyp_df = pd.DataFrame({
        "stage_code": stage_codes,
        "stage": [stage_names[c] for c in stage_codes]
    }, index=pd.to_datetime(times))
    return hyp_df

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
    std_thr_g=0.02, 
    range_thr_g=0.1):
    """
    Apply exclusion criteria: total sleep time (TST) bounds and non-wear time limits.
    Returns (exclude_flag, details_dict)
    """
    
    # Total sleep time
    tst_sample_count = np.sum(sleep_vector)
    tst_hours = (tst_sample_count / fs) / 3600

    # Divide into blocks and determine nonwear periods
    block_samples = int(block_minutes * 60 * fs)
    n_blocks = len(acc_df) // block_samples if block_samples > 0 else 0
    nonwear_mask = pd.Series(False, index=acc_df.index)

    for i in range(n_blocks):
        start_idx = i * block_samples
        end_idx = start_idx + block_samples
        block_df = acc_df.iloc[start_idx:end_idx]
        if block_df.empty:
            continue
        # If at least 2 axes show abnormally low std or movement range, flag as nonwear
        std_vals = block_df.std()
        range_vals = block_df.max() - block_df.min()
        std_count = np.sum(std_vals < std_thr_g)
        range_count = np.sum(range_vals < range_thr_g)
        if (std_count >= 2) or (range_count >= 2):
            nonwear_mask.iloc[start_idx:end_idx] = True

    # Determine nonwear in main analysis window (21:00 to 09:00)
    hours = acc_df.index.hour
    if nonwear_window_start <= nonwear_window_end:
        window_mask = (hours >= nonwear_window_start) & (hours < nonwear_window_end)
    else:
        window_mask = (hours >= nonwear_window_start) | (hours < nonwear_window_end)
    
    nonwear_in_window_mask = nonwear_mask & window_mask
    nonwear_sample_count = np.sum(nonwear_in_window_mask)
    nonwear_hours_in_window = (nonwear_sample_count / fs) / 3600

    # Exclude night if sleep duration out of bounds or too much nonwear
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
    details = {
        "exclude": exclude,
        "tst_hours": tst_hours,
        "nonwear_hours_in_window": nonwear_hours_in_window,
        "reasons": "; ".join(reasons) if exclude else "OK"
    }
    return exclude, details

#%% Main code

block_minutes = 30            # Non-wear detection block length (min)
std_thr_g = 0.02              # Standard deviation threshold (g)
range_thr_g = 0.1             # Range threshold (g)
nonwear_window_start = 21     # Window start hour (21:00)
nonwear_window_end = 9        # Window end hour (09:00)

data_folders = {
    "iRBD":   r"C:\Users\johan\Desktop\data_preprocessed\iRBD",
    "Controls": r"C:\Users\johan\Desktop\data_preprocessed\Controls"}
hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms"

# For each group (iRBD and Controls), assess all nights for nonwear and sleep content
rows = []
for group, data_folder in data_folders.items():
    all_night_files = sorted(glob.glob(os.path.join(data_folder, "*_night_*.h5")))
    print(f"\n[{group}] - Processing {len(all_night_files)} nights from {data_folder}")
    for filepath in all_night_files:
        base_name = os.path.basename(filepath)
        
        # Load accelerometer data and sample rate
        night_df, fs = load_h5_acc(filepath)
        if night_df is None or night_df.shape[0] == 0 or fs is None:
            continue
        person_prefix = base_name.split("_night_")[0]
        night_core_match = re.search(r'night_(\d{4}-\d{2}-\d{2})', base_name)
        if not night_core_match:
            print(f"  Could not parse night date in {base_name}")
            continue
        night_core = night_core_match.group(0)
        
        # Load per-night sleep staging
        hyp_file = os.path.join(hypnogram_folder, group, f"{person_prefix}_{night_core}.npy")
        if not os.path.exists(hyp_file):
            print(f"  Hypnogram missing for {base_name}")
            continue
        hyp_df = load_npy_hypnogram(hyp_file, epoch_sec=30)
        hyp_df_aligned = hyp_df.reindex(night_df.index, method='nearest')
        sleep_vector = hyp_df_aligned['stage_code'].isin([0, 1, 2]).astype(int)
        
        row = {
            "night_file": base_name,
            "group": group}
    
        # Use exclusion criteria function
        exc, det = night_exclusion_criteria(night_df, fs, sleep_vector,std_thr_g=std_thr_g, range_thr_g=range_thr_g, block_minutes=block_minutes, nonwear_window_start=nonwear_window_start, nonwear_window_end=nonwear_window_end)
        
        row["tst_hours"] = det["tst_hours"] 
        row["nonwear_hours_21_09"] = det["nonwear_hours_in_window"]  # Use returned value
        row["excluded"] = exc
        rows.append(row)
        
df = pd.DataFrame(rows)

# Produce heatmaps of nonwear vs. sleep time, before and after exclusions
excl_col = "excluded"
x_col = "nonwear_hours_21_09"

block_size_h = block_minutes / 60
x_max = 12
xbins = np.arange(0, x_max + block_size_h, block_size_h)  # Non-wear hours bins
ybins = np.linspace(0, 14, 40)  # Total sleep time bins

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
for ax in axes:
    ax.set_facecolor("white")
    ax.grid(False) 
    
# All nights (before exclusion)
x_all = df[x_col]
y_all = df['tst_hours']
h1, _, _, img1 = axes[0].hist2d(x_all, y_all, bins=[xbins, ybins], norm=LogNorm(), cmap='hot')
axes[0].set_title("Before exclusion")
axes[0].set_xlabel('Non-wear [h]')
axes[0].set_ylabel('Total sleep time [h]')
plt.colorbar(img1, ax=axes[0], label='Counts (log)')

# After exclusion: only keep nights which passed quality checks
mask = ~df[excl_col]
x_filt = df.loc[mask, x_col]
y_filt = df.loc[mask, 'tst_hours']
h2, _, _, img2 = axes[1].hist2d( x_filt, y_filt, bins=[xbins, ybins], norm=LogNorm(), cmap='hot')
axes[1].set_title("After exclusion")
axes[1].set_ylabel('Total sleep time [h]')
axes[1].set_xlabel('Non-wear [h]')
plt.colorbar(img2, ax=axes[1], label='Counts (log)')
plt.tight_layout()
plt.show()

# Print summary of night exclusion
n_all = len(mask)
n_exc = (~mask).sum()
print(f"{n_exc}/{n_all} nights excluded ({100*n_exc/n_all:.1f}%)")