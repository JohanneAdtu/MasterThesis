import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import welch
from scipy.stats import entropy
import re

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

def extract_frequency_features(signal, fs, nperseg, band_edges):
    """
    Extract frequency-domain features from the power spectral density of a given 1D signal.
    Includes total power, band powers, peak frequency, and spectral entropy.
    """

    # # Compute Welch PSD
    freq, psd = welch(signal, fs=fs, nperseg=nperseg, scaling='density')
    
    # Total power integration across spectrum
    total_power = np.trapezoid(psd, freq)
    
    # Band powers
    band_edges=[(0.1, 0.5), (0.5, 2.0), (2.0, 10.0)]
    band_powers = []
    for low, high in band_edges:
        idx_band = np.logical_and(freq >= low, freq < high)
        band_power = np.trapezoid(psd[idx_band], freq[idx_band])
        band_powers.append(band_power)
    
    # Maximum PSD peak frequency
    mask = freq > 0.2 # Mask out DC/low freq
    if np.all(np.isnan(psd)) or np.sum(mask) == 0:
        peak_freq = np.nan
    else:
        peak_freq = freq[mask][np.nanargmax(psd[mask])]    
    
    # Spectral entropy
    psd_norm = psd / np.sum(psd) # Normalize PSD
    spec_entropy = entropy(psd_norm) # Shannon spectral entropy
    
    # Compile features
    features = {
        "TotalPower": total_power,
        "BandPower_1": band_powers[0],
        "BandPower_2": band_powers[1],
        "BandPower_3": band_powers[2],
        "PeakFreq": peak_freq,
        "SpectralEntropy": spec_entropy}
    return features

#%% Main code

data_folder = r"C:\Users\johan\Desktop\data_preprocessed\Controls"
#data_folder = r"C:\Users\johan\Desktop\data_preprocessed\iRBD"
hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\Controls"
#hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\iRBD"
summary_folder = r"C:\Users\johan\Desktop\ML_features_sleep_stages"
os.makedirs(summary_folder, exist_ok=True)

stage_names = {0: "N3", 1: "N1/N2", 2: "REM", 3: "Wake"}
sleep_stage_groups = {
    "all_sleep": [0, 1, 2],
    "nrem": [0, 1],
    "rem": [2],
    "sleep_block": None
}
min_stage_minutes = 5
band_edges = [(0.1, 0.5), (0.5, 2.0), (2.0, 10.0)]
epoch_sec_for_welch = 30

all_night_files = sorted(glob.glob(os.path.join(data_folder, "*_night_*.h5")))
person_prefixes = sorted(set(os.path.basename(f).split("_night_")[0] for f in all_night_files))

# One feature-table per stage group
stage_rows = {stage: [] for stage in sleep_stage_groups.keys()}

# Process participant files
for person_prefix in person_prefixes:
    file_pattern = os.path.join(data_folder, f"{person_prefix}_night_*.h5")
    night_files = sorted(glob.glob(file_pattern))
    print(f"\nProcessing {person_prefix} ({len(night_files)} nights)")
    for filepath in night_files:
        # Load accelerometer data
        base_name = os.path.basename(filepath)
        night_df, fs = load_h5_acc(filepath)
        if night_df is None or night_df.shape[0] == 0 or fs is None:
            print(f"Skipping empty file: {base_name}")
            continue
        
        # # Find hypnogram and align
        night_core_m = re.search(r'(night_\d{4}-\d{2}-\d{2})', base_name)
        if not night_core_m:
            print(f"Could not parse night core from {base_name}")
            continue
        night_core = night_core_m.group(1)
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
        
        #For sleep_block: contiguous all sleep epochs from first to last
        sleep_codes = [0, 1, 2]
        sleep_indices = np.where(hyp_df_aligned['stage_code'].isin(sleep_codes))[0]
        sleep_block_vector = np.zeros_like(hyp_df_aligned['stage_code'])
        if len(sleep_indices) > 0:
            block_start = sleep_indices[0]
            block_end = sleep_indices[-1]
            sleep_block_vector[block_start:block_end + 1] = 1

        # Extract features for each stage group
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
            # Insufficient data
            if sleep_vector_for_window.sum() == 0 or minutes_of_data < min_stage_minutes:
                row['status'] = 'skipped'
                for f in ["total_power", "band1_power", "band2_power", "band3_power", "peak_freq", "spectral_entropy"]:
                    row[f] = np.nan
                stage_rows[stage_grp].append(row)
                continue

            # Prepare magnitude signal and compute features
            sleep_df = night_df[sleep_vector_for_window == 1].copy()
            sleep_df = sleep_df.ffill().bfill()
            mag = np.sqrt(sleep_df['x']**2 + sleep_df['y']**2 + sleep_df['z']**2).values
            mag_centered = mag - np.mean(mag)
            nperseg = int(epoch_sec_for_welch * fs)
            
            features = extract_frequency_features(mag_centered, fs, nperseg, band_edges)
            for key in features:
                row[key] = features[key]
            row['status'] = 'ok'
            stage_rows[stage_grp].append(row)
            
            # Plot features (optional)
            if stage_grp in {"sleep_block"} and sleep_df.shape[0] > 0:
                freq, psd = welch(mag_centered, fs=fs, nperseg=nperseg, scaling='density')
                features_str = (
                    f"Total power: {features['TotalPower']:.2e} g²\n"
                    f"Band 1 power: {features['BandPower_1']:.2e} g²\n"
                    f"Band 2 power: {features['BandPower_2']:.2e} g²\n"
                    f"Band 3 power: {features['BandPower_3']:.2e} g²\n"
                    f"Peak frequency: {features['PeakFreq']:.2f} Hz\n"
                    f"Spectral entropy: {features['SpectralEntropy']:.2f}")
            
                plt.figure(figsize=(10, 4))
                plt.plot(freq, psd, label='Magnitude')
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("PSD [g²/Hz]")
                plt.title(f"Power spectral density during {stage_grp}: {base_name}")
                plt.grid(True)
                plt.axvspan(0.1, 0.5, color='blue', alpha=0.15, label="Band 1 (0.1–0.5 Hz)")
                plt.axvspan(0.5, 2.0, color='orange', alpha=0.15, label="Band 2 (0.5–2 Hz)")
                plt.axvspan(2.0, 10.0, color='green', alpha=0.15, label="Band 3 (2–10 Hz)")
                # Put feature summary in upper right box
                plt.gca().text(0.98, 0.65, features_str, transform=plt.gca().transAxes, fontsize=11,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.show()

#%%
# --- Save per-stage-group CSV ---
for stage_grp in sleep_stage_groups.keys():
    df_stage = pd.DataFrame(stage_rows[stage_grp])
    csv_path = os.path.join(summary_folder, f"Controls_{stage_grp}_frequency.csv")
    #csv_path = os.path.join(summary_folder, f"iRBD_{stage_grp}_frequency.csv")
    df_stage.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} with {len(df_stage)} rows.")

 