import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import re
from scipy.signal import butter, filtfilt, stft, get_window, find_peaks, firwin
from scipy.ndimage import median_filter
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

#---- Nightbeat signal construction ----#
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter (step 1)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_nightbeat_signal(acc_df, fs):
    """
    Compute the Nightbeat signal (step 1):
        1. Bandpass filter each axis (5–14 Hz, 4th order Butterworth)
        2. Compute L2-norm of filtered axes
        3. Bandpass filter L2-norm (0.5–3.5 Hz, 4th order Butterworth)
    """
    # Fill/interpolate NaNs
    acc_df_filled = acc_df.interpolate(method='linear').ffill().bfill()
    x_filt = butter_bandpass_filter(acc_df_filled['x'].values, 5, 14, fs)
    y_filt = butter_bandpass_filter(acc_df_filled['y'].values, 5, 14, fs)
    z_filt = butter_bandpass_filter(acc_df_filled['z'].values, 5, 14, fs)
    l2_norm = np.sqrt(x_filt**2 + y_filt**2 + z_filt**2)
    nightbeat = butter_bandpass_filter(l2_norm, 0.5, 3.5, fs)
    return pd.Series(nightbeat, index=acc_df.index)

#---- Artifact/Motion detection ----#
def robust_sigma(x):
    """Robust spread measure: IQR/1.349 (for artifact threshold)"""
    q75, q25 = np.percentile(x, [75,25])
    iqr = q75 - q25
    return iqr / 1.349

def compute_stft_nightbeat(nb_vec, fs):
    """
    Short-Time Fourier Transform as specified:
     - Kaiser window (β=2), nperseg=1024
     - 10× zero padding (nfft)
     - 10 sample shift (noverlap = 1024-10)
    """
    nperseg = 1024
    pad_factor = 10
    nfft = nperseg * pad_factor
    noverlap = nperseg - 10
    window = get_window(('kaiser', 2), nperseg)
    f, t, Zxx = stft(nb_vec, fs=fs, window=window, nperseg=nperseg, 
                     noverlap=noverlap, nfft=nfft, padded=False, boundary=None)
    return f, t, Zxx

#---- HR peak finding (frequency domain) ----#
def find_hr_peaks(Sxx, f, t, freq_range=(0.5,3.5)):
    """
    For each time in STFT:
      - Find ALL peaks within freq_range, with height/prominence > 0.5×max
      - Remove harmonics (within 2% of mults)
    Returns:  List of lists of non-harmonic peak frequencies (Hz) for each time step.
    """
    peaks_all = []
    for i in range(Sxx.shape[1]):
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        spectrum = np.abs(Sxx[:, i])[freq_mask]
        freq = f[freq_mask]
        if len(spectrum) == 0:
            peaks_all.append([])
            continue
        maxval = np.max(spectrum)
        if maxval == 0:
            peaks_all.append([])
            continue
        # Find all candidate peaks
        peak_indices, props = find_peaks(spectrum, height=0.5*maxval, prominence=0.5*maxval)
        peak_freqs = freq[peak_indices]
        # Sort peaks by frequency (ascending)
        peak_freqs = np.sort(peak_freqs)
        # Robust harmonic rejection: only keep peaks not a harmonic of a lower-freq peak
        selected = []
        for fp in peak_freqs:
            is_harmonic = False
            for fj in selected:
                if np.abs(fp/fj - np.round(fp/fj)) < 0.02 and fp > fj:
                    is_harmonic = True
                    break
            if not is_harmonic:
                selected.append(fp)
        peaks_all.append(selected)
    return peaks_all # List of lists of freq (Hz), length = n_timesteps

def trace_hr_curves_with_interrupt(peaks_all, t, artifact_mask, gap_steps=5, freq_tol=0.05, min_curve_len=10):
    """
    Trace peaks over STFT time (merging if ≤ 0.05 Hz apart within ≤1s), cut at artifact points, only keep curves ≥10s. (step 3)
    """
    curves = []
    active_curves = []
    for ti, pks in enumerate(peaks_all):
        if artifact_mask[ti]:
            # Interrupt all running curves
            active_curves = []
            continue
        assigned = [False]*len(pks)
        # Try to extend active curves
        new_active_curves = []
        for curve in active_curves:
            last_t, last_f = curve[-1]
            if ti-last_t > gap_steps:
                continue # break in continuity
            # Extend if candidate within 0.05Hz
            candidates = [j for j, pk in enumerate(pks) if abs(pk-last_f)<=freq_tol]
            if candidates:
                idx = candidates[0]
                new_curve = curve + [(ti, pks[idx])]
                new_active_curves.append(new_curve)
                assigned[idx] = True
            else:
                # End curve, save if ≥10s
                if t[curve[-1][0]] - t[curve[0][0]] >= min_curve_len:
                    curves.append(curve)
        # Start new curves for unassigned peaks
        for j, pk in enumerate(pks):
            if not assigned[j]:
                new_active_curves.append([(ti, pk)])
        active_curves = new_active_curves
    # Add any remaining active curves if long enough
    for curve in active_curves:
        if t[curve[-1][0]] - t[curve[0][0]] >= min_curve_len:
            curves.append(curve)
    return curves

def select_best_curve(curves, t):
    """
    At each time, select value from longest/most robust curve(s) (step 3, outlier removal).
    Outlier rejection (median-robust_sigma, keep ones within 5σ).
    """
    min_curve_len=10
    if not curves:
        return np.array([]), np.array([])
    
    # 1. Only keep curves longer than min_curve_len seconds
    long_curves = [c for c in curves if (t[c[-1][0]] - t[c[0][0]]) >= min_curve_len]
    if not long_curves:
        return np.array([]), np.array([])
    
    # 2. Compute overall median and robust sigma from all long curve frequencies
    all_freqs = np.concatenate([[f for _, f in c] for c in long_curves])
    overall_median = np.median(all_freqs)
    sigma = robust_sigma(all_freqs)
    
    # 3. Compute median for each long curve
    medians = [np.median([f for _, f in c]) for c in long_curves]
    
    # 4. Filter out curves whose median is further than 5*sigma from the overall median
    curves_good = [c for c, m in zip(long_curves, medians) if abs(m - overall_median) <= 5*sigma]
    if not curves_good:
        return np.array([]), np.array([])
    
    # 5. At each t_idx (STFT time), pick the frequency from the remaining curve(s) closest to the overall median
    t_idx_curve = {}
    for c in curves_good:
        for ti, f in c:
            if ti not in t_idx_curve:
                t_idx_curve[ti] = []
            t_idx_curve[ti].append(f)
    final_curve = []
    for ti in sorted(t_idx_curve.keys()):
        f_list = t_idx_curve[ti]
        best_f = min(f_list, key=lambda fv: abs(fv - overall_median))
        final_curve.append((ti, best_f))
        
    # Convert to array of time/freq, sorted by time
    final_curve = sorted(final_curve)
    times = [t[ti] for ti, _ in final_curve]
    freqs = [f for _, f in final_curve]
    return np.array(times), np.array(freqs)

#---- Time-domain heartbeat detection ----#
def detect_heartbeats(nightbeat_segment, fs, HRC):
    """
    Heartbeat peak detection (step 4): 
     - Rolling median + rolling mean smoothing (win=0.3/HRC s)
     - 20th order FIR filter (0.9–1.1×HRC)
     - Peaks above rolling mean threshold (win=0.5/HRC s)
     - Min peak distance: 0.6×period
     - Return HRP (time-domain HR), np.nan if fails or |HRP-HRC|>10 bpm
    """
    # 1. Rolling median (0.3 s/HRC)
    period = 1.0 / HRC
    winlen_samples = int(0.3 * period * fs)
    if winlen_samples < 1:
        winlen_samples = 1
    if winlen_samples % 2 == 0:
        winlen_samples += 1  # make odd for median
    
    # Rolling median
    med_filtered = pd.Series(nightbeat_segment).rolling(winlen_samples, center=True, min_periods=1).median().values
    
    # 2. Rolling mean (same window)
    mean_filtered = pd.Series(med_filtered).rolling(winlen_samples, center=True, min_periods=1).mean().values
    
    # 3. Narrow bandpass filter (FIR, 20th order)
    lowcut = 0.9 * HRC
    highcut = 1.1 * HRC
    if highcut >= 0.5 * fs:
        highcut = 0.5 * fs - 0.01  # Nyquist
    fir_coeff = firwin(21, [lowcut/(0.5*fs), highcut/(0.5*fs)], pass_zero=False)
    filtered = filtfilt(fir_coeff, [1.0], mean_filtered)
    
    # 4. Peak threshold: rolling mean over 0.5 s/HRC
    thresh_win = int(0.5 * period * fs)
    if thresh_win < 1:
        thresh_win = 1
    threshold = pd.Series(filtered).rolling(thresh_win, center=True, min_periods=1).mean().values
    
    # 5. Peak detection
    # distance to avoid double-counting: using 0.6 of period
    min_distance = max(1, int(0.6 * period * fs))
    peaks, props = find_peaks(filtered, distance=min_distance)  
    if len(peaks) > 0:
        per_peak_thresh = threshold[peaks]
        keep = filtered[peaks] > per_peak_thresh
        peaks = peaks[keep]
    
    # 6. HRP calculation
    if len(peaks) >= 2:
        ibis = np.diff(peaks) / fs
        HRP = 60.0 / np.mean(ibis)
    else:
        HRP = np.nan
    
    # 7. Quality control
    bpm_HRC = HRC * 60
    if np.isnan(HRP) or abs(bpm_HRC - HRP) > 10:
        HRP = np.nan  # discard
    
    return HRP, peaks

#---- HR time series features ----#
def extract_hr_features(hr, times, spike_threshold=10, spike_min_duration_sec=10):
    """
    Compute mean, SD, RMSSD, and number of heart rate spikes from heart rate series.
    """
    hr = pd.Series(hr, index=pd.to_datetime(times))
    hr_sleep = hr.dropna()
    if len(hr_sleep) < 2:
        return { 'mean_hr': np.nan, 'sd_hr': np.nan, 'rmssd_hr': np.nan, 'num_hr_spikes': np.nan }

    # Mean HR
    mean_hr = hr_sleep.mean()
    # SD HR
    sd_hr = hr_sleep.std()
    # RMSSD
    diff_hr = np.diff(hr_sleep.values)
    rmssd_hr = np.sqrt(np.mean(diff_hr**2)) if len(diff_hr) > 0 else np.nan

    # HR spikes: above median + threshold for at least min_duration
    median_hr = hr_sleep.median()
    is_spike = (hr_sleep > (median_hr + spike_threshold)).astype(int)
    spike_regions = []
    start = None
    idxs = hr_sleep.index
    for i, val in enumerate(is_spike):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i-1
            duration = (idxs[end] - idxs[start]).total_seconds()
            if duration >= spike_min_duration_sec:
                spike_regions.append((start, end, duration))
            start = None
    if start is not None:
        end = len(is_spike)-1
        duration = (idxs[end] - idxs[start]).total_seconds()
        if duration >= spike_min_duration_sec:
            spike_regions.append((start, end, duration))
    num_hr_spikes = len(spike_regions)

    features = {
        'mean_hr': mean_hr,
        'sd_hr': sd_hr,
        'rmssd_hr': rmssd_hr,
        'num_hr_spikes': num_hr_spikes,
        'spike_times': spike_times,
    }
    return features

#%% Main code: runs over all nights and stage segments

# Configuration
#data_folder = r"C:\Users\johan\Desktop\data_preprocessed\iRBD"
data_folder = r"C:\Users\johan\Desktop\data_preprocessed\Controls"
#hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\iRBD"
hypnogram_folder = r"C:\Users\johan\Desktop\data_hypnograms\Controls"
summary_folder = r"C:\Users\johan\Desktop\ML_features_sleep_stages"
os.makedirs(summary_folder, exist_ok=True)

min_stage_minutes = 5
stage_names = {0: "N3", 1: "N1/N2", 2: "REM", 3: "Wake"}
sleep_stage_groups = {
    "all_sleep": [0, 1, 2],
    "nrem": [0, 1],
    "rem": [2],
    "sleep_block": None}

all_night_files = sorted(glob.glob(os.path.join(data_folder, "*_night_*.h5")))
person_prefixes = sorted(set(os.path.basename(f).split("_night_")[0] for f in all_night_files))
stage_rows = {stage: [] for stage in sleep_stage_groups.keys()}

# Loop through each participant, night and sleep segment
for person_prefix in person_prefixes:
    night_files = sorted(glob.glob(os.path.join(data_folder, f"{person_prefix}_night_*.h5")))
    print(f"\nProcessing {person_prefix} ({len(night_files)} nights)")
    for filepath in night_files:
        # Load accelerometer data
        base_name = os.path.basename(filepath)
        print(f"  Processing night: {base_name}")
        night_df, fs = load_h5_acc(filepath)
        if night_df is None or night_df.shape[0] == 0 or fs is None:
            print(f"Skipping empty file: {base_name}")
            continue
        night_core_m = re.search(r'(night_\d{4}-\d{2}-\d{2})', base_name)
        if not night_core_m:
            print(f"Could not parse night core from {base_name}")
            continue
        # Load hypnogram data
        night_core = night_core_m.group(1)
        hyp_file = os.path.join(hypnogram_folder, f"{person_prefix}_{night_core}.npy")
        if not os.path.exists(hyp_file):
            print(f"  No hypnogram found for {base_name}")
            continue
        hyp_df = load_npy_hypnogram(hyp_file, epoch_sec=30)
        hyp_df_aligned = hyp_df.reindex(night_df.index, method='nearest')
        
        # Non-wear detection
        sleep_vector_night = hyp_df_aligned['stage_code'].isin([0,1,2]).astype(int)
        exclude, details = night_exclusion_criteria(night_df, fs, sleep_vector_night)
        if exclude:
            print(f"    Excluding {base_name}: {details['reasons']}")
            continue
        
        # For sleep_block: create a vector from first to last epoch scored as sleep (contiguous)
        sleep_codes = [0, 1, 2]
        sleep_indices = np.where(hyp_df_aligned['stage_code'].isin(sleep_codes))[0]
        sleep_block_vector = np.zeros_like(hyp_df_aligned['stage_code'])
        if len(sleep_indices) > 0:
            block_start = sleep_indices[0]
            block_end = sleep_indices[-1]
            sleep_block_vector[block_start:block_end+1] = 1
            
        for stage_grp, stage_codes in sleep_stage_groups.items():
            # Mask for each segment
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
                "minutes_of_data": minutes_of_data }
            
            if sleep_vector_for_window.sum() == 0 or minutes_of_data < min_stage_minutes:
                row['status'] = 'skipped'
                for f in ['mean_hr', 'sd_hr', 'rmssd_hr', 'num_hr_spikes']:
                    row[f] = np.nan
                stage_rows[stage_grp].append(row)
                continue
            
            # --- A. NIGHTBEAT SIGNAL ---
            nightbeat = compute_nightbeat_signal(night_df, fs)
            nightbeat_stage = nightbeat[sleep_vector_for_window == 1]
            
            # ---- B: ARTIFACT REMOVAL ----
            window_len = int(3 * 60 * fs) # 3-minute window
            step_size = int(60 * fs)      # 1-minute step (2 min overlap)
            
            all_HRC_times = []
            all_HRC_bpm = [] # HRC in bpm
            all_HRP_bpm = [] # HRP in bpm
            
            for i, start in enumerate(range(0, len(nightbeat_stage) - window_len + 1, step_size)):
                nb_seg = nightbeat_stage.iloc[start:start+window_len].values
                
                # Compute STFT
                f, t, Zxx = compute_stft_nightbeat(nb_seg, fs)
                
                # Compute spectral energy 
                stft_energy = np.sum(np.abs(Zxx), axis=0)
                
                # Valid window
                valid = (t >= 0) & (t <= window_len/fs)
                t = t[valid]
                stft_energy = stft_energy[valid]
                Zxx = Zxx[:, valid]
                
                # Artifact mask above 5 SD
                sigma = robust_sigma(stft_energy)
                sigma = sigma if sigma > 0 else (np.std(stft_energy) if np.std(stft_energy) > 0 else 1e-8)
                stft_energy_sigma = (stft_energy - np.median(stft_energy)) / sigma
                artifact_mask = stft_energy_sigma > 5
                
                # ---- C: CURVE TRACING ----
                # Find peaks, but "cut" the heart rate curve during artifact intervals
                peaks_all = find_hr_peaks(Zxx, f, t)
                
                # Set peaks to [] in artifact intervals to interrupt curves
                peaks_all_artifacted = [p if not artifact_mask[ti] else [] for ti, p in enumerate(peaks_all)]
                
                # Trace curves
                curves = trace_hr_curves_with_interrupt(peaks_all_artifacted, t, artifact_mask)
                
                # Select best curve
                hr_times, hr_freqs = select_best_curve(curves, t)
                
                # For 20s windows with 10s overlap, only use artifact-free intervals
                dt = t[1] - t[0] if len(t)>1 else 0.2
                win20_len = int(np.round(20.0/dt))
                step20 = int(np.round(10.0/dt))
                for j in range(0, len(hr_times)-win20_len+1, step20):
                    seg_freqs = hr_freqs[j:j+win20_len]
                    seg_times = hr_times[j:j+win20_len]
                    # Only use window if all points are in artifact-free intervals
                    seg_mask = np.array([not artifact_mask[np.argmin(np.abs(t-st))] for st in seg_times])
                    
                    # Calculate HRC: frequency-based heart rate metric
                    if len(seg_freqs) > 0 and np.all(seg_mask):
                        HRC = np.mean(seg_freqs)
                        mean_sample = int(np.round(np.mean(seg_times) * fs)) + start
                        if mean_sample >= 0 and mean_sample < len(nightbeat_stage):
                            all_HRC_times.append(nightbeat_stage.index[mean_sample])
                            all_HRC_bpm.append(HRC * 60)
                            
                            # ---- D: HEART BEAT DETECTION ----
                            # Centered 20s segment from nightbeat_segment (artifact-free)
                            center = np.mean(seg_times)
                            s0 = max(0, int((center-10)*fs))
                            s1 = min(len(nb_seg), int((center+10)*fs))
                            nb_20s = nb_seg[s0:s1]
                            
                            # Only run if long enough
                            if len(nb_20s) < int(0.8 * 20 * fs):
                                all_HRP_bpm.append(np.nan)
                                continue
                            
                            # Detect heartbeats
                            HRP, _ = detect_heartbeats(nb_20s, fs, HRC)
                            all_HRP_bpm.append(HRP)
                        else:
                            all_HRP_bpm.append(np.nan)
                            
            # ---- E: POST PROCESSING ----
            all_HRC_times = np.array(all_HRC_times)
            all_HRC_bpm = np.array(all_HRC_bpm)
            all_HRP_bpm = np.array(all_HRP_bpm)
            
            # Only keep windows where both HRC and HRP are valid and |HRC-HRP| ≤ 10 bpm
            valid_mask = (~np.isnan(all_HRC_bpm)) & (~np.isnan(all_HRP_bpm)) & (np.abs(all_HRC_bpm-all_HRP_bpm)<=10)
            final_HRC_times = all_HRC_times[valid_mask]
            final_HRC_bpm = all_HRC_bpm[valid_mask]
            
            # Rolling 5-min median, HRM
            if len(final_HRC_times) > 0:
                df = pd.DataFrame({"HRC": final_HRC_bpm}, index=pd.DatetimeIndex(final_HRC_times))
                df = df.sort_index()
                hrm = df['HRC'].rolling('5min', center=True, min_periods=1).median()
                
                # Final check: |HRC-HRM| ≤ 10 bpm
                final_mask = np.abs(df['HRC']-hrm) <= 10
                final_HRC_times = df.index[final_mask]
                final_HRC_bpm = df['HRC'][final_mask].values
            
            # --- HR feature extraction ---
            if len(final_HRC_bpm) > 1:
                features = extract_hr_features(final_HRC_bpm, final_HRC_times)
            else:
                features = { 'mean_hr': np.nan, 'sd_hr': np.nan, 'rmssd_hr': np.nan, 'num_hr_spikes': np.nan, 'spike_times': []}
            for f in features:
                row[f] = features[f]
            row['status'] = 'ok'
            stage_rows[stage_grp].append(row)
            
            # # PLOTTING (optional)
            # if stage_grp == "sleep_block" and len(final_HRC_bpm) > 2:
            #     # --- Prepare HR series for plotting ---
            #     hr_series = pd.Series(final_HRC_bpm, index=pd.to_datetime(final_HRC_times)).sort_index()
            #     hr_series = hr_series[(hr_series > 20) & (hr_series < 140)].dropna()
            #     mean_hr = hr_series.mean()
            #     std_hr = hr_series.std()
            
            #     # --- RMSSD calculation (2-min window, 30s step) ---
            #     window_s = 120
            #     step_s = 30
            #     times_rmssd = []
            #     rmssd_vals = []
            #     hr_times = hr_series.index
            #     start = hr_times[0]
            #     end = hr_times[-1]
            #     cur = start
            #     while cur + pd.Timedelta(seconds=window_s) <= end:
            #         mask = (hr_times >= cur) & (hr_times < cur + pd.Timedelta(seconds=window_s))
            #         win_vals = hr_series[mask].values
            #         if len(win_vals) > 2:
            #             rmssd = np.sqrt(np.mean(np.diff(win_vals)**2))
            #             times_rmssd.append(cur + pd.Timedelta(seconds=window_s/2))
            #             rmssd_vals.append(rmssd)
            #         cur += pd.Timedelta(seconds=step_s)
            #     rmssd_series = pd.Series(rmssd_vals, index=times_rmssd)
            #     overall_rmssd = np.sqrt(np.mean(np.diff(hr_series.values)**2)) if len(hr_series) > 2 else np.nan
            
            #     # Plot actigraphy, hypnogram, HR, and RMSSD
            #     fig, (ax_top, ax_hyp, ax_hr, ax_rmssd) = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.5, 1.5]})
            
            #     # 1. Top plot: actigraphy
            #     ax_top.plot(night_df.index, night_df['x'], label='x')
            #     ax_top.plot(night_df.index, night_df['y'], label='y')
            #     ax_top.plot(night_df.index, night_df['z'], label='z')
            #     min_acc = night_df[['x','y','z']].min().min()
            #     max_acc = night_df[['x','y','z']].max().max()
            #     ax_top.fill_between(night_df.index, min_acc, max_acc, where=sleep_vector_for_window == 1, color='green', alpha=0.2, label=f"{stage_grp} window")
            #     ax_top.set_title(f"Accelerometer data & {stage_grp} inference: {base_name}")
            #     ax_top.set_ylabel("Acceleration [g]")
            #     ax_top.legend(loc='upper left')
            
            #     # 2. Hypnogram
            #     ax_hyp.step(hyp_df_aligned.index, hyp_df_aligned['stage_code'], where='post', color='k')
            #     ax_hyp.set_ylabel("Sleep stage")
            #     ax_hyp.set_yticks([0, 1, 2, 3])
            #     ax_hyp.set_yticklabels(["N3", "N1/N2", "REM", "Wake"])
            #     ax_hyp.set_title("Hypnogram")
            
            #     # 3. HR time series
            #     ax_hr.plot(hr_series.index, hr_series.values, 'o', color='tab:blue', label='Estimated heart rate', markersize=2)
            #     spike_times = features.get('spike_times', [])
            #     y_min = mean_hr - std_hr
            #     y_max = mean_hr + std_hr
            #     for i, st in enumerate(spike_times):
            #         st_num = mdates.date2num(st)
            #         if i == 0:
            #             ax_hr.vlines(st_num, y_min, y_max, color='k', linewidth=2, linestyle='-', label='Heart rate spike')
            #         else:
            #             ax_hr.vlines(st_num, y_min, y_max, color='k', linewidth=2, linestyle='-')
            #     ax_hr.axhline(mean_hr, color='tab:red', linestyle='--', label=f'Mean heart rate: {mean_hr:.2f} bpm')
            #     ax_hr.text(
            #         0.98, 0.95,
            #         f"Standard deviation: {std_hr:.2f} bpm\n Number of heart rate spikes: {features['num_hr_spikes']}",
            #         ha='right', va='top', transform=ax_hr.transAxes,
            #         fontsize=10,
            #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6)
            #         )
            #     ax_hr.set_ylabel('Heart rate [bpm]')
            #     ax_hr.set_title('Estimated heart rate')
            #     ax_hr.legend(loc='upper left')
            #     ax_hr.grid(True, axis='y')
            
            #     # 4. Rolling RMSSD
            #     ax_rmssd.plot(rmssd_series.index, rmssd_series.values, '-', label='RMSSD (2-min window)', color='tab:blue')
            #     ax_rmssd.axhline(overall_rmssd, color='tab:red', linestyle='--', label=f'Mean RMSSD: {overall_rmssd:.2f} bpm')
            #     ax_rmssd.set_ylabel('RMSSD [bpm]')
            #     ax_rmssd.set_title('Heart rate variability (RMSSD)')
            #     ax_rmssd.legend(loc='upper left')
            #     ax_rmssd.grid(True, axis='y')
            #     plt.xlabel("Time")
            #     plt.tight_layout()
            #     plt.show()

# ---- Save CSVs ----
for stage_grp in sleep_stage_groups.keys():
    df_stage = pd.DataFrame(stage_rows[stage_grp])
    csv_path = os.path.join(summary_folder, f"Controls_{stage_grp}_hr.csv")
    #csv_path = os.path.join(summary_folder, f"iRBD_{stage_grp}_hr.csv")
    df_stage.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} with {len(df_stage)} rows.")