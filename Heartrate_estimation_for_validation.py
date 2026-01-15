import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from scipy.signal import stft, get_window
from scipy.signal import find_peaks, firwin

#%% Functions

def load_h5_acc(filepath: str):
    """Load a single night from HDF5 into a pandas DataFrame."""
    with h5py.File(filepath, "r") as h5:
        acc = h5["data"]["accelerometry"][:]  # shape (3, N)
        fs = h5["data"]["accelerometry"].attrs["sample_frequency"]
        start_time = pd.to_datetime(h5.attrs["start_time"])
    acc_df = pd.DataFrame(acc.T, columns=["x", "y", "z"])
    time_index = pd.date_range(start=start_time, periods=acc_df.shape[0], freq=pd.to_timedelta(1/fs, unit="s"))
    acc_df.index = time_index

    return acc_df, fs

def write_h5_acc(
    outfile: str,
    accelerometry: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    acc_info: dict,
    annotations: pd.DataFrame,
    study_start: pd.Timestamp,
    chunk_size_sec: int = 600,
    ahi: float = None,
    stage_hours: dict = None,
) -> None:
    '''Write the data to a h5 file'''
    with h5py.File(outfile, 'w') as f:
        f.create_group('annotations')
        f.create_group('data')
        try:
            f.attrs.create('start_time', study_start.strftime('%Y-%m-%d %H:%M:%S'))
        except AttributeError:
            f.attrs.create('start_time', study_start)
        if ahi is not None:
            f.attrs.create('ahi', ahi)
        if stage_hours is not None:
            for stage, hours in stage_hours.items():
                f.attrs.create(f'{stage}_hours', hours)
        
        # Write the calibrated accelerometry data
        try:
            acc_fs = round(1/accelerometry.index.diff().mean().total_seconds(), 2)
        except:
            acc_fs = round(1/np.nanmean(accelerometry.index.diff()), 2)
        acc = accelerometry[[x_col, y_col, z_col]].values
        dataset = f['data'].create_dataset(
            'accelerometry',
            data=acc.T.astype(np.float32),
            chunks=(acc.shape[1], chunk_size_sec * acc_fs),
        )
        for key, value in acc_info.items():
            dataset.attrs.create(key, str(value))
        dataset.attrs.create('sample_frequency', acc_fs)
            
        # Write the annotations
        try:
            annot_fs = round(1/annotations.index.diff().mean().total_seconds(), 2) 
        except: 
            annot_fs = round(1/np.nanmean(annotations.index.diff()), 2)
        for field in annotations.columns:
            annotation = annotations[field]
            dataset = f['annotations'].create_dataset(
                f'{field}',
                data=annotation.to_numpy(),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(chunk_size_sec * annot_fs,),
            )
            dataset.attrs.create('sample_frequency', annot_fs)
            
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
    return ( pd.Series(x_filt, index=acc_df.index), pd.Series(y_filt, index=acc_df.index), pd.Series(z_filt, index=acc_df.index), pd.Series(l2_norm, index=acc_df.index), pd.Series(nightbeat, index=acc_df.index))

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


def get_artifact_intervals(t, mask):
    """Return time intervals where mask is True (for artifact rejection)."""
    # mask: boolean array, t: time axis (1D)
    # Returns list of (start, end) intervals where mask is True
    mask = np.asarray(mask)
    t = np.asarray(t)
    # Find rising and falling edges
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(t[s], t[e-1] if e > 0 else t[s]) for s, e in zip(starts, ends)]

#---- HR peak finding (frequency domain) ----#
def find_hr_peaks(Sxx, f, t, freq_range=(0.5, 3.5)):
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
        sorted_indices = np.argsort(peak_freqs)
        peak_freqs = peak_freqs[sorted_indices]
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
    return peaks_all  # List of lists of freq (Hz), length = n_timesteps

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
                continue
            candidates = [j for j, pk in enumerate(pks) if abs(pk-last_f) <= freq_tol]
            if candidates:
                idx = candidates[0]
                new_curve = curve + [(ti, pks[idx])]
                new_active_curves.append(new_curve)
                assigned[idx] = True
            else:
                # Curve not continued, store if long enough
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
def detect_heartbeats( nightbeat_segment, fs, HRC, plot_example):
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
    
    # Plot example (optional)
    if plot_example:
        t = np.arange(len(nightbeat_segment)) / fs
        plt.figure(figsize=(10,3))
        plt.plot(t, nightbeat_segment, label='Nightbeat (raw)', color='blue')
        plt.plot(t, filtered, label='Smoothed + filtered', color='orange')
        plt.plot(t, threshold, label='Threshold', color='green')
        plt.scatter(t[peaks], filtered[peaks], color='purple', marker='x', label='Detected peaks')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [g]')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.title(f"Heartbeat peak detection: {base_name} (window 1)")
        plt.show()

    return HRP, peaks



#%% Main code, iterate manually for each person

#data_folder = r"C:\Users\johan\Desktop\data_preprocessed\Controls"
data_folder = r"C:\Users\johan\Desktop\data_preprocessed\New"
#person_prefix = "AXRBD067"  
person_prefix = "AXRBD025"  
file_pattern = os.path.join(data_folder, f"{person_prefix}_night_*.h5")
night_files = sorted(glob.glob(file_pattern))
hr_output_path = r"C:\Users\johan\Desktop\activity_summary\HR"

if not night_files:
    raise ValueError(f"No night files found with pattern: {file_pattern}")

print(f"Found {len(night_files)} night files:")
for f in night_files:
    print(" ", f)
    

n_nights = len(night_files)

for f in night_files:
    print(" ", f)

for filepath in night_files:
    base_name = os.path.basename(filepath)
    
    # Load data
    night_df, fs = load_h5_acc(filepath)
    
    # --- A. NIGHTBEAT SIGNAL ---
    x_filt, y_filt, z_filt, l2_norm, nightbeat = compute_nightbeat_signal(night_df, fs)
    
    l2_norm_sleep = l2_norm
    nightbeat_sleep = nightbeat
    time_sleep = nightbeat_sleep.index

    # Plot nightbeat signal (optional)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    # 1. Raw accelerometer data
    axs[0].plot(night_df.index, night_df['x'], label='x', alpha=0.7)
    axs[0].plot(night_df.index, night_df['y'], label='y', alpha=0.7)
    axs[0].plot(night_df.index, night_df['z'], label='z', alpha=0.7)
    axs[0].set_title(f"Accelerometer data: {base_name}")
    axs[0].set_ylabel("Acceleration [g]")
    axs[0].legend(loc='upper right')
    # 2. Bandpass filtered axes (5-14 Hz)
    axs[1].plot(night_df.index, x_filt, label='x (5-14 Hz)')
    axs[1].plot(night_df.index, y_filt, label='y (5-14 Hz)')
    axs[1].plot(night_df.index, z_filt, label='z (5-14 Hz)')
    axs[1].set_title("Bandpass filtered axes (5–14 Hz)")
    axs[1].set_ylabel("Acceleration [g]")
    axs[1].legend(loc='upper right')
    # 3. L2-norm (full) and Nightbeat (sleep only) on same plot
    axs[2].plot(l2_norm.index, l2_norm, color='purple', label='L2-norm')
    axs[2].plot(time_sleep, nightbeat_sleep, color='red', label='Nightbeat signal (0.5-3.5 Hz)')
    axs[2].set_title("L2-norm of filtered axes & Nightbeat signal")
    axs[2].set_ylabel("Acceleration [g]")
    axs[2].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    
    # ---- B: ARTIFACT REMOVAL ----
    window_len = int(3 * 60 * fs)  # 3 minutes
    step_size = int(60 * fs)       # 1 minute stride (2 min overlap)
    n_windows = (len(nightbeat_sleep) - window_len) // step_size + 1 if len(nightbeat_sleep) >= window_len else 0

    all_HRC_times = []
    all_HRC_freqs = []
    all_HRC_bpm = []  # HRC in bpm
    all_HRP_bpm = []  # HRP in bpm
    
    for i, start in enumerate(range(0, len(nightbeat_sleep) - window_len + 1, step_size)):
        # Make window
        nightbeat_segment = nightbeat_sleep.iloc[start:start+window_len].values
        nb_times = np.arange(window_len) / fs
        
        # Compute STFT
        f, t, Zxx = compute_stft_nightbeat(nightbeat_segment, fs)
        
        # Compute spectral energy (within 5 SD)
        stft_energy = np.sum(np.abs(Zxx), axis=0)
        
        # Valid window 
        valid = (t >= 0) & (t <= window_len/fs)
        t = t[valid]
        stft_energy = stft_energy[valid]
        Zxx = Zxx[:, valid]
        
        # Artifact mask above 5 SD
        sigma = robust_sigma(stft_energy)
        if sigma <= 0:
            sigma = np.std(stft_energy) if np.std(stft_energy) > 0 else 1e-8
        stft_energy_sigma = (stft_energy - np.median(stft_energy)) / sigma
        artifact_mask = stft_energy_sigma > 5#5
        artifact_intervals = get_artifact_intervals(t, artifact_mask)
    
        # ---- C: CURVE TRACING ----
        # Find peaks, but "cut" the HR curve during artifact intervals
        peaks_all = find_hr_peaks(Zxx, f, t)
        # Set peaks to [] in artifact intervals to interrupt curves
        peaks_all_artifacted = [p if not artifact_mask[ti] else [] for ti, p in enumerate(peaks_all)]
        curves = trace_hr_curves_with_interrupt(peaks_all_artifacted, t, artifact_mask, gap_steps=5, freq_tol=0.05, min_curve_len=10)

        hr_times, hr_freqs = select_best_curve(curves, t)
        # For 20s windows with 10s overlap, only use artifact-free intervals
        dt = t[1] - t[0]
        win20_len = int(np.round(20.0 / dt))
        step20 = int(np.round(10.0 / dt))
        for j in range(0, len(hr_times) - win20_len + 1, step20):
            seg_freqs = hr_freqs[j:j+win20_len]
            seg_times = hr_times[j:j+win20_len]
            # Only use window if all points are in artifact-free intervals
            seg_mask = np.array([not artifact_mask[np.argmin(np.abs(t - st))] for st in seg_times])
            
            # Calculate HRC: frequency-based heart rate metric
            if len(seg_freqs) > 0 and np.all(seg_mask):
                HRC = np.mean(seg_freqs)
                mean_sample = int(np.round(np.mean(seg_times) * fs)) + start
                if mean_sample >= 0 and mean_sample < len(nightbeat_sleep):
                    all_HRC_times.append(nightbeat_sleep.index[mean_sample])
                    all_HRC_freqs.append(HRC)
                    all_HRC_bpm.append(HRC * 60)
                else:
                    continue
                
                # ---- D: HEART BEAT DETECTION ----
                # Centered 20s segment from nightbeat_segment (artifact-free)
                seg_center = np.mean(seg_times)
                seg_start = max(0, int((seg_center - 10) * fs))
                seg_end = min(len(nightbeat_segment), int((seg_center + 10) * fs))
                nb_20s = nightbeat_segment[seg_start:seg_end]
    
                # Only run if long enough
                if len(nb_20s) < int(0.8 * 20 * fs):
                    continue
    
                # Call the function
                HRP, peaks = detect_heartbeats(nb_20s, fs, HRC, plot_example=True)
                all_HRP_bpm.append(HRP)


        # Plot STFT for first 3 windows (optional)
        if i < 3:
            fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            axs[0].plot(nb_times, nightbeat_segment, color='tab:blue')
            axs[0].set_ylabel("Acceleration [g]")
            axs[0].set_xlim([0, nb_times[-1]])
            axs[0].set_title(f"Nightbeat Signal: {base_name} (window {i+1})")
            for t0, t1 in artifact_intervals:
                axs[0].axvspan(t0, t1, color='red', alpha=0.3)
            
            im = axs[1].pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
            axs[1].set_ylabel('STFT [Hz]')
            axs[1].set_ylim([0, 3.5])
            for t0, t1 in artifact_intervals:
               axs[1].axvspan(t0, t1, color='red', alpha=0.3)
            # Overlay HR curve if present
            if len(hr_times) > 0:
                axs[1].plot(hr_times, hr_freqs, color='yellow', lw=2, label='Estimated HR curve')
                axs[1].legend()
                        
            axs[2].plot(t, stft_energy_sigma, color='tab:blue')
            axs[2].axhline(5, color='red', linestyle='-', label='Motion artifact threshold')
            axs[2].set_ylabel("Total STFT energy (σ above median)")
            axs[2].set_xlabel("Time [s]")
            axs[2].legend()
            for t0, t1 in artifact_intervals:
                axs[2].axvspan(t0, t1, color='red', alpha=0.3)
            plt.tight_layout()
            plt.show()

    
    # ---- E: POST PROCESSING ----
    all_HRC_times = np.array(all_HRC_times)
    all_HRC_bpm = np.array(all_HRC_bpm)
    all_HRP_bpm = np.array(all_HRP_bpm)

    # 1. Only keep windows where both HRC and HRP are valid and |HRC-HRP| ≤ 10 bpm
    valid_mask = (~np.isnan(all_HRC_bpm)) & (~np.isnan(all_HRP_bpm)) & (np.abs(all_HRC_bpm - all_HRP_bpm) <= 10)
    valid_HRC_times = all_HRC_times[valid_mask]
    valid_HRC_bpm = all_HRC_bpm[valid_mask]
    
    # 2. Rolling 5-min median
    if len(valid_HRC_times) > 0:
        df = pd.DataFrame({"HRC": valid_HRC_bpm}, index=pd.DatetimeIndex(valid_HRC_times))
        df = df.sort_index()  # Ensure index is monotonic increasing
        hrm = df['HRC'].rolling('5min', center=True, min_periods=1).median()
    
        # 3. Final check: |HRC-HRM| ≤ 10 bpm
        final_mask = np.abs(df['HRC'] - hrm) <= 10
        final_HRC_times = df.index[final_mask]
        final_HRC_bpm = df['HRC'][final_mask].values
        final_HRM_bpm = hrm[final_mask].values

        # PLot HR predictions (optional)
        fig, axs = plt.subplots(2, 1, figsize=(15, 7), sharex=True) #figsize=(15, 16)
        # 1. Raw accelerometer data
        axs[0].plot(night_df.index, night_df['x'], label='x', alpha=0.7)
        axs[0].plot(night_df.index, night_df['y'], label='y', alpha=0.7)
        axs[0].plot(night_df.index, night_df['z'], label='z', alpha=0.7)
        axs[0].set_title(f"Accelerometer data: {base_name}")
        axs[0].set_ylabel("Acceleration [g]")
        axs[0].legend(loc='upper right')
        # 2. Final HR predictions
        axs[1].plot(final_HRC_times, final_HRC_bpm, 'o', markersize=1, label='Final HR Prediction', color='tab:blue')
        axs[1].set_ylabel("Heart rate [bpm]")
        axs[1].legend(loc='upper right')
        axs[1].set_title("Final heart rate predictions")
        axs[1].legend()
        axs[1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()
        
        # --- Save final HR time series ---
        final_hr_df = pd.DataFrame({'time': final_HRC_times, 'hr_bpm': final_HRC_bpm})
        hr_csv_path = os.path.join(hr_output_path, f"{base_name.replace('.h5', '')}_final_hr_full.csv")
        final_hr_df.to_csv(hr_csv_path, index=False)
        print(f"Saved final HR estimate to {hr_output_path}")
        