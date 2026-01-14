import os, glob, re, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
from scipy.signal import welch
from scipy.stats import entropy

#%% Functions

def load_h5_acc(filepath):
    """
    Load accelerometer (actigraphy) data and sample frequency from HDF5 file.
    """
    with h5py.File(filepath, "r") as h5:
        acc = h5["data"]["accelerometry"][:]
        fs = h5["data"]["accelerometry"].attrs["sample_frequency"]
        start_time = pd.to_datetime(h5.attrs["start_time"])
    acc_df = pd.DataFrame(acc.T, columns=["x", "y", "z"])
    time_index = pd.date_range(start=start_time, periods=acc_df.shape[0], freq=pd.to_timedelta(1/fs, unit="s"))
    acc_df.index = time_index
    return acc_df, fs

def find_txt_file(person_id, txt_folder):
    """
    Return the path to a hypnogram text file for a given participant.
    """
    pattern = os.path.join(txt_folder, f"*{person_id}*_hypnogram.txt")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def load_txt_hypnogram(filepath, first_stage_time, epoch_sec=30):
    """
    Load a manually scored hypnogram from plain text, aligning the first epoch to the provided time.
    """
    with open(filepath, "r") as f:
        stages = [line.strip() for line in f if line.strip()]
    idx_first = next((i for i, s in enumerate(stages) if s in ("N1", "N2")), None)
    if idx_first is None: raise ValueError("No N1 or N2 found in file!")
    first_stage_time = pd.Timestamp(first_stage_time)
    start_time = first_stage_time - pd.Timedelta(seconds=epoch_sec*idx_first)
    times = [start_time + pd.Timedelta(seconds=epoch_sec*i) for i in range(len(stages))]
    df = pd.DataFrame({"stage": stages}, index=pd.to_datetime(times))
    df["plot_stage"] = df["stage"].map({"N3":1, "N2":2, "N1":3, "R":4, "W":5})
    return df

def bin_hr_by_time(hr_times, hr_values, bin_size_s=10):
    """
    Bin heart rate series into fixed intervals (default: 10s).
    """
    bin_times = pd.to_datetime((hr_times.astype(np.int64) // (bin_size_s * 10**9)) * bin_size_s * 10**9)
    df = pd.DataFrame({"time": hr_times, "hr": hr_values, "bin_time": bin_times})
    binned = df.groupby("bin_time")["hr"].mean().reset_index().rename(columns={"bin_time": "time", "hr": "hr_binned"})
    return binned

def mask_by_epochs(times, epochs, epoch_length=pd.Timedelta(seconds=30)):
    """
    Boolean mask for samples that fall inside any set of epoch start times.
    """
    mask = np.zeros(len(times), dtype=bool)
    times_array = np.array(times)
    for start in epochs:
        mask |= (times_array >= start) & (times_array < start + epoch_length)
    return mask

def windowed_hrv_rmssd(hr_values, hr_times, window_length_s=120, step_s=30):
    """
    Calculate RMSSD (proxy HRV) in sliding windows for a given heart rate trace.
    """
    times, rmssd_vals = [], []
    if len(hr_times) < 3: return pd.Series([], index=[])
    start, end = hr_times.iloc[0], hr_times.iloc[-1]
    cur = start
    while cur + pd.Timedelta(seconds=window_length_s) <= end:
        mask = (hr_times >= cur) & (hr_times < cur + pd.Timedelta(seconds=window_length_s))
        win_hr = hr_values[mask]
        if len(win_hr) > 2:
            rmssd = np.sqrt(np.mean(np.diff(win_hr)**2))
            times.append(cur + pd.Timedelta(seconds=window_length_s/2))
            rmssd_vals.append(rmssd)
        cur += pd.Timedelta(seconds=step_s)
    return pd.Series(rmssd_vals, index=times)

def find_matching_file(directory, prefix, possible_dates, suffix):
    """
    Find file that matches prefix, one of several possible dates, and suffix.
    """
    for date in possible_dates:
        pattern = os.path.join(directory, f"{prefix}_night_{date}{suffix}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def get_possible_dates(night_datetime):
    """
    Return [date, date+1, date-1] for boundary matching.
    """
    night_date = pd.to_datetime(night_datetime).date()
    return [
        night_date.strftime("%Y-%m-%d"),
        (night_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        (night_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]


def get_frequency_features(hr_series, fs=1/10):
    """
    Calculate power features across HR frequency bands, peak freq, and spectral entropy.
    """
    hr = np.array(hr_series.dropna())
    if len(hr) < 10:
        return np.nan, np.nan
    hr = hr - np.mean(hr)
    f, Pxx = welch(hr, fs=fs, nperseg=min(256,len(hr)))
    band_a_mask = (f >= 0.003) & (f < 0.01)
    band_b_mask = (f >= 0.01) & (f <= 0.03)
    band_c_mask = (f >= 0.03) & (f <= 0.05)
    total_mask = (f >= 0.003) & (f <= 0.05)
    band_a_power = np.trapz(Pxx[band_a_mask], f[band_a_mask]) if np.any(band_a_mask) else np.nan
    band_b_power = np.trapz(Pxx[band_b_mask], f[band_b_mask]) if np.any(band_b_mask) else np.nan
    band_c_power = np.trapz(Pxx[band_c_mask], f[band_c_mask]) if np.any(band_c_mask) else np.nan
    total_power = np.trapz(Pxx[total_mask], f[total_mask]) if np.any(total_mask) else np.nan
    
    return band_a_power, band_b_power, band_c_power, total_power

#%% Configurations 
txt_times = {
    "SHAS004": "2023-10-18 22:21:03",
    "SHAS010": "2023-10-30 23:25:58",
    "SHAS012": "2025-01-15 00:22:45",
    "SHAS037": "2024-05-01 00:52:58",
    "SHAS052": "2024-05-21 00:40:17",
    "SHAS070": "2025-04-30 23:50:55",
    "SHAS071": "2025-01-14 23:58:47",
    "SHAS083": "2025-01-28 00:10:51",
    "SHAS094": "2025-03-26 23:24:09",}

all_person_prefixes = [
    "SHAS004_SleepStudy_10_23_L",
    "SHAS004_SleepStudy_10_23_R",
    "SHAS010_10_23_OVERNIGHT_L",
    "SHAS010_10_23_OVERNIGHT_R",
    #"SHAS012_OVERNIGHT_01_25_L",
    #"SHAS012_OVERNIGHT_01_25_R",
    "SHAS037_SleepStudy_05_24_L",
    "SHAS037_SleepStudy_05_24_R",
    "SHAS052_OVERNIGHT_05_24_L",
    "SHAS052_OVERNIGHT_05_24_R",
    "SHAS070_SleepStudy_04_25_L",
    "SHAS070_SleepStudy_04_25_R",
    "SHAS071_OVERNIGHT_01_25_L" ,
    "SHAS071_OVERNIGHT_01_25_R" ,
    "SHAS083_OVERNIGHT_01_25_L",
    "SHAS083_OVERNIGHT_01_25_R",]

edf_dir = r"C:\Users\johan\Desktop\stanford_cwa_clean\PSG"
h5_dir = r"C:\Users\johan\Desktop\data_preprocessed\New"
hr_dir = r"C:\Users\johan\Desktop\activity_summary\HR"
txt_folder = r"C:\Users\johan\Desktop\Hypnograms_vs_actigrams_txt"

#%% Preload ECG-derived HR for all participants and cache results
edf_cache = {}
for person_prefix in all_person_prefixes:
    m = re.match(r"(SHAS\d{3})", person_prefix)
    person_id_short = m.group(1) if m else person_prefix[:7]
    night_datetime = txt_times[person_id_short]
    possible_dates = get_possible_dates(night_datetime)
    edf_file = os.path.join(edf_dir, f"{person_id_short}_EDF.edf")
    if not os.path.exists(edf_file):
        edf_file = os.path.join(edf_dir, f"{person_id_short}_EDF_reduced.edf")
        if not os.path.exists(edf_file):
            continue
    if edf_file not in edf_cache:
        # Only include ECG1 for specific large/multi-channel files
        if os.path.basename(edf_file) in ("SHAS037_EDF.edf", "SHAS083_EDF.edf"):
            raw = mne.io.read_raw_edf(edf_file, preload=False, include=["ECG1"])
        else:
            raw = mne.io.read_raw_edf(edf_file, preload=False)
        ecg = raw.get_data(picks=["ECG1"])[0]
        fs = raw.info['sfreq']
        start_time = raw.info['meas_date']
        signals, info = nk.ecg_process(ecg, sampling_rate=fs)
        r_peaks = info['ECG_R_Peaks']
        r_peak_times = start_time + pd.to_timedelta(r_peaks / fs, unit="s")
        rr_intervals = np.diff(r_peak_times)
        rr_seconds = np.array([td.total_seconds() for td in rr_intervals])
        hr_from_rr = 60 / rr_seconds
        hr_times = r_peak_times[1:]
        edf_cache[edf_file] = {
            'ecg': ecg, 'fs': fs, 'start_time': start_time,
            'hr_times': pd.Series(pd.to_datetime(hr_times)).dt.tz_localize(None),
            'hr_from_rr': pd.Series(hr_from_rr)
        }

#%% Comparison loop: for each participant and segment

participant_metrics_allnight = []
participant_metrics_sleep = []
participant_metrics_rem = []
participant_metrics_nrem = []

for i, person_prefix in enumerate(all_person_prefixes):
    m = re.match(r"(SHAS\d{3})", person_prefix)
    person_id_short = m.group(1) if m else person_prefix[:7]
    night_datetime = txt_times[person_id_short]
    possible_dates = get_possible_dates(night_datetime)
    edf_file = os.path.join(edf_dir, f"{person_id_short}_EDF.edf")
    if not os.path.exists(edf_file):
        edf_file = os.path.join(edf_dir, f"{person_id_short}_EDF_reduced.edf")
        if not os.path.exists(edf_file):
            print(f"Missing EDF for {person_prefix}")
            continue
    accel_h5_file = find_matching_file(h5_dir, person_prefix, possible_dates, suffix=".h5")
    if accel_h5_file is None:
        print(f"Missing h5 for {person_prefix}")
        continue
    hr_csv_file = find_matching_file(hr_dir, person_prefix, possible_dates, suffix="_final_hr_full.csv")
    if hr_csv_file is None:
        print(f"Missing HR CSV for {person_prefix}")
        continue
    txt_path = find_txt_file(person_id_short, txt_folder)
    if txt_path is None:
        print(f"Missing hypnogram TXT for {person_prefix}")
        continue

    # ECG-derived HR from cache
    hr_times = edf_cache[edf_file]['hr_times']
    hr_from_rr = edf_cache[edf_file]['hr_from_rr']

    # Nightbeat-estimated HR
    acc_df, fs = load_h5_acc(accel_h5_file)
    hr_df = pd.read_csv(hr_csv_file, parse_dates=["time"])
    final_HRC_times = pd.Series(pd.to_datetime(hr_df["time"])).dt.tz_localize(None)
    final_HRC_bpm = pd.Series(hr_df["hr_bpm"])

    # Hypnogram (for segmentation)
    first_stage_time = txt_times[person_id_short]
    hypno_df = load_txt_hypnogram(txt_path, first_stage_time, epoch_sec=30)
    sleep_epochs = hypno_df.index[hypno_df["stage"].isin(["N1", "N2", "N3", "R"])]
    rem_epochs = hypno_df.index[hypno_df["stage"] == "R"]
    nrem_epochs = hypno_df.index[hypno_df["stage"].isin(["N1", "N2", "N3"])]
    epoch_length = pd.Timedelta(seconds=30)
    segment_defs = [
        ("All night", hr_times, []),
        ("Sleep window", sleep_epochs, ["sleep"]),
        ("REM window", rem_epochs, ["rem"]),
        ("NREM window", nrem_epochs, ["nrem"])]
    
    for segment_label, epochs, which in segment_defs:
        if len(epochs) == 0:
            # All night: mask everything as True
            mask_ecg = np.ones(len(hr_times), dtype=bool)
            mask_nb = np.ones(len(final_HRC_times), dtype=bool)
        else:
            mask_ecg = mask_by_epochs(hr_times, epochs, epoch_length)
            mask_nb = mask_by_epochs(final_HRC_times, epochs, epoch_length)
        # Mask into valid ranges
        valid_mask_ecg = (hr_from_rr > 20) & (hr_from_rr < 140)
        valid_mask_nb = (final_HRC_bpm > 20) & (final_HRC_bpm < 140)
        # Overlapping segments
        ecg_binned = bin_hr_by_time(hr_times[mask_ecg & valid_mask_ecg], hr_from_rr[mask_ecg & valid_mask_ecg])
        nb_binned = bin_hr_by_time(final_HRC_times[mask_nb & valid_mask_nb], final_HRC_bpm[mask_nb & valid_mask_nb])
        merged = pd.merge(ecg_binned, nb_binned, on="time", how="inner", suffixes=("_ecg", "_nightbeat"))
        valid = (merged.hr_binned_ecg > 20) & (merged.hr_binned_ecg < 140) & (merged.hr_binned_nightbeat > 20) & (merged.hr_binned_nightbeat < 140)
        x, y, t = merged.hr_binned_ecg[valid], merged.hr_binned_nightbeat[valid], merged.time[valid]
        
        # Frequency features (Welch, fs = 1/10 for 10s bins)
        band_a_ecg, band_b_ecg, band_c_ecg, total_power_ecg = get_frequency_features(x, fs=1/10)
        band_a_nb, band_b_nb, band_c_nb, total_power_nb = get_frequency_features(y, fs=1/10)
        
        # Compare/summary stats
        mae = np.mean(np.abs(y - x)) if len(x)>0 else np.nan
        rmse = np.sqrt(np.mean((y-x)**2)) if len(x)>0 else np.nan
        corr = np.corrcoef(y,x)[0,1] if len(x)>1 else np.nan
        rmssd_ecg = windowed_hrv_rmssd(x, t)
        rmssd_nb = windowed_hrv_rmssd(y, t)
        mean_ecg = x.mean()
        std_ecg = x.std()
        mean_nb = y.mean()
        std_nb = y.std()
        mean_rmssd_ecg = rmssd_ecg.mean()
        std_rmssd_ecg = rmssd_ecg.std()
        mean_rmssd_nb = rmssd_nb.mean()
        std_rmssd_nb = rmssd_nb.std()
        overall_rmssd_ecg = np.sqrt(np.mean(np.diff(x) ** 2)) if len(x)>1 else np.nan
        overall_rmssd_nb = np.sqrt(np.mean(np.diff(y) ** 2)) if len(y)>1 else np.nan
        metrics_dict = dict(
            person=person_prefix, segment=segment_label,
            mean_ecg=mean_ecg, std_ecg=std_ecg, mean_nb=mean_nb, std_nb=std_nb,
            rmssd_ecg_mean=mean_rmssd_ecg, rmssd_ecg_std=std_rmssd_ecg,
            rmssd_nb_mean=mean_rmssd_nb, rmssd_nb_std=std_rmssd_nb,
            rmssd_ecg_overall=overall_rmssd_ecg, rmssd_nb_overall=overall_rmssd_nb,
            mae=mae, rmse=rmse, corr=corr,
            band_a_ecg=band_a_ecg, band_b_ecg=band_b_ecg, band_c_ecg=band_c_ecg,
            band_a_nb=band_a_nb, band_b_nb=band_b_nb, band_c_nb=band_c_nb,
            total_power_ecg=total_power_ecg, total_power_nb=total_power_nb)
        
        # Save metrics per segment type for group summary
        if segment_label == "All night":
            participant_metrics_allnight.append(metrics_dict)
        elif "sleep" in which:
            participant_metrics_sleep.append(metrics_dict)
        elif "rem" in which:
            participant_metrics_rem.append(metrics_dict)
        elif "nrem" in which:
            participant_metrics_nrem.append(metrics_dict)
            
        # Visualization
        
        # a. Accelerometer data + hypnogram + heart rate
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1, 2]})
        axs[0].plot(acc_df.index, acc_df['x'], label='x', alpha=0.7)
        axs[0].plot(acc_df.index, acc_df['y'], label='y', alpha=0.7)
        axs[0].plot(acc_df.index, acc_df['z'], label='z', alpha=0.7)
        if "sleep" in which:
            min_acc = acc_df[['x','y','z']].min().min(); max_acc = acc_df[['x','y','z']].max().max()
            sleep_vector = pd.Series(0, index=acc_df.index)
            for epoch_start in sleep_epochs:
                sleep_vector[(acc_df.index >= epoch_start) & (acc_df.index < epoch_start + epoch_length)] = 1
            axs[0].fill_between(acc_df.index, min_acc, max_acc, where=sleep_vector==1, color='green', alpha=0.3, label='Sleep window')
        elif "rem" in which:
            min_acc = acc_df[['x','y','z']].min().min(); max_acc = acc_df[['x','y','z']].max().max()
            rem_vector = pd.Series(0, index=acc_df.index)
            for epoch_start in rem_epochs:
                rem_vector[(acc_df.index >= epoch_start) & (acc_df.index < epoch_start + epoch_length)] = 1
            axs[0].fill_between(acc_df.index, min_acc, max_acc, where=rem_vector==1, color='green', alpha=0.3, label='REM window')
        elif "nrem" in which:
            min_acc = acc_df[['x','y','z']].min().min(); max_acc = acc_df[['x','y','z']].max().max()
            nrem_vector = pd.Series(0, index=acc_df.index)
            for epoch_start in nrem_epochs:
                nrem_vector[(acc_df.index >= epoch_start) & (acc_df.index < epoch_start + epoch_length)] = 1
            axs[0].fill_between(acc_df.index, min_acc, max_acc, where=nrem_vector==1, color='green', alpha=0.3, label='NREM window')
        axs[0].set_title(f"Accelerometer data: {person_prefix}")
        axs[0].set_ylabel("Acceleration [g]")
        axs[0].legend(loc='upper right')
        axs[1].step(hypno_df.index, hypno_df["plot_stage"], where='post', color='black')
        axs[1].set_yticks([1,2,3,4,5])
        axs[1].set_yticklabels(["N3","N2","N1","REM","Wake"])
        axs[1].set_ylabel("Sleep stage")
        axs[1].set_title("Hypnogram")
        for epoch_start in rem_epochs:
            axs[1].plot([epoch_start, epoch_start + epoch_length], [4,4], color='red', linewidth=6, solid_capstyle='butt', zorder=5)
        axs[2].plot(t, x, 'x', markersize=2, label='ECG heart rate', color='tab:red')     
        axs[2].plot(t, y, 'o', markersize=2, label='Estimated heart rate', color='tab:blue')
        axs[2].set_ylabel("Heart rate [bpm]")
        axs[2].set_title(f"Heart rate comparison ({segment_label})")
        axs[2].legend(loc='upper right')
        axs[2].grid(True, axis='y')
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
        
        # b. Heart rate/RMSSD detailed plot
        fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        axs[0].plot(t, x, 'x', markersize=1, label='ECG HR (10s mean)', color='tab:red')
        axs[0].plot(t, y, 'o', markersize=1, label='Nightbeat HR', color='tab:blue')
        axs[0].axhline(mean_ecg, color='tab:red', linestyle='--', label=f'HR (ECG) Mean: {mean_ecg:.2f}, Std: {std_ecg:.2f}')
        axs[0].axhline(mean_nb, color='tab:blue', linestyle='--', label=f'HR (Nightbeat) Mean: {mean_nb:.2f}, Std: {std_nb:.2f}')
        axs[0].set_ylabel('HR (bpm)')
        axs[0].set_xlabel('Time')
        axs[0].set_title(f"Estimated HR vs. ECG HR (10s mean, {segment_label})")
        axs[0].legend(loc='upper right')
        axs[0].text(0.02, 0.98, f"MAE={mae:.2f} bpm\nRMSE={rmse:.2f} bpm\nCORR={corr:.3f}", ha='left', va='top', transform=axs[0].transAxes, fontsize=10)
        axs[1].plot(rmssd_ecg.index, rmssd_ecg.values, '-', label='RMSSD proxy (ECG)', color='tab:red')
        axs[1].plot(rmssd_nb.index, rmssd_nb.values, '-', label='RMSSD proxy (Nightbeat)', color='tab:blue')
        axs[1].axhline(mean_rmssd_ecg, color='tab:red', linestyle='--', label=f'RMSSD (ECG) Mean: {mean_rmssd_ecg:.2f}, Std: {std_rmssd_ecg:.2f}')
        axs[1].axhline(mean_rmssd_nb, color='tab:blue', linestyle='--', label=f'RMSSD (Nightbeat) Mean: {mean_rmssd_nb:.2f}, Std: {std_rmssd_nb:.2f}')
        axs[1].text(0.02, 0.98, f"Overall RMSSD (ECG): {overall_rmssd_ecg:.2f}\nOverall RMSSD (Nightbeat): {overall_rmssd_nb:.2f}",
                    ha='left', va='top', transform=axs[1].transAxes, fontsize=10)
        axs[1].set_ylabel("RMSSD (bpm)")
        axs[1].set_title(f"Windowed RMSSD ({segment_label}, 2 min windows)")
        axs[1].legend(loc='upper right')
        axs[1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()
        
        # c. Frequency/power (Welch) plot for each segment, with colored bands
        hr_ecg = x.dropna().values if hasattr(x, 'dropna') else np.array(x)
        hr_nb = y.dropna().values if hasattr(y, 'dropna') else np.array(y)
        fs = 1/10  # Sampling frequency in Hz (for 10s bins)
        
        if len(hr_ecg) >= 10 and len(hr_nb) >= 10:
            hr_ecg_detrended = hr_ecg - np.mean(hr_ecg)
            f_ecg, Pxx_ecg = welch(hr_ecg_detrended, fs=fs, nperseg=min(256, len(hr_ecg)))
            hr_nb_detrended = hr_nb - np.mean(hr_nb)
            f_nb, Pxx_nb = welch(hr_nb_detrended, fs=fs, nperseg=min(256, len(hr_nb)))
        
            # Band A: 0–0.02 Hz, Band B: 0.02–0.05 Hz
            band_a_mask = (f_ecg >= 0.003) & (f_ecg < 0.01)
            band_b_mask = (f_ecg >= 0.01) & (f_ecg <= 0.03)
            band_c_mask = (f_ecg >= 0.03) & (f_ecg <= 0.05)
            total_mask = (f_ecg >= 0.003) & (f_ecg <= 0.05)
        
            band_a_power_ecg = np.trapz(Pxx_ecg[band_a_mask], f_ecg[band_a_mask])
            band_b_power_ecg = np.trapz(Pxx_ecg[band_b_mask], f_ecg[band_b_mask])
            band_c_power_ecg = np.trapz(Pxx_ecg[band_c_mask], f_ecg[band_c_mask])
            total_power_ecg  = np.trapz(Pxx_ecg[total_mask], f_ecg[total_mask])
        
            band_a_power_nb = np.trapz(Pxx_nb[band_a_mask], f_nb[band_a_mask])
            band_b_power_nb = np.trapz(Pxx_nb[band_b_mask], f_nb[band_b_mask])
            band_c_power_nb = np.trapz(Pxx_nb[band_c_mask], f_nb[band_c_mask])
            total_power_nb  = np.trapz(Pxx_nb[total_mask], f_nb[total_mask])
                
            plt.figure(figsize=(10,5))
            plt.plot(f_ecg, Pxx_ecg, label='ECG HR', color='tab:red')
            plt.plot(f_nb, Pxx_nb, label='Nightbeat HR', color='tab:blue')
            plt.axvspan(0.003, 0.01, color='blue', alpha=0.15, label="Band A (0.003-0.01 Hz)")
            plt.axvspan(0.01, 0.03, color='orange', alpha=0.15, label="Band B (0.01-0.03 Hz)")
            plt.axvspan(0.03, 0.05, color='green', alpha=0.15, label="Band C (0.03-0.05 Hz)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (bpm²/Hz)")
            plt.title(f"HR Power Spectrum ({person_prefix}, {segment_label})")
            plt.legend(loc='upper right')
            plt.tight_layout()
            box_str = (
                f"ECG HR Total Power: {total_power_ecg:.2f} bpm²\n"
                f"Nightbeat HR Total Power: {total_power_nb:.2f} bpm²\n"
                f"ECG HR Power Band A: {band_a_power_ecg:.2f} bpm²\n"
                f"Nightbeat HR Power Band A: {band_a_power_nb:.2f} bpm²\n"
                f"ECG HR Power Band B: {band_b_power_ecg:.2f} bpm²\n"
                f"Nightbeat HR Power Band B: {band_b_power_nb:.2f} bpm²\n"
                f"ECG HR Power Band C: {band_c_power_ecg:.2f} bpm²\n"
                f"Nightbeat HR Power Band C: {band_c_power_nb:.2f} bpm²")        
            plt.text(0.98, 0.72, box_str, ha='right', va='top', transform=plt.gca().transAxes, fontsize=11,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
            plt.tight_layout()
            plt.show()
  
#%% Scatterplots

def plot_summary_scatter_multicondition(metric_dict, metric_name, xlab, ylab, title):
    """
    Produce summary scatter plots (ECG HR vs. estimated HR) for each segment type, with identity line and correlation.
    """
    colors = {'All night': 'tab:blue', 'Sleep window': 'tab:green', 'REM window': 'tab:red', 'NREM window': 'tab:orange'}
    plt.figure(figsize=(5,5))
    all_x = []
    all_y = []
    for label, df in metric_dict.items():
        if len(df) == 0: continue
        x = df[metric_name[0]].values
        y = df[metric_name[1]].values
        all_x.extend(x)
        all_y.extend(y)
        # Compute correlation for this group
        r = np.corrcoef(x, y)[0,1] if len(x) > 1 else np.nan
        plt.scatter(x, y, label=f"{label} (corr = {r:.2f})", color=colors.get(label, None), s=60, alpha=0.8, edgecolor='white')
    # Set axis limits
    min_val = min(all_x + all_y)
    max_val = max(all_x + all_y)
    margin = 0.05 * (max_val - min_val) if max_val > min_val else 1
    lim_low = min_val - margin
    lim_high = max_val + margin
    plt.xlim(lim_low, lim_high)
    plt.ylim(lim_low, lim_high)
    # Identity line
    plt.plot([lim_low, lim_high], [lim_low, lim_high], '--', color='gray', zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc='upper left')
    ## Overall correlation
    #if len(all_x) > 1:
    #    overall_corr = np.corrcoef(all_x, all_y)[0,1]
    #    plt.text(0.98, 0.02, f"Overall corr = {overall_corr:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    plt.show()

# Put main results into DataFrames
metric_dfs = {
    "All night": pd.DataFrame(participant_metrics_allnight),
    "Sleep window": pd.DataFrame(participant_metrics_sleep),
    "REM window": pd.DataFrame(participant_metrics_rem),
    "NREM window": pd.DataFrame(participant_metrics_nrem),}

# Grouped scatter plots for main heart rate features
plot_summary_scatter_multicondition(metric_dfs, ('mean_ecg', 'mean_nb'),xlab='Mean ECG heart rate [bpm]', ylab='Mean estimated heart rate [bpm]', title='Mean heart rate')
plot_summary_scatter_multicondition(metric_dfs, ('std_ecg', 'std_nb'), xlab='SD of ECG heart rate [bpm]', ylab='SD of estimated heart rate [bpm]', title='SD of heart rate')
plot_summary_scatter_multicondition(metric_dfs, ('rmssd_ecg_overall', 'rmssd_nb_overall'), xlab='RMSSD of ECG heart rate [bpm]', ylab='RMSSD of estimated heart rate [bpm]', title='RMSSD of heart rate')
plot_summary_scatter_multicondition(metric_dfs, ('total_power_ecg', 'total_power_nb'), xlab='Total power of ECG heart rate [bpm²]', ylab='Total power of estimated heart rate [bpm²]', title='Total power of heart rate (0.003 - 0.05 Hz)')
plot_summary_scatter_multicondition(metric_dfs, ('band_a_ecg', 'band_a_nb'), xlab='Band A power of ECG heart rate [bpm²]', ylab='Band A power of estimated heart rate [bpm²]', title='Band A power of heart rate (0.003-0.01 Hz)')
plot_summary_scatter_multicondition(metric_dfs, ('band_b_ecg', 'band_b_nb'), xlab='Band B power of ECG heart rate [bpm²]', ylab='Band B power of estimated heart rate [bpm²]', title='Band B power of heart rate (0.01-0.03 Hz)')
plot_summary_scatter_multicondition(metric_dfs, ('band_c_ecg', 'band_c_nb'), xlab='Band C power of ECG heart rate [bpm²]', ylab='Band C power of estimated heart rate [bpm²]', title='Band C power of heart rate (0.03-0.05 Hz)')

#%% Seperate plotting
def plot_separate_scatterplots_per_condition(metric_dfs, col_ecg, col_nb, xlab, ylab, title):
    """
    Plot scatterplots of ECG/Nightbeat metrics for each window (all night, sleep, REM, NREM) separately.
    """
    for label, df in metric_dfs.items():
        if len(df) == 0:
            continue
        plt.figure(figsize=(5,5))
        x = df[col_ecg].values
        y = df[col_nb].values
        plt.scatter(x, y, color='tab:blue', s=60, alpha=0.8, edgecolor='white')
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        margin = 0.05 * (max_val - min_val) if max_val > min_val else 1
        lim_low = min_val - margin
        lim_high = max_val + margin
        plt.xlim(lim_low, lim_high)
        plt.ylim(lim_low, lim_high)
        plt.plot([lim_low, lim_high], [lim_low, lim_high], '--', color='gray', zorder=0)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(f'{title} ({label})')
        if len(x) > 1:
            r = np.corrcoef(x, y)[0,1]
            plt.text(0.98, 0.02, f"Corr = {r:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes, 
                     fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        plt.tight_layout()
        plt.show()

metrics_to_plot = [
    ('mean_ecg', 'mean_nb', 'Mean HR ECG (bpm)', 'Mean HR Nightbeat (bpm)', 'Mean HR'),
    ('std_ecg', 'std_nb', 'Std HR ECG (bpm)', 'Std HR Nightbeat (bpm)', 'HR Std'),
    ('rmssd_ecg_overall', 'rmssd_nb_overall', 'RMSSD ECG (bpm)', 'RMSSD Nightbeat (bpm)', 'RMSSD'),
    ('rmssd_ecg_mean', 'rmssd_nb_mean', 'Mean RMSSD ECG (bpm)', 'Mean RMSSD Nightbeat (bpm)', 'Mean RMSSD'),
    ('rmssd_ecg_std', 'rmssd_nb_std', 'Std RMSSD ECG (bpm)', 'Std RMSSD Nightbeat (bpm)', 'Std RMSSD'),
    ('total_power_ecg', 'total_power_nb', 'Total power ECG HR (bpm²)', 'Total power Nightbeat HR (bpm²)', 'Total HR power'),
    ('band_a_ecg', 'band_a_nb', 'Band A power ECG HR (bpm²)', 'Band A power Nightbeat HR (bpm²)', 'Band A HR power'),
    ('band_b_ecg', 'band_b_nb', 'Band B power ECG HR (bpm²)', 'Band B power Nightbeat HR (bpm²)', 'Band B HR power'),
    ('band_c_ecg', 'band_c_nb', 'Band C power ECG HR (bpm²)', 'Band C power Nightbeat HR (bpm²)', 'Band C HR power')
]

for col_ecg, col_nb, xlab, ylab, title in metrics_to_plot:
    plot_separate_scatterplots_per_condition(metric_dfs, col_ecg, col_nb, xlab, ylab, title)