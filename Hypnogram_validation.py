import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import glob
import h5py

#%% Functions

def load_rtf_hypnogram(filepath):
    """
    Load manually scored hypnogram from an RTF-formatted file and parse into a DataFrame.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        rtf_content = f.read()
    
    # Remove RTF codes and extract only the sleep staging table
    text = re.sub(r"{\\.*?}", "", rtf_content)
    text = re.sub(r"\\[a-z]+\d*", "", text)
    text = re.sub(r"[{}]", "", text)
    text = text.replace("\r", "\n").replace("\n\n", "\n")
    match = re.search(r"Signal ID: SchlafProfil.*?Events list:.*?\n(.*?)(?:Signal ID:|\Z)", text, re.DOTALL)
    if not match:
        raise ValueError("Couldn't find SchlafProfil section in RTF file.")
    schlaf_text = match.group(1)
    lines = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3});\s*([A-Za-z0-9]+)", schlaf_text)
    start_match = re.search(r"Start Time:\s*(\d{1,2}/\d{1,2}/\d{4}) (\d{1,2}:\d{2}:\d{2}) (AM|PM)", text)
    if start_match:
        date_str = start_match.group(1)
        time_str = start_match.group(2)
        ampm = start_match.group(3)
        datetime_str = f"{date_str} {time_str} {ampm}"
        base_date = datetime.strptime(datetime_str, "%m/%d/%Y %I:%M:%S %p").date()
    else:
        base_date = datetime(9999, 99, 99).date()
    datetimes = []
    prev_time = None
    curr_date = base_date
    for time_str, stage_str in lines:
        t_clean = time_str.replace(",", ".")
        curr_time = datetime.strptime(t_clean, "%H:%M:%S.%f").time()
        if prev_time and curr_time < prev_time:
            curr_date += timedelta(days=1)
        dt = datetime.combine(curr_date, curr_time)
        datetimes.append(dt)
        prev_time = curr_time
    stages = [l[1] for l in lines]
    hypnogram_df = pd.DataFrame({"stage": stages}, index=pd.to_datetime(datetimes))
    # Map stage names to numeric codes
    hypnogram_df["stage_code"] = hypnogram_df["stage"].map({"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "Rem": 4})
    return hypnogram_df

def find_txt_file(person_id, txt_folder):
    """
    Return the path to a hypnogram text file for a given participant.
    """
    pattern = os.path.join(txt_folder, f"*{person_id}*_hypnogram.txt")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None

def load_txt_hypnogram(filepath, first_stage_time, epoch_sec=30):
    """
    Load a manually scored hypnogram from plain text, aligning the first epoch to the provided time.
    """
    with open(filepath, "r") as f:
        stages = [line.strip() for line in f if line.strip()]
    # Find first occurrence of N1/N2 to estimate clock time alignment
    idx_first = None
    for i, stage in enumerate(stages):
        if stage in ("N1", "N2"):
            idx_first = i
            break
    if idx_first is None:
        raise ValueError("No N1 or N2 found in file!")
    if not isinstance(first_stage_time, pd.Timestamp):
        first_stage_time = pd.Timestamp(first_stage_time)
    # Infer time for the first epoch, then build epoch-by-epoch index
    start_time = first_stage_time - pd.Timedelta(seconds=epoch_sec*idx_first)
    times = [start_time + pd.Timedelta(seconds=epoch_sec*i) for i in range(len(stages))]
    stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
    df = pd.DataFrame({"stage": stages}, index=pd.to_datetime(times))
    df["stage_code"] = df["stage"].map(stage_map)
    return df

def remap_stage_code(orig_code):
    """
    Remap 5-class annotation to 4-class (N3=0, N1/N2=1, REM=2, Wake=3).
    """
    if orig_code == 3: return 0     # N3/Deep Sleep
    if orig_code in [1, 2]: return 1 # Light Sleep N1/N2
    if orig_code == 4: return 2      # REM
    if orig_code == 0: return 3      # Wake
    return np.nan

def load_npy_predicted_hypnogram(filepath, epoch_sec=30):
    """
    Load a model-predicted sleep hypnogram (.npy), producing DataFrame over time.
    """
    m = re.search(r"_night_(\d{4}-\d{2}-\d{2})", filepath)
    if not m:
        raise ValueError("Could not extract date from npy filename.")
    date_str = m.group(1)
    start_time = pd.Timestamp(f"{date_str} 21:00:00")
    prob_arr = np.load(filepath)
    is_empty = np.nansum(prob_arr, axis=1) == 0
    stage_codes = np.full(prob_arr.shape[0], np.nan)
    stage_codes[~is_empty] = np.argmax(prob_arr[~is_empty], axis=1)
    stage_map = {0: "N3", 1: "N1/N2", 2: "REM", 3: "Wake"}
    times = [start_time + pd.Timedelta(seconds=epoch_sec * i) for i in range(len(stage_codes))]
    stages = [stage_map.get(code, np.nan) if not np.isnan(code) else np.nan for code in stage_codes]
    df = pd.DataFrame({
        "stage_code": stage_codes,
        "stage": stages
    }, index=pd.to_datetime(times))
    return df

def load_h5_acc_with_annotations(filepath: str):
    """
    Load accelerometer data and state annotation from HDF5 file, if present.
    """
    with h5py.File(filepath, "r") as h5:
        acc = h5["data"]["accelerometry"][:]  # shape (3, N)
        fs = h5["data"]["accelerometry"].attrs["sample_frequency"]
        start_time = pd.to_datetime(h5.attrs["start_time"])
        if "state" in h5["annotations"]:
            state = h5["annotations"]["state"][:]
            state = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in state]
        else:
            state = None
    acc_df = pd.DataFrame(acc.T, columns=["x", "y", "z"])
    time_index = pd.date_range(
        start=start_time,
        periods=acc_df.shape[0],
        freq=pd.to_timedelta(1/fs, unit="s"),
    )
    acc_df.index = time_index
    if state is not None:
        annot_df = pd.DataFrame({"state": state}, index=acc_df.index)
        return acc_df, fs, annot_df
    else:
        return acc_df, fs, None

#%% Main code

# Configurations
rtf_folder = r"C:\\Users\\johan\\Desktop\\Hypnograms_vs_actigrams"
txt_folder = r"C:\\Users\\johan\\Desktop\\Hypnograms_vs_actigrams_txt"
npy_folder = r"C:\\Users\\johan\\Desktop\\data_hypnograms_validation"
h5_folder  = r"C:\\Users\\johan\\Desktop\\wake_sleep_comp"

# Starttimes
txt_times = {
    "SHAS004": "2023-10-18 22:21:03",
    "SHAS010": "2023-10-30 23:25:58",
    "SHAS012": "2025-01-15 00:22:45",
    "SHAS037": "2024-05-01 00:52:58",
    "SHAS052": "2024-05-21 00:40:17",
    "SHAS070": "2025-04-30 23:50:55",
    "SHAS071": "2025-01-14 23:58:47",
    "SHAS083": "2025-01-28 00:10:51",
    "SHAS094": "2025-03-26 23:24:09",
}

# Loop through each file
npy_files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
summary_rows = []
for npy_file in npy_files:
    # Parse identifiers from filename
    m = re.match(r"(.+?)_night_(\d{4}-\d{2}-\d{2})\.npy", npy_file)
    if not m:
        print("Skipping npy file (pattern mismatch):", npy_file)
        continue
    person = m.group(1)
    night = m.group(2)
    base = f"{person}_night_{night}"

    # Locate and load ground truth annotation
    rtf_path = os.path.join(rtf_folder, f"{person}.rtf")
    match_short = re.match(r"(SHAS\d{3})", person)
    if match_short:
        person_id_short = match_short.group(1)
    else:
        person_id_short = person[:7]
    txt_path = find_txt_file(person_id_short, txt_folder)
    has_txt_file = txt_path is not None
    has_txt_time = person_id_short in txt_times

    if os.path.exists(rtf_path):
        gt_df = load_rtf_hypnogram(rtf_path)
    elif has_txt_file and has_txt_time:
        first_stage_time = txt_times[person_id_short]
        gt_df = load_txt_hypnogram(txt_path, first_stage_time, epoch_sec=30)
    else:
        print(f"No valid ground truth for {base}")
        continue

    gt_df = gt_df.copy()
    gt_df["stage_code_remap"] = gt_df["stage_code"].map(remap_stage_code)
    
    # Load model-predicted hypnodensity (softmax) and derive sleep stages
    pred_df = load_npy_predicted_hypnogram(os.path.join(npy_folder, npy_file), epoch_sec=30)

    # Align predictions and ground truth using the overlapping time
    min_time = max(pred_df.index.min(), gt_df.index.min())
    max_time = min(pred_df.index.max(), gt_df.index.max())
    all_bins = pd.date_range(min_time, max_time, freq='30s')
    pred_binned = pred_df['stage_code'].reindex(all_bins, method='nearest')
    gt_binned = gt_df['stage_code_remap'].reindex(all_bins, method='nearest')
    valid_idx = (~gt_binned.isna()) & (~pred_binned.isna())
    true_stages_final = gt_binned[valid_idx].astype(int)
    pred_stages_final = pred_binned[valid_idx].astype(int)
    labels = [0, 1, 2, 3]  # N3, N1/N2, REM, Wake
    names = ["N3", "N1/N2", "REM", "Wake"]

    # 4-class confusion matrix and metrics
    cm = confusion_matrix(true_stages_final, pred_stages_final, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'{base}')
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(true_stages_final, pred_stages_final)
    f1_macro = f1_score(true_stages_final, pred_stages_final, labels=labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_stages_final, pred_stages_final, labels=labels, average='macro', zero_division=0)
    precision_macro = precision_score(true_stages_final, pred_stages_final, labels=labels, average='macro', zero_division=0)
    recall_per_class_float_raw = recall_score(true_stages_final, pred_stages_final, labels=labels, average=None, zero_division=0)
    recall_per_class_float = [rec if (true_stages_final == lab).sum() > 0 else np.nan for rec, lab in zip(recall_per_class_float_raw, labels)]

    # For printing recall for each class
    recall_per_class = []
    for idx, label in enumerate(labels):
        n_true = (true_stages_final == label).sum()
        if n_true == 0:
            recall_per_class.append("N/A")
        else:
            rc_val = recall_score(true_stages_final, pred_stages_final, labels=[label], average='macro', zero_division=0)
            recall_per_class.append(f"{rc_val:.3f}")

    print(f"\n----- 4-class metrics for {base}:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro Recall:    {recall_macro:.3f}")
    print(f"Macro Precision: {precision_macro:.3f}")
    print(f"Macro F1:        {f1_macro:.3f}")
    for idx, name_lab in enumerate(names):
        print(f"Recall {name_lab}:   {recall_per_class[idx]}")

    # 3-class (REM/REM/Wake) confusion matrix and metrics
    def combine_nrem_rem_wake(code):
        if code in [0,1]: return 0   # NREM
        if code == 2: return 1       # REM
        if code == 3: return 2       # Wake
        return np.nan
    true_stages_nrem = true_stages_final.map(combine_nrem_rem_wake)
    pred_stages_nrem = pred_stages_final.map(combine_nrem_rem_wake)
    valid_idx_nrem = (~true_stages_nrem.isna()) & (~pred_stages_nrem.isna())
    true_stages_nrem = true_stages_nrem[valid_idx_nrem].astype(int)
    pred_stages_nrem = pred_stages_nrem[valid_idx_nrem].astype(int)
    labels_nrem = [0,1,2]
    names_nrem = ["NREM", "REM", "Wake"]
    cm_nrem = confusion_matrix(true_stages_nrem, pred_stages_nrem, labels=labels_nrem)
    disp_nrem = ConfusionMatrixDisplay(confusion_matrix=cm_nrem, display_labels=names_nrem)
    plt.figure(figsize=(6,6))
    disp_nrem.plot(cmap='Blues', values_format='d')
    plt.title(f'{base} Confusion Matrix (NREM/REM/Wake)')
    plt.tight_layout()
    plt.show()

    acc_nrem = accuracy_score(true_stages_nrem, pred_stages_nrem)
    f1_macro_nrem = f1_score(true_stages_nrem, pred_stages_nrem, labels=labels_nrem, average='macro', zero_division=0)
    recall_macro_nrem = recall_score(true_stages_nrem, pred_stages_nrem, labels=labels_nrem, average='macro', zero_division=0)
    precision_macro_nrem = precision_score(true_stages_nrem, pred_stages_nrem, labels=labels_nrem, average='macro', zero_division=0)
    recall_per_class_nrem_raw = recall_score(true_stages_nrem, pred_stages_nrem, labels=labels_nrem, average=None, zero_division=0)
    recall_per_class_nrem = [rec if (true_stages_nrem == lab).sum() > 0 else np.nan for rec, lab in zip(recall_per_class_nrem_raw, labels_nrem)]

    # 2-class (Sleep/Wake) confusion matrix and metrics 
    def combine_sleep_wake(code):
        if code in [0,1,2]: return 0    # Sleep
        if code == 3: return 1          # Wake
        return np.nan
    true_stages_sleep = true_stages_final.map(combine_sleep_wake)
    pred_stages_sleep = pred_stages_final.map(combine_sleep_wake)
    valid_idx_sleep = (~true_stages_sleep.isna()) & (~pred_stages_sleep.isna())
    true_stages_sleep = true_stages_sleep[valid_idx_sleep].astype(int)
    pred_stages_sleep = pred_stages_sleep[valid_idx_sleep].astype(int)
    labels_sleep = [0,1]
    names_sleep = ["Sleep", "Wake"]
    cm_sleep = confusion_matrix(true_stages_sleep, pred_stages_sleep, labels=labels_sleep)
    disp_sleep = ConfusionMatrixDisplay(confusion_matrix=cm_sleep, display_labels=names_sleep)
    plt.figure(figsize=(6,6))
    disp_sleep.plot(cmap='Blues', values_format='d')
    plt.title(f'{base} Confusion Matrix (Sleep/Wake)')
    plt.tight_layout()
    plt.show()

    acc_sleep = accuracy_score(true_stages_sleep, pred_stages_sleep)
    f1_macro_sleep = f1_score(true_stages_sleep, pred_stages_sleep, labels=labels_sleep, average='macro', zero_division=0)
    recall_macro_sleep = recall_score(true_stages_sleep, pred_stages_sleep, labels=labels_sleep, average='macro', zero_division=0)
    precision_macro_sleep = precision_score(true_stages_sleep, pred_stages_sleep, labels=labels_sleep, average='macro', zero_division=0)
    recall_per_class_sleep_raw = recall_score(true_stages_sleep, pred_stages_sleep, labels=labels_sleep, average=None, zero_division=0)
    recall_per_class_sleep = [rec if (true_stages_sleep == lab).sum() > 0 else np.nan for rec, lab in zip(recall_per_class_sleep_raw, labels_sleep)]

    # Plot section: visualize data for each night
    h5_pattern = os.path.join(h5_folder, f"{base}_wake_sleep.h5")
    h5_matches = glob.glob(h5_pattern)
    acc_df = None
    if h5_matches:
        h5_path = h5_matches[0]
        acc_df, acc_fs, annot_df = load_h5_acc_with_annotations(h5_path)
    prob_arr = np.load(os.path.join(npy_folder, npy_file))  # shape (epochs, n_stages)
    start_time = pd.Timestamp(f"{night} 21:00:00")
    times = [start_time + pd.Timedelta(seconds=30*i) for i in range(prob_arr.shape[0])]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    if acc_df is not None:
        axes[0].plot(acc_df.index, acc_df["x"], label='x')
        axes[0].plot(acc_df.index, acc_df["y"], label='y')
        axes[0].plot(acc_df.index, acc_df["z"], label='z')
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel("Acceleration [g]")
        axes[0].set_title("Accelerometer data")
    axes[1].stackplot(times, prob_arr.T, labels=names)
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Predicted hypnodensity")
    axes[1].legend(loc="upper right")
    axes[2].step(pred_df.index, pred_df['stage_code'], where='post', color="blue", label="Predicted sleep stages")
    axes[2].set_ylabel("Sleep stage")
    axes[2].set_yticks(labels)
    axes[2].set_yticklabels(names)
    axes[2].legend(loc='upper right')
    axes[2].set_title("Predicted hypnogram")
    axes[3].step(gt_df.index, gt_df["stage_code_remap"], where='post', color="red", label="Ground truth sleep stages")
    axes[3].set_ylabel("Sleep stage")
    axes[3].set_yticks(labels)
    axes[3].set_yticklabels(names)
    axes[3].legend(loc='upper right')
    axes[3].set_title("Ground truth hypnogram")
    axes[3].set_xlabel("Time")
    plt.suptitle(base)
    plt.tight_layout()
    plt.show()

    summary_rows.append({
        "person": person,
        "night": night,
        # 4-class
        "accuracy_4class": acc,
        "recall_macro_4class": recall_macro,
        "precision_macro_4class": precision_macro,
        "f1_macro_4class": f1_macro,
        "recall_N3": recall_per_class_float[0],
        "recall_N1N2": recall_per_class_float[1],
        "recall_REM": recall_per_class_float[2],
        "recall_Wake": recall_per_class_float[3],
        # 3-class
        "accuracy_3class": acc_nrem,
        "recall_macro_3class": recall_macro_nrem,
        "precision_macro_3class": precision_macro_nrem,
        "f1_macro_3class": f1_macro_nrem,
        "recall_NREM": recall_per_class_nrem[0],
        "recall_REM_3c": recall_per_class_nrem[1],
        "recall_Wake_3c": recall_per_class_nrem[2],
        # 2-class
        "accuracy_2class": acc_sleep,
        "recall_macro_2class": recall_macro_sleep,
        "precision_macro_2class": precision_macro_sleep,
        "f1_macro_2class": f1_macro_sleep,
        "recall_Sleep": recall_per_class_sleep[0],
        "recall_Wake_2c": recall_per_class_sleep[1],
    })

#%% Create and print summary tables

summary_all = pd.DataFrame(summary_rows)

# --- 4-class Table ---
table_4class = summary_all[[
    "person",
    "night",
    "accuracy_4class",
    "recall_macro_4class",
    "precision_macro_4class",
    "f1_macro_4class",
    "recall_N3",
    "recall_N1N2",
    "recall_REM",
    "recall_Wake"
]]

# --- 3-class Table ---
table_3class = summary_all[[
    "person",
    "night",
    "accuracy_3class",
    "recall_macro_3class",
    "precision_macro_3class",
    "f1_macro_3class",
    "recall_NREM",
    "recall_REM_3c",
    "recall_Wake_3c"
]]

# --- 2-class Table ---
table_2class = summary_all[[
    "person",
    "night",
    "accuracy_2class",
    "recall_macro_2class",
    "precision_macro_2class",
    "f1_macro_2class",
    "recall_Sleep",
    "recall_Wake_2c"
]]

print("\n--- 4-class summary table ---")
print(table_4class.round(3).to_string(index=False))

print("\n--- 3-class summary table ---")
print(table_3class.round(3).to_string(index=False))

print("\n--- 2-class summary table ---")
print(table_2class.round(3).to_string(index=False))

# --- Aggregate means and stds per table ---
def print_means_stds(table, label):
    print(f"\n{label} - Mean ± Std:")
    for col in table.columns[2:]:
        d = pd.to_numeric(table[col], errors='coerce')
        mean = d.mean(skipna=True)
        std = d.std(skipna=True)
        print(f"{col:<20}: {mean:.3f} ± {std:.3f}")

print_means_stds(table_4class, "4-class")
print_means_stds(table_3class, "3-class")
print_means_stds(table_2class, "2-class")