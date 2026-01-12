import os
import actipy
import pandas as pd
import warnings
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Folder containing all CWA files
#input_folder = r"C:\Users\johan\Desktop\stanford_cwa_clean\Controls"
input_folder = r"C:\Users\johan\Desktop\stanford_cwa_clean\New"
#output_folder = r"C:\Users\johan\Desktop\data_preprocessed\Controls"
output_folder = r"C:\Users\johan\Desktop\data_preprocessed\New"
#os.makedirs(output_folder, exist_ok=True)

# Nighttime window
start_hour = 21
end_hour = 9

#%% Preprocessing functions

def _safe_standardize(signal: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Standardizes the signal (z-score), handling potential NaNs and near-zero std dev.
       Returns standardized signal, mean, and std dev used."""
    nan_mask = np.isnan(signal)
    valid_signal = signal[~nan_mask] # Process only non-NaN values

    if valid_signal.size == 0:
        warnings.warn("Signal contains only NaNs. Cannot standardize.", UserWarning)
        return signal.copy().astype(float), np.nan, np.nan

    mean = np.mean(valid_signal)
    std_dev = np.std(valid_signal)

    standardized_signal = signal.copy().astype(float)
    std_dev_used = std_dev

    if std_dev < epsilon:
        warnings.warn(f"Standard deviation ({std_dev:.2e}) is close to zero. Setting standardized signal to zero where not NaN.", UserWarning)
        standardized_signal[~nan_mask] = 0.0
        std_dev_used = epsilon
    else:
        standardized_signal[~nan_mask] = (valid_signal - mean) / std_dev

    if np.isnan(standardized_signal[~nan_mask]).any():
         warnings.warn("NaNs produced during standardization calculation. Check input data.", UserWarning)

    return standardized_signal, mean, std_dev_used

def preprocess_actigraphy_df(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    fs_in: int = 100,
    fs_out: int = 30,
    gravity_calibration: bool = False,
    calib_cube: float = -1.0,
    output_suffix: str = '_preprocessed',
    input_unit_divisor: float = 1,
    scale_by_force: bool = False,
    standardize_axes: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Preprocesses 3-axis accelerometer data using actipy.
    This version is ROBUST: It handles both timezone-aware and timezone-naive
    input DataFrames by temporarily making the index naive for actipy processing.
    """
    if verbose:
        print(f"Preprocessing actigraphy data: Input Fs={fs_in}Hz, Output Fs={fs_out}Hz")
        print(f"Using columns: X='{x_col}', Y='{y_col}', Z='{z_col}'")
        print(f"Gravity Calibration Requested: {gravity_calibration}")
        print(f"Input Unit Divisor: {input_unit_divisor}")
        print(f"Standardize Axes Requested: {standardize_axes}")

    required_cols = [x_col, y_col, z_col]
    if not all(col in data_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data_df.columns]
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    all_info = {
        'input_fs': fs_in, 'output_fs': fs_out,
        'input_columns': {'x': x_col, 'y': y_col, 'z': z_col},
        'output_suffix': output_suffix, 'input_unit_divisor_used': input_unit_divisor,
        'calibration_requested': gravity_calibration, 'standardization_requested': standardize_axes,
    }

    process_df = data_df[[x_col, y_col, z_col]].copy()
    process_df.rename(columns={x_col: 'x', y_col: 'y', z_col: 'z'}, inplace=True)
    
    if not isinstance(process_df.index, pd.DatetimeIndex):
        raise ValueError(f'Dataframe index should be pd.DatetimeIndex got {type(process_df.index)}')

    # --- ROBUSTNESS CHECK: This block makes the function handle both cases ---
    # 1. Check if the input data has a timezone.
    original_tz = process_df.index.tz
    
    # 2. If it has a timezone, remove it TEMPORARILY for actipy.
    if original_tz is not None:
        if verbose:
            print(f"  Temporarily converting index from {original_tz} to timezone-naive for actipy processing.")
        process_df.index = process_df.index.tz_localize(None)
    # If original_tz is None, this block is skipped, and the naive index is used directly.
    # --- END OF ROBUSTNESS CHECK ---
    
    process_df.index.name = 'time'

    # 0. Apply Input Unit Scaling (Convert to 'g')
    if input_unit_divisor != 1.0:
        if input_unit_divisor == 0:
            raise ValueError("input_unit_divisor cannot be zero.")
        if verbose: print(f"  Applying input unit scaling: Dividing x, y, z by {input_unit_divisor}")
        process_df[['x', 'y', 'z']] = process_df[['x', 'y', 'z']].astype(float) / input_unit_divisor
    elif verbose:
            print("  Skipping input unit scaling (divisor is 1.0).")

    # 1. Handle potential signal dropout
    force = np.linalg.norm(process_df[['x', 'y', 'z']].values, axis=1)
    dropout_indices = np.where(force == 0)[0]
    if len(dropout_indices) > 0:
        # This is a significant event, consider warning always or if verbose
        message = f"  Detected {len(dropout_indices)} potential dropout points (zero magnitude) out of {len(force)} total samples. Setting to NaN."
        if verbose: print(message)
        else: warnings.warn(message, UserWarning)
        process_df.iloc[dropout_indices] = np.nan
        force[dropout_indices] = np.nan
        all_info['dropout_points_handled'] = len(dropout_indices)
    else:
        all_info['dropout_points_handled'] = 0

    if np.all(np.isnan(force)):
        warnings.warn("All force vector values are NaN after dropout handling. Scaling cannot be applied.", UserWarning)
        all_info['mean_force_scaling_skipped'] = 'All data NaN after dropout'

    # 2. Scale signal by the mean force (OPTIONAL)
    if scale_by_force:
        mean_force = np.nanmean(force)
        if np.isnan(mean_force) or mean_force == 0:
            message = f"  Skipping scaling by mean force (mean_force is {mean_force})."
            if verbose: print(message)
            else: warnings.warn(message, UserWarning)
            all_info['mean_force_scaling_skipped'] = f'Mean force was {mean_force}'
        else:
            if verbose: print(f"  Scaling data by mean force: {mean_force:.4f}")
            process_df[['x', 'y', 'z']] = process_df[['x', 'y', 'z']] / mean_force
            all_info['mean_force_scale_factor'] = float(mean_force)
        if len(dropout_indices) > 0:
            process_df.iloc[dropout_indices] = np.nan
    elif verbose:
        print("  Skipping scaling by mean force (scale_by_force=False).")
    all_info['mean_force_scale_factor'] = all_info.get('mean_force_scale_factor', None)

    # 3. Low-pass filter
    cutoff_freq = fs_out / 2.0
    if verbose: print(f"  Applying low-pass filter with cutoff ~{cutoff_freq:.2f} Hz...")
    try:
        process_df, filter_info = actipy.processing.lowpass(
            data=process_df, data_sample_rate=fs_in, cutoff_rate=cutoff_freq,
        )
        if verbose: print(f"  Filter info: {filter_info}")
        all_info.update(filter_info); all_info['filtering_applied'] = True
    except Exception as filter_err:
        warnings.warn(f"ERROR during low-pass filtering: {filter_err}. Proceeding with unfiltered data (relative to this step).", UserWarning)
        all_info['filtering_error'] = str(filter_err); all_info['filtering_applied'] = False

    # 4. Resample
    if verbose: print(f"  Resampling data to {fs_out} Hz...")
    try:
        process_df, resample_info = actipy.processing.resample(
            data=process_df, sample_rate=fs_out,
        )

        if verbose: print(f"  Resample info: {resample_info}")
          
        all_info.update(resample_info); all_info['resampling_applied'] = True

    except Exception as resample_err:
        warnings.warn(f"ERROR during resampling: {resample_err}. Proceeding with unresampled data (relative to this step).", UserWarning)
        all_info['resampling_error'] = str(resample_err); all_info['resampling_applied'] = False

    # 5. Calibrate Gravity (Conditional)
    calib_diagnostics = {'calibration_skipped_reason': 'Not Skipped'}
    calibration_successful = False; calibration_applied = False
    if gravity_calibration:
        if process_df.isnull().all().all():
            warnings.warn("Data is all NaN before calibration. Skipping calibration step.", UserWarning)
            calib_diagnostics['calibration_skipped_reason'] = 'All data NaN before calibration'
        else:
            if verbose: print("  Calibrating gravity...")
            calib_min_samples = 50
            calib_window = '10s'
            calib_stdtol = 0.013
            calib_chunksize = int(1e6)
			
            try:
                process_df_calibrated, calib_diagnostics_update = actipy.processing.calibrate_gravity(
                    data=process_df.copy(), 
					calib_cube=calib_cube, 
					calib_min_samples=calib_min_samples,
                    window=calib_window, 
					stdtol=calib_stdtol, 
					chunksize=calib_chunksize,
                )
                calib_diagnostics.update(calib_diagnostics_update)
				
                if calib_diagnostics_update.get('CalibOK', 0) == 1:
                    calibration_successful = True
                    process_df = process_df_calibrated
                    calibration_applied = calib_diagnostics_update.get('CalibNumIters', 0) > 0
                    if verbose: print(f'  Calibration successful and {"applied" if calibration_applied else "not applied (low initial error)"}.')
                else:
                    warnings.warn(f"Calibration failed (CalibOK != 1). Proceeding with uncalibrated data. Diagnostics: {calib_diagnostics_update}", UserWarning)
                    calibration_successful = False
                if verbose: print(f"  Calibration info: {calib_diagnostics}")
					
            except Exception as calib_err:
                warnings.warn(f"ERROR during gravity calibration call: {calib_err}. Proceeding with uncalibrated data.", UserWarning)
                calib_diagnostics['calibration_error'] = str(calib_err); calibration_successful = False
    elif verbose:
            print("  Skipping gravity calibration as requested.")
    if not gravity_calibration:
        calib_diagnostics['calibration_skipped_reason'] = 'Skipped by user flag (gravity_calibration=False)'

    all_info['calibration_successful'] = calibration_successful
    all_info['calibration_applied'] = calibration_applied
    all_info['calibration_diagnostics'] = calib_diagnostics

    # 6. Standardize Axes (Z-score)
    if standardize_axes:
        if verbose: print("  Applying per-axis Z-score standardization (post-calibration/resampling)...")
        stats = {}; standardization_successful_flag = True
        try:
            for axis in ['x', 'y', 'z']:
                if axis not in process_df.columns or process_df[axis].isnull().all():
                    stats[axis] = {'mean': None, 'std_dev_used': None, 'error': 'Missing or all NaN'}
                    standardization_successful_flag = False; continue
                standardized_signal, mean, std_dev = _safe_standardize(process_df[axis].values)
                process_df[axis] = standardized_signal
                stats[axis] = {'mean': float(mean) if not np.isnan(mean) else None, 'std_dev_used': float(std_dev) if not np.isnan(std_dev) else None}
            all_info['standardization_stats'] = stats
            if verbose: print("  Standardization " + ("complete." if standardization_successful_flag else "attempted with issues."))
        except Exception as e:
            warnings.warn(f"ERROR during standardization: {e}. Proceeding with unstandardized data.", UserWarning)
            all_info['standardization_error'] = str(e); standardization_successful_flag = False
        all_info['standardization_applied'] = standardization_successful_flag
    else:
        if verbose: print("  Skipping per-axis standardization.")
        all_info['standardization_applied'] = False; all_info['standardization_stats'] = None

    # --- ROBUSTNESS RESTORATION: This block restores the original timezone state ---
    # 3. If there WAS an original timezone, add it back to the processed data.
    if original_tz is not None:
        if verbose:
            print(f"  Re-applying original timezone ({original_tz}) to index after actipy processing.")
        process_df.index = process_df.index.tz_localize(original_tz)
    # If the original data was naive, this is skipped, and the output remains naive.
    # --- END OF ROBUSTNESS RESTORATION ---

    output_df = process_df.rename(columns={
        'x': f"{x_col}{output_suffix}",
        'y': f"{y_col}{output_suffix}",
        'z': f"{z_col}{output_suffix}"
    })
    all_info['output_columns'] = list(output_df.columns)
    all_info['output_shape'] = list(output_df.shape)

    return output_df, all_info

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
            f'accelerometry',
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

#%%

# Loop over all CWA files
for fname in os.listdir(input_folder):
    if not fname.lower().endswith(".cwa"):
        continue

    path = os.path.join(input_folder, fname)
    base_name = os.path.splitext(fname)[0]
    
    print(f"\nProcessing {fname}...")

    # --- Load data ---
    acti_data, info = actipy.read_device(path, lowpass_hz=None, calibrate_gravity=False, resample_hz=None)

    # --- Preprocess ---
    output_df, all_info = preprocess_actigraphy_df(
        data_df=acti_data,
        x_col='x',     
        y_col='y',     
        z_col='z',     
        fs_in=info['SampleRate'],
        fs_out=30,
        gravity_calibration=True,
        calib_cube=-1.0,
        output_suffix='_preprocessed',
        input_unit_divisor=1.0,
        scale_by_force=False, # also false
        standardize_axes=False, #set to false
        verbose=False
    )

    output_df.index = pd.to_datetime(output_df.index)

    # --- Split into nights ---
    def filter_night(df, start_hour, end_hour):
        nights = []
        current_date = df.index.min().normalize()
        end_date = df.index.max().normalize()
        while current_date <= end_date:
            night_start = current_date + pd.Timedelta(hours=start_hour)
            night_end = current_date + pd.Timedelta(days=1, hours=end_hour)
            night_df = df[(df.index >= night_start) & (df.index < night_end)]
            if not night_df.empty:
                nights.append((current_date.date(), night_df))
            current_date += pd.Timedelta(days=1)
        return nights

    all_nights = filter_night(output_df, start_hour, end_hour)

    # --- Filter and save each night ---
    saved_nights = []  # store nights for plotting
    for night_date, night_df in all_nights:
        total_samples = len(night_df)
        missing = night_df.isna().any(axis=1).sum()
        prop_missing = missing / total_samples

        if prop_missing > 0.5:
            print(f"Skipping night {night_date} ({base_name}): {prop_missing:.1%} missing data")
            continue

        outfile = os.path.join(output_folder, f"{base_name}_night_{night_date}.h5")
        annotations = pd.DataFrame()

        write_h5_acc(
            outfile=outfile,
            accelerometry=night_df,
            x_col="x_preprocessed",
            y_col="y_preprocessed",
            z_col="z_preprocessed",
            acc_info=all_info,
            annotations=annotations,
            study_start=night_df.index[0],
            chunk_size_sec=600,
        )

        #print(f"Saved {outfile}")
        saved_nights.append((night_date, night_df))  # keep for plotting
        
    # --- Plot all saved nights for this person ---
    #if saved_nights:
    num_nights = len(saved_nights)
    fig, axes = plt.subplots(num_nights, 1, figsize=(14, 3*num_nights), sharex=False)
    if num_nights == 1:
        axes = [axes]

    for ax, (night_date, night_df) in zip(axes, saved_nights):
        ax.plot(night_df.index, night_df['x_preprocessed'], label='X-axis')
        ax.plot(night_df.index, night_df['y_preprocessed'], label='Y-axis')
        ax.plot(night_df.index, night_df['z_preprocessed'], label='Z-axis')
        ax.set_title(f"{base_name} - Night {night_date} (21:00–09:00)")
        ax.set_ylabel("Acceleration")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()
