# MasterThesis
Repository for the master thesis "Early Detection of Neurodegenerative Disease Using Wrist Accelerometers" made by Johanne Niman Abildgaard (Student ID: s185380).

The repository contains the following files:
- Actigraphy_preprocessing.py: This script preprocesses raw actigraphy data from .cwa files, including missing data handling, filtering, resampling, and calibration, and saves nightly segments as HDF5 files.
- Nonwear_heatmaps.py: This script analyzes preprocessed accelerometer data and hypnogram data for non-wear detection, excludes poor-quality nights, and visualizes data distribution as heatmaps before and after exclusion.
- Hypnogram_validation.py: This script compares algorithm-predicted sleep stages with manual hypnogram scoring, computes performance metrics, plots confusion matrices, and produces per-night summary tables.
- Heartrate_estimation_for_validation.py: This script estimates heart rate from preprocessed accelerometer data using the Nightbeat method, visualizes the estimation process, and saves results for further validation.
- Heartrate_validation.py: This script compares ECG-derived and Nightbeat-estimated heart rate, segmented by sleep stage, and generates visual and quantitative performance comparisons for each night.
- Sleep_structure_feature_extraction.py: This script extracts sleep structure features from predicted nightly hypnograms and produces per-night CSV summaries.
- Heartrate_feature_extraction.py: This script extracts heart rate features and includes the Nightbeat method for estimating heart rate. Features are extracted from preprocessed accelerometer data and hypnogram data, and are saved per night and per sleep segment as CSV summaries.
- Activity_feature_extraction.py: This script extracts activity features from sleep epochs across sleep stages. Features are extracted from preprocessed accelerometer data and hypnogram data, and are saved per night and per sleep segment as CSV summaries.
- Frequency_feature_extraction.py: This script extracts frequency-domain features using Welch's PSD. Features are extracted from preprocessed accelerometer data and hypnogram data, and are saved per night and per sleep segment as CSV summaries.
- Feature_distribution.py: This script loads extracted feature CSVs and visualizes feature distributions between groups and sleep segments using split violin plots with subject-level means and effect size annotations.
- Machine_learning.py: This script builds and evaluates machine learning classifiers to distinguish iRBD from controls using night-level sleep features, with leave-one-participant-out cross-validation, diagnostic metrics, and permutation-based feature importance.
