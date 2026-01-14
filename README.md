# MasterThesis
Repository for the master thesis "Early Detection of Neurodegenerative Disease Using Wrist Accelerometers" made by Johanne Niman Abildgaard (Student ID: s185380).

The repository contains the following .py files:
- Actigraphy_preprocessing. This script preprocesses raw actigraphy data from .cwa files, including missing data handling, filtering, resampling, and calibrating the accelerometer signals, and saves nightly segments as HDF5 files.
- Nonwear_heatmaps. This script analyzes per-night accelerometer and hypnogram data to detect non-wear periods, excludes poor-quality nights, and visualizes the data distribution as heatmaps before and after exclusion.
- Hypnogram_validation. This script compares algorithm-predicted sleep stages with manual hypnogram scorings for nightly recordings, computes performance metrics, plots confusion matrices, and summarizes classification results for each night.
- Heartrate_validation. This script compares ECG-derived and actigraphy-estimated heart rate (estimated by the Nightbeat method), segmenting by sleep stage, and produces visual and quantitative performance comparisons for each night.
- Sleep_structure_feature_extraction. This script extracts sleep structure features from predicted per-night hypnograms, merges them with actigraphy data to ensure overlap and quality, and saves a summary table for further analysis.
- Heartrate_feature_extraction.
- Activity_feature_extraction.
- Frequency_feature_extraction.
- Feature_distribution.
- Machine_learning.
