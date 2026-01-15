import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns
from collections import namedtuple

#%% Functions

def plot_split_violin_with_subject_means( csv_folder, groups, stage_groups, stage_labels, feature_tpls, is_windowed=False, epoch_lengths=None, window_names=None,
    palette=['lightblue','lightcoral'], figsize=(6,5), hue_order=['Controls','iRBD'], subject_id_col='person_prefix'):
    """Function for plotting split violin plots and calculationg stats (p-value and cohens d).
    """
    
    sns.set(style="whitegrid")
    
    if is_windowed:
        # For windowed features, multiple epoch sizes
        assert epoch_lengths is not None and window_names is not None
    else:
        epoch_lengths = [None]
        window_names = [""] 

    # Loop through all features specified in feature_tpls
    for feat_tpl, feat_title, feat_ylabel in feature_tpls:
        for win, win_label in zip(epoch_lengths, window_names):
            # ------------- Data collection -----------------
            all_data = []
            for stage in stage_groups:
                for group in ['Controls', 'iRBD']:
                    file_path = groups[group].format(stage)
                    try:
                        df = pd.read_csv(file_path)
                        df['Group'] = group
                        df['Stage'] = stage_labels[stage]
                        df = df[df['status'] == 'ok']
                        if is_windowed:
                            feat_col = feat_tpl.format(win)
                        else:
                            feat_col = feat_tpl
                        if feat_col in df.columns:
                            df = df[['Group','Stage',feat_col,subject_id_col]].rename(
                                columns={feat_col:'Value'})
                            all_data.append(df)
                    except Exception as e:
                        print(f"Could not load {file_path}: {e}")

            if not all_data:
                continue

            # Concatenate across groups/stages
            df_long = pd.concat(all_data, ignore_index=True)
            if df_long['Value'].dropna().empty:
                continue

            # Prepare categorical label lists for plot order and for significant (*) labeling
            unique_stages = [stage_labels[s] for s in stage_groups]
            signif_labels = unique_stages.copy()
            perstage_p = []
            perstage_d = []

            # ---------- Per-subject means for stats ---------
            for ix, stagelab in enumerate(unique_stages):
                subset = df_long[df_long['Stage'] == stagelab]

                # Compute per-subject means per group
                subject_mean_ctrl = subset[subset['Group'] == 'Controls'].groupby(subject_id_col)['Value'].mean()
                subject_mean_irbd = subset[subset['Group'] == 'iRBD'].groupby(subject_id_col)['Value'].mean()
                ctrl_vals = subject_mean_ctrl.dropna()
                irbd_vals = subject_mean_irbd.dropna()

                # p-val & Cohen's d (Welch t-test)
                if len(ctrl_vals) > 1 and len(irbd_vals) > 1:
                    t_stat, p_val = ttest_ind(ctrl_vals, irbd_vals, nan_policy='omit', equal_var=False)
                    n1, n2 = len(ctrl_vals), len(irbd_vals)
                    s1, s2 = ctrl_vals.std(ddof=1), irbd_vals.std(ddof=1)
                    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
                    d = (irbd_vals.mean() - ctrl_vals.mean())/s_pooled if s_pooled>0 else np.nan
                    perstage_p.append(p_val)
                    perstage_d.append(d)
                    print(f"Feature: {feat_title}, Stage: {stagelab}, p-value (subject mean): {p_val:.2e}, Cohen's d: {d:.2f}")
                    if p_val < 0.05:
                        signif_labels[ix] = stagelab + '*'
                else:
                    perstage_p.append(np.nan)
                    perstage_d.append(np.nan)

            # ---------- Plot: Split violin ----------
            plt.figure(figsize=figsize)
            ax = sns.violinplot(
                x="Stage", y="Value", hue="Group", data=df_long,
                split=True, inner="quart", palette=palette, hue_order=hue_order)

            # Optional: Overlay subject means as black diamonds for each segment+group
            # for stagelab in unique_stages:
            #     for group, color in zip(hue_order, ['black','black']):
            #         group_data = df_long[(df_long['Group']==group)&(df_long['Stage']==stagelab)]
            #         if group_data.empty: continue
            #         subjmeans = group_data.groupby(subject_id_col)['Value'].mean()
            #         xpos = unique_stages.index(stagelab)
            #         # Offset for split violin horizontally
            #         x_jitter = -0.16 if group == hue_order[0] else 0.16
            #         y_vals = subjmeans.values
            #         ax.scatter([xpos + x_jitter]*len(y_vals), y_vals, 
            #                    color=color, marker='D', edgecolors='white', zorder=4, s=55, linewidth=0.6)

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(signif_labels)
            ylabel = feat_ylabel if not is_windowed else f"{feat_ylabel} ({win_label})"
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Sleep segment")
            title = feat_title if not is_windowed else f"{feat_title} ({win_label})"
            ax.set_title(title)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title='Group', loc='upper right')
            ax.grid(True, axis='y')
            # optional log scale:
            #ax.set_yscale('log')
            # optional set ylim
            #ax.set_ylim([-5e-5, 3e-4])
            plt.tight_layout()
            plt.show()

#%%-------- HR FEATURES --------

csv_folder_hr = r"C:\Users\johan\Desktop\ML_features_sleep_stages"

groups_hr = {
    'iRBD': csv_folder_hr + "/iRBD_{}_hr.csv",
    'Controls': csv_folder_hr + "/Controls_{}_hr.csv"}
stage_groups = ['all_sleep', 'nrem', 'rem', 'sleep_block']
stage_labels = {
    'all_sleep': 'All sleep',
    'nrem': 'NREM',
    'rem': 'REM',
    'sleep_block': 'Sleep block'}
feature_tpls_hr = [
    ('mean_hr', 'Mean heart rate', 'Mean heart rate [bpm]'),
    ('sd_hr', 'SD of heart rate', 'SD of heart rate [bpm]'),
    ('rmssd_hr', 'RMSSD of heart rate', 'RMSSD of heart rate [bpm]'),
    ('num_hr_spikes', 'Number of heart rate spikes', 'Number of heart rate spikes')]
plot_split_violin_with_subject_means(
    csv_folder_hr, groups_hr, stage_groups, stage_labels,
    feature_tpls_hr, is_windowed=False, palette=['lightblue','lightcoral'], figsize=(6,5), subject_id_col='person_prefix')

#%%-------- ACTIVITY FEATURES --------

csv_folder_act = r"C:\Users\johan\Desktop\ML_features_sleep_stages"

groups_act = {
    'iRBD': csv_folder_act + "/iRBD_{}_activity.csv",
    'Controls': csv_folder_act + "/Controls_{}_activity.csv"}
feature_tpls_act = [
    ('mean_activity_{}s', 'Mean activity', 'Mean activity'),
    ('activity_index_percent_{}s', 'Activity index', 'Activity Index [%]'),
    ('twitch_per_hour_{}s', 'Twitch per hour', 'Twitch per hour')]
epoch_lengths = [1, 15, 30]
window_names = ['1s', '15s', '30s']
plot_split_violin_with_subject_means(
    csv_folder_act, groups_act, stage_groups, stage_labels,
    feature_tpls_act, is_windowed=True, epoch_lengths=epoch_lengths, window_names=window_names, palette=['lightblue','lightcoral'], figsize=(6,5), subject_id_col='person_prefix')

#%%-------- FREQUENCY FEATURES --------

csv_folder_freq = r"C:\Users\johan\Desktop\ML_features_sleep_stages"

groups_freq = {
    'iRBD': csv_folder_freq + "/iRBD_{}_frequency.csv",
    'Controls': csv_folder_freq + "/Controls_{}_frequency.csv"}
feature_tpls_freq = [
    ('TotalPower', 'Total power', 'Total power [g²]'),
    ('BandPower_1', 'Band 1 power (0.1–0.5 Hz)', 'Band 1 power [g²]'),
    ('BandPower_2', 'Band 2 power (0.5–2 Hz)', 'Band 2 power [g²]'),
    ('BandPower_3', 'Band 3 power (2–10 Hz)', 'Band 3 power [g²]'),
    ('PeakFreq', 'Peak frequency', 'Peak frequency [Hz]'),
    ('SpectralEntropy', 'Spectral entropy', 'Spectral entropy')]
plot_split_violin_with_subject_means(
    csv_folder_freq, groups_freq, stage_groups, stage_labels,
    feature_tpls_freq, is_windowed=False, palette=['lightblue','lightcoral'], figsize=(6,5), subject_id_col='person_prefix')

#%%-------- SLEEP STRUCTURE FEATURES --------

csv_folder = r"C:\Users\johan\Desktop\ML_features_sleep_stages"

groups = {
    'iRBD': csv_folder + "/iRBD_sleep_structure.csv",
    'Controls': csv_folder + "/Controls_sleep_structure.csv"}

feature_list = [
    ('TST_minutes',            'Total sleep time',          'min'),
    ('WASO_minutes',           'Wake after sleep onset',    'min'),
    ('REMLatency_minutes',     'REM latency',               'min'),
    ('LongestSleepBout_minutes','Longest sleep bout',       'min'),
    ('SleepEfficiency',        'Sleep efficiency',          '%'),
    ('Percent_N3',             'Percent N3',                '%'),
    ('Percent_N1N2',           'Percent N1N2',              '%'),
    ('Percent_REM',            'Percent REM',               '%'),
    ('Awakenings',             'Number of awakenings',      ''),
    ('NumStageTransitions',    'Number of stage transitions',''),
    ('FragmentationIndex',     'Fragmentation index',       ''),
    ('NREM_MaxBout_minutes',   'Max NREM bout',             'min'),
    ('NREM_MeanBout_minutes',  'Mean NREM Bout',            'min'),
    ('NREM_NumBouts',          'Number of NREM bouts',      ''),
    ('REM_MaxBout_minutes',    'Max REM bout',              'min'),
    ('REM_MeanBout_minutes',   'Mean REM bout',             'min'),
    ('REM_NumBouts',           'Number of REM bouts',       '')]

palette = ['lightblue', 'lightcoral']
hue_order = ['Controls', 'iRBD']
violin_width = 0.6
jitter_amt = 0.16

# Load data and add group tag
try:
    df_controls = pd.read_csv(groups['Controls'])
    df_controls['Group'] = 'Controls'
except Exception as e:
    print(f"Cannot load Controls: {e}")
    df_controls = pd.DataFrame()

try:
    df_irbd = pd.read_csv(groups['iRBD'])
    df_irbd['Group'] = 'iRBD'
except Exception as e:
    print(f"Cannot load iRBD: {e}")
    df_irbd = pd.DataFrame()

df = pd.concat([df_controls, df_irbd], ignore_index=True)

# --- Optionally convert SleepEfficiency to percent if needed ---
if 'SleepEfficiency' in df.columns and df['SleepEfficiency'].max() <= 1.0:
    df['SleepEfficiency'] = df['SleepEfficiency'] * 100

sns.set(style="whitegrid")

# ----- COLLECT stats for ranking -----
FeatureStat = namedtuple("FeatureStat", ["col", "label", "unit", "pval", "d"])
feature_stats = []

for col, label, unit in feature_list:
    df_plot = df[['Group', col, 'person_prefix']].dropna().copy()
    if df_plot.empty:
        continue

    subjmeans_ctrl = df_plot[df_plot['Group'] == 'Controls'].groupby('person_prefix')[col].mean()
    subjmeans_irbd = df_plot[df_plot['Group'] == 'iRBD'].groupby('person_prefix')[col].mean()
    vals_ctrl = subjmeans_ctrl.values
    vals_irbd = subjmeans_irbd.values

    # Calculate stats
    if len(vals_ctrl) > 1 and len(vals_irbd) > 1:
        tstat, p_val = ttest_ind(vals_ctrl, vals_irbd, nan_policy='omit', equal_var=False)
        n1, n2 = len(vals_ctrl), len(vals_irbd)
        s1, s2 = subjmeans_ctrl.std(ddof=1), subjmeans_irbd.std(ddof=1)
        s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2)) if (n1 + n2 - 2) > 0 else np.nan
        cohens_d = (np.mean(vals_irbd)-np.mean(vals_ctrl))/s_pooled if s_pooled > 0 else np.nan
    else:
        p_val, cohens_d = np.nan, np.nan

    feature_stats.append(FeatureStat(col, label, unit, p_val, cohens_d))

# ----- SORT and print top 9 features by |d| -----
feature_stats_sorted = sorted(
    [fp for fp in feature_stats if not np.isnan(fp.d)],
    key=lambda x: abs(x.d), reverse=True)
top_9 = feature_stats_sorted[:9]

print("\nTop 9 sleep structure features (highest effect size, |d|):")
for i, fp in enumerate(top_9, 1):
    signif = "*" if fp.pval < 0.05 else ""
    print(f"{i}. {fp.label:30s} p = {fp.pval:.4g}, d = {fp.d:.2f} {signif}")

# ----- Plot all features -----
for fp in feature_stats:
    # Reload df_plot for plotting
    df_plot = df[['Group', fp.col, 'person_prefix']].dropna().copy()
    if df_plot.empty:
        continue
    df_plot['Feature'] = fp.label

    signif = fp.pval < 0.05
    title_str = fp.label + ('*' if signif else '')
    ylabel_str = f"{fp.label}" + (f" [{fp.unit}]" if fp.unit else "")

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.violinplot(x='Feature', y=fp.col, hue='Group', data=df_plot, split=True, inner="quart", palette=palette, width=violin_width, hue_order=hue_order, ax=ax)

    # Optional: Overlay per-subject means as black diamonds, offset to each side
    # for group, xoffset in zip(hue_order, [-jitter_amt, jitter_amt]):
    #     subjmeans = df_plot[df_plot['Group'] == group].groupby('person_prefix')[fp.col].mean()
    #     xpos = 0 + xoffset  # only one feature
    #     ax.scatter([xpos] * len(subjmeans), subjmeans.values, color='black', marker='D',
    #                zorder=4, s=55, edgecolors='white', linewidths=0.7)

    ax.set_title(title_str)
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel('')
    ax.set_xticks([])
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Group', loc='upper right')
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()