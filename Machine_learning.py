import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, precision_score, confusion_matrix, auc, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#%% Functions

def get_model(model_type, **kwargs):
    """ 
    Returns an unfiited model or pipeline. 
    kwargs: params like n_estimators, learning_rate, etc. 
    """
    if model_type == "GradientBoosting":
        # Only GB needs pipeline + imputer
        return make_pipeline(
            SimpleImputer(strategy="mean"),
            GradientBoostingClassifier(
                n_estimators=kwargs["n_estimators"],
                learning_rate=kwargs["learning_rate"],
                max_depth=kwargs["max_depth"],
                random_state=kwargs["random_state"]))
    
    elif model_type == "XGBoost":
        return XGBClassifier(
            n_estimators=kwargs["n_estimators"],
            learning_rate=kwargs["learning_rate"],
            max_depth=kwargs["max_depth"],
            eval_metric='logloss',
            random_state=kwargs["random_state"],
            missing=np.nan)
    else:
        raise ValueError("Invalid model_type: choose from 'GradientBoosting', 'XGBoost'.")

def load_and_merge_per_night_features(data_directory, sleep_stages, feature_suffixes):
    """
    Load and horizontally merge feature CSVs from multiple night-level feature sets
    """
    all_dfs = []

    for suffix in feature_suffixes:
        if suffix == 'sleep_structure':
            # Merge only once for sleep-structure (not per stage)
            control_file = os.path.join(data_directory, "Controls_sleep_structure.csv")
            irbd_file = os.path.join(data_directory, "iRBD_sleep_structure.csv")
            if not (os.path.exists(control_file) and os.path.exists(irbd_file)):
                print("Missing sleep_structure file(s).")
                continue
            df_control = pd.read_csv(control_file)
            df_irbd = pd.read_csv(irbd_file)
            df_control['group'] = 'Control'
            df_irbd['group'] = 'iRBD'
            # Add feature prefix to non-ID columns
            id_cols = ['person_prefix', 'night_file', 'night_date', 'group']
            feature_cols = [c for c in df_control.columns if c not in id_cols]
            df_control = df_control.rename(columns={col: f"sleep_structure_{col}" for col in feature_cols})
            df_irbd = df_irbd.rename(columns={col: f"sleep_structure_{col}" for col in feature_cols})
            merged = pd.concat([df_control, df_irbd], ignore_index=True)
            all_dfs.append(merged)
        else:
            # For stage-segmented features (activity, hr, frequency)
            for stage in sleep_stages:
                control_file = os.path.join(data_directory, f"Controls_{stage}_{suffix}.csv")
                irbd_file = os.path.join(data_directory, f"iRBD_{stage}_{suffix}.csv")
                if not (os.path.exists(control_file) and os.path.exists(irbd_file)):
                    print(f"Missing file for {stage}_{suffix}")
                    continue
                df_control = pd.read_csv(control_file)
                df_irbd = pd.read_csv(irbd_file)
                df_control['group'] = 'Control'
                df_irbd['group'] = 'iRBD'
                id_cols = ['person_prefix', 'night_file', 'night_date', 'status', 'group']
                feature_cols = [c for c in df_control.columns if c not in id_cols]
                # Prefix feature columns with stage and suffix for unique feature naming
                df_control = df_control.rename(columns={col: f"{stage}_{suffix}_{col}" for col in feature_cols})
                df_irbd = df_irbd.rename(columns={col: f"{stage}_{suffix}_{col}" for col in feature_cols})
                merged = pd.concat([df_control, df_irbd], ignore_index=True)
                merged = merged[merged['status'] == 'ok']
                merged = merged.drop(columns=['status'])
                all_dfs.append(merged)

    if not all_dfs:
        print("No data loaded.")
        return None

    # Merge all dataframes on standard keys (person/night/date/group)
    base_df = all_dfs[0]
    for add_df in all_dfs[1:]:
        shared_keys = [k for k in ['person_prefix', 'night_file', 'night_date', 'group']
                       if k in base_df.columns and k in add_df.columns]
        base_df = pd.merge(base_df, add_df, on=shared_keys, how='outer')

    return base_df

#%% Main code

DATA_DIRECTORY = r"C:\Users\johan\Desktop\ML_features_sleep_stages"

## SELECTIONS -- set here what model, stages, features to use
# Choose model
MODEL_CHOICE = "GradientBoosting"
#MODEL_CHOICE = "XGBoost"

# Choose sleep stage(s)
#SLEEP_STAGE_SELECT = ["all_sleep"]
#SLEEP_STAGE_SELECT = ["nrem"]
#SLEEP_STAGE_SELECT = ["rem"]
#SLEEP_STAGE_SELECT = ["sleep_block"]
SLEEP_STAGE_SELECT = ["all_sleep", "nrem", "rem", "sleep_block"]

# Choose features
#FEATURE_SUFFIXES = ['activity']
#FEATURE_SUFFIXES = ['hr']
#FEATURE_SUFFIXES = ['frequency']
#FEATURE_SUFFIXES = ['sleep_structure']
FEATURE_SUFFIXES = ['activity', 'hr', 'frequency', 'sleep_structure']

# Load data and rename feature columns to include sleep stage
ml_df = load_and_merge_per_night_features(DATA_DIRECTORY, SLEEP_STAGE_SELECT, FEATURE_SUFFIXES)

if ml_df is None or len(ml_df) == 0:
    raise ValueError("No merged data loaded. Check paths and stage selection.")

print("Data loaded and merged. Total nights for ML:", len(ml_df))

# PREPARE ML DATA
# Remove features irrelevant for ML
excluded_features = ['total_power', 'band1_power', 'band2_power', 'band3_power', 'peak_freq', 'spectral_entropy', 'minutes_of_data','person_prefix', 'night_file', 'night_date', 'status','group','stage_group','n_overlap_epochs', 'minutes_overlap', 'minutes_total_hypnogram', 'TIB_minutes','Percent_Wake', 'NREM_MinBout_minutes', 'REM_MinBout_minutes','SOL_minutes']

feature_columns = [
    col for col in ml_df.columns
    if not any([excl in col for excl in excluded_features])]
selected_df = ml_df[feature_columns]

X = ml_df[feature_columns]
y_raw = ml_df['group'] # Control/iRBD labels
persons = ml_df['person_prefix'].values

le = LabelEncoder()
y = le.fit_transform(y_raw)  # Control=0, iRBD=1

print(f"Classes: {le.classes_} (0: Control, 1: iRBD)")
print(f"Feature matrix shape (X): {X.shape}")
print(f"Target vector shape (y): {y.shape}")

# Hyperparameters and settings
n_estimators = 400
learning_rate = 0.1
#min_samples_leaf = 1
max_depth = 10
random_state = 39   # for reproducibility

unique_persons = np.unique(persons)
y_score = np.zeros(len(y))
y_pred = np.zeros(len(y))

print(f"Total participants: {len(unique_persons)}")

# --- LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION ---
for i, person in enumerate(unique_persons, 1):
    print(f"[{i}/{len(unique_persons)}] Processing participant: {person}")
    test_idx = np.where(persons == person)[0]
    train_idx = np.where(persons != person)[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = get_model(
        MODEL_CHOICE,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        #min_samples_leaf=min_samples_leaf,
        random_state=random_state)

    if MODEL_CHOICE == "GradientBoosting":
        # Needs imputer, has staged_predict_proba
        model.fit(X_train, y_train)
        y_score[test_idx] = model.predict_proba(X_test)[:, 1]
        y_pred[test_idx] = model.predict(X_test)
        
    elif MODEL_CHOICE == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        y_score[test_idx] = model.predict_proba(X_test)[:, 1]
        y_pred[test_idx] = model.predict(X_test)


#%% Per night and per subject plots and metrics 

# --- PER-NIGHT PERFORMANCE ---

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Per-night confusion matrix for {MODEL_CHOICE}Classifier\n(All sleep stages, all features)")
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y, y_score)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'Per-night ROC curve for {MODEL_CHOICE}Classifier\n(All sleep stages, all features)')
plt.grid(False)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Metrics
sensitivity = recall_score(y, y_pred, pos_label=1)
specificity = recall_score(y, y_pred, pos_label=0)
auc_val = roc_auc_score(y, y_score)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, pos_label=1)

print(f"\n--- Diagnostic Performance for iRBD Detection (Per-Night) [{', '.join(SLEEP_STAGE_SELECT)}] ---")
print(f"Sensitivity (Recall for iRBD):   {sensitivity:.3f}")
print(f"Specificity (Recall for Control): {specificity:.3f}")
print(f"Accuracy:                        {accuracy:.3f}")
print(f"Precision (for iRBD):            {precision:.3f}")
print(f"AUC (iRBD detection):            {auc_val:.3f}")

# --- PER-SUBJECT PERFORMANCE ---

person_df = pd.DataFrame({'person': persons, 'group': y_raw, 'score': y_score})
person_avg = person_df.groupby('person').agg({'score': 'mean', 'group': 'first'}).reset_index()
person_avg['pred_label'] = (person_avg['score'] > 0.5).astype(int)
person_avg['true_label'] = le.transform(person_avg['group'])

# Confusion matrix
cm_subj = confusion_matrix(person_avg['true_label'], person_avg['pred_label'])
plt.figure(figsize=(5,4))
sns.heatmap(cm_subj, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Per-subject confusion matrix for {MODEL_CHOICE}Classifier\n(All sleep stages, all features)")
plt.tight_layout()
plt.show()

# ROC curve
fpr_subj, tpr_subj, _ = roc_curve(person_avg['true_label'], person_avg['score'])
roc_auc_subj = auc(fpr_subj, tpr_subj)
plt.figure(figsize=(6,5))
plt.plot(fpr_subj, tpr_subj, color='navy', lw=2, label=f'ROC curve (AUC = {roc_auc_subj:.3f})')
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'Per-subject ROC curve for {MODEL_CHOICE}Classifier\n(All sleep stages, all features)')
plt.grid(False)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Metrics
subject_sensitivity = recall_score(person_avg['true_label'], person_avg['pred_label'], pos_label=1)
subject_specificity = recall_score(person_avg['true_label'], person_avg['pred_label'], pos_label=0)
subject_auc = roc_auc_score(person_avg['true_label'], person_avg['score'])
subject_accuracy = accuracy_score(person_avg['true_label'], person_avg['pred_label'])
subject_precision = precision_score(person_avg['true_label'], person_avg['pred_label'], pos_label=1)

print(f"\n--- Diagnostic Performance for iRBD Detection (Per-Subject) [{', '.join(SLEEP_STAGE_SELECT)}] ---")
print(f"Sensitivity (Recall for iRBD):   {subject_sensitivity:.3f}")
print(f"Specificity (Recall for Control): {subject_specificity:.3f}")
print(f"Accuracy:                        {subject_accuracy:.3f}")
print(f"Precision (for iRBD):            {subject_precision:.3f}")
print(f"AUC (iRBD detection):            {subject_auc:.3f}")


#%% Feature importance (using permutation importance)
X_full = X
y_full = y
model_full = get_model(
    MODEL_CHOICE,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=random_state)
model_full.fit(X_full, y_full)

# Permutation importance
result = permutation_importance(model_full, X_full, y_full, n_repeats=20, random_state=random_state)
importance_means = result.importances_mean
sorted_idx = np.argsort(importance_means)[::-1]

# Plot feature importance
plt.figure(figsize=(max(8, int(len(feature_columns)*0.6)),5))
plt.bar(np.array(feature_columns)[sorted_idx], importance_means[sorted_idx])
plt.xticks(rotation=45, ha='right')
plt.title(f'Permutation feature importance\n({", ".join(SLEEP_STAGE_SELECT)})')
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot top 15 features
top_n = 15
top_idx = sorted_idx[:top_n]
plt.figure(figsize=(8,5))
plt.bar(np.array(feature_columns)[top_idx], importance_means[top_idx])
plt.xticks(rotation=45, ha='right')
plt.title(f'Top {top_n} most important features for {MODEL_CHOICE}Classifier\n(All sleep stages, all features)')
plt.grid(False)
plt.tight_layout()
plt.show()



#%% Display misclassifications for review

misclassified = person_avg[person_avg['pred_label'] != person_avg['true_label']]
misclass_ids = misclassified['person']
night_counts = ml_df.groupby('person_prefix').size().rename('n_nights').reset_index()
misclassified = misclassified.merge(night_counts, left_on='person', right_on='person_prefix', how='left')
print("Misclassified persons:")
print(misclassified[['person', 'group', 'score', 'pred_label', 'true_label', 'n_nights']])