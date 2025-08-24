import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)

# -----------------------------
# Load data (EEG + Gaze + Video)
# -----------------------------
annotations = pd.read_csv("annotations.csv")
annotations = annotations.dropna(subset=['eeg', 'gaze'])
valid_indices = annotations.index  # only keep runs with EEG

# EEG
df_eeg = pd.read_csv("eeg_df_3_interpolated.csv")
X_eeg, y_eeg = df_eeg.drop('truth', axis=1).values, df_eeg['truth'].values

# Gaze
df_gaze = pd.read_csv("gaze_df_2.csv")
df_gaze = df_gaze.iloc[valid_indices].reset_index(drop=True)
X_gaze, y_gaze = df_gaze.drop('truth', axis=1).values, df_gaze['truth'].values

# Video
df_video = pd.read_csv("video_features.csv")
df_video = df_video.iloc[valid_indices].reset_index(drop=True)
X_video, y_video = df_video.drop(columns=['truth', 'path']).values, df_video['truth'].values

# -----------------------------
# Align labels across modalities
# -----------------------------
labels_all = np.vstack([y_eeg, y_gaze, y_video]).T
valid_mask = np.all(labels_all == labels_all[:, [0]], axis=1)
valid_indices_final = np.where(valid_mask)[0]

print(f"Samples before alignment: {len(y_eeg)}")
print(f"Samples after alignment: {len(valid_indices_final)}")

# Filter aligned samples
X_eeg, y_eeg = X_eeg[valid_indices_final], y_eeg[valid_indices_final]
X_gaze, y_gaze = X_gaze[valid_indices_final], y_gaze[valid_indices_final]
X_video, y_video = X_video[valid_indices_final], y_video[valid_indices_final]

# -----------------------------
# Scale + feature selection
# -----------------------------
k = 1000  # top k features

def scale_select(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    X_sel = SelectKBest(mutual_info_classif, k=min(k, X.shape[1])).fit_transform(X_scaled, y)
    return X_sel

X_eeg_sel   = scale_select(X_eeg, y_eeg)
X_gaze_sel  = scale_select(X_gaze, y_gaze)
X_video_sel = scale_select(X_video, y_video)

# -----------------------------
# Define classifiers per modality
# -----------------------------
video_clfs = [
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=1, min_samples_split=2,
                                  max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel="sigmoid", C=100, gamma=0.001, probability=True, random_state=42)),
    ("mlp", MLPClassifier(max_iter=2000, early_stopping=True, random_state=42, activation='tanh',
                          alpha=0.0001, hidden_layer_sizes=(500, 100), learning_rate='constant', learning_rate_init=0.01, n_iter_no_change=20))
]

eeg_clfs = [
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=2,
                                  min_samples_leaf=5, max_features='log2', class_weight='balanced', random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel='rbf', C=25, gamma='auto', class_weight='balanced', probability=True, random_state=42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(500, 500), alpha=0.1e-05, learning_rate='constant',
                          activation='relu', learning_rate_init=0.001, n_iter_no_change=5,
                          early_stopping=True, max_iter=2000, random_state=42))
]

gaze_clfs = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_split=2,
                                  min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(300, 300), alpha=1e-05, learning_rate='constant',
                          activation='tanh', learning_rate_init=0.005, n_iter_no_change=5,
                          early_stopping=True, max_iter=2000, random_state=42))
]

# -----------------------------
# Meta-classifier
# -----------------------------
meta_clf = GradientBoostingClassifier(random_state=42)

# -----------------------------
# Cross-validation setup
# -----------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
n_samples = X_video_sel.shape[0]

# Compute total meta-features length
n_meta_features = len(video_clfs) + len(eeg_clfs) + len(gaze_clfs)
X_meta_oof = np.zeros((n_samples, n_meta_features))

# -----------------------------
# Generate out-of-fold predictions per classifier
# -----------------------------
def generate_oof(clfs, X, y):
    X_meta_local = np.zeros((X.shape[0], len(clfs)))
    for i, (name, clf) in enumerate(clfs):
        oof_preds = np.zeros(X.shape[0])
        for train_idx, val_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        X_meta_local[:, i] = oof_preds
    return X_meta_local

idx = 0
X_meta_oof[:, idx:idx+len(video_clfs)] = generate_oof(video_clfs, X_video_sel, y_video); idx += len(video_clfs)
X_meta_oof[:, idx:idx+len(eeg_clfs)] = generate_oof(eeg_clfs, X_eeg_sel, y_eeg); idx += len(eeg_clfs)
X_meta_oof[:, idx:idx+len(gaze_clfs)] = generate_oof(gaze_clfs, X_gaze_sel, y_gaze)

# -----------------------------
# Train meta-classifier and evaluate with fold-wise metrics
# -----------------------------
accs, precs, recs, f1s, rocs = [], [], [], [], []

for train_idx, test_idx in skf.split(X_meta_oof, y_video):  # use video labels as reference
    X_meta_train, X_meta_test = X_meta_oof[train_idx], X_meta_oof[test_idx]
    y_train, y_test = y_video[train_idx], y_video[test_idx]
    
    meta_clf.fit(X_meta_train, y_train)
    p_final = meta_clf.predict_proba(X_meta_test)[:, 1]
    y_pred = (p_final >= 0.5).astype(int)
    
    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred))
    recs.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    rocs.append(roc_auc_score(y_test, p_final))

# -----------------------------
# Final averaged results with margin of error (std)
# -----------------------------
def mean_std(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

print("\nCross-validated performance (Stacked EEG + Gaze + Video):")
print(f"Accuracy : {mean_std(accs)}")
print(f"Precision: {mean_std(precs)}")
print(f"Recall   : {mean_std(recs)}")
print(f"F1 Score : {mean_std(f1s)}")
print(f"ROC AUC  : {mean_std(rocs)}")
