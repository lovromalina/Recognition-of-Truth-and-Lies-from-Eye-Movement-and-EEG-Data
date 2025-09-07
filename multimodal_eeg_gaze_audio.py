import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc, confusion_matrix)

np.random.seed(42)

# -------------------------------
# Load and align data
# -------------------------------
annotations = pd.read_csv("annotations.csv")
annotations = annotations.dropna(subset=['eeg', 'gaze'])
valid_indices = annotations.index

df_eeg = pd.read_csv("eeg_df_3_interpolated.csv")
X_eeg, y_eeg = df_eeg.drop('truth', axis=1).values, df_eeg['truth'].values

df_gaze = pd.read_csv("gaze_df_2.csv")
df_gaze = df_gaze.iloc[valid_indices].reset_index(drop=True)
X_gaze, y_gaze = df_gaze.drop('truth', axis=1).values, df_gaze['truth'].values

df_video = pd.read_csv("video_features.csv")
df_video = df_video.iloc[valid_indices].reset_index(drop=True)
X_video, y_video = df_video.drop(columns=['truth', 'path']).values, df_video['truth'].values

# Align samples across all modalities
labels_all = np.vstack([y_eeg, y_gaze, y_video]).T
valid_mask = np.all(labels_all == labels_all[:, [0]], axis=1)
valid_indices_final = np.where(valid_mask)[0]

X_eeg, y_eeg = X_eeg[valid_indices_final], y_eeg[valid_indices_final]
X_gaze, y_gaze = X_gaze[valid_indices_final], y_gaze[valid_indices_final]
X_video, y_video = X_video[valid_indices_final], y_video[valid_indices_final]

# -------------------------------
# Feature scaling & selection
# -------------------------------
k = 1000
def scale_select(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    X_sel = SelectKBest(mutual_info_classif, k=min(k, X.shape[1])).fit_transform(X_scaled, y)
    return X_sel

X_eeg_sel = scale_select(X_eeg, y_eeg)
X_gaze_sel = scale_select(X_gaze, y_gaze)
X_video_sel = scale_select(X_video, y_video)

# -------------------------------
# Base classifiers
# -------------------------------
video_clfs = [
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=1,
                                  min_samples_split=2, max_features='sqrt',
                                  class_weight='balanced', random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel="sigmoid", C=100, gamma=0.001, probability=True, random_state=42)),
    ("mlp", MLPClassifier(max_iter=2000, early_stopping=True, random_state=42,
                          activation='tanh', alpha=0.0001, hidden_layer_sizes=(500, 100),
                          learning_rate='constant', learning_rate_init=0.01, n_iter_no_change=20))
]

eeg_clfs = [
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=2,
                                  min_samples_leaf=5, max_features='log2', class_weight='balanced',
                                  random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel='rbf', C=25, gamma='auto', class_weight='balanced',
                probability=True, random_state=42)),
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

# -------------------------------
# Generate OOF meta-features
# -------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
n_samples = X_video_sel.shape[0]
n_meta_features = len(video_clfs) + len(eeg_clfs) + len(gaze_clfs)
X_meta_oof = np.zeros((n_samples, n_meta_features))

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

# -------------------------------
# Grid Search for Gradient Boosting
# -------------------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'loss': ['deviance', 'exponential']
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_meta_oof, y_gaze)
best_meta_clf = grid_search.best_estimator_
print("Best parameters for Gradient Boosting:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

# -------------------------------
# Final evaluation
# -------------------------------
accs, precs, recs, f1s, rocs = [], [], [], [], []
y_probs_full = np.zeros_like(y_gaze, dtype=float)

for train_idx, test_idx in skf.split(X_meta_oof, y_gaze):
    X_train, X_test = X_meta_oof[train_idx], X_meta_oof[test_idx]
    y_train, y_test = y_gaze[train_idx], y_gaze[test_idx]

    best_meta_clf.fit(X_train, y_train)
    y_prob = best_meta_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    y_probs_full[test_idx] = y_prob
    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred))
    recs.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    rocs.append(roc_auc_score(y_test, y_prob))

def mean_std(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

print("\nFinal Evaluation with Best GB Parameters:")
print(f"Accuracy : {mean_std(accs)}")
print(f"Precision: {mean_std(precs)}")
print(f"Recall   : {mean_std(recs)}")
print(f"F1 Score : {mean_std(f1s)}")
print(f"ROC AUC  : {mean_std(rocs)}")

# -------------------------------
# Plot ROC curve
# -------------------------------
def plot_roc(y_true, y_prob, prefix="eeg_gaze_video"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Chance')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_roc.png", dpi=300)
    plt.close()
    print(f"ROC curve saved as {prefix}_roc.png")

plot_roc(y_gaze, y_probs_full, prefix="stacked_eeg_gaze_video")

# -------------------------------
# Plot confusion matrix
# -------------------------------
def plot_confusion_matrix_oof(y_true, y_pred, prefix="eeg_gaze_video"):

    cm = confusion_matrix(y_true, y_pred, normalize='true')  # normalized CM
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))

    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca(), colorbar=True)
    plt.tight_layout()
    
    filename = f"{prefix}_normalized_cm.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

# Usage with your final predictions
y_pred_final = (y_probs_full >= 0.5).astype(int)
plot_confusion_matrix_oof(y_gaze, y_pred_final)
