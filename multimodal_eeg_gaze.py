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

# --- Load annotations ---
annotations = pd.read_csv("annotations.csv")
annotations = annotations.dropna(subset=['eeg', 'gaze'])

# --- EEG data ---
df_eeg = pd.read_csv("eeg_df_3_interpolated.csv")
X_eeg, y_eeg = df_eeg.drop('truth', axis=1).values, df_eeg['truth'].values

# --- Gaze data aligned to EEG runs ---
df_gaze = pd.read_csv("gaze_df_2.csv")
valid_indices = annotations.index  # only keep runs with EEG
df_gaze = df_gaze.iloc[valid_indices].reset_index(drop=True)
X_gaze, y_gaze = df_gaze.drop('truth', axis=1).values, df_gaze['truth'].values

# --- Feature selection ---
k = 1000

# EEG
scaler_eeg = StandardScaler()
X_eeg_scaled = scaler_eeg.fit_transform(X_eeg)
selector_eeg = SelectKBest(score_func=mutual_info_classif, k=min(k, X_eeg.shape[1]))
X_eeg_selected = selector_eeg.fit_transform(X_eeg_scaled, y_eeg)

# Gaze
scaler_gaze = StandardScaler()
X_gaze_scaled = scaler_gaze.fit_transform(X_gaze)
selector_gaze = SelectKBest(score_func=mutual_info_classif, k=min(k, X_gaze.shape[1]))
X_gaze_selected = selector_gaze.fit_transform(X_gaze_scaled, y_gaze)

# --- Base classifiers ---
base_classifiers_eeg = [
    ("rf", RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_split=2,
        min_samples_leaf=5, max_features='log2',
        class_weight='balanced', random_state=42, n_jobs=-1)),
    ("svm", SVC(
        kernel='rbf', C=25, gamma='auto', class_weight='balanced',
        probability=True, random_state=42)),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(500, 500), alpha=0.1e-05,
        learning_rate='constant', activation='relu',
        learning_rate_init=0.001, n_iter_no_change=5,
        early_stopping=True, max_iter=2000, random_state=42))
]

base_classifiers_gaze = [
    ("rf", RandomForestClassifier(
        n_estimators=100, max_depth=2, min_samples_split=2,
        min_samples_leaf=2, max_features='sqrt',
        class_weight=None, random_state=42, n_jobs=-1)),
    ("svm", SVC(
        kernel='rbf', C=1, gamma='auto', class_weight=None,
        probability=True, random_state=42)),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(300, 300), alpha=1e-05,
        learning_rate='constant', activation='tanh',
        learning_rate_init=0.005, n_iter_no_change=5,
        early_stopping=True, max_iter=2000, random_state=42))
]

# --- Cross-validation setup ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# --- Prepare out-of-fold predictions for stacking ---
n_samples = X_gaze_selected.shape[0]
n_meta_features = len(base_classifiers_eeg) + len(base_classifiers_gaze)
X_meta_oof = np.zeros((n_samples, n_meta_features))

for i, (name, clf) in enumerate(base_classifiers_eeg):
    oof_preds = np.zeros(n_samples)
    for train_idx, val_idx in skf.split(X_eeg_selected, y_eeg):
        clf.fit(X_eeg_selected[train_idx], y_eeg[train_idx])
        oof_preds[val_idx] = clf.predict_proba(X_eeg_selected[val_idx])[:, 1]
    X_meta_oof[:, i] = oof_preds

for j, (name, clf) in enumerate(base_classifiers_gaze):
    oof_preds = np.zeros(n_samples)
    for train_idx, val_idx in skf.split(X_gaze_selected, y_gaze):
        clf.fit(X_gaze_selected[train_idx], y_gaze[train_idx])
        oof_preds[val_idx] = clf.predict_proba(X_gaze_selected[val_idx])[:, 1]
    X_meta_oof[:, len(base_classifiers_eeg) + j] = oof_preds

# --- Grid Search for Gradient Boosting (meta-classifier) ---
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

print("Best parameters for Gradient Boosting:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

# --- Evaluate with StratifiedKFold using the best meta-classifier ---
best_meta_clf = grid_search.best_estimator_

accs, precs, recs, f1s, rocs = [], [], [], [], []
y_probs_full = np.zeros_like(y_gaze, dtype=float)

for train_idx, test_idx in skf.split(X_meta_oof, y_gaze):
    X_meta_train, X_meta_test = X_meta_oof[train_idx], X_meta_oof[test_idx]
    y_train, y_test = y_gaze[train_idx], y_gaze[test_idx]

    best_meta_clf.fit(X_meta_train, y_train)
    p_final = best_meta_clf.predict_proba(X_meta_test)[:, 1]
    y_pred = (p_final >= 0.5).astype(int)

    y_probs_full[test_idx] = p_final
    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred))
    recs.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    rocs.append(roc_auc_score(y_test, p_final))

print(f"\nFinal Evaluation with Best GB Parameters:")
print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1 Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"ROC AUC : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")

# -------------------------------
# Plot ROC curve
# -------------------------------
def plot_roc(y_true, y_prob, prefix="eeg_gaze"):
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

plot_roc(y_gaze, y_probs_full, prefix="stacked_eeg_gaze")

# -------------------------------
# Plot confusion matrix
# -------------------------------
def plot_confusion_matrix_oof(y_true, y_pred, prefix="eeg_gaze"):

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
