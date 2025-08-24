import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)

np.random.seed(42)


annotations = pd.read_csv("annotations.csv")
annotations = annotations.dropna(subset=['eeg', 'gaze'])


df_eeg = pd.read_csv("eeg_df_3_interpolated.csv")
X_eeg, y_eeg = df_eeg.drop('truth', axis=1).values, df_eeg['truth'].values


df_gaze = pd.read_csv("gaze_df_2.csv")
valid_indices = annotations.index  # only keep runs with EEG
df_gaze = df_gaze.iloc[valid_indices].reset_index(drop=True)
X_gaze, y_gaze = df_gaze.drop('truth', axis=1).values, df_gaze['truth'].values


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

# -----------------------------
# Base classifiers
# -----------------------------
base_classifiers_eeg = [
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=2,
                                  min_samples_leaf=5, max_features='log2',
                                  class_weight='balanced', random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel='rbf', C=25, gamma='auto', class_weight='balanced',
                probability=True, random_state=42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(500, 500), alpha=0.1e-05,
                          learning_rate='constant', activation='relu',
                          learning_rate_init=0.001, n_iter_no_change=5,
                          early_stopping=True, max_iter=2000, random_state=42))
]

base_classifiers_gaze = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_split=2,
                                  min_samples_leaf=2, max_features='sqrt',
                                  class_weight=None, random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel='rbf', C=1, gamma='auto', class_weight=None,
                probability=True, random_state=42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(300, 300), alpha=1e-05,
                          learning_rate='constant', activation='tanh',
                          learning_rate_init=0.005, n_iter_no_change=5,
                          early_stopping=True, max_iter=2000, random_state=42))
]

# Meta-classifier
meta_clf = GradientBoostingClassifier(random_state=42)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# To store final predictions for all folds
y_true_all = []
y_pred_all = []
y_score_all = []

n_samples = X_gaze_selected.shape[0]
n_meta_features = len(base_classifiers_eeg) + len(base_classifiers_gaze)
X_meta_oof = np.zeros((n_samples, n_meta_features))

# -----------------------------
# Generate out-of-fold predictions for meta-features
# -----------------------------
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

# -----------------------------
# Train meta-classifier fold by fold
# -----------------------------
for train_idx, test_idx in skf.split(X_meta_oof, y_gaze):
    X_meta_train, X_meta_test = X_meta_oof[train_idx], X_meta_oof[test_idx]
    y_train, y_test = y_gaze[train_idx], y_gaze[test_idx]
    
    # Train meta-classifier
    meta_clf.fit(X_meta_train, y_train)
    p_final = meta_clf.predict_proba(X_meta_test)[:, 1]
    y_pred = (p_final >= 0.5).astype(int)
    
    # Collect predictions
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    y_score_all.extend(p_final)

# -----------------------------
# Metrics
# -----------------------------
acc = accuracy_score(y_true_all, y_pred_all)
prec = precision_score(y_true_all, y_pred_all)
rec = recall_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all)
roc_auc = roc_auc_score(y_true_all, y_score_all)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
plt.figure(figsize=(6,6))
disp.plot(cmap=plt.cm.Blues, values_format='.2f')
plt.title("Normalized Confusion Matrix (Stacked EEG + Gaze)")
plt.savefig("stacked_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Stacked EEG + Gaze)')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("stacked_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()
