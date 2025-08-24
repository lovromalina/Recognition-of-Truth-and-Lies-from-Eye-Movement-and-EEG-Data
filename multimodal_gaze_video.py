import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

np.random.seed(42)


df_gaze = pd.read_csv("gaze_df_2.csv")
X_gaze, y_gaze = df_gaze.drop('truth', axis=1).values, df_gaze['truth'].values


df_video = pd.read_csv("video_features.csv")
X_video, y_video = df_video.drop(columns=['truth', 'path'], errors='ignore').values, df_video['truth'].values


if not np.array_equal(y_gaze, y_video):
    raise ValueError("⚠️ Gaze and Video labels mismatch detected!")


k = 1000

def scale_select(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    X_sel = SelectKBest(mutual_info_classif, k=min(k, X.shape[1])).fit_transform(X_scaled, y)
    return X_sel

X_gaze_sel  = scale_select(X_gaze, y_gaze)
X_video_sel = scale_select(X_video, y_video)


video_clfs = [
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=1, min_samples_split=2,
                                  max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel="sigmoid", C=100, gamma=0.001, probability=True, random_state=42)),
    ("mlp", MLPClassifier(max_iter=2000, early_stopping=True, random_state=42, activation='tanh',
                          alpha=0.0001, hidden_layer_sizes=(500, 100), learning_rate='constant', learning_rate_init=0.01, n_iter_no_change=20))
]

gaze_clfs = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_split=2,
                                  min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(300, 300), alpha=1e-05, learning_rate='constant',
                          activation='tanh', learning_rate_init=0.005, n_iter_no_change=5,
                          early_stopping=True, max_iter=2000, random_state=42))
]


# Meta-classifier
meta_clf = GradientBoostingClassifier(random_state=42)


# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
n_samples = X_video_sel.shape[0]
n_meta_features = len(video_clfs) + len(gaze_clfs)
X_meta_oof = np.zeros((n_samples, n_meta_features))


# Generate out-of-fold predictions
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
X_meta_oof[:, idx:idx+len(video_clfs)] = generate_oof(video_clfs, X_video_sel, y_video)
idx += len(video_clfs)
X_meta_oof[:, idx:idx+len(gaze_clfs)] = generate_oof(gaze_clfs, X_gaze_sel, y_gaze)


acc_list, prec_list, rec_list, f1_list, roc_list = [], [], [], [], []

for train_idx, test_idx in skf.split(X_meta_oof, y_video):
    X_meta_train, X_meta_test = X_meta_oof[train_idx], X_meta_oof[test_idx]
    y_train, y_test = y_video[train_idx], y_video[test_idx]
    
    meta_clf.fit(X_meta_train, y_train)
    p_final = meta_clf.predict_proba(X_meta_test)[:, 1]
    y_pred = (p_final >= 0.5).astype(int)
    
    acc_list.append(accuracy_score(y_test, y_pred))
    prec_list.append(precision_score(y_test, y_pred))
    rec_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))
    roc_list.append(roc_auc_score(y_test, p_final))


# Compute mean ± std for all metrics

acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
prec_mean, prec_std = np.mean(prec_list), np.std(prec_list)
rec_mean, rec_std = np.mean(rec_list), np.std(rec_list)
f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
roc_mean, roc_std = np.mean(roc_list), np.std(roc_list)

print(f"Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Precision: {prec_mean:.4f} ± {prec_std:.4f}")
print(f"Recall   : {rec_mean:.4f} ± {rec_std:.4f}")
print(f"F1 Score : {f1_mean:.4f} ± {f1_std:.4f}")
print(f"ROC AUC  : {roc_mean:.4f} ± {roc_std:.4f}")
