import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("gaze_df_2.csv")
X = df.drop('truth', axis=1)
y = df['truth']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 1000
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
accuracies = []
precisions = []
recalls = []
f1s = []
rocs = []

for train_index, test_index in skf.split(X_selected, y):
    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=2,
        min_samples_split=50,
        min_samples_leaf=5,
        max_features='log2',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    rocs.append(roc)

    print("\nFold classification report:\n", classification_report(y_test, y_pred, zero_division=0))

# Final cross-validated scores
print("\n==== Cross-Validated Results ====")
print(f"Accuracy     : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision    : {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall       : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score     : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"ROC AUC      : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
