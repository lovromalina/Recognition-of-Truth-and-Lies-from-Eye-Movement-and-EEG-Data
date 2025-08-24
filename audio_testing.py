import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

df = pd.read_csv("audio_features_with_truth.csv")
X = df.drop(columns=["path", "truth"])
y = df["truth"]

# Set up Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

accuracies_rf = []
accuracies_knn = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Random Forest
    clf = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    accuracies_rf.append(acc_rf)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    accuracies_knn.append(acc_knn)

# Report
print(f"RandomForest cross-validated accuracies: {accuracies_rf}")
print(f"RandomForest Mean accuracy: {np.mean(accuracies_rf):.4f}")
print(f"RandomForest Std deviation: {np.std(accuracies_rf):.4f}")
print("----------------------------------------------------------")
print(f"KNeighborsClassifier cross-validated accuracies: {accuracies_knn}")
print(f"KNeighborsClassifier Mean accuracy: {np.mean(accuracies_knn):.4f}")
print(f"KNeighborsClassifier Std deviation: {np.std(accuracies_knn):.4f}")
