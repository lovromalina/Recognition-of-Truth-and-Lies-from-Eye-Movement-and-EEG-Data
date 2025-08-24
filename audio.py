import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

df = pd.read_csv("audio_features_with_truth.csv")
X = df.drop(columns=["path", "truth"])
y = df["truth"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base classifiers
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    n_estimators=50,
    max_depth=None,
    max_features='sqrt',
    min_samples_split=20,
    min_samples_leaf=5
)

knn = KNeighborsClassifier(
    n_neighbors=15,
    algorithm='auto',
    metric='chebyshev',
    weights='uniform',
    p=1
)

svm = SVC(
    kernel='linear',
    C=25,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(500, 300, 100),
    activation='tanh',
    max_iter=2000,
    early_stopping=True,
    alpha=1e-05,
    learning_rate='constant',
    learning_rate_init=0.005,
    random_state=42
)

# Ensemble classifiers
voting = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
    voting='soft',
    n_jobs=-1
)

stacking = StackingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=skf,
    passthrough=False,
    n_jobs=-1
)

# Model dictionary
models = {
    "Random Forest": rf,
    "KNN": knn,
    "SVM": svm,
    "MLP": mlp,
    "Voting Classifier": voting,
    "Stacking Classifier": stacking
}

# Evaluation loop
for model_name, model in models.items():
    accs, precs, recs, f1s, rocs = [], [], [], [], []
    proba_all = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-y_proba))
        else:
            y_proba = y_pred

        proba_all[test_idx] = y_proba

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        rocs.append(roc_auc_score(y_test, y_proba))

    # Print metrics
    print(f"\n==== {model_name} Cross-Validated Results ====")
    print(f"Accuracy     : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision    : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall       : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1 Score     : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"ROC AUC      : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")

