import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

df = pd.read_csv("gaze_df_2.csv")
X = df.drop('truth', axis=1)
y = df['truth']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 1000
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base models
rf = RandomForestClassifier(
    n_estimators=100, max_depth=2,
    min_samples_split=2, min_samples_leaf=2,
    max_features='sqrt', class_weight=None,
    random_state=42, n_jobs=-1
)
svm = SVC(
    kernel='rbf',
    C=1,
    gamma='auto',
    class_weight=None,
    probability=True,
    random_state=42
)
mlp = MLPClassifier(
    hidden_layer_sizes=(300, 300),
    alpha=1e-05,
    learning_rate='constant',
    activation='tanh',
    learning_rate_init=0.005,
    n_iter_no_change=5,
    early_stopping=True,
    max_iter=2000,
    random_state=42
)

# Model dictionary
models = {
    "Random Forest": rf,
    "SVM": svm,
    "MLP": mlp,
    "Voting Classifier": VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("mlp", mlp)],
        voting='soft',
        n_jobs=-1
    ),
    "Stacking Classifier": StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("mlp", mlp)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=skf,
        passthrough=False,
        n_jobs=-1
    )
}

# Evaluation with saving probabilities
for model_name, model in models.items():
    accs, precs, recs, f1s, rocs = [], [], [], [], []
    proba_all = np.zeros(len(y))  # store out-of-fold predictions

    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            # convert scores into probabilities
            y_proba = 1 / (1 + np.exp(-y_proba))
        else:
            y_proba = y_pred

        # Save probs in correct test positions
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

    # Save predicted probabilities
    prob_filename = f"gaze_{model_name.replace(' ', '_').lower()}_proba.npy"
    np.save(prob_filename, proba_all)
    print(f"Saved probabilities to {prob_filename}")

