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

df = pd.read_csv("eeg_df_3_interpolated.csv")
X = df.drop('truth', axis=1)
y = df['truth']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 500
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define base models
rf = RandomForestClassifier(
    n_estimators=200, max_depth=5,
    min_samples_split=2, min_samples_leaf=5,
    max_features='log2', class_weight='balanced',
    random_state=42, n_jobs=-1
)
svm = SVC(
    kernel='rbf',
    C=25,
    gamma='auto',
    class_weight='balanced',
    probability=True,
    random_state=42
)
mlp = MLPClassifier(
    hidden_layer_sizes=(500, 500),
    alpha=0.1e-05,
    learning_rate='constant',
    activation='relu',
    learning_rate_init=0.001,
    n_iter_no_change=5,
    early_stopping=True,
    max_iter=2000,
    random_state=42
)

# Base models dictionary
base_models = {
    "Random Forest": rf,
    "SVM": svm,
    "MLP": mlp
}

# Voting and Stacking classifiers
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='soft',
    n_jobs=-1
)

stacking_clf = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=skf,
    n_jobs=-1,
    passthrough=False,
)

all_models = {
    **base_models,
    "Voting Classifier": voting_clf,
    "Stacking Classifier": stacking_clf
}

# Models to run
models_to_run = [
    "Random Forest",
    "SVM",
    "MLP",
    "Voting Classifier",
    "Stacking Classifier"
]

# Evaluation loop with saving probabilities
for model_name in models_to_run:
    model = all_models[model_name]
    accs, precs, recs, f1s, rocs = [], [], [], [], []

    # Array to save all predicted probabilities for later use
    proba_all = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Get probabilities for ROC AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            # Scale decision_function to [0,1]
            y_proba = 1 / (1 + np.exp(-y_proba))
        else:
            y_proba = y_pred

        # Save probabilities in correct positions
        proba_all[test_idx] = y_proba

        # Metrics
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        rocs.append(roc_auc_score(y_test, y_proba))

    # Print cross-validated results
    print(f"\n==== {model_name} Cross-Validated Results ====")
    print(f"Accuracy     : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision    : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall       : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1 Score     : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"ROC AUC      : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")

    # Save predicted probabilities for later fusion
    # prob_filename = f"{model_name.replace(' ', '_').lower()}_proba.npy"
    # np.save(prob_filename, proba_all)
    # print(f"Predicted probabilities saved to {prob_filename}")
