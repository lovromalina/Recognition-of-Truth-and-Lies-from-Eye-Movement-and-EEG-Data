import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.read_csv("video_features.csv")
X = df.drop(columns=["truth", "path"])
y = df["truth"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k = 1000
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base classifiers
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=1, min_samples_split=2,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)
svm = SVC(kernel="sigmoid", C=100, gamma=0.001, probability=True, random_state=42)
mlp = MLPClassifier(
    max_iter=2000, early_stopping=True, random_state=42,
    activation='tanh', alpha=0.0001, hidden_layer_sizes=(500, 100),
    learning_rate='constant', learning_rate_init=0.01, n_iter_no_change=20
)

# Ensemble classifiers
voting = VotingClassifier(
    estimators=[("rf", rf), ("svm", svm), ("mlp", mlp)],
    voting="soft", n_jobs=-1
)
stacking = StackingClassifier(
    estimators=[("rf", rf), ("svm", svm), ("mlp", mlp)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=skf, passthrough=False, n_jobs=-1
)

# Model dictionary
models = {
    "Random Forest": rf,
    "SVM": svm,
    "MLP": mlp,
    "Voting Classifier": voting,
    "Stacking Classifier": stacking
}

# Loop through models and save ROC curves
for name, model in models.items():
    y_true_all = []
    y_proba_all = []

    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-y_proba))  # scale to [0,1]
        else:
            y_proba = model.predict(X_test)

        y_true_all.extend(y_test)
        y_proba_all.extend(y_proba)

    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true_all, y_proba_all)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend(loc="lower right")

    filename = f"{name.replace(' ', '_').lower()}_roc_curve_video.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved ROC curve as {filename}")
