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

# Base models
rf = RandomForestClassifier(
    n_estimators=200, max_depth=5,
    min_samples_split=2, min_samples_leaf=5,
    max_features='log2', class_weight='balanced',
    random_state=42, n_jobs=-1
)
svm = SVC(
    kernel='rbf', C=25, gamma='auto',
    class_weight='balanced', probability=True, random_state=42
)
mlp = MLPClassifier(
    hidden_layer_sizes=(500, 500), alpha=0.1e-05,
    learning_rate='constant', activation='relu',
    learning_rate_init=0.001, n_iter_no_change=5,
    early_stopping=True, max_iter=2000, random_state=42
)

# Voting and Stacking classifiers
base_models = {"Random Forest": rf, "SVM": svm, "MLP": mlp}
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='soft', n_jobs=-1
)
stacking_clf = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=skf, n_jobs=-1, passthrough=False
)

all_models = {**base_models, "Voting Classifier": voting_clf, "Stacking Classifier": stacking_clf}

# Function to collect out-of-fold probabilities
def collect_oof_probabilities(model, X, y, skf):
    y_true_all = []
    y_proba_all = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = 1 / (1 + np.exp(-model.decision_function(X_test)))
        else:
            y_proba = model.predict(X_test)

        y_true_all.extend(y_test)
        y_proba_all.extend(y_proba)

    return np.array(y_true_all), np.array(y_proba_all)

# Plot all ROC curves on the same figure
plt.figure(figsize=(8, 6))

for name, model in all_models.items():
    y_true_oof, y_proba_oof = collect_oof_probabilities(model, X_selected, y, skf)
    fpr, tpr, _ = roc_curve(y_true_oof, y_proba_oof)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

# Random chance line
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for EEG Models')
plt.legend(loc='lower right')
plt.tight_layout()

# Save combined ROC curve
plt.savefig("all_eeg_models_roc_curve.png", dpi=300)
plt.close()
print("Saved combined ROC curve as all_eeg_models_roc_curve.png")
