import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base classifiers
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=1, min_samples_split=2,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)
svm = SVC(
    kernel="sigmoid", C=100, gamma=0.001,
    class_weight=None, probability=True, random_state=42
)
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

# Function to collect out-of-fold predictions
def collect_cv_predictions(model, X, y, skf):
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    return np.array(y_true_all), np.array(y_pred_all)

# Loop through models and save individual confusion matrices
for name, model in models.items():
    y_true_cv, y_pred_cv = collect_cv_predictions(model, X_selected, y, skf)

    cm = confusion_matrix(y_true_cv, y_pred_cv, normalize='true')  # normalized
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca(), colorbar=True)

    # Save figure
    filename = f"{name.replace(' ', '_').lower()}_normalized_cm_video.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")
