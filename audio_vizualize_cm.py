import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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
    y_true_cv, y_pred_cv = collect_cv_predictions(model, X_scaled, y, skf)

    cm = confusion_matrix(y_true_cv, y_pred_cv, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca(), colorbar=True)

    # Save figure
    filename = f"{name.replace(' ', '_').lower()}_normalized_cm_audio.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")
