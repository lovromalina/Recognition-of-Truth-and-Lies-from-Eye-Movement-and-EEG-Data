import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Models to visualize
selected_models = {"Random Forest": rf, "SVM": svm, "MLP": mlp}

# Loop through models and save individual confusion matrices
for name, model in selected_models.items():
    y_true_cv, y_pred_cv = collect_cv_predictions(model, X_selected, y, skf)
    
    cm = confusion_matrix(y_true_cv, y_pred_cv, normalize='true')  # normalized
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    
    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca(), colorbar=True)
    plt.title(f"{name} Normalized CM")
    
    # Save individual figure
    filename = f"{name.replace(' ', '_').lower()}_normalized_cm.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()
