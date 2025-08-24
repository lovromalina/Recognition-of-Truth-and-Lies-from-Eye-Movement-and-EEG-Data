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

# Voting and Stacking classifiers
base_models = {"Random Forest": rf, "SVM": svm, "MLP": mlp}
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
    passthrough=False
)

all_models = {**base_models, "Voting Classifier": voting_clf, "Stacking Classifier": stacking_clf}

# Function to collect out-of-fold predictions
def collect_oof_predictions(model, X, y, skf):
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

# Loop through models and save confusion matrices
for name, model in all_models.items():
    y_true_oof, y_pred_oof = collect_oof_predictions(model, X_selected, y, skf)
    
    cm = confusion_matrix(y_true_oof, y_pred_oof, normalize='true')  # normalized CM
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    
    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca(), colorbar=True)
    plt.title(f"{name} Normalized Confusion Matrix")
    
    filename = f"{name.replace(' ', '_').lower()}_normalized_cm_eeg.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")
