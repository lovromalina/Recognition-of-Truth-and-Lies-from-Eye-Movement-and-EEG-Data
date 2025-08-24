import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

np.random.seed(42)

df = pd.read_csv("audio_features_with_truth.csv")

X = df.drop(columns=["truth", "path"], errors='ignore')
y = df["truth"]

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

k = 500 
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.1, 1, 10, 25, 50, 75, 100],
    'gamma': ['scale', 'auto', 1e-3, 1e-4],
    'class_weight': [None, 'balanced']
}

svm = SVC(random_state=42, probability=True)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_selected, y)

print("\nBest SVM Accuracy: {:.4f}".format(grid_search.best_score_))
print("Best Parameters:", grid_search.best_params_)
