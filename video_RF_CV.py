import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold


df = pd.read_csv("video_features.csv")
X = df.drop(columns=["truth", "path"])
y = df["truth"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_leaf': [1, 2, 3, 5],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV with Stratified K-Fold and verbose output
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2  
)

# Fit the model
grid_search.fit(X_scaled, y)

# Final best results
print("\nBest RF Accuracy: {:.4f}".format(grid_search.best_score_))
print("Best Parameters:", grid_search.best_params_)
