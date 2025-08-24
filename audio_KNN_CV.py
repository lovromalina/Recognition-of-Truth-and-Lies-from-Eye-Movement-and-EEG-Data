import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

np.random.seed(42)

df = pd.read_csv("audio_features_with_truth.csv")

# Features (exclude 'path' and 'truth')
X = df.drop(columns=["path", "truth"])
y = df["truth"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
k = 500
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 25, 31],   
    'weights': ['uniform', 'distance'],                  
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine'],  
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']                          
}

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_selected, y)

print("\nBest KNN Accuracy (with SelectKBest): {:.4f}".format(grid_search.best_score_))
print("Best Parameters:", grid_search.best_params_)
