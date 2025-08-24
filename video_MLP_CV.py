import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


df = pd.read_csv("video_features.csv")
X = df.drop(columns=["truth", "path"])
y = df["truth"]

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=2000, early_stopping=True, random_state=42))
])


param_grid = {
    'mlp__hidden_layer_sizes': [(300, 100), (300, 200), (500, 100), (500, 200), (500, 300)],
    'mlp__alpha': [1e-5, 1e-4, 1e-3],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__learning_rate_init': [0.001, 0.005, 0.01],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__n_iter_no_change': [5, 10, 20]
}

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X, y)

# Results
print(f"Best Accuracy: {grid.best_score_:.4f}")
print(f"Best Parameters: {grid.best_params_}")
