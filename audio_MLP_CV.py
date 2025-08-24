import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

np.random.seed(42)

df = pd.read_csv("audio_features_with_truth.csv")

X = df.drop(columns=["path", "truth"])
y = df["truth"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 500
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

param_grid = {
    'hidden_layer_sizes': [(300, 200), (500, 100), (500, 200), (500, 300), (500, 300, 100), (500, 500), (500, 500, 100)],
    'alpha': [1e-5, 1e-4],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.005],
    'activation': ['relu', 'tanh'],
}

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    MLPClassifier(random_state=42, max_iter=2000, early_stopping=True),
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_selected, y)

print("\nBest MLP Accuracy (with SelectKBest): {:.4f}".format(grid_search.best_score_))
print("Best Parameters:", grid_search.best_params_)
