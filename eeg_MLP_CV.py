import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

np.random.seed(42)

df = pd.read_csv("eeg_df_3_interpolated.csv")
X = df.drop(columns=["truth", "path"], errors='ignore')  
y = df["truth"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 500
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

mlp = MLPClassifier(max_iter=2000, early_stopping=True, random_state=42)

# Parameter grid for tuning
param_grid = {
    'hidden_layer_sizes': [(300, 200), (500, 100), (500, 200), (500, 300), (500, 300, 100), (500, 500), (500, 500, 100)],
    'alpha': [1e-5, 1e-4],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.005],
    'activation': ['relu', 'tanh'],
    'n_iter_no_change': [5, 10]
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run grid search
grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid.fit(X_selected, y)

# Results
print(f"\nBest Accuracy: {grid.best_score_:.4f}")
print(f"Best Parameters: {grid.best_params_}")
