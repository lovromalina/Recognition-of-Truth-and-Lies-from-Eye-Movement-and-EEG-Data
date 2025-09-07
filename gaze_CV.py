import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold

np.random.seed(42)

df = pd.read_csv("gaze_df_2.csv")
X = df.drop(columns=["truth"])
y = df["truth"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k = min(1000, X.shape[1]) 
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_grids = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, None],
            'min_samples_leaf': [1, 2, 5],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "param_grid": {
            'C': [0.1, 1, 10, 25, 50, 100],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced', None]
        }
    },
    "MLP": {
        "model": MLPClassifier(random_state=42, max_iter=2000, early_stopping=True),
        "param_grid": {
            'hidden_layer_sizes': [(100,), (100, 100), (300, 300), (500, 500), (1000), (1000, 500), (1000, 500, 100)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.005, 0.01],
            'alpha': [1e-5, 1e-4],
            'n_iter_no_change': [5, 10]
        }
    }
}


models_to_run = ["MLP"]

for model_name in models_to_run:
    print(f"\n🔍 Running Grid Search for: {model_name}")
    
    clf = model_grids[model_name]["model"]
    param_grid = model_grids[model_name]["param_grid"]
    
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_selected, y)
    
    print(f"\n✅ Best {model_name} Accuracy: {grid_search.best_score_:.4f}")
    print(f"🏆 Best Parameters: {grid_search.best_params_}")
