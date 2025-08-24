import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

df = pd.read_csv("eeg_df_3_interpolated.csv")
X = df.drop(columns=["truth"])
y = df["truth"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 500  
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)

param_grid = {
    'n_estimators': [50, 100, 200, 300],    
    'max_depth': [2, 3, 5, None], 
    'min_samples_leaf': [1, 2, 3, 5],
    'min_samples_split': [2, 5, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2  
)

grid_search.fit(X_selected, y)

# Final best results
print("\nBest RF Accuracy (with SelectKBest): {:.4f}".format(grid_search.best_score_))
print("Best Parameters:", grid_search.best_params_)
