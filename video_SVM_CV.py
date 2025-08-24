import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("video_features.csv")

X = df.drop(columns=["truth", "path"])
y = df["truth"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 1e-3, 1e-4],
    'class_weight': [None, 'balanced']
}


svm = SVC(random_state=42)

# Define Stratified K-Fold cross-validator
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

grid_search.fit(X_scaled, y)

print("Best SVM Accuracy:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)
