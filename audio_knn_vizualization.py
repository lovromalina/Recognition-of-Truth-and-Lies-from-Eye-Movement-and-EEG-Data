import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


df = pd.read_csv("audio_features_with_truth.csv")

# Drop non-feature columns
X = df.drop(columns=["path", "truth"])
y = df["truth"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

knn = KNeighborsClassifier(
    n_neighbors=15,
    metric='chebyshev',
    weights='uniform',
    p=1
)
knn.fit(X_2d, y)


x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Predict over the grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 7))
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])


plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)

plt.title("KNN Decision Boundary (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")


plt.savefig("knn_decision_boundary.png", dpi=300, bbox_inches='tight')

plt.show()
