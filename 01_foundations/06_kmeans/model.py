"""k-Means聚类模型实现。"""
import numpy as np

class KMeans:
    """k-Means聚类算法。"""
    def __init__(self, k=3, max_iter=100, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices].copy()
        for _ in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        self.labels = labels
        return self

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def inertia(self, X):
        distances = np.sqrt(((X - self.centroids[self.labels, :]) ** 2).sum(axis=1))
        return np.sum(distances ** 2)