"""k近邻模型实现。"""
import numpy as np
from collections import Counter

class KNN:
    """k近邻分类器。"""
    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _distance(self, x1, x2):
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"不支持的距离度量: {self.distance}")

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)