"""感知机模型实现。"""
import numpy as np

class Perceptron:
    """感知机模型。"""
    def __init__(self, learning_rate=0.01, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def _activation(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self._activation(np.dot(xi, self.weights) + self.bias)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                self.bias += update
                if update != 0:
                    errors += 1
            if errors == 0:
                break
        return self

    def predict(self, X):
        return np.array([self._activation(np.dot(xi, self.weights) + self.bias) for xi in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)