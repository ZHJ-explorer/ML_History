"""多层感知机（MLP）与反向传播实现。"""
import numpy as np

class MLP:
    """多层感知机，使用反向传播训练。"""
    def __init__(self, layer_sizes, learning_rate=0.01, max_iter=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros(self.layer_sizes[i+1]))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _forward(self, X):
        activations = [X]
        z_values = []
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = z
            activations.append(a)
        return activations, z_values

    def _backward(self, X, y, activations, z_values):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        output = activations[-1]
        deltas[-1] = (output - y) / m
        for i in range(len(self.weights) - 2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * self._relu_derivative(z_values[i])
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0)

    def fit(self, X, y):
        for _ in range(self.max_iter):
            activations, z_values = self._forward(X)
            self._backward(X, y, activations, z_values)
        return self

    def predict(self, X):
        activations, _ = self._forward(X)
        return activations[-1]

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))