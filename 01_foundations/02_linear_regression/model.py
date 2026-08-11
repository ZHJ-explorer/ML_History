"""线性回归模型实现。

使用最小二乘法求解，提供解析解和梯度下降两种方法。
"""

import numpy as np


class LinearRegression:
    """线性回归模型。

    使用最小二乘法求解，支持解析解和梯度下降两种方法。
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """初始化线性回归模型。

        Args:
            learning_rate: 学习率（梯度下降使用）。
            max_iter: 最大迭代次数（梯度下降使用）。
            tol: 收敛容差。
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.history = None

    def _fit_closed_form(self, X, y):
        """使用正规方程求解。

        解析解：w = (X^T X)^(-1) X^T y

        Args:
            X: 特征矩阵，添加偏置列。
            y: 目标变量。
        """
        XTX = X.T @ X

        if np.linalg.det(XTX) == 0:
            self.weights = np.linalg.pinv(XTX) @ X.T @ y
        else:
            self.weights = np.linalg.inv(XTX) @ X.T @ y

        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def _fit_gradient_descent(self, X, y):
        """使用梯度下降求解。

        Args:
            X: 特征矩阵。
            y: 目标变量。
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.history = {"loss": [], "weights": []}

        for i in range(self.max_iter):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.history["loss"].append(loss)

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.history["weights"].append(self.weights.copy())

            if i > 0 and abs(self.history["loss"][-2] - self.history["loss"][-1]) < self.tol:
                break

    def fit(self, X, y, method="closed_form"):
        """拟合模型。

        Args:
            X: 训练数据，shape (n_samples, n_features)。
            y: 目标变量，shape (n_samples,)。
            method: 求解方法，"closed_form" 或 "gradient_descent"。
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if method == "closed_form":
            X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            self._fit_closed_form(X_with_bias, y)
        elif method == "gradient_descent":
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError(f"不支持的求解方法: {method}")

        return self

    def predict(self, X):
        """预测。

        Args:
            X: 测试数据，shape (n_samples, n_features)。

        Returns:
            ndarray: 预测值，shape (n_samples,)。
        """
        if self.weights is None:
            raise ValueError("模型未训练，请先调用fit方法")

        return X @ self.weights + self.bias

    def score(self, X, y):
        """计算R²分数。

        Args:
            X: 测试数据。
            y: 真实标签。

        Returns:
            float: R²分数。
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0

        return 1 - (ss_res / ss_tot)