"""MP神经元模型实现。

McCulloch-Pitts神经元是第一个数学形式的神经网络模型，
使用阶跃函数作为激活函数，只能处理二值输入输出。
"""

import numpy as np


class MPNeuron:
    """MP神经元模型。

    McCulloch-Pitts neuron是二值分类器，输入和输出都是0或1。
    通过设置权重和阈值可以实现基本逻辑门。
    """

    def __init__(self, n_features):
        """初始化MP神经元。

        Args:
            n_features: 输入特征数量。
        """
        self.n_features = n_features
        self.weights = None
        self.threshold = None

    def _step_function(self, z):
        """阶跃激活函数。

        Args:
            z: 加权求和结果。

        Returns:
            二值输出（0或1）。
        """
        return 1 if z >= self.threshold else 0

    def predict(self, X):
        """预测输出。

        Args:
            X: 输入矩阵，shape (n_samples, n_features)，值域为{0, 1}。

        Returns:
            预测输出，shape (n_samples,)。
        """
        if self.weights is None:
            raise ValueError("模型未初始化，请先设置权重和阈值")

        # 加权求和
        z = np.dot(X, self.weights)

        # 阶跃函数
        return np.array([self._step_function(z_i) for z_i in z])

    def set_logic_gate(self, gate_type="AND"):
        """设置逻辑门权重。

        Args:
            gate_type: 逻辑门类型，可选 "AND", "OR", "NOT"。
        """
        if gate_type == "AND":
            # AND门：两个输入都为1时输出1
            self.weights = np.array([1.0, 1.0])
            self.threshold = 1.5
        elif gate_type == "OR":
            # OR门：任一输入为1时输出1
            self.weights = np.array([1.0, 1.0])
            self.threshold = 0.5
        elif gate_type == "NOT":
            # NOT门：取反
            self.weights = np.array([-1.0])
            self.threshold = -0.5
        else:
            raise ValueError(f"不支持的逻辑门类型: {gate_type}")

    def train(self, X, y, learning_rate=0.1, max_epochs=100):
        """训练MP神经元。

        MP神经元本身没有学习机制，此方法仅用于演示。

        Args:
            X: 训练数据，shape (n_samples, n_features)。
            y: 训练标签，shape (n_samples,)。
            learning_rate: 学习率（未使用）。
            max_epochs: 最大训练轮数（未使用）。

        Returns:
            dict: 训练历史记录。
        """
        # MP神经元无法自动学习，返回空历史
        return {"loss": [], "accuracy": []}

    def fit(self, X, y):
        """拟合模型（占位方法）。

        Args:
            X: 训练数据。
            y: 训练标签。
        """
        # MP神经元没有自动学习机制
        pass

    def score(self, X, y):
        """计算准确率。

        Args:
            X: 测试数据。
            y: 真实标签。

        Returns:
            float: 准确率。
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)