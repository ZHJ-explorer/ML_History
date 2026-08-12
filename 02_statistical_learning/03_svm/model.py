"""支持向量机（SVM）实现。"""
import numpy as np

class SVM:
    """使用简化版SMO算法求解的SVM分类器。"""
    def __init__(self, C=1.0, kernel='linear', gamma=None):
        """初始化SVM分类器。

        Args:
            C: 正则化参数，控制间隔最大化与误分类惩罚的平衡，默认为1.0。
            kernel: 核函数类型，支持'linear'和'rbf'，默认为'linear'。
            gamma: RBF核函数系数，默认为None（自动计算为1/n_features）。
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alphas = None
        self.b = 0
        self.support_vectors = None
        self.labels = None

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma else 1.0 / X1.shape[1]
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * sq_dists)
        return np.dot(X1, X2.T)

    def fit(self, X, y):
        """训练SVM分类器。

        Args:
            X: 训练特征矩阵，形状为(n_samples, n_features)。
            y: 训练标签，形状为(n_samples,)，取值为1或-1。

        Returns:
            self: 训练后的分类器实例。
        """
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.b = 0
        kernel_matrix = self._kernel(X, X)
        for epoch in range(100):
            for i in range(n_samples):
                ei = np.dot(self.alphas * y, kernel_matrix[i]) + self.b - y[i]
                if (y[i] * ei < -1e-5 and self.alphas[i] < self.C) or \
                   (y[i] * ei > 1e-5 and self.alphas[i] > 0):
                    j = np.random.randint(n_samples)
                    ej = np.dot(self.alphas * y, kernel_matrix[j]) + self.b - y[j]
                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if L == H:
                        continue
                    eta = 2 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j]
                    if eta >= 0:
                        continue
                    self.alphas[j] -= y[j] * (ei - ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    b1 = self.b - ei - y[i] * (self.alphas[i] - alpha_i_old) * kernel_matrix[i][i] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * kernel_matrix[i][j]
                    b2 = self.b - ej - y[i] * (self.alphas[i] - alpha_i_old) * kernel_matrix[i][j] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * kernel_matrix[j][j]
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
        sv_mask = self.alphas > 1e-5
        if np.sum(sv_mask) == 0:
            self.support_vectors = X
            self.labels = y
            self.alphas = np.ones(n_samples) / n_samples
        else:
            self.support_vectors = X[sv_mask]
            self.labels = y[sv_mask]
            self.alphas = self.alphas[sv_mask]
        return self

    def _decision_function(self, X):
        return np.dot(X, self.support_vectors.T).dot(self.alphas * self.labels) + self.b

    def predict(self, X):
        """对输入数据进行预测。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。

        Returns:
            预测标签数组，形状为(n_samples,)，取值为1或-1。
        """
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        """计算分类准确率。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。
            y: 真实标签，形状为(n_samples,)，取值为1或-1。

        Returns:
            分类准确率（0到1之间的浮点数）。
        """
        return np.mean(self.predict(X) == y)