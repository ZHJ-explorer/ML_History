"""决策树（ID3算法）实现。"""
import numpy as np
from collections import Counter

class DecisionTree:
    """ID3决策树分类器。"""
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [c / len(y) for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def _info_gain(self, y, y_left, y_right):
        parent_entropy = self._entropy(y)
        n = len(y)
        if n == 0:
            return 0
        child_entropy = (len(y_left) / n) * self._entropy(y_left) + \
                        (len(y_right) / n) * self._entropy(y_right)
        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                gain = self._info_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        feature, threshold, gain = self._best_split(X, y)
        if gain <= 0:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(np.array(X), np.array(y))
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)