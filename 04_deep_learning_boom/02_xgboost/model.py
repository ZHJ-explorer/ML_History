"""XGBoost实现。"""
import numpy as np
from collections import defaultdict


class XGBoostTree:
    """XGBoost决策树。"""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        self.leaf_value = None
    
    def fit(self, X, gradients, hessians):
        """构建树。"""
        self.tree = self._build_tree(X, gradients, hessians, depth=0)
        return self
    
    def _build_tree(self, X, g, h, depth):
        node = {'children': {}}
        
        if depth >= self.max_depth or len(np.unique(g)) <= 1:
            self.leaf_value = np.sum(g) / (np.sum(h) + 1e-16)
            return {'leaf': True, 'value': self.leaf_value}
        
        best_gain = -1
        best_split = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                
                left_g = np.sum(g[left_mask])
                left_h = np.sum(h[left_mask])
                right_g = np.sum(g[right_mask])
                right_h = np.sum(h[right_mask])
                
                gain = (left_g ** 2) / (left_h + 1e-16) + (right_g ** 2) / (right_h + 1e-16)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature': feature, 'threshold': threshold}
        
        if best_split is None:
            self.leaf_value = np.sum(g) / (np.sum(h) + 1e-16)
            return {'leaf': True, 'value': self.leaf_value}
        
        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        node['feature'] = best_split['feature']
        node['threshold'] = best_split['threshold']
        node['left'] = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        node['right'] = self._build_tree(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1)
        
        return node
    
    def predict_one(self, x, node):
        if node.get('leaf'):
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


class XGBoost:
    """XGBoost分类器。"""
    
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X, y):
        self.trees = []
        n = len(y)
        predictions = np.zeros(n)
        
        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(predictions)
            gradients = probabilities - y
            hessians = probabilities * (1 - probabilities)
            
            tree = XGBoostTree(max_depth=self.max_depth)
            tree.fit(X, gradients, hessians)
            
            predictions += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        n = X.shape[0]
        predictions = np.zeros(n)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return (self._sigmoid(predictions) >= 0.5).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)