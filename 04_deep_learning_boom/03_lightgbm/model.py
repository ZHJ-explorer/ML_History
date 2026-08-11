"""LightGBM实现。"""
import numpy as np
from collections import defaultdict


class LightGBMTree:
    """LightGBM决策树（使用直方图优化）。"""
    
    def __init__(self, max_depth=3, num_leaves=31):
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.tree = None
    
    def fit(self, X, gradients, hessians):
        self.tree = self._build_tree(X, gradients, hessians, depth=0)
        return self
    
    def _build_tree(self, X, g, h, depth):
        if depth >= self.max_depth or len(np.unique(g)) <= 1:
            return {'leaf': True, 'value': np.sum(g) / (np.sum(h) + 1e-16)}
        
        best_gain = -1
        best_split = None
        
        for feature in range(X.shape[1]):
            # 使用直方图优化
            bins = np.histogram(X[:, feature], bins=10)[1]
            feature_values = X[:, feature]
            
            for i in range(len(bins) - 1):
                left_mask = feature_values <= bins[i]
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
                    best_split = {'feature': feature, 'threshold': bins[i]}
        
        if best_split is None:
            return {'leaf': True, 'value': np.sum(g) / (np.sum(h) + 1e-16)}
        
        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        node = {'feature': best_split['feature'], 'threshold': best_split['threshold'], 'children': {}}
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


class LightGBM:
    """LightGBM分类器。"""
    
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
            
            tree = LightGBMTree(max_depth=self.max_depth)
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