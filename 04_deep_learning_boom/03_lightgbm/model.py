"""LightGBM实现。"""
import numpy as np


class LightGBMTree:
    """LightGBM决策树。"""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, gradients, hessians):
        self.tree = self._build_tree(X, gradients, hessians, depth=0)
        return self
    
    def _build_tree(self, X, g, h, depth):
        result = {}
        
        if depth >= self.max_depth or len(np.unique(g)) <= 1:
            result['leaf'] = True
            result['value'] = np.sum(g) / (np.sum(h) + 1e-8)
            return result
        
        best_gain = -1
        best_split = None
        
        n = len(g)
        for feature in range(X.shape[1]):
            sorted_idx = np.argsort(X[:, feature])
            sorted_g = g[sorted_idx]
            sorted_h = h[sorted_idx]
            sorted_X = X[sorted_idx, feature]
            
            left_g, left_h = 0, 0
            for i in range(n - 1):
                left_g += sorted_g[i]
                left_h += sorted_h[i]
                right_g = np.sum(sorted_g) - left_g
                right_h = np.sum(sorted_h) - left_h
                
                if sorted_X[i] == sorted_X[i + 1]:
                    continue
                if left_h < 1e-8 or right_h < 1e-8:
                    continue
                
                gain = (left_g ** 2) / (left_h + 1e-8) + (right_g ** 2) / (right_h + 1e-8)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature': feature, 'threshold': (sorted_X[i] + sorted_X[i + 1]) / 2}
        
        if best_split is None or best_gain <= 0:
            result['leaf'] = True
            result['value'] = np.sum(g) / (np.sum(h) + 1e-8)
            return result
        
        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        result['feature'] = best_split['feature']
        result['threshold'] = best_split['threshold']
        result['left'] = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        result['right'] = self._build_tree(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1)
        
        return result
    
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
    
    def __init__(self, n_estimators=50, max_depth=3, learning_rate=0.01):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.base_score = 0.5
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X, y):
        self.trees = []
        n = len(y)
        
        self.base_score = np.mean(y)
        initial_pred = np.log(self.base_score / (1 - self.base_score + 1e-16) + 1e-16)
        predictions = np.full(n, initial_pred)
        
        for i in range(self.n_estimators):
            probabilities = self._sigmoid(predictions)
            gradients = probabilities - y
            hessians = probabilities * (1 - probabilities) + 1e-8
            
            tree = LightGBMTree(max_depth=self.max_depth)
            tree.fit(X, gradients, hessians)
            
            predictions += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            
            if (i + 1) % 10 == 0:
                prob = self._sigmoid(predictions)
                preds = (prob >= 0.5).astype(int)
                acc = np.mean(preds == y)
                print(f"  Iteration {i+1}/{self.n_estimators}, Train Acc: {acc:.4f}")
        
        return self
    
    def predict(self, X):
        n = X.shape[0]
        initial_pred = np.log(self.base_score / (1 - self.base_score + 1e-16) + 1e-16)
        predictions = np.full(n, initial_pred)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return (self._sigmoid(predictions) >= 0.5).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)