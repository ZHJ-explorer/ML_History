"""k-Means训练脚本。"""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import KMeans
from common.utils import load_iris_data, standardize

def demo():
    print("=" * 50)
    print("k-Means聚类演示")
    print("=" * 50)
    X, _ = load_iris_data()
    X_std, _, _, _ = standardize(X, X)
    model = KMeans(k=3, random_state=42)
    model.fit(X_std)
    print(f"聚类中心:\n{model.centroids}")
    print(f"惯性（簇内平方和）: {model.inertia(X_std):.2f}")

if __name__ == "__main__":
    demo()