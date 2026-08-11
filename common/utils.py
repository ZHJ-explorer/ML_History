"""通用工具函数模块。

提供数据加载、评估指标、可视化等基础功能。
所有函数使用纯NumPy实现，不依赖外部ML库。
"""

import numpy as np
import os


def load_iris_data(data_dir="data"):
    """加载Iris数据集。

    使用sklearn内置数据集（仅用于加载，不用于建模）。

    Args:
        data_dir: 数据目录路径。

    Returns:
        tuple: (X, y) - 特征矩阵和标签向量。
    """
    try:
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target
    except ImportError:
        raise ImportError("请安装sklearn: pip install scikit-learn")


def load_breast_cancer_data(data_dir="data"):
    """加载Breast Cancer数据集。

    Args:
        data_dir: 数据目录路径。

    Returns:
        tuple: (X, y) - 特征矩阵和标签向量。
    """
    try:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data, data.target
    except ImportError:
        raise ImportError("请安装sklearn: pip install scikit-learn")


def load_mnist_data(data_dir="data"):
    """加载MNIST数据集。

    Args:
        data_dir: 数据目录路径。

    Returns:
        tuple: (X, y) - 特征矩阵和标签向量。
    """
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("mnist_784", version=1, as_frame=False)
        return data.data.astype(np.float32), data.target.astype(np.int32)
    except ImportError:
        raise ImportError("请安装sklearn: pip install scikit-learn")


def train_test_split(X, y, test_size=0.2, random_state=42):
    """划分训练集和测试集。

    Args:
        X: 特征矩阵，shape (n_samples, n_features)。
        y: 标签向量，shape (n_samples,)。
        test_size: 测试集比例。
        random_state: 随机种子。

    Returns:
        tuple: (X_train, X_test, y_train, y_test)。
    """
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    """标准化特征（零均值、单位方差）。

    使用训练集的统计量标准化训练集和测试集。

    Args:
        X_train: 训练集特征矩阵。
        X_test: 测试集特征矩阵。

    Returns:
        tuple: (X_train_std, X_test_std, mean, std)。
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # 避免除零

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_test_std, mean, std


def accuracy_score(y_true, y_pred):
    """计算准确率。

    Args:
        y_true: 真实标签。
        y_pred: 预测标签。

    Returns:
        float: 准确率。
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """计算混淆矩阵。

    Args:
        y_true: 真实标签。
        y_pred: 预测标签。

    Returns:
        ndarray: 混淆矩阵。
    """
    n_classes = len(np.unique(y_true))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm


def compute_metrics(y_true, y_pred):
    """计算多种评估指标。

    Args:
        y_true: 真实标签。
        y_pred: 预测标签。

    Returns:
        dict: 包含准确率、精确率、召回率、F1分数。
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # 计算精确率、召回率、F1（macro平均）
    precision = np.zeros(len(np.unique(y_true)))
    recall = np.zeros(len(np.unique(y_true)))
    f1 = np.zeros(len(np.unique(y_true)))

    for i in range(len(np.unique(y_true))):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return {
        "accuracy": acc,
        "precision_macro": np.mean(precision),
        "recall_macro": np.mean(recall),
        "f1_macro": np.mean(f1),
        "confusion_matrix": cm
    }


def plot_learning_curve(history, title="Learning Curve"):
    """绘制学习曲线。

    Args:
        history: 包含'loss'和'accuracy'的历史记录。
        title: 图表标题。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示：安装matplotlib可绘制学习曲线: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history["loss"], "b-", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(history["accuracy"], "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("学习曲线已保存到: learning_curve.png")


def plot_decision_boundary(X, y, model, resolution=200):
    """绘制决策边界。

    Args:
        X: 特征矩阵（需要是2D）。
        y: 标签向量。
        model: 模型对象，需要有predict方法。
        resolution: 网格分辨率。
    """
    if X.shape[1] != 2:
        print("仅支持2D特征的决策边界可视化")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示：安装matplotlib可绘制决策边界: pip install matplotlib")
        return

    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))

    # 预测网格
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("决策边界已保存到: decision_boundary.png")


def ensure_dir(path):
    """确保目录存在。

    Args:
        path: 目录路径。
    """
    os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    """设置随机种子以确保可复现性。

    Args:
        seed: 随机种子。
    """
    np.random.seed(seed)


def download_if_not_exists(url, filepath):
    """下载文件（如果不存在）。

    Args:
        url: 下载URL。
        filepath: 本地保存路径。
    """
    if not os.path.exists(filepath):
        print(f"下载中: {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, filepath)
            print(f"已保存到: {filepath}")
        except Exception as e:
            print(f"下载失败: {e}")
    else:
        print(f"文件已存在: {filepath}")


# 数据文件URL（可选下载）
DATA_URLS = {
    "iris.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "breast_cancer.csv": "https://raw.githubusercontent.com/mlin/breast-cancer-dataset/master/data/breast-cancer.csv",
}