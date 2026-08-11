"""LeNet-5 训练脚本。

演示如何在MNIST数据集上训练LeNet-5卷积神经网络。
关键模块（卷积层、池化层）使用手写实现。
"""
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import LeNet5
from common.utils import load_mnist_data, train_test_split, compute_metrics, set_seed


def demo():
    """训练并评估LeNet-5模型。"""
    set_seed(42)
    device = torch.device('cpu')

    print("=" * 50)
    print("LeNet-5 卷积神经网络演示")
    print("=" * 50)

    # 加载数据
    print("\n[1] 加载MNIST数据集...")
    X, y = load_mnist_data()
    # 归一化到[0, 1]
    X = X / 255.0
    # 重塑为 28x28 图像 (N, 1, 28, 28)
    X = X.reshape(-1, 1, 28, 28)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 转换为Tensor
    X_train = torch.FloatTensor(X_train)  # (N, 1, 28, 28)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # 构建模型
    print("\n[2] 构建LeNet-5模型...")
    model = LeNet5().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")
    print(model)

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    print(f"\n[3] 开始训练（{epochs} epochs）...")
    history = {"loss": [], "accuracy": []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (outputs.argmax(1) == y_batch).sum().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / len(X_train)
        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.1f}s")

    # 测试
    print("\n[4] 评估模型...")
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    print(f"测试集准确率: {metrics['accuracy']:.4f}")
    print(f"精确率(Macro): {metrics['precision_macro']:.4f}")
    print(f"召回率(Macro): {metrics['recall_macro']:.4f}")
    print(f"F1分数(Macro): {metrics['f1_macro']:.4f}")

    # 打印混淆矩阵
    cm = metrics["confusion_matrix"]
    print("\n混淆矩阵:")
    print(cm)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "data", "lenet5.pth"))
    print(f"\n模型已保存到: data/lenet5.pth")

    return model, metrics


if __name__ == "__main__":
    demo()
