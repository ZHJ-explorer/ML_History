"""RNN训练脚本。

演示如何在文本数据上训练RNN循环神经网络。
使用IMDB情感分析数据集（简化版）。
关键模块（RNN单元）使用手写实现。
"""
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import TextRNN, ManualRNN
from common.utils import set_seed


# 简化版数据集：使用数字序列进行分类
class SimpleSeqDataset(Dataset):
    """简化的序列数据集，用于演示RNN。

    生成随机序列，根据序列特征进行分类。
    任务：判断序列中奇偶数的数量差异
    """

    def __init__(self, num_samples=2000, seq_len=20, vocab_size=10, num_classes=2):
        """初始化数据集。

        Args:
            num_samples: 样本数量。
            seq_len: 序列长度。
            vocab_size: 词汇表大小。
            num_classes: 分类数。
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        # 生成随机数据
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            # 生成随机序列
            seq = torch.randint(0, vocab_size, (seq_len,))
            # 计算序列中偶数的数量
            even_count = (seq % 2 == 0).sum().item()
            # 如果偶数数量超过一半则为正类
            label = 1 if even_count > seq_len // 2 else 0
            self.data.append(seq)
            self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 文本情感分析数据集（简化版IMDB）
class SentimentDataset(Dataset):
    """简化版情感分析数据集。

    使用简单的词汇表进行情感分类。
    """

    def __init__(self, num_samples=2000, seq_len=30, vocab_size=500, max_len=30):
        """初始化数据集。

        Args:
            num_samples: 样本数量。
            seq_len: 序列长度。
            vocab_size: 词汇表大小。
            max_len: 最大序列长度。
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.max_len = max_len

        # 生成模拟文本数据
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            # 生成随机词索引序列
            actual_len = np.random.randint(10, max_len + 1)
            seq = torch.randint(1, vocab_size, (actual_len,))  # 0是padding

            # 简单情感规则：根据序列中特定范围的词来决定标签
            # 后半部分使用"积极词"或"消极词"
            half_len = actual_len // 2
            pos_start, pos_end = vocab_size // 2, vocab_size
            neg_start, neg_end = 1, vocab_size // 2

            if np.random.random() > 0.5:
                # 正类：后半部分使用积极词
                pos_words = torch.randint(pos_start, pos_end, (actual_len - half_len,))
                seq[half_len:actual_len] = pos_words
                label = 1
            else:
                # 负类：后半部分使用消极词
                neg_words = torch.randint(neg_start, neg_end, (actual_len - half_len,))
                seq[half_len:actual_len] = neg_words
                label = 0

            # 填充到固定长度
            if len(seq) < max_len:
                seq = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])

            self.data.append(seq)
            self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def demo_basic_rnn():
    """演示基础RNN单元。"""
    print("=" * 50)
    print("基础RNN单元演示")
    print("=" * 50)

    # 创建简单数据集
    dataset = SimpleSeqDataset(num_samples=2000, seq_len=20, vocab_size=10, num_classes=2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 创建模型
    model = TextRNN(
        vocab_size=10,
        embed_dim=16,
        hidden_size=32,
        output_size=2,
        num_layers=1
    )

    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 30

    print(f"\n开始训练（{epochs} epochs）...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(1) == y_batch).sum().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / len(train_dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.1f}s")

    # 测试
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_correct = 0
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            test_loss += criterion(logits, y_batch).item()
            test_correct += (logits.argmax(1) == y_batch).sum().item()

        test_loss /= len(test_loader)
        test_acc = test_correct / len(test_dataset)

    print(f"\n测试集准确率: {test_acc:.4f}")
    print(f"测试集损失: {test_loss:.4f}")

    return model, test_acc


def demo_sentiment_rnn():
    """演示文本情感分类RNN。"""
    print("\n" + "=" * 50)
    print("文本情感分类RNN演示")
    print("=" * 50)

    # 创建数据集
    dataset = SentimentDataset(num_samples=2000, seq_len=30, vocab_size=500)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 创建模型
    model = TextRNN(
        vocab_size=500,
        embed_dim=32,
        hidden_size=64,
        output_size=2,
        num_layers=2
    )

    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20

    print(f"\n开始训练（{epochs} epochs）...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(1) == y_batch).sum().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / len(train_dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.1f}s")

    # 测试
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_correct = 0
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            test_loss += criterion(logits, y_batch).item()
            test_correct += (logits.argmax(1) == y_batch).sum().item()

        test_loss /= len(test_loader)
        test_acc = test_correct / len(test_dataset)

    print(f"\n测试集准确率: {test_acc:.4f}")
    print(f"测试集损失: {test_loss:.4f}")

    return model, test_acc


def demo():
    """主演示函数。"""
    set_seed(42)

    print("=" * 60)
    print("RNN 循环神经网络演示")
    print("=" * 60)

    # 演示基础RNN
    model1, acc1 = demo_basic_rnn()

    # 演示情感分类RNN
    model2, acc2 = demo_sentiment_rnn()

    print("\n" + "=" * 60)
    print("演示完成！")
    print(f"基础RNN测试准确率: {acc1:.4f}")
    print(f"情感分类RNN测试准确率: {acc2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    demo()
