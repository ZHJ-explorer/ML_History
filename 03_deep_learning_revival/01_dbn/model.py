"""深度信念网络（DBN）实现。"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RBM(nn.Module):
    """受限玻尔兹曼机。"""
    def __init__(self, n_visible, n_hidden):
        """初始化受限玻尔兹曼机。

        Args:
            n_visible: 可见层神经元数量。
            n_hidden: 隐藏层神经元数量。
        """
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.hb = nn.Parameter(torch.zeros(n_hidden))
        self.vb = nn.Parameter(torch.zeros(n_visible))

    def forward(self, v):
        """执行RBM前向传播。

        Args:
            v: 可见层输入，形状为(batch_size, n_visible)。

        Returns:
            v_prob: 可见层重建概率，形状为(batch_size, n_visible)。
            h_prob: 隐藏层激活概率，形状为(batch_size, n_hidden)。
        """
        h_prob = torch.sigmoid(torch.matmul(v, self.w) + self.hb)
        h = torch.bernoulli(h_prob)
        v_prob = torch.sigmoid(torch.matmul(h, self.w.t()) + self.vb)
        return v_prob, h_prob

    def train_step(self, v, lr=0.01):
        """执行一步CD-1训练。

        Args:
            v: 可见层输入，形状为(batch_size, n_visible)。
            lr: 学习率，默认为0.01。

        Returns:
            self: 训练后的RBM实例。
        """
        # 正相
        h_prob_pos = torch.sigmoid(torch.matmul(v, self.w) + self.hb)
        h_pos = torch.bernoulli(h_prob_pos)

        # 负相（k=1 CD）
        v_neg, h_prob_neg = self(v)
        h_neg = torch.bernoulli(h_prob_neg)

        # 更新
        self.w.data += lr * (torch.matmul(v.t(), h_prob_pos) - torch.matmul(v_neg.t(), h_neg)) / v.size(0)
        self.hb.data += lr * (h_prob_pos.mean(0) - h_neg.mean(0))
        self.vb.data += lr * (v_neg.mean(0) - v_neg.mean(0))
        return self


class DBN(nn.Module):
    """深度信念网络。"""
    def __init__(self, layer_sizes):
        """初始化深度信念网络。

        Args:
            layer_sizes: 各层神经元数量列表，如[784, 256, 128, 10]。
        """
        super().__init__()
        self.rbms = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i + 1]))

    def pretrain(self, X, epochs=10, lr=0.01):
        """逐层预训练DBN。

        Args:
            X: 训练数据，形状为(n_samples, n_visible)。
            epochs: 每层预训练迭代次数，默认为10。
            lr: 学习率，默认为0.01。

        Returns:
            self: 训练后的DBN实例。
        """
        current = X
        for rbm in self.rbms:
            for _ in range(epochs):
                rbm.train_step(current, lr)
            _, h_prob = rbm(current)
            current = torch.bernoulli(h_prob)
        return self

    def forward(self, x):
        """执行DBN前向传播。

        Args:
            x: 输入数据，形状为(batch_size, n_visible)。

        Returns:
            x: 最终隐藏层激活，形状为(batch_size, n_hidden)。
        """
        for rbm in self.rbms:
            _, h_prob = rbm(x)
            x = torch.bernoulli(h_prob)
        return x