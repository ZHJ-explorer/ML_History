"""深度信念网络（DBN）实现。"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RBM(nn.Module):
    """受限玻尔兹曼机。"""
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.hb = nn.Parameter(torch.zeros(n_hidden))
        self.vb = nn.Parameter(torch.zeros(n_visible))

    def forward(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.w) + self.hb)
        h = torch.bernoulli(h_prob)
        v_prob = torch.sigmoid(torch.matmul(h, self.w.t()) + self.vb)
        return v_prob, h_prob

    def train_step(self, v, lr=0.01):
        # 正相
        h_prob_pos = torch.sigmoid(torch.matmul(v, self.w) + self.hb)
        h_pos = torch.bernoulli(h_prob_pos)

        # 负相（k=1 CD）
        v_neg, h_prob_neg = self(v)
        h_neg = torch.bernoulli(h_prob_neg)

        # 更新
        self.w.data += lr * (torch.matmul(v.t(), h_prob_pos) - torch.matmul(v_neg.t(), h_neg)) / v.size(0)
        self.hb.data += lr * (h_prob_pos.mean(0) - h_neg.mean(0))
        self.vb.data += lr * (v_prob.mean(0) - v_neg.mean(0))
        return self


class DBN(nn.Module):
    """深度信念网络。"""
    def __init__(self, layer_sizes):
        super().__init__()
        self.rbms = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i + 1]))

    def pretrain(self, X, epochs=10, lr=0.01):
        """逐层预训练。"""
        current = X
        for rbm in self.rbms:
            for _ in range(epochs):
                rbm.train_step(current, lr)
            _, h_prob = rbm(current)
            current = torch.bernoulli(h_prob)
        return self

    def forward(self, x):
        for rbm in self.rbms:
            _, h_prob = rbm(x)
            x = torch.bernoulli(h_prob)
        return x