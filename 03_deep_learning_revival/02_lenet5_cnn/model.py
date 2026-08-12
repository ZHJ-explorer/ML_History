"""LeNet-5 卷积神经网络实现。

LeNet-5 由 Yann LeCun 于1998年提出，是最早成功的卷积神经网络之一，
用于手写数字识别（MNIST数据集）。

网络结构（公式与代码一一对应）：
    输入 28x28 -> Conv1(6@5x5,pad=2) -> ReLU -> AvgPool(2x2)
             -> Conv2(16@5x5,pad=2) -> ReLU -> AvgPool(2x2)
             -> Flatten -> FC1(120) -> ReLU
             -> FC2(84) -> ReLU -> FC3(10)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ManualConv2D(nn.Module):
    """手动实现二维卷积层（向量化版本）。

    公式：
        output[b, o, i, j] = sum_{c,m,n} input[b, c, i+m, j+n] * weight[o, c, m, n] + bias[o]

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        kernel_size: 卷积核大小。
        stride: 步长。
        padding: 填充。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """前向传播（向量化卷积）。

        Args:
            x: 输入张量，shape (B, C, H, W)。

        Returns:
            卷积输出，shape (B, out_channels, H', W')。
        """
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # 填充
        if p > 0:
            x = nn.functional.pad(x, (p, p, p, p))
            H, W = x.shape[2], x.shape[3]

        # 计算输出尺寸
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        # 使用im2col方法将输入转换为2D矩阵
        # 输出形状: (B * out_h * out_w, C * k * k)
        cols = []
        for b in range(B):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * s
                    h_end = h_start + k
                    w_start = j * s
                    w_end = w_start + k
                    col = x[b, :, h_start:h_end, w_start:w_end].reshape(C * k * k)
                    cols.append(col)
        cols = torch.stack(cols).reshape(B * out_h * out_w, C * k * k)

        # 权重展平
        weight_flat = self.weight.reshape(self.out_channels, -1)

        # 矩阵乘法实现卷积
        output = torch.matmul(cols, weight_flat.t())  # (B * out_h * out_w, out_channels)
        output = output + self.bias.unsqueeze(0)  # 添加偏置

        # 重塑回4D
        output = output.reshape(B, out_h, out_w, self.out_channels)
        output = output.permute(0, 3, 1, 2)  # (B, out_channels, out_h, out_w)

        return output


class ManualAvgPool2D(nn.Module):
    """手动实现平均池化层。

    公式：
        output[b, c, i, j] = mean(input[b, c, i*stride:i*stride+k, j*stride:j*stride+k])

    Args:
        kernel_size: 池化窗口大小。
        stride: 步长。
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入张量，shape (B, C, H, W)。

        Returns:
            池化输出，shape (B, C, H', W')。
        """
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        # 使用im2col方法提取池化窗口
        cols = []
        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s
                        h_end = h_start + k
                        w_start = j * s
                        w_end = w_start + k
                        col = x[b, c, h_start:h_end, w_start:w_end].reshape(k * k)
                        cols.append(col)
        cols = torch.stack(cols).reshape(B * C * out_h * out_w, k * k)

        # 计算均值
        output = cols.mean(dim=1)  # (B * C * out_h * out_w)

        # 重塑回4D
        output = output.reshape(B, C, out_h, out_w)

        return output


class LeNet5(nn.Module):
    """LeNet-5 卷积神经网络。

    网络结构：
        Input(1x28x28) -> Conv1(6x5x5,pad=2) -> ReLU -> AvgPool(2x2)
                       -> Conv2(16x5x5,pad=2) -> ReLU -> AvgPool(2x2)
                       -> Flatten -> FC1(120) -> ReLU
                       -> FC2(84) -> ReLU
                       -> FC3(10)

    对应论文：LeCun, Y., et al. (1998). Gradient-based learning applied
              to document recognition. Proceedings of the IEEE.
    """

    def __init__(self):
        super().__init__()
        # 手动卷积层（使用padding保持尺寸）
        self.conv1 = ManualConv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = ManualConv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2)

        # 手动池化层
        self.pool = ManualAvgPool2D(kernel_size=2, stride=2)

        # 全连接层（使用标准nn.Linear）
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # 池化后14x14 -> 7x7
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入张量，shape (B, 1, 28, 28)。

        Returns:
            分类logits，shape (B, 10)。
        """
        # Conv1 -> ReLU -> AvgPool
        x = self.conv1(x)      # (B, 6, 28, 28)
        x = torch.relu(x)      # (B, 6, 28, 28)
        x = self.pool(x)       # (B, 6, 14, 14)

        # Conv2 -> ReLU -> AvgPool
        x = self.conv2(x)      # (B, 16, 14, 14)
        x = torch.relu(x)      # (B, 16, 14, 14)
        x = self.pool(x)       # (B, 16, 7, 7)

        # Flatten
        x = x.view(x.size(0), -1)  # (B, 784)

        # FC layers
        x = torch.relu(self.fc1(x))  # (B, 120)
        x = torch.relu(self.fc2(x))  # (B, 84)
        x = self.fc3(x)              # (B, 10)

        return x
