"""LeNet-5 卷积神经网络测试。"""
import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import LeNet5, ManualConv2D, ManualAvgPool2D


class TestManualConv2D:
    """手动卷积层测试类。"""

    def test_init_shapes(self):
        """测试卷积层初始化参数形状。"""
        conv = ManualConv2D(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        assert conv.weight.shape == (16, 3, 5, 5)
        assert conv.bias.shape == (16,)

    def test_forward_shape_same_padding(self):
        """测试带padding时输出尺寸保持不变。"""
        conv = ManualConv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        x = torch.rand(2, 1, 28, 28)
        out = conv(x)
        assert out.shape == (2, 6, 28, 28)

    def test_forward_shape_no_padding(self):
        """测试无padding时输出尺寸缩小。"""
        conv = ManualConv2D(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        x = torch.rand(1, 1, 28, 28)
        out = conv(x)
        assert out.shape == (1, 16, 24, 24)

    def test_forward_strides(self):
        """测试不同步长输出尺寸。"""
        conv = ManualConv2D(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=0)
        x = torch.rand(1, 1, 10, 10)
        out = conv(x)
        assert out.shape == (1, 4, 4, 4)


class TestManualAvgPool2D:
    """手动平均池化层测试类。"""

    def test_init(self):
        """测试池化层初始化。"""
        pool = ManualAvgPool2D(kernel_size=2, stride=2)
        assert pool.kernel_size == 2
        assert pool.stride == 2

    def test_forward_shape(self):
        """测试池化输出尺寸减半。"""
        pool = ManualAvgPool2D(kernel_size=2, stride=2)
        x = torch.rand(2, 6, 14, 14)
        out = pool(x)
        assert out.shape == (2, 6, 7, 7)

    def test_forward_values(self):
        """测试池化值是窗口的均值。"""
        pool = ManualAvgPool2D(kernel_size=2, stride=2)
        x = torch.ones(1, 1, 4, 4)
        out = pool(x)
        assert out.shape == (1, 1, 2, 2)
        assert torch.allclose(out, torch.ones(1, 1, 2, 2))


class TestLeNet5:
    """LeNet-5模型测试类。"""

    def test_init(self):
        """测试LeNet-5初始化。"""
        model = LeNet5()
        assert isinstance(model.conv1, ManualConv2D)
        assert isinstance(model.conv2, ManualConv2D)
        assert isinstance(model.pool, ManualAvgPool2D)
        assert model.fc1.in_features == 784
        assert model.fc1.out_features == 120
        assert model.fc2.out_features == 84
        assert model.fc3.out_features == 10

    def test_forward_shape(self):
        """测试前向传播输出形状。"""
        model = LeNet5()
        x = torch.rand(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_single_sample(self):
        """测试单样本前向传播。"""
        model = LeNet5()
        x = torch.rand(1, 1, 28, 28)
        out = model(x)
        assert out.shape == (1, 10)

    def test_parameter_count(self):
        """测试模型参数数量合理。"""
        model = LeNet5()
        total_params = sum(p.numel() for p in model.parameters())
        # LeNet-5大约有10万参数（手动卷积层参数量较大）
        assert total_params > 50000
        assert total_params < 200000

    def test_training_mode(self):
        """测试训练模式下可以正常前向传播。"""
        model = LeNet5()
        model.train()
        x = torch.rand(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    def test_evaluation_mode(self):
        """测试评估模式下可以正常前向传播。"""
        model = LeNet5()
        model.eval()
        x = torch.rand(8, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 10)

    def test_loss_computation(self):
        """测试可以计算交叉熵损失。"""
        model = LeNet5()
        criterion = nn.CrossEntropyLoss()
        x = torch.rand(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        out = model(x)
        loss = criterion(out, y)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_backward_pass(self):
        """测试反向传播可以正常执行。"""
        model = LeNet5()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.rand(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        optimizer.step()
        # 检查参数是否有更新
        has_updates = False
        for param in model.parameters():
            if param.grad is not None:
                has_updates = True
                break
        assert has_updates

    def test_convergence_on_simple_task(self):
        """测试模型在简单任务上可以学习。"""
        model = LeNet5()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 创建一个简单的分类任务：根据输入均值分类
        torch.manual_seed(42)
        X = torch.rand(100, 1, 28, 28)
        y = (X.mean(dim=(1, 2, 3)) > 0.5).long()

        # 训练几个epoch
        for epoch in range(5):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # 检查是否在学习
        with torch.no_grad():
            out = model(X)
            preds = out.argmax(1)
            acc = (preds == y).float().mean().item()
            # 应该在随机准确率(0.1)以上
            assert acc > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
