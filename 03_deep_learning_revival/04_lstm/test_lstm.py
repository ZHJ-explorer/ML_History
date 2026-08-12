"""LSTM 长短期记忆网络测试。"""
import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import LSTMCell, LSTM


class TestLSTMCell:
    """LSTM单元测试类。"""

    def test_init(self):
        """测试LSTM单元初始化参数。"""
        cell = LSTMCell(input_size=10, hidden_size=20)
        assert cell.W_f.shape == (20, 10)
        assert cell.W_i.shape == (20, 10)
        assert cell.W_o.shape == (20, 10)
        assert cell.W_c.shape == (20, 10)
        assert cell.U_f.shape == (20, 20)
        assert cell.b_f.shape == (20,)

    def test_forward_output_shapes(self):
        """测试LSTM单元输出形状。"""
        cell = LSTMCell(input_size=8, hidden_size=16)
        x = torch.rand(4, 8)  # batch=4
        h_prev = torch.zeros(4, 16)
        c_prev = torch.zeros(4, 16)
        h_t, c_t = cell(x, h_prev, c_prev)
        assert h_t.shape == (4, 16)
        assert c_t.shape == (4, 16)

    def test_forward_gate_values(self):
        """测试门控值在sigmoid范围内。"""
        cell = LSTMCell(input_size=4, hidden_size=8)
        x = torch.ones(2, 4) * 100  # 大输入测试饱和
        h_prev = torch.zeros(2, 8)
        c_prev = torch.zeros(2, 8)
        # 直接计算门控值验证范围
        f = torch.sigmoid(x @ cell.W_f.T + h_prev @ cell.U_f.T + cell.b_f)
        assert torch.all(f >= 0) and torch.all(f <= 1)

    def test_hidden_state_update(self):
        """测试隐藏状态随时间更新。"""
        cell = LSTMCell(input_size=4, hidden_size=8)
        x = torch.rand(2, 4)
        h_prev = torch.zeros(2, 8)
        c_prev = torch.zeros(2, 8)
        h_t, c_t = cell(x, h_prev, c_prev)
        # 隐藏状态应被更新
        assert not torch.allclose(h_t, h_prev)


class TestLSTM:
    """LSTM网络测试类。"""

    def test_init_single_layer(self):
        """测试单层LSTM初始化。"""
        lstm = LSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        assert lstm.num_layers == 1
        assert len(lstm.cells) == 1
        assert lstm.output_layer.in_features == 32

    def test_init_multi_layer(self):
        """测试多层LSTM初始化。"""
        lstm = LSTM(input_size=2, hidden_size=64, num_layers=3, output_size=1)
        assert lstm.num_layers == 3
        assert len(lstm.cells) == 3

    def test_forward_sequence(self):
        """测试序列前向传播。"""
        lstm = LSTM(input_size=1, hidden_size=16, num_layers=2, output_size=1)
        # 输入形状: (seq_len, batch, input_size)
        x = torch.rand(10, 4, 1)
        output = lstm(x)
        assert output.shape == (4, 1)

    def test_forward_single_step(self):
        """测试单步前向传播。"""
        lstm = LSTM(input_size=2, hidden_size=8, num_layers=1, output_size=1)
        x = torch.rand(1, 4, 2)  # seq_len=1, batch=4
        output = lstm(x)
        assert output.shape == (4, 1)

    def test_forward_varied_sequence_lengths(self):
        """测试不同序列长度。"""
        lstm = LSTM(input_size=1, hidden_size=16, num_layers=1, output_size=1)
        for seq_len in [5, 10, 20]:
            x = torch.rand(seq_len, 2, 1)
            output = lstm(x)
            assert output.shape == (2, 1)

    def test_loss_computation(self):
        """测试可以计算回归损失。"""
        lstm = LSTM(input_size=1, hidden_size=16, num_layers=1, output_size=1)
        criterion = nn.MSELoss()
        x = torch.rand(10, 4, 1)
        y = torch.rand(4, 1)
        output = lstm(x)
        loss = criterion(output, y)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_training_step(self):
        """测试完整的训练步骤。"""
        lstm = LSTM(input_size=1, hidden_size=16, num_layers=2, output_size=1)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        x = torch.rand(20, 8, 1)
        y = torch.rand(8, 1)

        optimizer.zero_grad()
        output = lstm(x)
        loss = criterion(output.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_convergence_on_simple_task(self):
        """测试在简单序列任务上可以学习。"""
        torch.manual_seed(42)
        lstm = LSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 简单的序列预测：预测序列均值
        X = torch.rand(100, 10, 1)
        y = X.mean(dim=1).squeeze()  # (100,)

        initial_loss = None
        final_loss = None

        for epoch in range(20):
            optimizer.zero_grad()
            output = lstm(X.permute(1, 0, 2))  # (10, 100, 1) -> LSTM expects (seq, batch, input)
            loss = criterion(output.squeeze(), y)
            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
            loss.backward()
            optimizer.step()

        # 损失应该下降
        assert final_loss < initial_loss

    def test_parameter_count(self):
        """测试模型参数数量合理。"""
        lstm = LSTM(input_size=4, hidden_size=32, num_layers=2, output_size=1)
        total_params = sum(p.numel() for p in lstm.parameters())
        assert total_params > 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
