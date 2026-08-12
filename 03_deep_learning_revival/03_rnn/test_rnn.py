"""RNN 循环神经网络测试。"""
import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ManualRNNCell, ManualRNN, TextRNN


class TestManualRNNCell:
    """手动RNN单元测试类。"""

    def test_init(self):
        """测试RNN单元初始化参数形状。"""
        cell = ManualRNNCell(input_size=10, hidden_size=20)
        assert cell.w_xh.shape == (10, 20)
        assert cell.w_hh.shape == (20, 20)
        assert cell.b_h.shape == (20,)

    def test_forward_single_step(self):
        """测试单步前向传播。"""
        cell = ManualRNNCell(input_size=8, hidden_size=16)
        x_t = torch.rand(4, 8)  # batch=4, input=8
        h_prev = torch.zeros(4, 16)
        h_t = cell(x_t, h_prev)
        assert h_t.shape == (4, 16)

    def test_forward_output_range(self):
        """测试输出在tanh范围内。"""
        cell = ManualRNNCell(input_size=4, hidden_size=8)
        x_t = torch.ones(2, 4) * 100  # 大输入
        h_prev = torch.zeros(2, 8)
        h_t = cell(x_t, h_prev)
        assert torch.all(h_t >= -1) and torch.all(h_t <= 1)

    def test_different_hidden_states(self):
        """测试不同隐藏状态产生不同输出。"""
        cell = ManualRNNCell(input_size=4, hidden_size=8)
        x_t = torch.rand(2, 4)
        h1 = torch.zeros(2, 8)
        h2 = torch.ones(2, 8)
        out1 = cell(x_t, h1)
        out2 = cell(x_t, h2)
        assert not torch.allclose(out1, out2)


class TestManualRNN:
    """手动RNN测试类。"""

    def test_init_single_layer(self):
        """测试单层RNN初始化。"""
        rnn = ManualRNN(input_size=10, hidden_size=20, output_size=5, num_layers=1)
        assert rnn.num_layers == 1
        assert len(rnn.layers) == 1
        assert isinstance(rnn.output_layer, nn.Linear)
        assert rnn.output_layer.in_features == 20
        assert rnn.output_layer.out_features == 5

    def test_init_multi_layer(self):
        """测试多层RNN初始化。"""
        rnn = ManualRNN(input_size=10, hidden_size=32, output_size=5, num_layers=3)
        assert rnn.num_layers == 3
        assert len(rnn.layers) == 3

    def test_forward_with_hidden(self):
        """测试带隐藏状态的前向传播。"""
        rnn = ManualRNN(input_size=8, hidden_size=16, output_size=4, num_layers=1)
        seq_len, batch, input_size = 10, 4, 8
        x = torch.rand(seq_len, batch, input_size)
        hidden = torch.zeros(1, batch, 16)
        output, hidden_out = rnn(x, hidden)
        assert output.shape == (seq_len, batch, 4)
        assert hidden_out.shape == (1, batch, 16)

    def test_forward_without_hidden(self):
        """测试不带隐藏状态的前向传播（自动初始化）。"""
        rnn = ManualRNN(input_size=8, hidden_size=16, output_size=4, num_layers=2)
        x = torch.rand(5, 2, 8)
        output, hidden = rnn(x)
        assert output.shape == (5, 2, 4)
        assert hidden.shape == (2, 2, 16)

    def test_forward_deterministic(self):
        """测试相同输入产生相同输出。"""
        rnn = ManualRNN(input_size=4, hidden_size=8, output_size=2, num_layers=1)
        x = torch.rand(3, 2, 4)
        out1, _ = rnn(x)
        out2, _ = rnn(x)
        assert torch.allclose(out1, out2)


class TestTextRNN:
    """文本RNN测试类。"""

    def test_init(self):
        """测试TextRNN初始化。"""
        model = TextRNN(
            vocab_size=100,
            embed_dim=32,
            hidden_size=64,
            output_size=2,
            num_layers=1
        )
        assert model.embedding.num_embeddings == 100
        assert model.embedding.embedding_dim == 32
        assert model.hidden_size == 64

    def test_forward_single_sequence(self):
        """测试单序列前向传播。"""
        model = TextRNN(vocab_size=50, embed_dim=16, hidden_size=32, output_size=2)
        x = torch.randint(0, 50, (1, 10))  # batch=1, seq_len=10
        logits = model(x)
        assert logits.shape == (1, 2)

    def test_forward_batch(self):
        """测试批量前向传播。"""
        model = TextRNN(vocab_size=100, embed_dim=32, hidden_size=64, output_size=3)
        x = torch.randint(0, 100, (8, 20))  # batch=8, seq_len=20
        logits = model(x)
        assert logits.shape == (8, 3)

    def test_forward_with_padding(self):
        """测试带padding索引的前向传播。"""
        model = TextRNN(vocab_size=50, embed_dim=16, hidden_size=32, output_size=2, padding_idx=0)
        x = torch.tensor([[0, 1, 2, 3, 0]])  # 含padding
        logits = model(x)
        assert logits.shape == (1, 2)
        assert not torch.isnan(logits).any()

    def test_loss_computation(self):
        """测试可以计算交叉熵损失。"""
        model = TextRNN(vocab_size=50, embed_dim=16, hidden_size=32, output_size=2)
        criterion = nn.CrossEntropyLoss()
        x = torch.randint(0, 50, (4, 10))
        y = torch.randint(0, 2, (4,))
        logits = model(x)
        loss = criterion(logits, y)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_training_step(self):
        """测试一个完整的训练步骤。"""
        model = TextRNN(vocab_size=50, embed_dim=16, hidden_size=32, output_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        x = torch.randint(0, 50, (8, 10))
        y = torch.randint(0, 2, (8,))

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_multi_layer(self):
        """测试多层TextRNN。"""
        model = TextRNN(
            vocab_size=50,
            embed_dim=16,
            hidden_size=32,
            output_size=2,
            num_layers=2
        )
        x = torch.randint(0, 50, (4, 10))
        logits = model(x)
        assert logits.shape == (4, 2)

    def test_parameter_count(self):
        """测试模型参数数量合理。"""
        model = TextRNN(vocab_size=100, embed_dim=32, hidden_size=64, output_size=2)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
