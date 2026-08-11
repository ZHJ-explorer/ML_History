"""循环神经网络（RNN）实现。

基础RNN单元，用于序列建模任务。
关键模块（RNN单元）使用手写实现，展示内部计算细节。

公式与代码对应：
    h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    y_t = W_hy * h_t + b_y
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ManualRNNCell(nn.Module):
    """手动实现单个RNN单元。

    公式：
        h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)

    其中：
        - x_t: 当前时刻输入
        - h_{t-1}: 上一时刻隐藏状态
        - h_t: 当前时刻隐藏状态
        - W_xh: 输入到隐藏的权重矩阵
        - W_hh: 隐藏到隐藏的权重矩阵
        - b_h: 隐藏层偏置
    """

    def __init__(self, input_size, hidden_size):
        """初始化RNN单元。

        Args:
            input_size: 输入维度。
            hidden_size: 隐藏状态维度。
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 权重矩阵初始化
        self.w_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.w_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        """单步RNN计算。

        Args:
            x_t: 当前时刻输入，shape (batch, input_size)。
            h_prev: 上一时刻隐藏状态，shape (batch, hidden_size)。

        Returns:
            h_t: 当前时刻隐藏状态，shape (batch, hidden_size)。
        """
        # 公式：h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        h_t = torch.tanh(
            torch.matmul(x_t, self.w_xh) +
            torch.matmul(h_prev, self.w_hh) +
            self.b_h
        )
        return h_t


class ManualRNN(nn.Module):
    """手动实现完整RNN（多层RNN单元）。

    公式：
        h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)

    Args:
        input_size: 输入维度。
        hidden_size: 隐藏状态维度。
        output_size: 输出维度。
        num_layers: RNN层数。
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化RNN。

        Args:
            input_size: 输入维度。
            hidden_size: 隐藏状态维度。
            output_size: 输出维度。
            num_layers: RNN层数（默认1层）。
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 创建多层RNN单元
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(ManualRNNCell(layer_input_size, hidden_size))

        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        """前向传播。

        Args:
            x: 输入序列，shape (seq_len, batch, input_size)。
            hidden: 初始隐藏状态，shape (num_layers, batch, hidden_size)。
                   如果为None，则初始化为全零。

        Returns:
            output: 输出序列，shape (seq_len, batch, output_size)。
            hidden: 最终隐藏状态。
        """
        seq_len, batch_size, _ = x.shape

        # 初始化隐藏状态
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        outputs = []

        # 遍历时间步
        for t in range(seq_len):
            x_t = x[t]  # (batch, input_size)
            new_hidden = []

            # 遍历每一层
            for layer_idx, rnn_cell in enumerate(self.layers):
                h_prev = hidden[layer_idx]
                h_t = rnn_cell(x_t, h_prev)
                new_hidden.append(h_t)
                x_t = h_t  # 下一层的输入是当前层的输出

            hidden = torch.stack(new_hidden, dim=0)
            output_t = self.output_layer(h_t)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=0)  # (seq_len, batch, output_size)
        return output, hidden


class TextRNN(nn.Module):
    """基于RNN的文本分类模型。

    用于演示RNN在序列数据上的应用。

    网络结构：
        Input Embedding -> RNN -> Last Hidden -> FC -> Output
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, num_layers=1, padding_idx=0):
        """初始化TextRNN。

        Args:
            vocab_size: 词汇表大小。
            embed_dim: 词嵌入维度。
            hidden_size: RNN隐藏状态维度。
            output_size: 输出类别数。
            num_layers: RNN层数。
            padding_idx: 填充索引。
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = ManualRNN(embed_dim, hidden_size, output_size, num_layers=num_layers)
        self.hidden_size = hidden_size

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入序列，shape (batch, seq_len)。

        Returns:
            logits: 分类logits，shape (batch, output_size)。
        """
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch, embed_dim)

        # RNN
        output, hidden = self.rnn(x)

        # 使用最后一层的最后一个时间步
        last_hidden = hidden[-1]  # (batch, hidden_size)
        logits = self.rnn.output_layer(last_hidden)

        return logits
