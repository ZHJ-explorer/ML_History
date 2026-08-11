"""Transformer实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制，允许模型同时关注不同位置的不同表示子空间。"""

    def __init__(self, d_model, num_heads):
        """初始化多头注意力层。

        Args:
            d_model: 模型维度。
            num_heads: 注意力头数。
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        """执行多头注意力计算。

        Args:
            q: 查询张量，形状为(batch_size, seq_len, d_model)。
            k: 键张量，形状为(batch_size, seq_len, d_model)。
            v: 值张量，形状为(batch_size, seq_len, d_model)。
            mask: 可选的掩码张量。

        Returns:
            注意力输出，形状为(batch_size, seq_len, d_model)。
        """
        batch_size = q.size(0)
        
        # 线性变换
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        # 重塑为多头
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        
        # 加权求和
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(out)


class PositionWiseFeedForward(nn.Module):
    """位置前馈网络，对每个位置独立应用相同的线性变换。"""

    def __init__(self, d_model, d_ff):
        """初始化位置前馈网络。

        Args:
            d_model: 模型维度。
            d_ff: 前馈网络隐藏层维度。
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Transformer编码器层，包含多头自注意力和前馈网络。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """初始化编码器层。

        Args:
            d_model: 模型维度。
            num_heads: 注意力头数。
            d_ff: 前馈网络隐藏层维度。
            dropout: Dropout比率，默认为0.1。
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 多头注意力
        attention_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class Transformer(nn.Module):
    """Transformer编码器模型，用于序列建模任务。"""

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=512):
        """初始化Transformer模型。

        Args:
            vocab_size: 词表大小。
            d_model: 模型维度，默认为512。
            num_heads: 注意力头数，默认为8。
            num_layers: 编码器层数，默认为6。
            d_ff: 前馈网络隐藏层维度，默认为2048。
            max_seq_len: 最大序列长度，默认为512。
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """执行前向传播。

        Args:
            x: 输入 token ID 张量，形状为(batch_size, seq_len)。

        Returns:
            输出 logits，形状为(batch_size, seq_len, vocab_size)。
        """
        batch_size, seq_len = x.size()
        
        # 嵌入
        x = self.embedding(x)
        
        # 位置编码
        x = x + self.position_encoding[:, :seq_len, :]
        
        # 编码器层
        for layer in self.layers:
            x = layer(x)
        
        return self.output_layer(x)