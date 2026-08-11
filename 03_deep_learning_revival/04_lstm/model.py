"""LSTM长短期记忆网络实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """单个LSTM单元。
    
    公式：
        f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)  # 遗忘门
        i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)  # 输入门
        o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)  # 输出门
        C_tilda = tanh(W_xc * x_t + W_hc * h_{t-1} + b_c) # 候选细胞
        C_t = f_t * C_{t-1} + i_t * C_tilda                # 细胞状态
        h_t = o_t * tanh(C_t)                              # 隐藏状态
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 权重矩阵
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.U_c = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        
        # 偏置
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, h_prev, c_prev):
        """前向传播。
        
        Args:
            x: 输入，shape (batch, input_size)
            h_prev: 上一时刻隐藏状态
            c_prev: 上一时刻细胞状态
            
        Returns:
            h_t, c_t
        """
        # 遗忘门
        f = torch.sigmoid(x @ self.W_f.T + h_prev @ self.U_f.T + self.b_f)
        # 输入门
        i = torch.sigmoid(x @ self.W_i.T + h_prev @ self.U_i.T + self.b_i)
        # 输出门
        o = torch.sigmoid(x @ self.W_o.T + h_prev @ self.U_o.T + self.b_o)
        # 候选细胞
        c_tilda = torch.tanh(x @ self.W_c.T + h_prev @ self.U_c.T + self.b_c)
        
        # 细胞状态和隐藏状态
        c_t = f * c_prev + i * c_tilda
        h_t = o * torch.tanh(c_t)
        
        return h_t, c_t


class LSTM(nn.Module):
    """多层LSTM网络。"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """前向传播。
        
        Args:
            x: 输入序列，shape (seq_len, batch, input_size)
            
        Returns:
            输出，shape (batch, output_size)
        """
        batch_size = x.size(1) if x.dim() > 2 else 1
        
        # 初始化隐藏状态和细胞状态
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        
        # 逐时间步处理
        for t in range(x.size(0)):
            x_t = x[t]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]
        
        return self.output_layer(h[-1])