"""扩散模型实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码。"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    """扩散模型残差块。"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.scale_emb = SinusoidalPositionEmbeddings(hidden_size)
    
    def forward(self, x, t):
        """前向传播。
        
        Args:
            x: 输入噪声
            t: 时间步
        """
        emb = self.scale_emb(t.float())
        h = self.net(x) * (1 + emb) + emb
        return x + h


class SimpleDiffusionModel(nn.Module):
    """简化版扩散模型。"""
    
    def __init__(self, input_dim=784, hidden_size=256, num_steps=100):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            ResBlock(hidden_size),
            nn.Linear(hidden_size, input_dim),
        )
        
        # 噪声调度
        self.register_buffer('alphas', self._cosine_noise_schedule(num_steps))
    
    def _cosine_noise_schedule(self, num_steps):
        """余弦噪声调度。"""
        steps = torch.arange(num_steps, dtype=torch.float32)
        alphas = torch.cos((steps / num_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas = alphas / alphas.max()
        return alphas
    
    def forward(self, x, t):
        """前向传播。
        
        Args:
            x: 噪声图像
            t: 时间步
        """
        noise = torch.randn_like(x)
        sqrt_alphas = torch.sqrt(self.alphas[t])
        sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas[t])
        
        x_noisy = sqrt_alphas[:, None] * x + sqrt_one_minus_alphas[:, None] * noise
        predicted_noise = self.net(x_noisy)
        
        return predicted_noise, noise
    
    def sample(self, shape, device='cpu'):
        """从扩散模型生成样本。"""
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((shape[0],), t, device=device)
            predicted_noise = self.net(x)
            
            alpha = self.alphas[t]
            alpha_prev = self.alphas[t - 1] if t > 0 else torch.tensor([1.0], device=device)
            
            sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            
            x = x.clone()
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            x = x + sigma * torch.randn_like(x)
        
        return x