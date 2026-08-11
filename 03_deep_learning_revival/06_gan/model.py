"""GAN生成对抗网络实现。"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Generator(nn.Module):
    """生成器：将随机噪声映射到数据空间。"""
    
    def __init__(self, latent_dim, img_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        """前向传播。
        
        Args:
            z: 随机噪声，shape (batch, latent_dim)
            
        Returns:
            生成样本，shape (batch, img_dim)
        """
        return self.model(z)


class Discriminator(nn.Module):
    """判别器：区分真实样本和生成样本。"""
    
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播。
        
        Args:
            x: 输入样本，shape (batch, img_dim)
            
        Returns:
            判定概率，shape (batch, 1)
        """
        return self.model(x)


class GAN(nn.Module):
    """GAN模型。"""
    
    def __init__(self, latent_dim, img_dim):
        super().__init__()
        self.generator = Generator(latent_dim, img_dim)
        self.discriminator = Discriminator(img_dim)