"""ViT（Vision Transformer）实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """图像块嵌入。"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """初始化图像块嵌入层。

        Args:
            img_size: 输入图像尺寸（方形），默认为224。
            patch_size: 块大小，默认为16。
            in_channels: 输入通道数，默认为3（RGB）。
            embed_dim: 嵌入维度，默认为768。
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        """前向传播。
        
        Args:
            x: 输入图像，shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力。"""
    
    def __init__(self, embed_dim=768, num_heads=12):
        """初始化多头自注意力层。

        Args:
            embed_dim: 嵌入维度，默认为768。
            num_heads: 注意力头数，默认为12。
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        """前向传播（注意力计算）。

        Args:
            x: 输入张量，形状为(batch_size, n_patches+1, embed_dim)。

        Returns:
            注意力输出，形状为(batch_size, n_patches+1, embed_dim)。
        """
        B, n, C = x.shape
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, n, C)
        x = self.proj(x)
        
        return x


class ViT(nn.Module):
    """Vision Transformer。"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, num_heads=12, num_layers=12):
        """初始化ViT模型。

        Args:
            img_size: 输入图像尺寸（方形），默认为224。
            patch_size: 块大小，默认为16。
            in_channels: 输入通道数，默认为3（RGB）。
            num_classes: 分类类别数，默认为1000。
            embed_dim: 嵌入维度，默认为768。
            num_heads: 注意力头数，默认为12。
            num_layers: Transformer层数，默认为12。
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        
        # Transformer编码器层
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': MultiHeadSelfAttention(embed_dim, num_heads),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.GELU(),
                    nn.Linear(4 * embed_dim, embed_dim),
                )
            })
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        """前向传播。

        Args:
            x: 输入图像，形状为(batch_size, in_channels, img_size, img_size)。

        Returns:
            分类logits，形状为(batch_size, num_classes)。
        """
        B = x.shape[0]
        
        # 图像块嵌入
        x = self.patch_embed(x)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 位置编码
        x = x + self.pos_embed
        
        # Transformer编码器层
        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['mlp'](layer['norm2'](x))
        
        # 分类头
        x = self.norm(x)
        cls_output = self.head(x[:, 0])
        
        return cls_output