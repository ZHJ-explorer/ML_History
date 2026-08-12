"""ViT (Vision Transformer) pytest测试。"""
import torch
import pytest
from model import ViT, PatchEmbedding, MultiHeadSelfAttention


class TestViT:
    """ViT测试类。"""
    
    @pytest.fixture
    def small_vit(self):
        """创建小型ViT用于快速测试。"""
        return ViT(
            img_size=32,
            patch_size=8,
            in_channels=3,
            num_classes=10,
            embed_dim=64,
            num_heads=4,
            num_layers=2
        )
    
    def test_forward_pass(self, small_vit):
        """测试前向传播。"""
        x = torch.randn(2, 3, 32, 32)
        output = small_vit(x)
        
        assert output.shape == (2, 10)
        assert torch.all(torch.isfinite(output))
    
    def test_different_batch_sizes(self, small_vit):
        """测试不同批次大小。"""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = small_vit(x)
            assert output.shape == (batch_size, 10)
    
    def test_patch_embedding(self):
        """测试图像块嵌入。"""
        patch_embed = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
        x = torch.randn(2, 3, 32, 32)
        
        output = patch_embed(x)
        
        # 输出形状: (B, n_patches, embed_dim)
        n_patches = (32 // 8) ** 2
        assert output.shape == (2, n_patches, 64)
    
    def test_multihead_attention(self):
        """测试多头自注意力。"""
        attn = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)  # B, n_patches+1, embed_dim
        
        output = attn(x)
        assert output.shape == x.shape
    
    def test_vit_with_cls_token(self, small_vit):
        """测试CLS token处理。"""
        x = torch.randn(1, 3, 32, 32)
        output = small_vit(x)
        
        # 输出维度应为num_classes
        assert output.shape == (1, 10)


class TestViTSizes:
    """测试不同尺寸输入。"""
    
    def test_imagenet_size(self):
        """测试ImageNet尺寸输入(224x224)。"""
        vit = ViT(
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=1000,
            embed_dim=768,
            num_heads=12,
            num_layers=1
        )
        
        x = torch.randn(1, 3, 224, 224)
        output = vit(x)
        
        assert output.shape == (1, 1000)
