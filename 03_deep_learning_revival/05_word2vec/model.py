"""Word2Vec词向量实现。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    """CBOW模型：通过上下文预测中心词。"""
    
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.output = nn.Linear(embed_size, vocab_size)
    
    def forward(self, context_indices):
        """前向传播。
        
        Args:
            context_indices: 上下文词索引，shape (batch, context_size)
            
        Returns:
            预测得分，shape (batch, vocab_size)
        """
        embeds = self.embed(context_indices)  # (batch, context_size, embed_size)
        vector = embeds.mean(dim=1)           # (batch, embed_size)
        output = self.output(vector)          # (batch, vocab_size)
        return output


class SkipGram(nn.Module):
    """Skip-gram模型：通过中心词预测上下文。"""
    
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.output = nn.Linear(embed_size, vocab_size)
    
    def forward(self, center_indices):
        """前向传播。
        
        Args:
            center_indices: 中心词索引，shape (batch,)
            
        Returns:
            预测得分，shape (batch, vocab_size)
        """
        embeds = self.embed(center_indices)  # (batch, embed_size)
        output = self.output(embeds)         # (batch, vocab_size)
        return output