"""GPT-1实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT1(nn.Module):
    """GPT-1语言模型。"""
    
    def __init__(self, vocab_size=50257, d_model=768, num_heads=12, num_layers=12, max_seq_len=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 使用DecoderLayer需要self-attention，简化为EncoderLayer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=3072, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.encoder = nn.TransformerEncoder(self.layers, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # 嵌入
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        
        # 叠加
        x = token_embeddings + position_embeddings
        
        # 编码器层（自注意力）
        x = self.encoder(x)
        
        # 最终层
        x = self.ln_final(x)
        
        # 预测
        logits = self.head(x)
        
        return logits