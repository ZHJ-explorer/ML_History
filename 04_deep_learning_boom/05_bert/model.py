"""BERT实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT(nn.Module):
    """BERT预训练语言模型。"""
    
    def __init__(self, vocab_size=30522, d_model=768, num_heads=12, num_layers=12, max_seq_len=512):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=3072, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.cls_head = nn.Linear(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # 嵌入
        word_embeddings = self.word_embedding(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        
        # 叠加
        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        
        # 编码
        output = embeddings
        for layer in self.encoder_layers:
            output = layer(output)
        
        # CLS token输出
        cls_output = output[:, 0, :]
        
        # 预训练任务
        mlm_output = self.cls_head(cls_output)
        nsp_output = self.nsp_head(cls_output)
        
        return mlm_output, nsp_output