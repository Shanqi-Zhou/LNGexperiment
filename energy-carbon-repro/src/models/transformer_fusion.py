"""
Transformer跨模态融合模型
基于技术路线的标准Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)  
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = CrossModalAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class LNGTransformerFusion(nn.Module):
    """LNG Transformer跨模态融合模型"""
    
    def __init__(self, 
                 dynamic_input_dim: int,
                 static_input_dim: int = 32,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 180,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            dynamic_input_dim: 动态特征维度 (9*K)
            static_input_dim: 静态特征维度 (32)
            d_model: 模型维度 (128)
            n_heads: 注意力头数 (4) 
            n_layers: 编码器层数 (3)
            d_ff: 前馈网络维度 (512)
            dropout: Dropout率
            max_seq_len: 最大序列长度 (180)
        """
        super().__init__()
        
        self.d_model = d_model
        self.logger = logger or logging.getLogger(__name__)
        
        # 输入投影层
        self.dynamic_projection = nn.Linear(dynamic_input_dim, d_model)
        self.static_projection = nn.Linear(static_input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 跨模态注意力
        self.cross_modal_attn = CrossModalAttention(d_model, n_heads, dropout)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dynamic_features: torch.Tensor, 
                static_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dynamic_features: (batch_size, seq_len, dynamic_input_dim)
            static_features: (batch_size, static_input_dim)
        Returns:
            predictions: (batch_size, 1)
        """
        batch_size, seq_len, _ = dynamic_features.shape
        
        # 投影到模型维度
        dynamic_emb = self.dynamic_projection(dynamic_features)  # (B, L, d_model)
        static_emb = self.static_projection(static_features)     # (B, d_model)
        
        # 为静态特征添加序列维度
        static_emb = static_emb.unsqueeze(1).repeat(1, seq_len, 1)  # (B, L, d_model)
        
        # 位置编码
        dynamic_emb = self.pos_encoding(dynamic_emb.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码器处理动态特征
        dynamic_encoded = dynamic_emb
        for layer in self.encoder_layers:
            dynamic_encoded = layer(dynamic_encoded)
        
        # 跨模态注意力融合
        fused_features = self.cross_modal_attn(
            query=dynamic_encoded,
            key=static_emb, 
            value=static_emb
        )
        
        # 全局平均池化
        dynamic_global = torch.mean(dynamic_encoded, dim=1)  # (B, d_model)
        fused_global = torch.mean(fused_features, dim=1)     # (B, d_model)
        
        # 拼接特征
        final_features = torch.cat([dynamic_global, fused_global], dim=-1)  # (B, 2*d_model)
        
        # 输出预测
        predictions = self.output_projection(final_features)  # (B, 1)
        
        return predictions
    
    def get_attention_weights(self, dynamic_features: torch.Tensor, 
                             static_features: torch.Tensor) -> torch.Tensor:
        """获取注意力权重用于可解释性分析"""
        with torch.no_grad():
            batch_size, seq_len, _ = dynamic_features.shape
            
            dynamic_emb = self.dynamic_projection(dynamic_features)
            static_emb = self.static_projection(static_features).unsqueeze(1).repeat(1, seq_len, 1)
            
            dynamic_emb = self.pos_encoding(dynamic_emb.transpose(0, 1)).transpose(0, 1)
            
            # 通过编码器
            dynamic_encoded = dynamic_emb
            for layer in self.encoder_layers:
                dynamic_encoded = layer(dynamic_encoded)
            
            # 计算跨模态注意力权重
            Q = self.cross_modal_attn.w_q(dynamic_encoded)
            K = self.cross_modal_attn.w_k(static_emb)
            
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            return attention_weights