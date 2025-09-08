"""
TCN+线性注意力架构 - 优化方案深度模型组件
结合时序卷积网络(TCN)和线性注意力机制，针对RTX 4060 8GB优化
支持混合精度训练、梯度检查点、动态批处理等内存优化技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List
import math
import warnings

warnings.filterwarnings('ignore')


class LinearAttention(nn.Module):
    """
    线性注意力机制 - 降低注意力计算复杂度
    O(n²d) → O(nd²)，适合长序列和内存受限环境
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0
        
        # 线性变换
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Dropout和LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        for module in [self.q_linear, self.k_linear, self.v_linear, self.out_linear]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def linear_attention_function(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        线性注意力计算
        q: (batch, heads, seq_len, head_dim)
        k, v: (batch, heads, seq_len, head_dim)
        """
        # 应用ELU+1激活函数确保非负性
        q = F.elu(q) + 1  # (B, H, N, D)
        k = F.elu(k) + 1  # (B, H, N, D)
        
        # 线性注意力：O(nd²) 复杂度
        # 计算 k^T * v (D x D 矩阵)
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)  # (B, H, D, D)
        
        # 计算 q * (k^T * v)
        out = torch.einsum('bhnd,bhdm->bhnm', q, kv)  # (B, H, N, D)
        
        # 归一化因子
        k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        normalizer = torch.einsum('bhnd,bhmd->bhn', q, k_sum.transpose(-2, -1))  # (B, H, N)
        
        # 避免除零
        normalizer = torch.clamp(normalizer, min=1e-6)
        out = out / normalizer.unsqueeze(-1)
        
        return out
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len) - 可选的填充掩码
        """
        batch_size, seq_len, _ = x.shape
        
        # 残差连接输入
        residual = x
        
        # 线性变换
        q = self.q_linear(x)  # (B, N, D)
        k = self.k_linear(x)  # (B, N, D)
        v = self.v_linear(x)  # (B, N, D)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        
        # 应用掩码（如果提供）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            q = q.masked_fill(mask == 0, 0)
            k = k.masked_fill(mask == 0, 0)
            v = v.masked_fill(mask == 0, 0)
        
        # 线性注意力计算
        attn_out = self.linear_attention_function(q, k, v)  # (B, H, N, D)
        
        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出线性变换
        out = self.out_linear(attn_out)
        out = self.dropout(out)
        
        # 残差连接和层归一化
        out = self.layer_norm(residual + out)
        
        return out


class TemporalConvBlock(nn.Module):
    """
    时序卷积块 - TCN的基本构建单元
    包含因果卷积、残差连接、权重归一化
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        
        # 第一层卷积
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                     padding=padding, dilation=dilation)
        )
        
        # 第二层卷积
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=1, 
                     padding=padding, dilation=dilation)
        )
        
        # 激活函数和dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: (batch_size, channels, seq_len)
        """
        residual = x
        
        # 第一层卷积 + 激活 + dropout
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二层卷积 + dropout
        out = self.conv2(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # 因果填充处理（移除未来信息）
        if out.size(2) > residual.size(2):
            out = out[:, :, :residual.size(2)]
        
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """
    时序卷积网络(TCN) - 专为时序建模设计
    使用膨胀因果卷积捕获长期依赖
    """
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # 指数增长的膨胀率
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # 因果填充
            
            layers.append(TemporalConvBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size, padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: (batch_size, channels, seq_len)
        """
        return self.network(x)


class TCNLinearAttention(nn.Module):
    """
    TCN+线性注意力融合架构
    结合TCN的时序建模能力和线性注意力的全局建模能力
    针对RTX 4060 8GB优化设计
    """
    
    def __init__(self, 
                 input_size: int,
                 d_model: int = 128,
                 tcn_channels: List[int] = [64, 64, 128],
                 num_heads: int = 4,
                 num_attention_layers: int = 2,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # TCN特征提取器
        self.tcn = TemporalConvNet(
            num_inputs=d_model, 
            num_channels=tcn_channels,
            kernel_size=kernel_size, 
            dropout=dropout
        )
        
        # 从TCN输出维度到模型维度的投影
        self.tcn_projection = nn.Linear(tcn_channels[-1], d_model)
        
        # 多层线性注意力
        self.attention_layers = nn.ModuleList([
            LinearAttention(d_model, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # 最终输出层
        self.output_projection = nn.Linear(d_model, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.tcn_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        
        # 偏置初始化为0
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.tcn_projection.bias)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        x: (batch_size, seq_len, input_size)
        mask: (batch_size, seq_len) - 填充掩码
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (B, N, d_model)
        x = self.dropout(x)
        
        # TCN特征提取 (需要转换维度)
        x_tcn = x.transpose(1, 2)  # (B, d_model, N)
        
        if self.use_gradient_checkpointing and self.training:
            x_tcn = torch.utils.checkpoint.checkpoint(self.tcn, x_tcn)
        else:
            x_tcn = self.tcn(x_tcn)  # (B, tcn_channels[-1], N)
        
        x_tcn = x_tcn.transpose(1, 2)  # (B, N, tcn_channels[-1])
        
        # TCN输出投影
        x_tcn = self.tcn_projection(x_tcn)  # (B, N, d_model)
        
        # 残差连接TCN和原始特征
        x = x + x_tcn
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 多层线性注意力
        for attention_layer in self.attention_layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(attention_layer, x, mask)
            else:
                x = attention_layer(x, mask)
        
        # 全局平均池化 (处理变长序列)
        if mask is not None:
            # 掩码加权平均
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)  # (B, d_model)
        else:
            x = x.mean(dim=1)  # (B, d_model)
        
        # 输出投影
        output = self.output_projection(x)  # (B, 1)
        
        return output.squeeze(-1)  # (B,)


class PositionalEncoding(nn.Module):
    """
    正弦位置编码 - 为序列添加位置信息
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class OptimizedTCNModel(nn.Module):
    """
    优化的TCN模型 - 专为RTX 4060 8GB设计
    集成所有内存优化技术
    """
    
    def __init__(self, 
                 input_size: int,
                 d_model: int = 128,
                 tcn_channels: List[int] = None,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_mixed_precision: bool = True,
                 use_gradient_checkpointing: bool = True,
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        
        self.logger = logger or logging.getLogger(__name__)
        
        # 默认TCN通道配置
        if tcn_channels is None:
            tcn_channels = [d_model // 2, d_model // 2, d_model]
        
        # 核心模型
        self.model = TCNLinearAttention(
            input_size=input_size,
            d_model=d_model,
            tcn_channels=tcn_channels,
            num_heads=num_heads,
            num_attention_layers=num_layers,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        self.use_mixed_precision = use_mixed_precision
        
        # 模型信息
        self.model_info = {
            'input_size': input_size,
            'd_model': d_model,
            'tcn_channels': tcn_channels,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'total_params': sum(p.numel() for p in self.parameters())
        }
        
        self.logger.info(f"TCN+线性注意力模型初始化完成，总参数量: {self.model_info['total_params']:,}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        if self.use_mixed_precision and self.training:
            with torch.cuda.amp.autocast():
                return self.model(x, mask)
        else:
            return self.model(x, mask)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_info
    
    def estimate_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """估计内存使用量"""
        # 粗略估计（MB）
        param_memory = self.model_info['total_params'] * 4 / (1024 * 1024)  # FP32参数
        
        # 激活内存估计
        d_model = self.model_info['d_model']
        activation_memory = (batch_size * seq_len * d_model * 8) / (1024 * 1024)  # 中间激活
        
        # 梯度内存
        gradient_memory = param_memory  # 梯度与参数同样大小
        
        total_memory = param_memory + activation_memory + gradient_memory
        
        # 混合精度减少约40%内存
        if self.use_mixed_precision:
            total_memory *= 0.6
        
        return {
            'parameter_memory_mb': param_memory,
            'activation_memory_mb': activation_memory, 
            'gradient_memory_mb': gradient_memory,
            'total_memory_mb': total_memory
        }


# 便捷函数
def create_tcn_model(input_size: int,
                    d_model: int = 128,
                    num_heads: int = 4,
                    num_layers: int = 2,
                    use_mixed_precision: bool = True,
                    logger: Optional[logging.Logger] = None) -> OptimizedTCNModel:
    """创建优化的TCN模型"""
    return OptimizedTCNModel(
        input_size=input_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        use_mixed_precision=use_mixed_precision,
        logger=logger
    )