
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    """TCN的基本构建块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = lambda x: x[:, :, :-padding].contiguous()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = lambda x: x[:, :, :-padding].contiguous()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """时序卷积网络 (TCN) 主干"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LightweightDeepModel(nn.Module):
    """
    TCN + Transformer Encoder 模型。
    架构遵循 `论文复现技术路线.md` 的描述。
    """
    def __init__(self, input_dim, output_dim, d_model=128, n_heads=4, n_layers=3, ffn_dim=256, dropout=0.1):
        super(LightweightDeepModel, self).__init__()
        
        # TCN 主干
        # TCN的通道数可以根据d_model来设定
        tcn_channels = [d_model] * n_layers
        self.tcn = TemporalConvNet(input_dim, tcn_channels, kernel_size=3, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, 
                                                   dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出头
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, input_dim)
        """
        # PyTorch的卷积层期望 (batch, channels, length)
        x = x.permute(0, 2, 1)
        
        # TCN提取特征
        tcn_out = self.tcn(x)
        
        # 准备输入到Transformer
        # Transformer期望 (batch, length, channels)
        trans_in = tcn_out.permute(0, 2, 1)
        
        # Transformer编码
        trans_out = self.transformer_encoder(trans_in)
        
        # 池化
        # 池化前需要将维度换回 (batch, channels, length)
        pooled = self.pooling(trans_out.permute(0, 2, 1))
        pooled = pooled.squeeze(-1)
        
        # 输出预测
        output = self.output_head(pooled)
        return output

# 使用示例
if __name__ == '__main__':
    # 模型参数
    batch_size = 16
    seq_length = 180 # 30分钟窗口，10s采样
    input_features = 96
    output_size = 1

    # 创建模型
    model = LightweightDeepModel(input_dim=input_features, output_dim=output_size)
    print(model)

    # 创建模拟输入
    dummy_input = torch.randn(batch_size, seq_length, input_features)

    # 前向传播测试
    output = model(dummy_input)

    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == (batch_size, output_size)
    print("\n模型结构和维度测试通过！")
