# -*- coding: utf-8 -*-
"""
一维卷积神经网络 (1D CNN) 时序多步预测模型：
利用局部时间卷积核提取多通道时间序列的局部波动与趋势特征。
采用自适应池化以保证在历史窗口敏感性实验中（不同 seq_len）网络权重结构稳定可复用。
"""

import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """
    1D CNN 多步时序预测模型
    """
    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        conv_channels: int = 64,
        kernel_size: int = 3
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 卷积特征提取网络
        self.feature_net = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            # 自适应池化到固定时序长度 16，适应不同 seq_len 输入
            nn.AdaptiveAvgPool1d(output_size=16)
        )
        
        # 多步预测投影头
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * 16, pred_len * num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> 转置为通道在前的 [B, C, L]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        B, L, C = x.shape
        x_trans = x.transpose(1, 2)
        features = self.feature_net(x_trans)  # [B, conv_channels, 16]
        preds = self.fc(features)             # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)
