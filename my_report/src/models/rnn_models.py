# -*- coding: utf-8 -*-
"""
循环神经网络模型：
1. LSTM: 长短期记忆网络，具备遗忘门、输入门和输出门，捕获历史时序依赖。
2. GRU: 门控循环单元，通过重置门和更新门平衡计算开销与长时记忆能力。
统一采用多通道输入 [B, L, C]，多通道直接多步输出 [B, H, C]。
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    单层 LSTM 时序多步预测模型
    """
    def __init__(self, num_channels: int = 16, seq_len: int = 48, pred_len: int = 24, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, pred_len * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        B = x.size(0)
        out, (hn, _) = self.lstm(x)  # hn: [num_layers, B, hidden_size]
        last_hidden = hn[-1]         # [B, hidden_size]
        preds = self.fc(last_hidden) # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)


class GRUModel(nn.Module):
    """
    单层 GRU 时序多步预测模型
    """
    def __init__(self, num_channels: int = 16, seq_len: int = 48, pred_len: int = 24, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, pred_len * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        B = x.size(0)
        out, hn = self.gru(x)        # hn: [num_layers, B, hidden_size]
        last_hidden = hn[-1]         # [B, hidden_size]
        preds = self.fc(last_hidden) # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)
