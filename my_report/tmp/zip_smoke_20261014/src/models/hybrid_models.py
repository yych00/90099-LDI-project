# -*- coding: utf-8 -*-
"""
混合网络模型：CNN-LSTM
结构流向：历史序列 -> 1D Conv 局部时间特征提取 -> LSTM 全局序列编码 -> 线性多步预测头。
设计假设：卷积提取局部细节与微观波动，LSTM 建模长程时序依赖。
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM 混合预测模型
    """
    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        kernel_size: int = 3
    ):
        super().__init__()
        if num_channels <= 0 or seq_len <= 0 or pred_len <= 0 or conv_channels <= 0 or lstm_hidden <= 0 or kernel_size <= 0:
            raise ValueError("num_channels, seq_len, pred_len, conv_channels, lstm_hidden, kernel_size 必须均大于 0")
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden

        # 1. 局部特征提取前端 (Conv1D)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU()
        )

        # 2. 循环时间编码网络 (LSTM)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        # 3. 线性多步输出头
        self.fc = nn.Linear(lstm_hidden, pred_len * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("输入时序长度 L 不能为 0")
        if x.size(-1) != self.num_channels:
            raise ValueError(f"输入通道数 ({x.size(-1)}) 必须与初始化通道数 ({self.num_channels}) 一致")
        B, L, C = x.shape
        # 传入 Conv1D: [B, C, L]
        conv_out = self.conv(x.transpose(1, 2))  # [B, conv_channels, L]
        # 转回时序在前的形状传给 LSTM: [B, L, conv_channels]
        lstm_in = conv_out.transpose(1, 2)
        out, (hn, _) = self.lstm(lstm_in)       # hn: [1, B, lstm_hidden]
        last_hidden = hn[-1]                    # [B, lstm_hidden]
        preds = self.fc(last_hidden)            # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)
