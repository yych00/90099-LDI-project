# -*- coding: utf-8 -*-
"""
现代时序模型 (来自 Time-Series-Library 规范)：
1. DLinear: 基于经典时序分解（趋势与季节项）的轻量级双分支线性模型。
2. PatchTST: 基于通道独立性 (Channel-Independence) 与时间块 (Patching) 注意力机制的现代 Transformer 模型。
"""

import math
import torch
import torch.nn as nn


class MovingAvgBlock(nn.Module):
    """
    移动平均池化层，用于提取时序趋势项
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size 必须大于 0，收到: {kernel_size}")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> 转置为 [B, C, L]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("输入时序长度 L 不能为 0")
        x_trans = x.transpose(1, 2)
        # 两侧对称 padding 保持时序长度不变
        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        front_val = x_trans[:, :, 0:1].repeat(1, 1, pad_front)
        end_val = x_trans[:, :, -1:].repeat(1, 1, pad_end)
        padded = torch.cat([front_val, x_trans, end_val], dim=-1)
        trend = self.avg(padded)  # [B, C, L]
        return trend.transpose(1, 2)


class SeriesDecomp(nn.Module):
    """
    时序分解模块：提取趋势项 (Trend) 与季节残差项 (Seasonal)
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvgBlock(kernel_size)

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("输入时序长度 L 不能为 0")
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearModel(nn.Module):
    """
    DLinear 预测模型 (AAAI 2023)
    通过移动平均分解时序，分别用单层全连接映射季节性与趋势项。
    """
    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        moving_avg: int = 25,
        individual: bool = False
    ):
        super().__init__()
        if num_channels <= 0 or seq_len <= 0 or pred_len <= 0 or moving_avg <= 0:
            raise ValueError("num_channels, seq_len, pred_len, moving_avg 必须均大于 0")
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.decompsition = SeriesDecomp(moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(num_channels)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(num_channels)
            ])
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"输入时序长度 ({x.size(1)}) 必须与初始化 seq_len ({self.seq_len}) 一致"
            )
        if self.individual and x.size(-1) != self.num_channels:
            raise ValueError(
                f"在 individual=True 模式下输入通道数 ({x.size(-1)}) 必须与初始化通道数 ({self.num_channels}) 一致"
            )
        seasonal_init, trend_init = self.decompsition(x)
        # 转置为 [B, C, L] 进行线性映射
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), self.num_channels, self.pred_len],
                dtype=x.dtype, device=x.device
            )
            trend_output = torch.zeros(
                [trend_init.size(0), self.num_channels, self.pred_len],
                dtype=x.dtype, device=x.device
            )
            for i in range(self.num_channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        x_out = seasonal_output + trend_output  # [B, C, H]
        return x_out.transpose(1, 2)            # [B, H, C]


class PatchTSTModel(nn.Module):
    """
    PatchTST 预测模型 (ICLR 2023)
    通道独立 (Channel Independence) + 局部时间 Patch 分块 + Transformer 编码器。
    """
    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        if seq_len <= 0 or pred_len <= 0 or patch_len <= 0 or stride <= 0 or num_channels <= 0:
            raise ValueError("num_channels, seq_len, pred_len, patch_len, stride 必须均大于 0")
        if stride > patch_len:
            raise ValueError(
                f"stride ({stride}) 不能大于 patch_len ({patch_len})，否则会导致时序观测盲区与反向传播梯度断流"
            )
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # 保证 patch 能够覆盖输入序列
        if seq_len < patch_len:
            self.patch_len = seq_len
            self.stride = max(1, seq_len // 2)

        # 计算 patch 数量与尾部 padding 长度（确保历史序列末尾最关键时间步被完整覆盖）
        if (self.seq_len - self.patch_len) % self.stride != 0:
            self.num_patches = (self.seq_len - self.patch_len) // self.stride + 2
            self.pad_len = (self.num_patches - 1) * self.stride + self.patch_len - self.seq_len
        else:
            self.num_patches = max(1, (self.seq_len - self.patch_len) // self.stride + 1)
            self.pad_len = 0

        # Patch 线性投影
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        # 可学习位置编码
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 展平与预测输出头
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"输入时序长度 ({x.size(1)}) 必须与初始化 seq_len ({self.seq_len}) 一致"
            )
        B, L, C = x.shape

        # 通道独立：展开为 [B * C, L, 1]
        x_ci = x.transpose(1, 2).reshape(B * C, L, 1)

        # Padding (复制末尾时间步保证覆盖完整历史序列)
        if self.pad_len > 0:
            pad = x_ci[:, -1:, :].repeat(1, self.pad_len, 1)
            x_ci = torch.cat([x_ci, pad], dim=1)

        # 切片提取 patches: [B * C, num_patches, patch_len]
        patches = x_ci[:, :, 0].unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # 截断或对齐 patch 数量
        if patches.size(1) > self.num_patches:
            patches = patches[:, :self.num_patches, :]

        # 线性映射到 d_model: [B * C, num_patches, d_model]
        enc_in = self.patch_embedding(patches) + self.pos_embedding[:, :patches.size(1), :]

        # Transformer 特征编码
        enc_out = self.transformer(enc_in)  # [B * C, num_patches, d_model]

        # 展平并映射到 pred_len: [B * C, pred_len]
        enc_flat = enc_out.reshape(B * C, -1)
        preds_ci = self.head(enc_flat)

        # 恢复通道维度: [B, C, H] -> [B, H, C]
        preds = preds_ci.view(B, C, self.pred_len).transpose(1, 2)
        return preds
