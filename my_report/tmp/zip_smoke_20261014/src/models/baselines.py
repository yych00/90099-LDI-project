# -*- coding: utf-8 -*-
"""
简单非深度学习基线模型：
1. Persistence (最近观测持久化): 假设未来各步等于最后一步观测。
2. Daily Seasonal Naive (日周期朴素基线): 假设未来24小时与历史最后一天（24小时前）完全相同。
"""

import torch
import torch.nn as nn


class PersistenceModel(nn.Module):
    """
    Persistence 基线模型：
    输入: [batch, seq_len, num_channels]
    输出: [batch, pred_len, num_channels]
    """
    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("输入时序长度 L 必须大于 0")
        last_step = x[:, -1:, :]  # [B, 1, C]
        out = last_step.repeat(1, self.pred_len, 1)  # [B, H, C]
        return out


class DailySeasonalNaiveModel(nn.Module):
    """
    Daily Seasonal Naive 基线模型：
    输入: [batch, seq_len, num_channels]
    输出: [batch, pred_len, num_channels]
    利用前一周期 (周期 m 步) 对应的历史观测作为未来预测。
    """
    def __init__(self, pred_len: int, seasonal_m: int):
        super().__init__()
        self.pred_len = pred_len
        self.seasonal_m = seasonal_m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是 3 维 [B, L, C]，收到形状: {tuple(x.shape)}")
        B, L, C = x.shape
        if L < self.seasonal_m:
            raise ValueError(f"输入历史长度 L={L} 小于季节周期 m={self.seasonal_m}")
        
        # 提取最近一个完整周期: [B, m, C]
        last_cycle = x[:, -self.seasonal_m:, :]
        
        if self.pred_len <= self.seasonal_m:
            out = last_cycle[:, :self.pred_len, :]
        else:
            repeats = (self.pred_len + self.seasonal_m - 1) // self.seasonal_m
            tiled = last_cycle.repeat(1, repeats, 1)
            out = tiled[:, :self.pred_len, :]
            
        return out
