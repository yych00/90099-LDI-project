# -*- coding: utf-8 -*-
"""Parameter-free controls establish how much continuity and daily repetition explain before learning a larger model."""

import torch
import torch.nn as nn


class PersistenceModel(nn.Module):
    """Repeat the most recent observation to test whether learning improves on temporal continuity."""

    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("Input history length L must be positive")
        last_step = x[:, -1:, :]  # [B, 1, C]
        out = last_step.repeat(1, self.pred_len, 1)  # [B, H, C]
        return out


class DailySeasonalNaiveModel(nn.Module):
    """Repeat the latest observed daily cycle; this strong control uses no future values or fitted parameters."""

    def __init__(self, pred_len: int, seasonal_m: int):
        super().__init__()
        self.pred_len = pred_len
        self.seasonal_m = seasonal_m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        B, L, C = x.shape
        if L < self.seasonal_m:
            raise ValueError(
                f"Input history L={L} is shorter than seasonal period m={self.seasonal_m}"
            )

        # Use the last fully observed daily cycle so no target values enter the baseline.
        last_cycle = x[:, -self.seasonal_m :, :]

        if self.pred_len <= self.seasonal_m:
            out = last_cycle[:, : self.pred_len, :]
        else:
            repeats = (self.pred_len + self.seasonal_m - 1) // self.seasonal_m
            tiled = last_cycle.repeat(1, repeats, 1)
            out = tiled[:, : self.pred_len, :]

        return out
