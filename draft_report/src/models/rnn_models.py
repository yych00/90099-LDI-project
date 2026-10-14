# -*- coding: utf-8 -*-
"""Gated recurrent states test learned memory. A direct multi-step head avoids feeding prediction errors back as future inputs."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Use gated memory and a direct multi-step head; final-state compression is a design assumption to test."""

    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, pred_len * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        B = x.size(0)
        out, (hn, _) = self.lstm(x)  # hn: [num_layers, B, hidden_size]
        last_hidden = hn[-1]  # [B, hidden_size]
        preds = self.fc(last_hidden)  # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)


class GRUModel(nn.Module):
    """Use a simpler gated state to compare recurrent memory with fewer parameters."""

    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, pred_len * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        B = x.size(0)
        out, hn = self.gru(x)  # hn: [num_layers, B, hidden_size]
        last_hidden = hn[-1]  # [B, hidden_size]
        preds = self.fc(last_hidden)  # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)
