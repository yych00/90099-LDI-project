# -*- coding: utf-8 -*-
"""Temporal filters detect local changes. Fixed-size pooling keeps the output head unchanged when history length varies."""

import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """Pool local features to a fixed head size so changing history length does not change this model's parameter count."""

    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        conv_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Share temporal filters to learn recurring local changes.
        self.feature_net = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            # Keep the dense head fixed across input lengths to avoid a capacity change in this CNN.
            nn.AdaptiveAvgPool1d(output_size=16),
        )

        # Predict the entire horizon directly to avoid recursive error feedback.
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(conv_channels * 16, pred_len * num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d treats channels as features, so move time to the final axis.
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        B, L, C = x.shape
        x_trans = x.transpose(1, 2)
        features = self.feature_net(x_trans)  # [B, conv_channels, 16]
        preds = self.fc(features)  # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)
