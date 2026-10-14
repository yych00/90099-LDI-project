# -*- coding: utf-8 -*-
"""A convolutional frontend tests whether local features help recurrent encoding; its extra capacity must be controlled in ablations."""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """Encode local convolutional features in time order, then forecast the whole horizon without recursive feedback."""

    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        if (
            num_channels <= 0
            or seq_len <= 0
            or pred_len <= 0
            or conv_channels <= 0
            or lstm_hidden <= 0
            or kernel_size <= 0
        ):
            raise ValueError(
                "num_channels, seq_len, pred_len, conv_channels, lstm_hidden, kernel_size must all be positive"
            )
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden

        # Extract local changes before asking the recurrent state to summarise history.
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )

        # Preserve feature time order for recurrent encoding.
        self.lstm = nn.LSTM(
            input_size=conv_channels, hidden_size=lstm_hidden, num_layers=1, batch_first=True
        )

        # A direct head makes the forecast protocol match the other models.
        self.fc = nn.Linear(lstm_hidden, pred_len * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("Input history cannot be empty")
        if x.size(-1) != self.num_channels:
            raise ValueError(
                f"Input channels ({x.size(-1)}) must match configured channels ({self.num_channels})"
            )
        B, L, C = x.shape
        # Conv1d expects features before the time axis.
        conv_out = self.conv(x.transpose(1, 2))  # [B, conv_channels, L]
        # Restore time-major features for batch-first recurrence.
        lstm_in = conv_out.transpose(1, 2)
        out, (hn, _) = self.lstm(lstm_in)  # hn: [1, B, lstm_hidden]
        last_hidden = hn[-1]  # [B, lstm_hidden]
        preds = self.fc(last_hidden)  # [B, H * C]
        return preds.view(B, self.pred_len, self.num_channels)
