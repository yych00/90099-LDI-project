# -*- coding: utf-8 -*-
"""Self-written DLinear and PatchTST-style implementations preserve decomposition and patching ideas without claiming an exact upstream reproduction."""

import math
import torch
import torch.nn as nn


class MovingAvgBlock(nn.Module):
    """Replicate endpoints before averaging to preserve length without inventing observations outside the history."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive; received: {kernel_size}")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average along time without mixing channel observations.
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("Input history cannot be empty")
        x_trans = x.transpose(1, 2)
        # Replicate endpoints to preserve length using observed values only.
        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        front_val = x_trans[:, :, 0:1].repeat(1, 1, pad_front)
        end_val = x_trans[:, :, -1:].repeat(1, 1, pad_end)
        padded = torch.cat([front_val, x_trans, end_val], dim=-1)
        trend = self.avg(padded)  # [B, C, L]
        return trend.transpose(1, 2)


class SeriesDecomp(nn.Module):
    """Separate a smooth moving average and its remainder; the remainder is not necessarily a pure seasonal component."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvgBlock(kernel_size)

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        if x.size(1) == 0:
            raise ValueError("Input history cannot be empty")
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearModel(nn.Module):
    """Project smooth and remaining components separately. Shared maps keep the default parameter count independent of channel count."""

    def __init__(
        self,
        num_channels: int = 16,
        seq_len: int = 48,
        pred_len: int = 24,
        moving_avg: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        if num_channels <= 0 or seq_len <= 0 or pred_len <= 0 or moving_avg <= 0:
            raise ValueError("num_channels, seq_len, pred_len, moving_avg must all be positive")
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.decompsition = SeriesDecomp(moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(num_channels)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(num_channels)]
            )
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Input history length ({x.size(1)}) must match configured seq_len ({self.seq_len})"
            )
        if self.individual and x.size(-1) != self.num_channels:
            raise ValueError(
                f"In individual=True mode, input channels ({x.size(-1)}) must match configured channels ({self.num_channels})"
            )
        seasonal_init, trend_init = self.decompsition(x)
        # Apply the same history-to-horizon map separately to each channel.
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), self.num_channels, self.pred_len],
                dtype=x.dtype,
                device=x.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), self.num_channels, self.pred_len],
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(self.num_channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        x_out = seasonal_output + trend_output  # [B, C, H]
        return x_out.transpose(1, 2)  # [B, H, C]


class PatchTSTModel(nn.Module):
    """Use shared channel-independent patch attention. Pad the tail when necessary so the latest observed points are not dropped."""

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
        dropout: float = 0.1,
    ):
        super().__init__()
        if seq_len <= 0 or pred_len <= 0 or patch_len <= 0 or stride <= 0 or num_channels <= 0:
            raise ValueError(
                "num_channels, seq_len, pred_len, patch_len, stride must all be positive"
            )
        if stride > patch_len:
            raise ValueError(
                f"stride ({stride}) must not exceed patch_len ({patch_len}); otherwise some history positions would be skipped"
            )
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # Short inputs still need at least one patch containing every observed point.
        if seq_len < patch_len:
            self.patch_len = seq_len
            self.stride = max(1, seq_len // 2)

        # Pad incomplete tail patches so recent observations are never silently dropped.
        if (self.seq_len - self.patch_len) % self.stride != 0:
            self.num_patches = (self.seq_len - self.patch_len) // self.stride + 2
            self.pad_len = (self.num_patches - 1) * self.stride + self.patch_len - self.seq_len
        else:
            self.num_patches = max(1, (self.seq_len - self.patch_len) // self.stride + 1)
            self.pad_len = 0

        # Embed a whole local segment as one attention token.
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        # Encode patch order because attention alone has no time ordering.
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Compare historical patches with shared channel-independent weights.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Map all encoded patches to the whole future horizon.
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, L, C] input; received: {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Input history length ({x.size(1)}) must match configured seq_len ({self.seq_len})"
            )
        B, L, C = x.shape

        # Fold channels into batch so sharing weights does not introduce cross-channel attention.
        x_ci = x.transpose(1, 2).reshape(B * C, L, 1)

        # Replicate the last observed value to cover the tail without reading future targets.
        if self.pad_len > 0:
            pad = x_ci[:, -1:, :].repeat(1, self.pad_len, 1)
            x_ci = torch.cat([x_ci, pad], dim=1)

        # Overlapping patches preserve local structure while reducing token count.
        patches = x_ci[:, :, 0].unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Keep token count aligned with the fixed forecast head.
        if patches.size(1) > self.num_patches:
            patches = patches[:, : self.num_patches, :]

        # Combine patch content with its temporal position.
        enc_in = self.patch_embedding(patches) + self.pos_embedding[:, : patches.size(1), :]

        # Attend only to patches drawn from the observed history.
        enc_out = self.transformer(enc_in)  # [B * C, num_patches, d_model]

        # Produce one direct horizon per independent channel.
        enc_flat = enc_out.reshape(B * C, -1)
        preds_ci = self.head(enc_flat)

        # Restore the common output interface used by evaluation.
        preds = preds_ci.view(B, C, self.pred_len).transpose(1, 2)
        return preds
