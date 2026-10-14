# -*- coding: utf-8 -*-
"""Load only verified release data and isolate future targets. Training-only scaling prevents validation and test information from influencing the fitted representation."""

import os
import math
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
from typing import Tuple, Dict, Optional, List, Any
from config import DatasetConfig, ELECTRICITY_CONFIG, SOLAR_CONFIG, DATA_DIR, TRAIN_RATIO, VAL_RATIO


class StandardScaler:
    """Fit per-channel statistics on training observations only, so evaluation data cannot alter the representation."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        """Fit a [training time, channels] array already restricted to the available training interval."""
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # Clamp constant-channel scales to avoid division by zero.
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transformation.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Restore original release scale before scoring; broadcasting also supports [batch, horizon, channels]."""
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transformation.")
        # Broadcast channel statistics across origins and lead times without mixing scales.
        mean = self.mean
        std = self.std
        if data.ndim == 3 and mean.ndim == 2:
            mean = np.expand_dims(mean, axis=0)  # [1, 1, C]
            std = np.expand_dims(std, axis=0)
        elif data.ndim == 1 and mean.ndim == 2:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        return (data * std) + mean

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transformation.")
        mean_t = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std_t = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean_t) / std_t

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transformation.")
        mean_t = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std_t = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        if tensor.dim() == 3 and mean_t.dim() == 2:
            mean_t = mean_t.unsqueeze(0)
            std_t = std_t.unsqueeze(0)
        return (tensor * std_t) + mean_t


def load_dataset_matrix(config: DatasetConfig) -> np.ndarray:
    """Accept only a hash-verified cache with the expected original release IDs."""
    import json
    from pathlib import Path
    from prepare_data import sha256

    path = Path(DATA_DIR) / f"{config.name}_16ch.npy"
    manifest_path = Path(DATA_DIR) / f"{config.name}_manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run python prepare_data.py first; synthetic fallback is disabled.")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        metadata["source_kind"] != "Monash original release"
        or metadata["channel_ids"] != config.channel_ids
    ):
        raise ValueError("Unverified source or channel IDs")
    if sha256(path) != metadata["cache_sha256"]:
        raise ValueError("Dataset cache checksum mismatch")
    data = np.load(path, allow_pickle=False)
    if data.shape != (config.total_length, config.num_channels) or not np.isfinite(data).all():
        raise ValueError("Invalid dataset matrix")
    return data


class TimeSeriesWindowDataset(Dataset):
    """Return [history, channels] inputs and [horizon, channels] targets from origins whose target intervals were isolated first."""

    def __init__(
        self, data_scaled: np.ndarray, start_indices: np.ndarray, seq_len: int, pred_len: int
    ):
        self.data = torch.from_numpy(data_scaled).float() if torch is not None else data_scaled
        self.start_indices = start_indices
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        t = self.start_indices[idx]
        # End input before the target origin to preserve causality.
        x = self.data[t - self.seq_len : t, :]
        # Use only the complete target already assigned to this split.
        y = self.data[t : t + self.pred_len, :]
        return x, y


def get_data_splits(
    config: DatasetConfig,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    custom_seq_len: Optional[int] = None,
    train_data_fraction: float = 1.0,
    common_start_idx: Optional[int] = None,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[Dict[str, Any], StandardScaler, np.ndarray, Tuple[int, int, int]]:
    """Keep complete future targets inside their chronological split. Fractions use the newest training segment; common_start_idx aligns origins across history lengths to avoid comparing different samples."""
    if not 0 < train_data_fraction <= 1:
        raise ValueError("train_data_fraction must lie in (0, 1]")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid chronological split ratios")
    seq_len = custom_seq_len if custom_seq_len is not None else config.seq_len
    pred_len = config.pred_len

    raw_data = load_dataset_matrix(config)  # [T, C]
    T = raw_data.shape[0]

    b1 = int(math.floor(train_ratio * T))
    b2 = int(math.floor(round(train_ratio + val_ratio, 12) * T))

    # Use the newest contiguous fraction so the training end and evaluation intervals stay fixed.
    train_interval_len = int(b1 * train_data_fraction)
    train_start_b = b1 - train_interval_len

    # Align all history lengths to the maximum admissible origin set to avoid sample drift.
    if common_start_idx is not None and common_start_idx < seq_len:
        raise ValueError(
            f"common_start_idx ({common_start_idx}) must be at least the requested history seq_len ({seq_len})，"
            f"to prevent negative input indices or empty histories."
        )
    if b1 < seq_len or seq_len <= 0 or batch_size <= 0:
        raise ValueError("Invalid context length or batch size")
    start_offset = common_start_idx if common_start_idx is not None else seq_len
    if train_interval_len < start_offset + pred_len:
        raise ValueError(
            f"Available training length ({train_interval_len}) cannot fit context ({start_offset}) plus horizon ({pred_len})"
        )
    train_starts = np.arange(train_start_b + start_offset, b1 - pred_len + 1, config.train_stride)
    # Validation targets must fit completely inside validation.
    val_starts = np.arange(b1, b2 - pred_len + 1, config.val_stride)
    # Test targets must fit completely inside test.
    test_starts = np.arange(b2, T - pred_len + 1, config.test_stride)

    # Fit only available training observations; reduced-data runs must not reuse unavailable statistics.
    scaler = StandardScaler()
    scaler.fit(raw_data[train_start_b:b1, :])
    data_scaled = scaler.transform(raw_data)

    train_ds = TimeSeriesWindowDataset(data_scaled, train_starts, seq_len, pred_len)
    val_ds = TimeSeriesWindowDataset(data_scaled, val_starts, seq_len, pred_len)
    test_ds = TimeSeriesWindowDataset(data_scaled, test_starts, seq_len, pred_len)

    if not all(len(a) for a in (train_starts, val_starts, test_starts)):
        raise ValueError("An evaluation split has no complete target windows")
    if DataLoader is not None:
        # Shuffle only after complete windows have been isolated chronologically;
        # order inside each history and all validation/test origins stays fixed.
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            generator=torch.Generator().manual_seed(seed),
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        train_loader, val_loader, test_loader = None, None, None

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_starts": train_starts,
        "val_starts": val_starts,
        "test_starts": test_starts,
    }

    return loaders, scaler, raw_data, (train_start_b, b1, b2)
