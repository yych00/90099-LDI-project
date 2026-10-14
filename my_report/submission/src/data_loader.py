# -*- coding: utf-8 -*-
"""
数据加载、预处理与滑动窗口划分模块。
严格遵循 ML Best Practices：
1. 严格时间顺序切分 (Train 70%, Val 10%, Test 20%)，杜绝时间穿越与数据泄漏。
2. 严格特征工程顺序：标准化 (Z-score) 统计量仅在训练集目标区间计算，并保留 inverse_transform 接口。
3. 仅接受 prepare_data.py 生成且通过校验和验证的正式发布数据，缺失时明确报错。
"""

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
    """
    单通道/多通道 Z-score 标准化器。
    严格仅基于训练区间计算均值和标准差，防止验证集/测试集信息泄漏。
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        """
        data: 形状为 [T_train, C] 的二维数组
        """
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # 防止常数序列导致除以零
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 尚未进行 fit 操作！")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        逆标准化：将预测结果转回原始物理尺度
        data: [..., C]
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 尚未进行 fit 操作！")
        # 匹配形状（支持 [B, H, C] 或 [T, C]）
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
            raise RuntimeError("StandardScaler 尚未进行 fit 操作！")
        mean_t = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std_t = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean_t) / std_t

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 尚未进行 fit 操作！")
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
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    if metadata['source_kind'] != 'Monash original release' or metadata['channel_ids'] != config.channel_ids:
        raise ValueError('Unverified source or channel IDs')
    if sha256(path) != metadata['cache_sha256']:
        raise ValueError('Dataset cache checksum mismatch')
    data = np.load(path, allow_pickle=False)
    if data.shape != (config.total_length, config.num_channels) or not np.isfinite(data).all():
        raise ValueError('Invalid dataset matrix')
    return data


class TimeSeriesWindowDataset(Dataset):
    """
    滑动窗口时序数据集：
    提取 input [L, C] 和 target [H, C]
    严格基于预先划分的可用预测起点索引，无未来泄漏。
    """
    def __init__(self, data_scaled: np.ndarray, start_indices: np.ndarray, seq_len: int, pred_len: int):
        self.data = torch.from_numpy(data_scaled).float() if torch is not None else data_scaled
        self.start_indices = start_indices
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        t = self.start_indices[idx]
        # 历史输入: [t - seq_len, t)
        x = self.data[t - self.seq_len : t, :]
        # 未来目标: [t, t + pred_len)
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
    seed: int = 42
) -> Tuple[Dict[str, Any], StandardScaler, np.ndarray, Tuple[int, int, int]]:
    """
    按协议划分数据集并构建 DataLoaders：
    b1 = floor(0.70 * T), b2 = floor(0.80 * T)
    - 训练目标区间: [train_start_b, b1)
    - 验证目标区间: [b1, b2)
    - 测试目标区间: [b2, T)
    
    参数:
    - custom_seq_len: 支持消融实验动态调整历史窗口长度
    - train_data_fraction: 支持敏感性分析缩减训练集数据量 (例如 0.25, 0.50, 1.0)
    - common_start_idx: (大纲 6.2 铁律) 多窗口对比时使用最大窗口所允许的共同起点集合，彻底消除样本漂移偏差
    """
    if not 0 < train_data_fraction <= 1:
        raise ValueError('train_data_fraction must lie in (0, 1]')
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError('Invalid chronological split ratios')
    seq_len = custom_seq_len if custom_seq_len is not None else config.seq_len
    pred_len = config.pred_len
    
    raw_data = load_dataset_matrix(config)  # [T, C]
    T = raw_data.shape[0]
    
    b1 = int(math.floor(train_ratio * T))
    b2 = int(math.floor(round(train_ratio + val_ratio, 12) * T))
    
    # 训练集区间考虑数据量缩减 (大纲 6.3 实验 B: 固定结束时间为 b1，从其末尾向前取连续片段)
    train_interval_len = int(b1 * train_data_fraction)
    train_start_b = b1 - train_interval_len

    # 训练集起点: 若指定共同起点偏移 (例如最大窗口 168)，则所有对比窗口在该相同集合上对齐
    if common_start_idx is not None and common_start_idx < seq_len:
        raise ValueError(
            f"common_start_idx ({common_start_idx}) 必须大于等于当前模型历史窗口 seq_len ({seq_len})，"
            f"以确保因果律约束且杜绝负索引导致的时间反向穿越或空张量切片！"
        )
    if b1 < seq_len or seq_len <= 0 or batch_size <= 0:
        raise ValueError('Invalid context length or batch size')
    start_offset = common_start_idx if common_start_idx is not None else seq_len
    if train_interval_len < start_offset + pred_len:
        raise ValueError(
            f"训练集可用数据长度 ({train_interval_len}) 不足以支持当前窗口 ({start_offset}) 与预测步长 ({pred_len})！"
        )
    train_starts = np.arange(train_start_b + start_offset, b1 - pred_len + 1, config.train_stride)
    # 验证集起点: t >= b1, t + pred_len <= b2
    val_starts = np.arange(b1, b2 - pred_len + 1, config.val_stride)
    # 测试集起点: t >= b2, t + pred_len <= T
    test_starts = np.arange(b2, T - pred_len + 1, config.test_stride)

    # 标准化拟合：严格只在可用的训练区间 [train_start_b, b1) 上计算
    scaler = StandardScaler()
    scaler.fit(raw_data[train_start_b:b1, :])
    data_scaled = scaler.transform(raw_data)

    train_ds = TimeSeriesWindowDataset(data_scaled, train_starts, seq_len, pred_len)
    val_ds = TimeSeriesWindowDataset(data_scaled, val_starts, seq_len, pred_len)
    test_ds = TimeSeriesWindowDataset(data_scaled, test_starts, seq_len, pred_len)

    if not all(len(a) for a in (train_starts, val_starts, test_starts)):
        raise ValueError('An evaluation split has no complete target windows')
    if DataLoader is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, generator=torch.Generator().manual_seed(seed), shuffle=True, drop_last=False)
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
        "test_starts": test_starts
    }
    
    return loaders, scaler, raw_data, (train_start_b, b1, b2)
