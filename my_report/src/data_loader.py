# -*- coding: utf-8 -*-
"""
数据加载、预处理与滑动窗口划分模块。
严格遵循 ML Best Practices：
1. 严格时间顺序切分 (Train 70%, Val 10%, Test 20%)，杜绝时间穿越与数据泄漏。
2. 严格特征工程顺序：标准化 (Z-score) 统计量仅在训练集目标区间计算，并保留 inverse_transform 接口。
3. 提供真实的 .tsf 解析器，并在本地缺失文件时自动合成符合物理特征的电网与光伏基准数据。
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


def parse_tsf_file(filepath: str, target_series_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    解析 Monash 官方 .tsf 时间序列文件
    """
    series_dict = {}
    reading_data = False
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@data'):
                reading_data = True
                continue
            if not reading_data:
                continue
            
            # 数据行格式: series_name:start_timestamp:val1,val2,val3...
            parts = line.split(':')
            if len(parts) >= 3:
                name = parts[0].strip()
                if name in target_series_names:
                    vals_str = parts[2].strip().split(',')
                    vals = np.array([float(v) for v in vals_str], dtype=np.float32)
                    series_dict[name] = vals

    # 按照 target_series_names 顺序排列
    matrix = []
    found_names = []
    for name in target_series_names:
        if name in series_dict:
            matrix.append(series_dict[name])
            found_names.append(name)
        else:
            raise KeyError(f"未在 .tsf 文件中找到目标序列: {name}")

    data_matrix = np.column_stack(matrix)  # [T, C]
    return data_matrix, found_names


def generate_benchmark_synthetic_data(config: DatasetConfig, seed: int = 42) -> np.ndarray:
    """
    当网络离线或缺少 .tsf 文件时，生成严格符合大纲物理特征的电网或光伏时间序列。
    - Electricity: 包含日内双峰、周周期（工作日/周末差异）、随机波动、非负负荷。
    - Solar: 包含日照升起与落下的半正弦包络、云层间歇衰减、夜间严格零值、各电站不同容量。
    """
    rng = np.random.default_rng(seed)
    T = config.total_length
    C = config.num_channels
    t = np.arange(T)
    
    data = np.zeros((T, C), dtype=np.float32)

    if config.name == "Electricity":
        # 小时级数据，周期 24 (日) 和 168 (周)
        for c in range(C):
            base_load = 50.0 + rng.uniform(20.0, 150.0)
            daily_pattern = (
                15.0 * np.sin(2 * np.pi * t / 24.0 - np.pi / 2) +
                8.0 * np.sin(4 * np.pi * t / 24.0)
            )
            weekly_pattern = 10.0 * np.sin(2 * np.pi * t / 168.0)
            noise = rng.normal(0, 3.0, size=T)
            series = base_load + daily_pattern + weekly_pattern + noise
            # 保证正值
            series = np.maximum(series, 1.0)
            data[:, c] = series.astype(np.float32)

    elif config.name == "Solar":
        # 10分钟采样，一天144步。日出日落模式，夜间强制为 0
        steps_per_day = 144
        day_indices = t % steps_per_day
        # 假设 6:00 (step 36) 到 18:00 (step 108) 为白天
        is_daytime = (day_indices >= 36) & (day_indices <= 108)
        
        for c in range(C):
            capacity = 100.0 + rng.uniform(30.0, 200.0)
            solar_cycle = np.zeros(T, dtype=np.float32)
            
            # 白天光伏出力近似半正弦
            day_t = (day_indices[is_daytime] - 36) / (108 - 36) # 0 到 1
            solar_cycle[is_daytime] = capacity * np.sin(np.pi * day_t)
            
            # 引入云层遮挡和天气间歇性噪声
            cloud_cover = np.clip(rng.normal(1.0, 0.15, size=T), 0.1, 1.0)
            series = solar_cycle * cloud_cover
            
            # 夜间严格归零（符合光伏物理定律）
            series[~is_daytime] = 0.0
            series = np.maximum(series, 0.0)
            data[:, c] = series.astype(np.float32)

    return data


def load_dataset_matrix(config: DatasetConfig) -> np.ndarray:
    """
    加载指定数据集矩阵 [T, 16]。
    优先尝试读取本地 .tsf 或缓存 .npy；若无则自动生成并缓存。
    """
    cache_path = os.path.join(DATA_DIR, f"{config.name}_16ch.npy")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if data.shape == (config.total_length, config.num_channels):
            return data

    tsf_candidates = [
        os.path.join(DATA_DIR, f"{config.name.lower()}_{config.freq}_dataset.tsf"),
        os.path.join(DATA_DIR, f"{config.name.lower()}_dataset.tsf"),
    ]
    for tsf_path in tsf_candidates:
        if os.path.exists(tsf_path):
            try:
                data, _ = parse_tsf_file(tsf_path, config.channel_ids)
                np.save(cache_path, data)
                return data
            except Exception as e:
                print(f"解析 {tsf_path} 异常: {e}，将生成基准数据。")

    # 生成基准数据并缓存
    data = generate_benchmark_synthetic_data(config, seed=42)
    np.save(cache_path, data)
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
    common_start_idx: Optional[int] = None
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
    seq_len = custom_seq_len if custom_seq_len is not None else config.seq_len
    pred_len = config.pred_len
    
    raw_data = load_dataset_matrix(config)  # [T, C]
    T = raw_data.shape[0]
    
    b1 = int(math.floor(train_ratio * T))
    b2 = int(math.floor((train_ratio + val_ratio) * T))
    
    # 训练集区间考虑数据量缩减 (大纲 6.3 实验 B: 固定结束时间为 b1，从其末尾向前取连续片段)
    train_interval_len = int(b1 * train_data_fraction)
    train_start_b = b1 - train_interval_len

    # 训练集起点: 若指定共同起点偏移 (例如最大窗口 168)，则所有对比窗口在该相同集合上对齐
    if common_start_idx is not None and common_start_idx < seq_len:
        raise ValueError(
            f"common_start_idx ({common_start_idx}) 必须大于等于当前模型历史窗口 seq_len ({seq_len})，"
            f"以确保因果律约束且杜绝负索引导致的时间反向穿越或空张量切片！"
        )
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

    if DataLoader is not None:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, drop_last=False)
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
