# -*- coding: utf-8 -*-
"""
评价指标与切片分析模块：
1. 原尺度 MAE, RMSE 与 Seasonal MASE。
2. 季节性分母严格在训练集目标区间计算，保护零分母。
3. 提供按预测距离 (Lead-time) 逐步误差曲线统计。
4. 提供工程切片分析：电网高负荷峰值时段误差、光伏活跃出力时段误差与夜间零值分析。
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Dict, Tuple, Optional, List, Any


def compute_seasonal_naive_denominators(train_data: np.ndarray, seasonal_m: int, eps: float = 1e-8) -> np.ndarray:
    """
    计算 Seasonal MASE 的分母基准误差：
    严格仅基于训练数据区间计算: mean(|y[t] - y[t-m]|)
    train_data: [T_train, C] 或 [T_train]
    返回: [C] 每个通道的分母
    """
    if train_data.ndim == 1:
        train_data = train_data[:, np.newaxis]
    T, C = train_data.shape
    if T <= seasonal_m:
        raise ValueError(f"训练集长度 {T} 不足以计算季节周期 {seasonal_m} 的差分！")
    
    diff = np.abs(train_data[seasonal_m:, :] - train_data[:-seasonal_m, :])
    denominators = np.mean(diff, axis=0)  # [C]
    # 零分母保护 (大纲 2.5: 分母接近零时提供保护并预警)
    if np.any(denominators <= eps):
        print(f"[Warning] 检测到部分通道周期差分接近零，已启用 {eps} 零分母保护。")
    denominators = np.maximum(denominators, eps)
    return denominators


def calculate_metrics_raw_scale(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    denominators: np.ndarray
) -> Dict[str, Any]:
    """
    在原始尺度上计算聚合指标：
    y_pred: [N, H, C]
    y_true: [N, H, C]
    denominators: [C]
    严格对齐报告公式 (3) MAE、(4) RMSE、(5) MASE
    """
    denominators = np.asarray(denominators, dtype=np.float64).reshape(-1)
    # 逐通道计算
    mae_per_channel = np.mean(np.abs(y_pred - y_true), axis=(0, 1))  # [C]
    rmse_per_channel = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(0, 1)))  # [C]
    mase_per_channel = mae_per_channel / denominators  # [C]

    # 全局聚合指标（按元素整体统计，彻底杜绝 Jensen 不等式导致的均方根偏差）
    overall_mae = float(np.mean(np.abs(y_pred - y_true)))
    overall_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    overall_mase = float(np.mean(mase_per_channel))

    return {
        "mae": overall_mae,
        "rmse": overall_rmse,
        "mase": overall_mase,
        "mae_per_channel": mae_per_channel.tolist(),
        "rmse_per_channel": rmse_per_channel.tolist(),
        "mase_per_channel": mase_per_channel.tolist()
    }


def compute_lead_time_errors(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """
    按预测距离 (Lead-time) 计算逐步 MAE 和 RMSE
    y_pred, y_true: [N, H, C]
    返回:
    mae_curve: [H]
    rmse_curve: [H]
    """
    # 在所有样本 N 和所有通道 C 上平均
    mae_curve = np.mean(np.abs(y_pred - y_true), axis=(0, 2))  # [H]
    rmse_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(0, 2)))  # [H]
    return {
        "mae_curve": mae_curve,
        "rmse_curve": rmse_curve
    }


def compute_slice_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dataset_name: str,
    train_data: np.ndarray
) -> Dict[str, float]:
    """
    业务切片误差分析 (大纲 7.3)：
    - Electricity: 高负荷时段 (真实值 > 训练集各通道 85% 分位数)
    - Solar: 活跃出力时段 (真实值 > 训练集各通道最大值的 10%)
    """
    if dataset_name == "Electricity":
        # 85% 分位数作为峰值阈值
        thresholds = np.percentile(train_data, 85, axis=0, keepdims=True)  # [1, C]
        thresholds = np.expand_dims(thresholds, axis=0)  # [1, 1, C]
        mask = (y_true >= thresholds)
        slice_name = "peak_load"
    elif dataset_name == "Solar":
        # 10% 最大出力作为活跃出力阈值
        thresholds = 0.10 * np.max(train_data, axis=0, keepdims=True)
        thresholds = np.expand_dims(thresholds, axis=0)
        mask = (y_true >= thresholds)
        slice_name = "active_solar"
    else:
        return {}

    if np.sum(mask) == 0:
        return {f"{slice_name}_mae": 0.0, f"{slice_name}_rmse": 0.0, "slice_sample_ratio": 0.0}

    slice_mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    slice_rmse = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
    slice_ratio = float(np.sum(mask) / y_true.size)

    return {
        f"{slice_name}_mae": slice_mae,
        f"{slice_name}_rmse": slice_rmse,
        "slice_sample_ratio": slice_ratio
    }
