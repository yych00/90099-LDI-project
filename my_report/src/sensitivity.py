# -*- coding: utf-8 -*-
"""
消融与敏感性分析模块 (大纲第 4 部分)：
1. 实验 A: 历史窗口长度敏感性分析 (24h, 48h, 168h)，对比 LSTM 与 PatchTST。
2. 实验 B: 训练数据量敏感性分析 (25%, 50%, 100%)，固定结束时间于 b1。
3. 实验 C: CNN-LSTM 结构消融实验 (对比 CNN-LSTM, LSTM 与 CNN1D)。
"""

from typing import List, Dict, Any
from config import DatasetConfig, TrainConfig, ELECTRICITY_CONFIG, SOLAR_CONFIG
from data_loader import get_data_splits
from train import train_and_evaluate


def run_window_sensitivity_experiment(
    config: DatasetConfig,
    models: List[str] = ["LSTM", "PatchTST"],
    epochs: int = 5
) -> List[Dict[str, Any]]:
    """
    实验 A: 历史观察窗口长度敏感性分析
    Electricity: 24 (24h), 48 (48h), 168 (168h)
    Solar: 144 (24h), 288 (48h), 1008 (168h)
    """
    if config.name == "Electricity":
        window_lengths = [24, 48, 168]
        max_seq = 168
    else:
        window_lengths = [144, 288, 1008]
        max_seq = 1008

    results = []
    train_cfg = TrainConfig(max_epochs=epochs)

    for seq_l in window_lengths:
        loaders, scaler, raw_data, bounds = get_data_splits(
            config, custom_seq_len=seq_l, common_start_idx=max_seq
        )
        for model_name in models:
            print(f"--- 窗口敏感性: {config.name} | 模型: {model_name} | 窗口: {seq_l} ---")
            res = train_and_evaluate(
                model_name=model_name,
                dataset_cfg=config,
                loaders=loaders,
                scaler=scaler,
                raw_data=raw_data,
                train_bounds=bounds,
                train_cfg=train_cfg,
                custom_seq_len=seq_l
            )
            results.append(res)

    return results


def run_data_size_sensitivity_experiment(
    config: DatasetConfig,
    models: List[str] = ["LSTM", "PatchTST"],
    fractions: List[float] = [0.25, 0.50, 1.0],
    epochs: int = 5
) -> List[Dict[str, Any]]:
    """
    实验 B: 训练数据量敏感性分析 (25%, 50%, 100%)
    """
    results = []
    train_cfg = TrainConfig(max_epochs=epochs)

    for frac in fractions:
        loaders, scaler, raw_data, bounds = get_data_splits(config, train_data_fraction=frac)
        for model_name in models:
            print(f"--- 数据量敏感性: {config.name} | 模型: {model_name} | 比例: {frac*100:.0f}% ---")
            res = train_and_evaluate(
                model_name=model_name,
                dataset_cfg=config,
                loaders=loaders,
                scaler=scaler,
                raw_data=raw_data,
                train_bounds=bounds,
                train_cfg=train_cfg,
                train_fraction=frac
            )
            results.append(res)

    return results


def run_ablation_experiment(
    config: DatasetConfig,
    epochs: int = 5
) -> List[Dict[str, Any]]:
    """
    实验 C: 局部卷积模块消融分析
    对比 CNN-LSTM (混合)、LSTM (无卷积前端) 与 CNN1D (无循环编码)
    """
    models = ["CNN1D", "LSTM", "CNNLSTM"]
    results = []
    train_cfg = TrainConfig(max_epochs=epochs)
    loaders, scaler, raw_data, bounds = get_data_splits(config)

    for model_name in models:
        print(f"--- 模块消融: {config.name} | 模型: {model_name} ---")
        res = train_and_evaluate(
            model_name=model_name,
            dataset_cfg=config,
            loaders=loaders,
            scaler=scaler,
            raw_data=raw_data,
            train_bounds=bounds,
            train_cfg=train_cfg
        )
        results.append(res)

    return results
