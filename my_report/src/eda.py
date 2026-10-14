# -*- coding: utf-8 -*-
"""
数据探索性分析 (EDA) 与可视化模块：
1. 输出时序矩阵维度、统计特性 (均值、标准差、零值比例)。
2. 生成连续 7 天时序走势图（负荷双峰与光伏昼夜特性）。
3. 绘制通道间相关性热力图与日内平均曲线。
所有图片自动保存至 figures/ 目录，供报告直接引用。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any

from config import DatasetConfig, ELECTRICITY_CONFIG, SOLAR_CONFIG, FIGURES_DIR
from data_loader import load_dataset_matrix

# 设置图表样式与中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_eda(config: DatasetConfig) -> Dict[str, Any]:
    """
    执行完整 EDA 分析并生成统计图
    """
    data = load_dataset_matrix(config)  # [T, C]
    T, C = data.shape

    # 1. 计算基础统计量
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    zero_ratios = np.mean(data <= 1e-5, axis=0)

    stats_summary = {
        "dataset": config.name,
        "total_samples": T,
        "channels": C,
        "overall_mean": float(np.mean(means)),
        "overall_std": float(np.mean(stds)),
        "overall_zero_ratio": float(np.mean(zero_ratios)),
        "channel_stats": []
    }

    for i in range(C):
        stats_summary["channel_stats"].append({
            "id": config.channel_ids[i],
            "mean": float(means[i]),
            "std": float(stds[i]),
            "min": float(mins[i]),
            "max": float(maxs[i]),
            "zero_ratio": float(zero_ratios[i])
        })

    # 2. 绘制连续 7 天时序曲线图
    # 7 天包含的步数: Electricity 7*24 = 168; Solar 7*144 = 1008
    steps_7d = 168 if config.name == "Electricity" else 1008
    plot_data = data[:steps_7d, :4]  # 选取前 4 个通道展示
    
    plt.figure(figsize=(12, 5))
    time_axis = np.arange(steps_7d)
    if config.name == "Electricity":
        time_axis = time_axis / 24.0  # 转换为天
        plt.xlabel("Time (Days)")
    else:
        time_axis = time_axis / 144.0
        plt.xlabel("Time (Days)")

    for c in range(4):
        plt.plot(time_axis, plot_data[:, c], label=f"Channel {config.channel_ids[c]}", alpha=0.85)

    plt.title(f"{config.name} - 7-Day Continuous Profiles (First 4 Channels)", fontsize=13)
    plt.ylabel(f"Value ({config.unit_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path_7d = os.path.join(FIGURES_DIR, f"{config.name.lower()}_7day_profile.png")
    plt.savefig(plot_path_7d, dpi=200)
    plt.close()

    # 3. 绘制通道间相关性热力图
    corr_matrix = np.corrcoef(data.T)
    plt.figure(figsize=(8, 7))
    plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Pearson Correlation")
    plt.xticks(ticks=np.arange(C), labels=config.channel_ids, rotation=45)
    plt.yticks(ticks=np.arange(C), labels=config.channel_ids)
    plt.title(f"{config.name} - Channel Correlation Matrix", fontsize=12)
    plt.tight_layout()
    corr_plot_path = os.path.join(FIGURES_DIR, f"{config.name.lower()}_correlation.png")
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()

    return stats_summary


if __name__ == "__main__":
    print("正在生成 Electricity EDA 分析...")
    res_elec = run_eda(ELECTRICITY_CONFIG)
    print(f"Electricity: 样本数={res_elec['total_samples']}, 均值={res_elec['overall_mean']:.2f}, 零值比例={res_elec['overall_zero_ratio']:.2%}")

    print("正在生成 Solar EDA 分析...")
    res_solar = run_eda(SOLAR_CONFIG)
    print(f"Solar: 样本数={res_solar['total_samples']}, 均值={res_solar['overall_mean']:.2f}, 零值比例={res_solar['overall_zero_ratio']:.2%}")
    print("EDA 分析与可视化图表生成完毕！")
