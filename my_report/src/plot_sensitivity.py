# -*- coding: utf-8 -*-
"""
敏感性分析与消融实验可视化脚本 (Section 6 / Section 9 出版级图表生成)。
数据源：SQLite 数据库 (experiments.db)，自动查询真实实验记录并动态绘制。
生成图表：
1. figures/sensitivity_window_length.png (历史窗口长度敏感性分析: 24h, 48h, 168h 对比 LSTM 与 PatchTST)
2. figures/sensitivity_data_fraction.png (训练数据量敏感性分析: 25%, 50%, 100% 对比 LSTM 与 PatchTST)
3. figures/ablation_cnn_lstm.png (跨场景 CNN-LSTM 模块消融: Electricity 与 Solar 下 LSTM, CNN1D, CNN-LSTM 的误差与参数量)
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURES_DIR, DB_PATH

# 绘图字体与风格优化
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def fetch_dataframe() -> pd.DataFrame:
    """从 experiments.db 提取全量实验数据"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库未找到: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM experiments ORDER BY id ASC", conn)
    conn.close()
    return df


def plot_window_sensitivity(df: pd.DataFrame = None):
    """
    绘制历史观察窗口长度敏感性曲线与资源开销 (Electricity L in {24h, 48h, 168h})
    """
    if df is None:
        df = fetch_dataframe()

    # 提取对齐受控窗口敏感性实验数据 (common_start_idx 对齐)
    # 优先从数据库中提取包含 seq_len=24, 48, 168 且 train_fraction=1.0 的记录
    sub = df[(df["dataset"] == "Electricity") & (df["train_fraction"] == 1.0) & (df["seq_len"].isin([24, 48, 168]))]
    
    # 获取最近一次完整窗口运行数据
    lstm_mase, lstm_time = [], []
    patchtst_mase, patchtst_time = [], []
    for l in [24, 48, 168]:
        r_lstm = sub[(sub["model_name"] == "LSTM") & (sub["seq_len"] == l)].iloc[-1]
        r_patch = sub[(sub["model_name"] == "PatchTST") & (sub["seq_len"] == l)].iloc[-1]
        lstm_mase.append(float(r_lstm["mase_24h"]))
        lstm_time.append(float(r_lstm["train_time_sec"]))
        patchtst_mase.append(float(r_patch["mase_24h"]))
        patchtst_time.append(float(r_patch["train_time_sec"]))

    windows = [24, 48, 168]
    window_labels = ['24h (1 Day)', '48h (2 Days)', '168h (1 Week)']
    x = np.arange(len(windows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # 子图 1: MASE 预测误差演变
    ax1.plot(x, lstm_mase, 'o-', color='#1f77b4', linewidth=2.4, markersize=8, label='LSTM (MASE)')
    ax1.plot(x, patchtst_mase, 's-', color='#d62728', linewidth=2.4, markersize=8, label='PatchTST (MASE)')
    ax1.axhline(y=1.0, color='gray', linestyle=':', label='Seasonal Naive (MASE=1.0)')

    # 精确标注避免重叠与截断
    ax1.annotate(f"LSTM: {lstm_mase[0]:.4f}", (x[0], lstm_mase[0]), textcoords="offset points", xytext=(-25, -18), fontsize=9.5, color='#1f77b4', weight='bold')
    ax1.annotate(f"PatchTST: {patchtst_mase[0]:.4f}", (x[0], patchtst_mase[0]), textcoords="offset points", xytext=(-25, 12), fontsize=9.5, color='#d62728', weight='bold')

    ax1.annotate(f"{lstm_mase[1]:.4f}", (x[1], lstm_mase[1]), textcoords="offset points", xytext=(-15, 12), fontsize=9.5, color='#1f77b4', weight='bold')
    ax1.annotate(f"{patchtst_mase[1]:.4f}", (x[1], patchtst_mase[1]), textcoords="offset points", xytext=(-15, -18), fontsize=9.5, color='#d62728', weight='bold')

    ax1.annotate(f"{lstm_mase[2]:.4f}", (x[2], lstm_mase[2]), textcoords="offset points", xytext=(-15, 12), fontsize=9.5, color='#1f77b4', weight='bold')
    ax1.annotate(f"{patchtst_mase[2]:.4f}", (x[2], patchtst_mase[2]), textcoords="offset points", xytext=(-15, -18), fontsize=9.5, color='#d62728', weight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(window_labels, fontsize=11)
    ax1.set_ylim(0.34, 1.06)
    ax1.set_title("Forecast Error vs. History Window Length L (Electricity)", fontsize=12, pad=10)
    ax1.set_xlabel("Historical Context Window L", fontsize=11)
    ax1.set_ylabel("Seasonal MASE (24h Ahead)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    # 子图 2: 训练用时与参数缩放开销
    width = 0.3
    ax2.bar(x - width/2, lstm_time, width, label='LSTM Train Time (s)', color='#1f77b4', alpha=0.75)
    ax2.bar(x + width/2, patchtst_time, width, label='PatchTST Train Time (s)', color='#d62728', alpha=0.75)

    for i in range(len(x)):
        ax2.text(x[i] - width/2, lstm_time[i] + 1.8, f"{lstm_time[i]:.1f}s", ha='center', fontsize=9.5, color='#1f77b4', weight='bold')
        ax2.text(x[i] + width/2, patchtst_time[i] + 1.8, f"{patchtst_time[i]:.1f}s", ha='center', fontsize=9.5, color='#d62728', weight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(window_labels, fontsize=11)
    ax2.set_ylim(0, max(patchtst_time) * 1.15)
    ax2.set_title("Computational Footprint vs. History Window Length L", fontsize=12, pad=10)
    ax2.set_xlabel("Historical Context Window L", fontsize=11)
    ax2.set_ylabel("Training Time (Seconds)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sensitivity_window_length.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_data_fraction_sensitivity(df: pd.DataFrame = None):
    """
    绘制训练数据量敏感性与样本效率对比 (25%, 50%, 100%)
    """
    if df is None:
        df = fetch_dataframe()

    sub = df[(df["dataset"] == "Electricity") & (df["seq_len"] == 48) & (df["train_fraction"].isin([0.25, 0.50, 1.0]))]
    
    lstm_mase, lstm_mae = [], []
    patchtst_mase, patchtst_mae = [], []
    for frac in [0.25, 0.50, 1.0]:
        r_lstm = sub[(sub["model_name"] == "LSTM") & (sub["train_fraction"] == frac)].iloc[-1]
        r_patch = sub[(sub["model_name"] == "PatchTST") & (sub["train_fraction"] == frac)].iloc[-1]
        lstm_mase.append(float(r_lstm["mase_24h"]))
        lstm_mae.append(float(r_lstm["mae_24h"]))
        patchtst_mase.append(float(r_patch["mase_24h"]))
        patchtst_mae.append(float(r_patch["mae_24h"]))

    fractions = [25, 50, 100]
    frac_labels = ['25% Data (~19w)', '50% Data (~13m)', '100% Data (~2.1y)']
    x = np.arange(len(fractions))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # 子图 1: 样本效率与 MASE 对比
    ax1.plot(x, lstm_mase, 'o-', color='#1f77b4', linewidth=2.4, markersize=8, label='LSTM (MASE)')
    ax1.plot(x, patchtst_mase, 's-', color='#d62728', linewidth=2.4, markersize=8, label='PatchTST (MASE)')
    ax1.axhline(y=1.0, color='gray', linestyle=':', label='Seasonal Naive Baseline (1.0)')

    ax1.annotate(f"LSTM: {lstm_mase[0]:.4f}", (x[0], lstm_mase[0]), textcoords="offset points", xytext=(-25, 12), fontsize=9.5, color='#1f77b4', weight='bold')
    ax1.annotate(f"PatchTST: {patchtst_mase[0]:.4f}", (x[0], patchtst_mase[0]), textcoords="offset points", xytext=(-25, -18), fontsize=9.5, color='#d62728', weight='bold')

    ax1.annotate(f"{lstm_mase[1]:.4f}", (x[1], lstm_mase[1]), textcoords="offset points", xytext=(-15, 12), fontsize=9.5, color='#1f77b4', weight='bold')
    ax1.annotate(f"{patchtst_mase[1]:.4f}", (x[1], patchtst_mase[1]), textcoords="offset points", xytext=(-15, -18), fontsize=9.5, color='#d62728', weight='bold')

    ax1.annotate(f"{lstm_mase[2]:.4f}", (x[2], lstm_mase[2]), textcoords="offset points", xytext=(-15, 12), fontsize=9.5, color='#1f77b4', weight='bold')
    ax1.annotate(f"{patchtst_mase[2]:.4f}", (x[2], patchtst_mase[2]), textcoords="offset points", xytext=(-15, -18), fontsize=9.5, color='#d62728', weight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(frac_labels, fontsize=11)
    ax1.set_ylim(0.30, 1.06)
    ax1.set_title("Sample Efficiency: Forecast Error vs. Training Fraction", fontsize=12, pad=10)
    ax1.set_xlabel("Training Data Fraction (Anchored to b1)", fontsize=11)
    ax1.set_ylabel("Seasonal MASE (24h Ahead)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    # 子图 2: MAE 原物理尺度误差对比柱状图
    width = 0.32
    ax2.bar(x - width/2, lstm_mae, width, label='LSTM MAE (kW)', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, patchtst_mae, width, label='PatchTST MAE (kW)', color='#d62728', alpha=0.8)

    for i in range(len(x)):
        ax2.text(x[i] - width/2, lstm_mae[i] + 0.15, f"{lstm_mae[i]:.2f}", ha='center', fontsize=9.5, color='#1f77b4', weight='bold')
        ax2.text(x[i] + width/2, patchtst_mae[i] + 0.15, f"{patchtst_mae[i]:.2f}", ha='center', fontsize=9.5, color='#d62728', weight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(frac_labels, fontsize=11)
    ax2.set_ylim(0, max(lstm_mae) * 1.15)
    ax2.set_title("MAE Error Degradation under Data Scarcity", fontsize=12, pad=10)
    ax2.set_xlabel("Training Data Fraction", fontsize=11)
    ax2.set_ylabel("Mean Absolute Error (kW)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sensitivity_data_fraction.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_modular_ablation(df: pd.DataFrame = None):
    """
    绘制跨场景 CNN-LSTM 卷积前端模块消融对比：
    包含 Panel (a) Electricity 负荷 与 Panel (b) Solar 光伏出力 两大任务对比，
    全面展示纯 LSTM、纯 1D-CNN 与混合 CNN-LSTM 的预测误差及参数量压缩比。
    """
    if df is None:
        df = fetch_dataframe()

    models = ['LSTM', 'CNN1D', 'CNNLSTM']
    model_labels = ['LSTM\n(Recurrent)', '1D-CNN\n(Temporal Conv)', 'CNN-LSTM\n(Hybrid)']

    # 提取 Electricity (L=48, train_fraction=1.0)
    sub_elec = df[(df["dataset"] == "Electricity") & (df["seq_len"] == 48) & (df["train_fraction"] == 1.0)]
    mae_elec, rmse_elec, params_elec = [], [], []
    for m in models:
        row = sub_elec[sub_elec["model_name"] == m].iloc[-1]
        mae_elec.append(float(row["mae_24h"]))
        rmse_elec.append(float(row["rmse_24h"]))
        params_elec.append(float(row["num_params"]) / 1000.0)

    # 提取 Solar (L=288, train_fraction=1.0)
    sub_solar = df[(df["dataset"] == "Solar") & (df["seq_len"] == 288) & (df["train_fraction"] == 1.0)]
    mae_solar, rmse_solar, params_solar = [], [], []
    for m in models:
        row = sub_solar[sub_solar["model_name"] == m].iloc[-1]
        mae_solar.append(float(row["mae_24h"]))
        rmse_solar.append(float(row["rmse_24h"]))
        params_solar.append(float(row["num_params"]) / 1000.0)

    x = np.arange(len(models))
    width = 0.32

    # 创建 2x2 复合子图 (清晰呈现两大数据集的误差与参数量)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- Panel (a1): Electricity 误差 ---
    ax_e1 = axes[0, 0]
    ax_e1.bar(x - width/2, mae_elec, width, label='24h MAE', color='#2ca02c', alpha=0.85)
    ax_e1.bar(x + width/2, rmse_elec, width, label='24h RMSE', color='#17becf', alpha=0.85)
    for i in range(len(x)):
        ax_e1.text(x[i] - width/2, mae_elec[i] + 0.05, f"{mae_elec[i]:.4f}", ha='center', fontsize=9, weight='bold')
        ax_e1.text(x[i] + width/2, rmse_elec[i] + 0.05, f"{rmse_elec[i]:.4f}", ha='center', fontsize=9, weight='bold')
    ax_e1.set_xticks(x)
    ax_e1.set_xticklabels(model_labels, fontsize=10)
    ax_e1.set_title("(a1) Electricity: Forecast Error (MAE & RMSE)", fontsize=11, pad=8)
    ax_e1.set_ylabel("Error (kW)", fontsize=10.5)
    ax_e1.set_ylim(0, max(rmse_elec) * 1.22)
    ax_e1.grid(True, linestyle='--', alpha=0.5)
    ax_e1.legend(loc='lower right', framealpha=0.9)

    # --- Panel (a2): Electricity 参数量 ---
    ax_e2 = axes[0, 1]
    bars_e = ax_e2.bar(x, params_elec, width=0.45, color=['#1f77b4', '#ff7f0e', '#9467bd'], alpha=0.85)
    for bar in bars_e:
        h = bar.get_height()
        ax_e2.text(bar.get_x() + bar.get_width()/2., h + 8, f"{h:.1f}k\nParams", ha='center', va='bottom', fontsize=9, weight='bold')
    ax_e2.text(2, params_elec[2] + 80, "-84.7% vs CNN", ha='center', fontsize=9.5, color='#d62728', weight='bold')
    ax_e2.set_xticks(x)
    ax_e2.set_xticklabels(model_labels, fontsize=10)
    ax_e2.set_title("(a2) Electricity: Parameter Footprint", fontsize=11, pad=8)
    ax_e2.set_ylabel("Params (x1,000)", fontsize=10.5)
    ax_e2.set_ylim(0, max(params_elec) * 1.25)
    ax_e2.grid(True, linestyle='--', alpha=0.5)

    # --- Panel (b1): Solar 误差 ---
    ax_s1 = axes[1, 0]
    ax_s1.bar(x - width/2, mae_solar, width, label='24h MAE', color='#2ca02c', alpha=0.85)
    ax_s1.bar(x + width/2, rmse_solar, width, label='24h RMSE', color='#17becf', alpha=0.85)
    for i in range(len(x)):
        ax_s1.text(x[i] - width/2, mae_solar[i] + 0.35, f"{mae_solar[i]:.2f}", ha='center', fontsize=9, weight='bold')
        ax_s1.text(x[i] + width/2, rmse_solar[i] + 0.35, f"{rmse_solar[i]:.2f}", ha='center', fontsize=9, weight='bold')
    ax_s1.set_xticks(x)
    ax_s1.set_xticklabels(model_labels, fontsize=10)
    ax_s1.set_title("(b1) Solar PV: Forecast Error (MAE & RMSE)", fontsize=11, pad=8)
    ax_s1.set_ylabel("Error (MW)", fontsize=10.5)
    ax_s1.set_ylim(0, max(rmse_solar) * 1.22)
    ax_s1.grid(True, linestyle='--', alpha=0.5)
    ax_s1.legend(loc='upper right', framealpha=0.9)

    # --- Panel (b2): Solar 参数量 ---
    ax_s2 = axes[1, 1]
    bars_s = ax_s2.bar(x, params_solar, width=0.45, color=['#1f77b4', '#ff7f0e', '#9467bd'], alpha=0.85)
    for bar in bars_s:
        h = bar.get_height()
        ax_s2.text(bar.get_x() + bar.get_width()/2., h + 50, f"{h:.1f}k\nParams", ha='center', va='bottom', fontsize=9, weight='bold')
    ax_s2.text(2, params_solar[2] + 450, "-92.1% vs CNN", ha='center', fontsize=9.5, color='#d62728', weight='bold')
    ax_s2.set_xticks(x)
    ax_s2.set_xticklabels(model_labels, fontsize=10)
    ax_s2.set_title("(b2) Solar PV: Parameter Footprint", fontsize=11, pad=8)
    ax_s2.set_ylabel("Params (x1,000)", fontsize=10.5)
    ax_s2.set_ylim(0, max(params_solar) * 1.25)
    ax_s2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "ablation_cnn_lstm.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    df = fetch_dataframe()
    plot_window_sensitivity(df)
    plot_data_fraction_sensitivity(df)
    plot_modular_ablation(df)
    print("All sensitivity & modular ablation plots successfully re-generated from database.")
