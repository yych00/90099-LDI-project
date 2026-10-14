# -*- coding: utf-8 -*-
"""
主实验驱动脚本：端到端执行全部实验流程并生成图表与数据库记录。
涵盖：
1. 探索性数据分析 (EDA)
2. 主模型对比实验 (Persistence, DailySeasonalNaive, LSTM, GRU, CNN1D, CNN-LSTM, DLinear, PatchTST)
3. 敏感性与消融实验 (历史窗口长度、训练数据量、CNN模块消融)
4. 生成高质量对比图表并保存至 figures/ 目录
5. 导出 LaTeX / Markdown 格式的实验结果表格
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from config import (
    ELECTRICITY_CONFIG, SOLAR_CONFIG, FIGURES_DIR, CHECKPOINT_DIR, DB_PATH, TrainConfig
)
from eda import run_eda
from data_loader import get_data_splits
from train import train_and_evaluate, create_model
from storage import get_experiments_dataframe, init_db
from sensitivity import (
    run_window_sensitivity_experiment,
    run_data_size_sensitivity_experiment,
    run_ablation_experiment
)

# 绘图字体与风格设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_model_comparisons(df_elec: pd.DataFrame, df_solar: pd.DataFrame):
    """生成主模型误差与资源开销对比柱状图"""
    for df, name in [(df_elec, "Electricity"), (df_solar, "Solar")]:
        if df.empty:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        models = df["model_name"].tolist()
        x = np.arange(len(models))
        width = 0.35

        # 误差指标对比
        ax1.bar(x - width/2, df["mae_24h"], width, label="24h MAE", color="#1f77b4", alpha=0.85)
        ax1.bar(x + width/2, df["rmse_24h"], width, label="24h RMSE", color="#ff7f0e", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=30, ha="right")
        ax1.set_title(f"{name} - Forecasting Errors (MAE & RMSE)", fontsize=12)
        ax1.set_ylabel("Error (Original Unit)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        # 计算开销 (参数量与推理延迟)
        color = 'tab:red'
        ax2.set_title(f"{name} - Computational Trade-offs", fontsize=12)
        ax2.set_xlabel("Model")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=30, ha="right")
        ax2.bar(x, df["num_params"] / 1000.0, width=0.4, color="#2ca02c", alpha=0.7, label="Params (x1k)")
        ax2.set_ylabel("Params (x1,000)", color="#2ca02c")
        ax2.tick_params(axis='y', labelcolor="#2ca02c")
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, df["infer_time_ms"], color="tab:purple", marker="o", linewidth=2, label="Infer Latency (ms)")
        ax2_twin.set_ylabel("Inference Latency (ms/batch)", color="tab:purple")
        ax2_twin.tick_params(axis='y', labelcolor="tab:purple")

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f"model_comparison_{name.lower()}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()


def plot_lead_time_curves(results_list: list, dataset_name: str):
    """绘制按预测跨度递增的 Lead-time 误差曲线"""
    plt.figure(figsize=(10, 5))
    for res in results_list:
        model_name = res["model_name"]
        mae_curve = res.get("lead_time_mae", [])
        if mae_curve:
            # 横轴按小时计算
            if dataset_name == "Electricity":
                hours = np.arange(1, len(mae_curve) + 1)
            else:
                hours = np.arange(1, len(mae_curve) + 1) * (10.0 / 60.0) # 转换为小时
            plt.plot(hours, mae_curve, label=model_name, linewidth=1.8, alpha=0.85)

    plt.xlabel("Forecast Horizon / Lead-Time (Hours)")
    plt.ylabel("MAE (Original Scale)")
    plt.title(f"{dataset_name} - Error Progression Over 24-Hour Horizon", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"lead_time_curves_{dataset_name.lower()}.png"), dpi=200)
    plt.close()


def plot_forecast_sample_case(
    model,
    loader,
    scaler,
    dataset_cfg,
    device,
    sample_idx: int = 0
):
    """可视化单个测试样本上的真实值与各模型预测对比"""
    model.eval()
    with torch.no_grad():
        for i, (x_b, y_b) in enumerate(loader):
            if i == 0:
                x_sample = x_b[sample_idx:sample_idx+1].to(device)
                y_sample = y_b[sample_idx:sample_idx+1]
                pred_sample = model(x_sample).cpu().numpy()
                true_sample = y_sample.numpy()
                break

    # 逆标准化
    pred_raw = scaler.inverse_transform(pred_sample)[0]  # [H, C]
    true_raw = scaler.inverse_transform(true_sample)[0]  # [H, C]

    H, C = pred_raw.shape
    plt.figure(figsize=(11, 5))
    if dataset_cfg.name == "Electricity":
        t_axis = np.arange(1, H + 1)
        plt.xlabel("Future Step (Hours)")
    else:
        t_axis = np.arange(1, H + 1) * (10.0 / 60.0)
        plt.xlabel("Future Step (Hours)")

    # 绘制第 0 号通道
    plt.plot(t_axis, true_raw[:, 0], 'k-', label="Ground Truth", linewidth=2.2)
    plt.plot(t_axis, pred_raw[:, 0], 'r--', label=f"Predicted ({model.__class__.__name__})", linewidth=2.0)
    plt.title(f"{dataset_cfg.name} - 24-Hour Forecast Case (Channel {dataset_cfg.channel_ids[0]})", fontsize=12)
    plt.ylabel(f"Value ({dataset_cfg.unit_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"forecast_case_{dataset_cfg.name.lower()}.png"), dpi=200)
    plt.close()


def main(fast_mode: bool = True):
    """
    主实验流程执行入口。
    fast_mode=True 时每个模型训练 2-3 个 epochs 以快速验证全流程与生成数据图表；
    fast_mode=False 时完整训练 20 epochs。
    """
    print("=" * 60)
    print("ELEN90099 LDI 项目 - 全流程预测实验启动")
    print(f"模式: {'Fast Draft 验证模式 (3 Epochs)' if fast_mode else 'Full Training 模式 (20 Epochs)'}")
    print("=" * 60)

    epochs = 3 if fast_mode else 20
    train_cfg = TrainConfig(max_epochs=epochs, patience=2 if fast_mode else 3)
    init_db()

    # 阶段 1: EDA 探索性数据分析
    print("\n>>> 阶段 1: 执行 EDA 探索性数据分析...")
    run_eda(ELECTRICITY_CONFIG)
    run_eda(SOLAR_CONFIG)

    # 阶段 2: 主模型对比实验
    all_models = [
        "Persistence", "DailySeasonalNaive",
        "LSTM", "GRU", "CNN1D", "CNNLSTM",
        "DLinear", "PatchTST"
    ]

    for cfg in [ELECTRICITY_CONFIG, SOLAR_CONFIG]:
        print(f"\n>>> 阶段 2: 正在运行 {cfg.name} 数据集主对比实验...")
        loaders, scaler, raw_data, bounds = get_data_splits(cfg)
        dataset_results = []
        best_dl_model_name = None
        best_dl_mae = float("inf")
        for model_name in all_models:
            print(f"  -> 训练并评估模型: {model_name}...")
            res = train_and_evaluate(
                model_name=model_name,
                dataset_cfg=cfg,
                loaders=loaders,
                scaler=scaler,
                raw_data=raw_data,
                train_bounds=bounds,
                train_cfg=train_cfg
            )
            dataset_results.append(res)
            if model_name not in ["Persistence", "DailySeasonalNaive"] and res["mae_24h"] < best_dl_mae:
                best_dl_mae = res["mae_24h"]
                best_dl_model_name = model_name

        # 绘制该数据集 Lead-time 曲线
        plot_lead_time_curves(dataset_results, cfg.name)

        # 绘制典型 24 小时预测案例对比图 (大纲 7.3: 展示时序预测与物理边界)
        if best_dl_model_name is not None:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{cfg.name}_{best_dl_model_name}_best.pt")
            device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.device == "cuda" else "cpu")
            eval_model = create_model(best_dl_model_name, cfg)
            if os.path.exists(ckpt_path):
                eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            eval_model.to(device)
            plot_forecast_sample_case(eval_model, loaders["test"], scaler, cfg, device, sample_idx=0)

    # 阶段 3: 敏感性与消融实验 (在 Electricity 上进行精细分析)
    print("\n>>> 阶段 3: 运行敏感性与消融实验...")
    print("  -> 实验 A: 历史窗口长度敏感性 (24h, 48h, 168h)...")
    run_window_sensitivity_experiment(ELECTRICITY_CONFIG, models=["LSTM", "PatchTST"], epochs=epochs)

    print("  -> 实验 B: 训练数据量敏感性 (25%, 50%, 100%)...")
    run_data_size_sensitivity_experiment(ELECTRICITY_CONFIG, models=["LSTM", "PatchTST"], fractions=[0.25, 0.50, 1.0], epochs=epochs)

    print("  -> 实验 C: CNN 模块消融实验 (CNN1D vs LSTM vs CNNLSTM)...")
    run_ablation_experiment(ELECTRICITY_CONFIG, epochs=epochs)

    # 阶段 4: 生成对比分析图与结果报告
    print("\n>>> 阶段 4: 汇总实验数据并绘制图表...")
    df_all = get_experiments_dataframe()
    # 筛选主实验记录 (seq_len=48/288 且 train_fraction=1.0)
    df_elec_main = df_all[(df_all["dataset"] == "Electricity") & (df_all["seq_len"] == 48) & (df_all["train_fraction"] == 1.0)].drop_duplicates(subset=["model_name"], keep="last")
    df_solar_main = df_all[(df_all["dataset"] == "Solar") & (df_all["seq_len"] == 288) & (df_all["train_fraction"] == 1.0)].drop_duplicates(subset=["model_name"], keep="last")

    plot_model_comparisons(df_elec_main, df_solar_main)

    print("\n" + "=" * 60)
    print("【Electricity 负荷预测 主实验汇总】")
    cols = ["model_name", "mae_24h", "rmse_24h", "mase_24h", "mae_4h", "rmse_4h", "num_params", "train_time_sec", "infer_time_ms"]
    print(df_elec_main[cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("【Solar 光伏出力预测 主实验汇总】")
    print(df_solar_main[cols].to_string(index=False))

    print("\n所有实验执行完毕！数据库记录保存在:", DB_PATH)
    print("生成图表保存在:", FIGURES_DIR)


if __name__ == "__main__":
    main(fast_mode=True)
