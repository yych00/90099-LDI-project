# -*- coding: utf-8 -*-
"""
实验交叉复现与比对验证工具 (Experimental Cross-Verification and Comparison Engine)
功能：
1. 加载 checkpoints/ 目录下的各模型最优权重或执行启发式基准；
2. 在测试集上进行原尺度前向推理，计算完整 24h MAE/RMSE/MASE、4h MAE 及业务切片 MAE；
3. 自动化连接 experiments.db，比对重新实验评测值与数据库存储元数据；
4. 输出逐项误差差值 (Delta) 与合规断言状态，支持独立运行与单元测试集成。
"""

import os
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional

# 确保导入模块路径正确
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    ELECTRICITY_CONFIG, SOLAR_CONFIG, CHECKPOINT_DIR, DB_PATH, DatasetConfig, TrainConfig
)
from data_loader import get_data_splits, StandardScaler
from train import create_model, train_and_evaluate
from eval import (
    compute_seasonal_naive_denominators,
    calculate_metrics_raw_scale,
    compute_slice_metrics
)


def get_resolved_db_path(db_path: Optional[str] = None) -> str:
    """自动探测数据库路径，支持 src 内部或上一级目录"""
    if db_path and os.path.exists(db_path):
        return db_path
    if os.path.exists(DB_PATH):
        return DB_PATH
    local_db = os.path.join(CURRENT_DIR, "experiments.db")
    if os.path.exists(local_db):
        return local_db
    return DB_PATH


def get_resolved_ckpt_path(dataset_name: str, model_name: str) -> str:
    """自动探测模型检查点路径"""
    path1 = os.path.join(CHECKPOINT_DIR, f"{dataset_name}_{model_name}_best.pt")
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(CURRENT_DIR, "checkpoints", f"{dataset_name}_{model_name}_best.pt")
    if os.path.exists(path2):
        return path2
    return path1


def evaluate_model_on_test(
    model_name: str,
    dataset_cfg: DatasetConfig,
    loaders: Dict[str, Any],
    scaler: StandardScaler,
    raw_data: np.ndarray,
    bounds: Tuple[int, ...],
    retrain: bool = False
) -> Dict[str, float]:
    """对单个模型在测试集上执行前向推理并计算物理原尺度指标，支持重新训练 (--retrain) 模式"""
    if retrain and model_name not in ["Persistence", "DailySeasonalNaive"]:
        # 执行基准受控重新训练实验 (3 epochs, patience=2, seed=42)
        train_cfg = TrainConfig(max_epochs=3, patience=2, seed=42)
        res = train_and_evaluate(
            model_name=model_name,
            dataset_cfg=dataset_cfg,
            loaders=loaders,
            scaler=scaler,
            raw_data=raw_data,
            train_bounds=bounds,
            train_cfg=train_cfg,
            db_path=None,
            save_checkpoint=False
        )
        slice_mae = res.get("slice_mae", 0.0)
        return {
            "mae_24h": round(res["mae_24h"], 4),
            "rmse_24h": round(res["rmse_24h"], 4),
            "mase_24h": round(res["mase_24h"], 4),
            "mae_4h": round(res["mae_4h"], 4),
            "slice_mae": round(slice_mae, 4)
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, dataset_cfg).to(device)

    # 深度学习模型加载对应权重文件
    if model_name not in ["Persistence", "DailySeasonalNaive"]:
        ckpt_path = get_resolved_ckpt_path(dataset_cfg.name, model_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到检查点文件: {ckpt_path}。若需从头重新训练，请使用 --retrain 参数。")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_test, y_test in loaders["test"]:
            x_test = x_test.to(device)
            preds = model(x_test)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(y_test.numpy())

    y_pred_norm = np.concatenate(all_preds, axis=0)
    y_true_norm = np.concatenate(all_trues, axis=0)

    # 反标准化回真实物理单位
    y_pred_raw = scaler.inverse_transform(y_pred_norm)
    y_true_raw = scaler.inverse_transform(y_true_norm)

    # 季节性分母（仅训练集目标区间）
    if len(bounds) == 3:
        train_start_b, b1, _ = bounds
    else:
        train_start_b, b1 = 0, bounds[0]
    train_slice = raw_data[train_start_b:b1, :]
    denominators = compute_seasonal_naive_denominators(train_slice, dataset_cfg.seasonal_m)

    # 计算 24h 指标
    metrics_24h = calculate_metrics_raw_scale(y_pred_raw, y_true_raw, denominators)

    # 计算 4h 指标
    short_h = dataset_cfg.short_horizon
    metrics_4h = calculate_metrics_raw_scale(
        y_pred_raw[:, :short_h, :],
        y_true_raw[:, :short_h, :],
        denominators
    )

    # 计算业务切片指标 (峰值负荷 / 活跃光伏)
    slice_metrics = compute_slice_metrics(y_pred_raw, y_true_raw, dataset_cfg.name, train_slice)
    slice_mae = slice_metrics.get("peak_load_mae", slice_metrics.get("active_solar_mae", 0.0))

    return {
        "mae_24h": round(metrics_24h["mae"], 4),
        "rmse_24h": round(metrics_24h["rmse"], 4),
        "mase_24h": round(metrics_24h["mase"], 4),
        "mae_4h": round(metrics_4h["mae"], 4),
        "slice_mae": round(slice_mae, 4)
    }


def fetch_db_benchmark_metrics(db_path: str, dataset_name: str, model_name: str) -> Dict[str, float]:
    """从 experiments.db 查询基准实验存储指标，与报告 Table 5.1 & 5.2 严格对齐"""
    resolved_db = get_resolved_db_path(db_path)
    if not os.path.exists(resolved_db):
        raise FileNotFoundError(f"数据库文件未找到: {resolved_db}")

    conn = sqlite3.connect(resolved_db)
    cur = conn.cursor()
    # 优先查询报告基准运行 (ID 32-47 严格对应主对比表)
    cur.execute("""
        SELECT mae_24h, rmse_24h, mase_24h, mae_4h, slice_mae
        FROM experiments
        WHERE id BETWEEN 32 AND 47 AND dataset = ? AND model_name = ?
    """, (dataset_name, model_name))
    row = cur.fetchone()

    # 备选查询：若 ID 32-47 不在此数据库，按全量基准配置查询最新一条
    if not row:
        cur.execute("""
            SELECT mae_24h, rmse_24h, mase_24h, mae_4h, slice_mae
            FROM experiments
            WHERE dataset = ? AND model_name = ? AND train_fraction = 1.0
            ORDER BY id DESC LIMIT 1
        """, (dataset_name, model_name))
        row = cur.fetchone()

    conn.close()

    if not row:
        raise ValueError(f"数据库中未检索到实验记录: {dataset_name} - {model_name}")

    return {
        "mae_24h": row[0],
        "rmse_24h": row[1],
        "mase_24h": row[2],
        "mae_4h": row[3],
        "slice_mae": row[4]
    }


def run_cross_verification(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    tolerance: float = 1e-3,
    retrain: bool = False
) -> pd.DataFrame:
    """
    全量重新实验与数据库比对验证
    tolerance: 允许的最大绝对差值，默认为 1e-3 (0.001)
    retrain: 是否重新从头训练模型进行比对
    """
    if datasets is None:
        datasets = ["Electricity", "Solar"]
    all_models = [
        "Persistence", "DailySeasonalNaive", "LSTM", "GRU",
        "CNN1D", "CNNLSTM", "DLinear", "PatchTST"
    ]
    if models is None:
        models = all_models

    records = []

    for ds_name in datasets:
        cfg = ELECTRICITY_CONFIG if ds_name == "Electricity" else SOLAR_CONFIG
        print(f"\n=======================================================")
        print(f"正在准备数据集与防泄漏切分: {ds_name}...")
        loaders, scaler, raw_data, bounds = get_data_splits(cfg)

        for m_name in models:
            action_tag = "重训并推演" if retrain and m_name not in ["Persistence", "DailySeasonalNaive"] else "重验前向推演"
            print(f"--> [{action_tag}] 数据集: {ds_name:11s} | 模型: {m_name:18s} ... ", end="", flush=True)
            re_metrics = evaluate_model_on_test(m_name, cfg, loaders, scaler, raw_data, bounds, retrain=retrain)
            db_metrics = fetch_db_benchmark_metrics(DB_PATH, ds_name, m_name)

            # 比对关键指标
            mae_delta = abs(re_metrics["mae_24h"] - db_metrics["mae_24h"])
            rmse_delta = abs(re_metrics["rmse_24h"] - db_metrics["rmse_24h"])
            mase_delta = abs(re_metrics["mase_24h"] - db_metrics["mase_24h"])
            mae4h_delta = abs(re_metrics["mae_4h"] - db_metrics["mae_4h"])
            slice_delta = abs(re_metrics["slice_mae"] - db_metrics["slice_mae"])

            is_pass = all(d <= tolerance for d in [mae_delta, rmse_delta, mase_delta, mae4h_delta, slice_delta])
            status = "PASS" if is_pass else "FAIL"

            print(f"[{status}] (MAE Delta: {mae_delta:.4f}, MASE Delta: {mase_delta:.4f})")

            records.append({
                "Dataset": ds_name,
                "Model": m_name,
                "DB_MAE_24h": db_metrics["mae_24h"],
                "Re_MAE_24h": re_metrics["mae_24h"],
                "Delta_MAE": round(mae_delta, 4),
                "DB_RMSE_24h": db_metrics["rmse_24h"],
                "Re_RMSE_24h": re_metrics["rmse_24h"],
                "DB_MASE_24h": db_metrics["mase_24h"],
                "Re_MASE_24h": re_metrics["mase_24h"],
                "Delta_MASE": round(mase_delta, 4),
                "DB_MAE_4h": db_metrics["mae_4h"],
                "Re_MAE_4h": re_metrics["mae_4h"],
                "DB_Slice_MAE": db_metrics["slice_mae"],
                "Re_Slice_MAE": re_metrics["slice_mae"],
                "Status": status
            })

    df_result = pd.DataFrame(records)
    return df_result


def main():
    parser = argparse.ArgumentParser(description="ELEN90099 LDI 重新实验交叉比对与元数据验证工具")
    parser.add_argument("--dataset", type=str, choices=["Electricity", "Solar", "both"], default="both", help="待验证数据集")
    parser.add_argument("--model", type=str, default=None,
                        choices=["Persistence", "DailySeasonalNaive", "LSTM", "GRU", "CNN1D", "CNNLSTM", "DLinear", "PatchTST"],
                        help="指定待重新实验比对的单一模型 (默认全部模型)")
    parser.add_argument("--quick", action="store_true", help="快速模式：仅验证轻量级与基线模型")
    parser.add_argument("--retrain", action="store_true", help="重新实验模式：从头重新训练模型并评测，验证训练收敛可复现性并与数据库比对")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="判定一致性的最大绝对误差阈值")
    args = parser.parse_args()

    target_datasets = ["Electricity", "Solar"] if args.dataset == "both" else [args.dataset]
    if args.model:
        target_models = [args.model]
    elif args.quick:
        target_models = ["Persistence", "DailySeasonalNaive", "DLinear", "CNN1D"]
    else:
        target_models = None

    mode_str = "重新训练实验并比对" if args.retrain else "权重重载推演并比对"
    print(f"启动 8 智能体全方位交叉实验比对验证流程 (模式: {mode_str})...")
    df_result = run_cross_verification(
        datasets=target_datasets,
        models=target_models,
        tolerance=args.tolerance,
        retrain=args.retrain
    )

    print("\n" + "=" * 90)
    print("交叉重验与数据库比对汇总结果：")
    print("=" * 90)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1200)
    display_cols = ["Dataset", "Model", "DB_MAE_24h", "Re_MAE_24h", "Delta_MAE", "DB_MASE_24h", "Re_MASE_24h", "Delta_MASE", "Status"]
    print(df_result[display_cols].to_string(index=False))

    failed_runs = df_result[df_result["Status"] != "PASS"]
    if len(failed_runs) > 0:
        print(f"\n[ALERT] 存在 {len(failed_runs)} 项比对未通过阈值！")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] 全部 {len(df_result)} 项实验重验指标与数据库完全吻合，最大绝对差值均在 {args.tolerance} 容差以内！")


if __name__ == "__main__":
    main()
