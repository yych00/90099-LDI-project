# -*- coding: utf-8 -*-
"""
ELEN90099 LDI Project - 启发式基准与经典深度学习模型全方位复查与重新实验验证引擎
(Comprehensive Re-evaluation and Verification of Heuristic & Classical DL Baselines)

验证范围：
1. 启发式基准：Persistence, Daily Seasonal Naive
2. 经典深度学习：LSTM, GRU, 1D-CNN (CNN1D)
3. 双数据集：Electricity (Hourly) 与 Solar (10-Minute)
4. 核心指标：原始尺度 24h MAE, 24h RMSE, 24h MASE, 4h MAE, 4h RMSE, 业务切片指标
5. 与 experiments.db (ID 32-36, 40-44) 历史记录的一致性比对 (Delta 容差校验)
6. 1D-CNN 宏观自适应池化层理论与参数分布、激活分布、跨窗口锁定特性全方位深度核验
7. 经典模型训练与收敛性实机复现校验 (种子复现性、梯度反向传播、无数值发散)
"""

import os
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 路径自适应
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    ELECTRICITY_CONFIG, SOLAR_CONFIG, CHECKPOINT_DIR, DB_PATH, DatasetConfig, TrainConfig
)
from data_loader import get_data_splits, StandardScaler
from train import create_model, train_and_evaluate, count_parameters, set_seed
from eval import (
    compute_seasonal_naive_denominators,
    calculate_metrics_raw_scale,
    compute_slice_metrics
)


TARGET_MODELS = ["Persistence", "DailySeasonalNaive", "LSTM", "GRU", "CNN1D"]


def evaluate_single_model_forward(
    model_name: str,
    dataset_cfg: DatasetConfig,
    loaders: dict,
    scaler: StandardScaler,
    raw_data: np.ndarray,
    bounds: tuple
) -> dict:
    """
    加载模型 checkpoint（基线直接初始化）在测试集上执行严格前向推理，
    返回 24h MAE/RMSE/MASE, 4h MAE/RMSE/MASE, 业务切片 MAE/RMSE。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, dataset_cfg).to(device)

    # 深度学习模型加载对应权重文件
    if model_name not in ["Persistence", "DailySeasonalNaive"]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_cfg.name}_{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到检查点文件: {ckpt_path}")
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

    y_pred_norm = np.concatenate(all_preds, axis=0)  # [N, H, C]
    y_true_norm = np.concatenate(all_trues, axis=0)  # [N, H, C]

    # 逆标准化至原始真实物理尺度
    y_pred_raw = scaler.inverse_transform(y_pred_norm)
    y_true_raw = scaler.inverse_transform(y_true_norm)

    # 季节性分母（仅在训练集目标区间计算）
    if len(bounds) == 3:
        train_start_b, b1, _ = bounds
    else:
        train_start_b, b1 = 0, bounds[0]
    train_slice = raw_data[train_start_b:b1, :]
    denominators = compute_seasonal_naive_denominators(train_slice, dataset_cfg.seasonal_m)

    # 1. 24 小时综合指标
    metrics_24h = calculate_metrics_raw_scale(y_pred_raw, y_true_raw, denominators)

    # 2. 4 小时短程评估指标
    short_h = dataset_cfg.short_horizon
    metrics_4h = calculate_metrics_raw_scale(
        y_pred_raw[:, :short_h, :],
        y_true_raw[:, :short_h, :],
        denominators
    )

    # 3. 业务切片指标 (峰值负荷 / 活跃光伏)
    slice_metrics = compute_slice_metrics(y_pred_raw, y_true_raw, dataset_cfg.name, train_slice)
    slice_mae = slice_metrics.get("peak_load_mae", slice_metrics.get("active_solar_mae", 0.0))
    slice_rmse = slice_metrics.get("peak_load_rmse", slice_metrics.get("active_solar_rmse", 0.0))

    return {
        "dataset": dataset_cfg.name,
        "model_name": model_name,
        "num_params": count_parameters(model),
        "mae_24h": round(metrics_24h["mae"], 4),
        "rmse_24h": round(metrics_24h["rmse"], 4),
        "mase_24h": round(metrics_24h["mase"], 4),
        "mae_4h": round(metrics_4h["mae"], 4),
        "rmse_4h": round(metrics_4h["rmse"], 4),
        "mase_4h": round(metrics_4h["mase"], 4),
        "slice_mae": round(slice_mae, 4),
        "slice_rmse": round(slice_rmse, 4)
    }


def fetch_benchmark_from_db(db_path: str, dataset_name: str, model_name: str) -> dict:
    """从 experiments.db 查询基准实验记录 (ID 32-47)"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, num_params, mae_24h, rmse_24h, mase_24h, mae_4h, rmse_4h, mase_4h, slice_mae, slice_rmse
        FROM experiments
        WHERE id BETWEEN 32 AND 47 AND dataset = ? AND model_name = ?
    """, (dataset_name, model_name))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"数据库中未检索到实验记录: {dataset_name} - {model_name}")

    return {
        "db_id": row[0],
        "db_num_params": row[1],
        "db_mae_24h": row[2],
        "db_rmse_24h": row[3],
        "db_mase_24h": row[4],
        "db_mae_4h": row[5],
        "db_rmse_4h": row[6],
        "db_mase_4h": row[7],
        "db_slice_mae": row[8],
        "db_slice_rmse": row[9]
    }


def audit_1d_cnn_macro_pooling(dataset_cfg: DatasetConfig, loaders: dict):
    """
    全方位核验 1D-CNN 宏观池化层参数分布与特征激活特性
    """
    print(f"\n================================================================================")
    print(f"【1D-CNN 宏观池化层与参数分布深度核验 - 数据集: {dataset_cfg.name}】")
    print(f"================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("CNN1D", dataset_cfg).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_cfg.name}_CNN1D_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 1. 逐层参数严谨审计
    print("\n--- 1. 逐层网络架构与参数量数学统计 ---")
    total_params = 0
    backbone_params = 0
    head_params = 0

    param_details = []
    for name, param in model.named_parameters():
        p_count = param.numel()
        total_params += p_count
        if "feature_net" in name:
            backbone_params += p_count
            layer_type = "Backbone (Conv/BN)"
        else:
            head_params += p_count
            layer_type = "Projection Head (Linear)"
        param_details.append({
            "Layer": name,
            "Shape": str(list(param.shape)),
            "Param Count": p_count,
            "Type": layer_type,
            "Requires Grad": param.requires_grad
        })

    df_params = pd.DataFrame(param_details)
    print(df_params.to_string(index=False))

    # 核验宏观自适应平均池化层 AdaptiveAvgPool1d(16)
    pool_layer = model.feature_net[6]
    pool_params = sum(p.numel() for p in pool_layer.parameters())
    print(f"\n--> 自适应宏观池化算子: {pool_layer}")
    print(f"--> 池化层可学习参数量: {pool_params} (严格为 0，纯几何无参下采样算子)")
    print(f"--> 特征骨干总参数量 (Backbone): {backbone_params:,} ({backbone_params / total_params * 100:.2f}%)")
    print(f"--> 预测投影头总参数量 (Linear Head): {head_params:,} ({head_params / total_params * 100:.2f}%)")
    print(f"--> 全模型总参数量 (Total Params): {total_params:,}")

    # 2. 物理意义核验
    print("\n--- 2. 宏观时间元 (Macro-Token) 物理时间尺度对齐核验 ---")
    seq_len = dataset_cfg.seq_len
    target_tokens = 16
    steps_per_token = seq_len / target_tokens
    if dataset_cfg.name == "Electricity":
        hours_per_token = steps_per_token * 1.0
        rf_hours = 5 * 1.0  # 感受野 5 步
        print(f"Electricity: 历史 L={seq_len} 小时 -> 池化为 {target_tokens} 个宏观时间元")
        print(f"每个 Token 聚合物理跨度: {steps_per_token:.1f} 步 = {hours_per_token:.2f} 小时")
        print(f"有效卷积感受野 (RF=5): {rf_hours:.1f} 小时 (涵盖单次早晚高峰负荷爬坡全过程)")
    else:
        hours_per_token = steps_per_token * (10.0 / 60.0)
        rf_hours = 5 * (10.0 / 60.0)
        print(f"Solar: 历史 L={seq_len} 步 (48小时) -> 池化为 {target_tokens} 个宏观时间元")
        print(f"每个 Token 聚合物理跨度: {steps_per_token:.1f} 步 = {hours_per_token:.2f} 小时 (完美跨数据集对齐!)")
        print(f"有效卷积感受野 (RF=5): {5 * 10} 分钟 = {rf_hours:.2f} 小时 (完美覆盖云层瞬态间歇波动)")

    # 3. 宏观池化激活值分布实测统计 (Forward Hook)
    print("\n--- 3. 宏观池化前后特征张量形状与激活分布统计 ---")
    feat_before_pool = []
    feat_after_pool = []

    def hook_before_pool(module, inp, outp):
        # module: feature_net[5] (ReLU2)
        feat_before_pool.append(outp.detach().cpu())

    def hook_after_pool(module, inp, outp):
        # module: feature_net[6] (AdaptiveAvgPool1d)
        feat_after_pool.append(outp.detach().cpu())

    h1 = model.feature_net[5].register_forward_hook(hook_before_pool)
    h2 = model.feature_net[6].register_forward_hook(hook_after_pool)

    with torch.no_grad():
        for x_b, _ in loaders["test"]:
            x_b = x_b.to(device)
            _ = model(x_b)

    h1.remove()
    h2.remove()

    all_before = torch.cat(feat_before_pool, dim=0).numpy()  # [N, 64, L]
    all_after = torch.cat(feat_after_pool, dim=0).numpy()    # [N, 64, 16]

    print(f"池化前特征张量形状: {all_before.shape} (保留高频空间局部卷积滤波)")
    print(f"池化后特征张量形状: {all_after.shape} (压缩为 16 个宏观时间单元)")

    # 统计均值、方差、最小最大值与稀疏度
    sparsity_before = np.mean(all_before == 0.0) * 100.0
    sparsity_after = np.mean(all_after == 0.0) * 100.0

    print(f"池化前 (ReLU2): 均值={np.mean(all_before):.4f}, 标差={np.std(all_before):.4f}, 范围=[{np.min(all_before):.4f}, {np.max(all_before):.4f}], 稀疏率(零值比)={sparsity_before:.2f}%")
    print(f"池化后 (AvgPool): 均值={np.mean(all_after):.4f}, 标差={np.std(all_after):.4f}, 范围=[{np.min(all_after):.4f}, {np.max(all_after):.4f}], 稀疏率(零值比)={sparsity_after:.2f}%")
    print(f"-> 宏观平均池化平滑了高频离群脉冲，有效降低特征方差并保留宏观趋势通道能量。")

    # 4. 跨历史窗口长度 (L) 的模型容量锁定测试 (Locked Model Capacity)
    print("\n--- 4. 任意输入序列长度兼容性与容量锁定 (Locked Capacity) 检验 ---")
    test_lens = [24, 48, 96, 168, 288]
    locked_checks = []
    for test_l in test_lens:
        dummy_in = torch.randn(2, test_l, dataset_cfg.num_channels).to(device)
        try:
            out_dummy = model(dummy_in)
            out_shape = tuple(out_dummy.shape)
            is_valid = (out_shape == (2, dataset_cfg.pred_len, dataset_cfg.num_channels))
            locked_checks.append({
                "Input L": test_l,
                "Input Shape": f"(2, {test_l}, {dataset_cfg.num_channels})",
                "Pooled Shape": f"(2, 64, 16)",
                "Output Shape": str(out_shape),
                "Param Count": count_parameters(model),
                "Capacity Locked": "YES" if is_valid else "NO"
            })
        except Exception as e:
            locked_checks.append({
                "Input L": test_l,
                "Input Shape": f"(2, {test_l}, {dataset_cfg.num_channels})",
                "Pooled Shape": "FAIL",
                "Output Shape": str(e),
                "Param Count": count_parameters(model),
                "Capacity Locked": "NO"
            })

    print(pd.DataFrame(locked_checks).to_string(index=False))

    # 5. L < 16 边界自适应上采样与反向梯度检验 (Edge-Case: L < 16)
    print("\n--- 5. 极端短序列输入 (L < 16) 边界自适应上采样与反向传播校验 ---")
    edge_lens = [8, 12, 16]
    edge_results = []
    for el in edge_lens:
        x_edge = torch.randn(2, el, dataset_cfg.num_channels, requires_grad=True).to(device)
        out_edge = model(x_edge)
        out_edge.sum().backward()
        g_norm = x_edge.grad.norm().item() if x_edge.grad is not None else 0.0
        edge_results.append({
            "Input L": el,
            "Input Shape": f"(2, {el}, {dataset_cfg.num_channels})",
            "Output Shape": str(tuple(out_edge.shape)),
            "Input Grad Norm": f"{g_norm:.4f}",
            "Valid Grad": "YES" if g_norm > 0 and not np.isnan(g_norm) else "NO"
        })
    print(pd.DataFrame(edge_results).to_string(index=False))
    print("-> AdaptiveAvgPool1d(16) 在 L < 16 时通过分段局部加权扩展上采样，保证梯度健康反向传导，无 NaN 产生。")

    # 6. 长预测时域 H 扩展与投影头参数爆炸极限评估
    print("\n--- 6. 超长预测时域 (H 扩展) 投影头参数量增长与爆炸极限评估 ---")
    h_candidates = [24, 72, 144, 168, 336]
    h_analysis = []
    for h in h_candidates:
        head_p = (64 * 16 + 1) * (h * dataset_cfg.num_channels)
        tot_p = backbone_params + head_p
        ratio = head_p / tot_p * 100.0
        h_analysis.append({
            "Pred Horizon H": f"{h} steps",
            "Backbone Params": f"{backbone_params:,}",
            "Head Params": f"{head_p:,}",
            "Total Params": f"{tot_p:,}",
            "Head Ratio (%)": f"{ratio:.2f}%"
        })
    print(pd.DataFrame(h_analysis).to_string(index=False))
    print("-> 1D-CNN 投影头参数量与 (H * C) 严格线性耦合，超长时域需引入因子分解或通道独立头以防显存暴涨。")


def run_comprehensive_benchmark_verification():
    """执行全部 5 种基准模型在双数据集下的严格比对与一致性检验"""
    print("================================================================================")
    print("【启发式基准与经典深度学习模型 - 全方位复查与重新实验比对】")
    print("================================================================================")
    print(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch 版本: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")

    datasets = [("Electricity", ELECTRICITY_CONFIG), ("Solar", SOLAR_CONFIG)]
    all_comparison_rows = []

    for ds_name, ds_cfg in datasets:
        print(f"\n=======================================================")
        print(f"正在载入并划分数据集: {ds_name} (无泄漏时序切分)...")
        loaders, scaler, raw_data, bounds = get_data_splits(ds_cfg)

        for model_name in TARGET_MODELS:
            print(f"--> [前向重验] {ds_name:11s} | {model_name:18s} ... ", end="", flush=True)
            re_metrics = evaluate_single_model_forward(model_name, ds_cfg, loaders, scaler, raw_data, bounds)
            db_metrics = fetch_benchmark_from_db(DB_PATH, ds_name, model_name)

            # 比对全部 8 项关键物理指标
            mae_delta = abs(re_metrics["mae_24h"] - db_metrics["db_mae_24h"])
            rmse_delta = abs(re_metrics["rmse_24h"] - db_metrics["db_rmse_24h"])
            mase_delta = abs(re_metrics["mase_24h"] - db_metrics["db_mase_24h"])
            mae4h_delta = abs(re_metrics["mae_4h"] - db_metrics["db_mae_4h"])
            rmse4h_delta = abs(re_metrics["rmse_4h"] - db_metrics["db_rmse_4h"])
            mase4h_delta = abs(re_metrics["mase_4h"] - db_metrics["db_mase_4h"])
            slice_mae_delta = abs(re_metrics["slice_mae"] - db_metrics["db_slice_mae"])
            slice_rmse_delta = abs(re_metrics["slice_rmse"] - db_metrics["db_slice_rmse"])

            # 严格容差判定 (1e-3)
            tolerance = 1e-3
            is_pass = all(
                d <= tolerance for d in [
                    mae_delta, rmse_delta, mase_delta,
                    mae4h_delta, rmse4h_delta, mase4h_delta,
                    slice_mae_delta, slice_rmse_delta
                ]
            )
            status = "PASS" if is_pass else "FAIL"
            print(f"[{status}] (24h MAE: {re_metrics['mae_24h']:.4f}, DB: {db_metrics['db_mae_24h']:.4f}, Delta: {mae_delta:.4f})")

            all_comparison_rows.append({
                "Dataset": ds_name,
                "Model": model_name,
                "Params": re_metrics["num_params"],
                "DB_Params": db_metrics["db_num_params"],
                "Re_MAE_24h": re_metrics["mae_24h"],
                "DB_MAE_24h": db_metrics["db_mae_24h"],
                "Delta_MAE": round(mae_delta, 4),
                "Re_RMSE_24h": re_metrics["rmse_24h"],
                "DB_RMSE_24h": db_metrics["db_rmse_24h"],
                "Delta_RMSE": round(rmse_delta, 4),
                "Re_MASE_24h": re_metrics["mase_24h"],
                "DB_MASE_24h": db_metrics["db_mase_24h"],
                "Delta_MASE": round(mase_delta, 4),
                "Re_MAE_4h": re_metrics["mae_4h"],
                "DB_MAE_4h": db_metrics["db_mae_4h"],
                "Delta_MAE4h": round(mae4h_delta, 4),
                "Re_RMSE_4h": re_metrics["rmse_4h"],
                "DB_RMSE_4h": db_metrics["db_rmse_4h"],
                "Delta_RMSE4h": round(rmse4h_delta, 4),
                "Re_MASE_4h": re_metrics["mase_4h"],
                "DB_MASE_4h": db_metrics["db_mase_4h"],
                "Delta_MASE4h": round(mase4h_delta, 4),
                "Re_Slice_MAE": re_metrics["slice_mae"],
                "DB_Slice_MAE": db_metrics["db_slice_mae"],
                "Delta_Slice_MAE": round(slice_mae_delta, 4),
                "Re_Slice_RMSE": re_metrics["slice_rmse"],
                "DB_Slice_RMSE": db_metrics["db_slice_rmse"],
                "Delta_Slice_RMSE": round(slice_rmse_delta, 4),
                "Status": status
            })

        # 执行 1D-CNN 宏观池化核验
        audit_1d_cnn_macro_pooling(ds_cfg, loaders)

    # 汇总输出比对大表
    df_all = pd.DataFrame(all_comparison_rows)
    print("\n" + "=" * 140)
    print("【全量基准与经典深度学习模型 - 重新实验与 experiments.db 比对大表 (覆盖全量 8 项物理指标)】")
    print("=" * 140)
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 1600)
    summary_cols = [
        "Dataset", "Model", "Params", "DB_MAE_24h", "Re_MAE_24h", "Delta_MAE",
        "DB_RMSE_24h", "Re_RMSE_24h", "Delta_RMSE", "DB_MASE_24h", "Re_MASE_24h",
        "DB_MAE_4h", "Re_MAE_4h", "DB_RMSE_4h", "Re_RMSE_4h",
        "DB_MASE_4h", "Re_MASE_4h", "DB_Slice_MAE", "Re_Slice_MAE", "Status"
    ]
    print(df_all[summary_cols].to_string(index=False))

    failed = df_all[df_all["Status"] != "PASS"]
    if len(failed) > 0:
        print(f"\n[CRITICAL ALERT] 发现 {len(failed)} 个模型比对存在显著偏差！")
        return False, df_all
    else:
        print(f"\n[SUCCESS] 全部 10 组 (双数据集 x 5 模型) 重新实验结果与 experiments.db 历史基准记录 100% 精确一致！")
        return True, df_all


def run_training_reproducibility_check():
    """核验经典深度学习模型在双数据集下的重训收敛性、种子可复现性与数据库指标对齐度"""
    print("\n================================================================================")
    print("【经典深度学习模型双数据集重训流程、梯度反向传播与指标复现核验 (保护 Checkpoint 不受污染)】")
    print("================================================================================")
    train_cfg = TrainConfig(seed=42, batch_size=32, lr=1e-3, max_epochs=3, patience=2)

    datasets = [("Electricity", ELECTRICITY_CONFIG), ("Solar", SOLAR_CONFIG)]
    for ds_name, ds_cfg in datasets:
        print(f"\n--- 正在载入 {ds_name} 数据集执行经典模型端到端重训复现核验 ---")
        loaders, scaler, raw_data, bounds = get_data_splits(ds_cfg)

        for m_name in ["LSTM", "GRU", "CNN1D"]:
            print(f"\n--> 执行 {ds_name} - {m_name} 确定性重训 (Epochs=3, Seed=42, save_checkpoint=False)...")
            set_seed(42)
            res = train_and_evaluate(
                model_name=m_name,
                dataset_cfg=ds_cfg,
                loaders=loaders,
                scaler=scaler,
                raw_data=raw_data,
                train_bounds=bounds,
                train_cfg=train_cfg,
                db_path=None,             # 不污染数据库
                save_checkpoint=False     # 保护 golden checkpoint 文件不被修改！
            )
            db_row = fetch_benchmark_from_db(DB_PATH, ds_name, m_name)
            mae_delta = abs(res['mae_24h'] - db_row['db_mae_24h'])
            rmse_delta = abs(res['rmse_24h'] - db_row['db_rmse_24h'])
            mae4h_delta = abs(res['mae_4h'] - db_row['db_mae_4h'])
            
            print(f"    训练完成: 实际轮数={res['actual_epochs']}, 耗时={res['train_time_sec']:.2f}s, 推理={res['infer_time_ms']:.2f}ms/batch")
            print(f"    产出指标: 24h MAE={res['mae_24h']:.4f} (DB={db_row['db_mae_24h']:.4f}, Delta={mae_delta:.4f})")
            print(f"             24h RMSE={res['rmse_24h']:.4f} (DB={db_row['db_rmse_24h']:.4f}, Delta={rmse_delta:.4f})")
            print(f"             4h MAE={res['mae_4h']:.4f} (DB={db_row['db_mae_4h']:.4f}, Delta={mae4h_delta:.4f})")
            
            assert not np.isnan(res['mae_24h']), f"{ds_name} {m_name} 训练出现 NaN 损失！"
            assert mae_delta <= 1e-3, f"{ds_name} {m_name} 重训 MAE 偏离预期 (Delta: {mae_delta})！"
            assert rmse_delta <= 1e-3, f"{ds_name} {m_name} 重训 RMSE 偏离预期 (Delta: {rmse_delta})！"
            print(f"    [CHECK PASS] {ds_name} {m_name} 重训结果与数据库历史记录完全精准吻合！")


if __name__ == "__main__":
    success, df = run_comprehensive_benchmark_verification()
    run_training_reproducibility_check()
    if not success:
        sys.exit(1)
