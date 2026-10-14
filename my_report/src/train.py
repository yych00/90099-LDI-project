# -*- coding: utf-8 -*-
"""
模型训练、早停验证与全链路测试评估引擎。
严格执行：
1. 监控验证集 MSE，执行 Early Stopping (patience=3)。
2. 加载最佳验证 Checkpoint，并在测试集进行严格原尺度评测。
3. 统计参数量、实际轮数、训练耗时与推理延迟。
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from config import DatasetConfig, TrainConfig, CHECKPOINT_DIR
from data_loader import StandardScaler
from models import (
    PersistenceModel, DailySeasonalNaiveModel,
    LSTMModel, GRUModel, CNN1DModel,
    CNNLSTMModel, DLinearModel, PatchTSTModel
)
from eval import (
    compute_seasonal_naive_denominators,
    calculate_metrics_raw_scale,
    compute_lead_time_errors,
    compute_slice_metrics
)
from storage import log_experiment_run


def set_seed(seed: int = 42):
    """设置随机种子保证可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(
    model_name: str,
    dataset_cfg: DatasetConfig,
    custom_seq_len: Optional[int] = None
) -> nn.Module:
    """
    模型工厂函数：根据模型名与数据集配置实例化模型
    """
    seq_len = custom_seq_len if custom_seq_len is not None else dataset_cfg.seq_len
    pred_len = dataset_cfg.pred_len
    num_channels = dataset_cfg.num_channels

    if model_name == "Persistence":
        return PersistenceModel(pred_len=pred_len)
    elif model_name == "DailySeasonalNaive":
        return DailySeasonalNaiveModel(pred_len=pred_len, seasonal_m=dataset_cfg.seasonal_m)
    elif model_name == "LSTM":
        return LSTMModel(num_channels=num_channels, seq_len=seq_len, pred_len=pred_len, hidden_size=64, num_layers=1)
    elif model_name == "GRU":
        return GRUModel(num_channels=num_channels, seq_len=seq_len, pred_len=pred_len, hidden_size=64, num_layers=1)
    elif model_name == "CNN1D":
        return CNN1DModel(num_channels=num_channels, seq_len=seq_len, pred_len=pred_len, conv_channels=64, kernel_size=3)
    elif model_name == "CNNLSTM":
        return CNNLSTMModel(num_channels=num_channels, seq_len=seq_len, pred_len=pred_len, conv_channels=64, lstm_hidden=64)
    elif model_name == "DLinear":
        return DLinearModel(num_channels=num_channels, seq_len=seq_len, pred_len=pred_len, moving_avg=25, individual=False)
    elif model_name == "PatchTST":
        # Solar 10min 采样率高，patch_len 设置为 16，stride 为 8
        return PatchTSTModel(
            num_channels=num_channels,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=16,
            stride=8,
            d_model=64,
            n_heads=4,
            e_layers=2,
            d_ff=128
        )
    else:
        raise ValueError(f"未知模型名称: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate(
    model_name: str,
    dataset_cfg: DatasetConfig,
    loaders: Dict[str, Any],
    scaler: StandardScaler,
    raw_data: np.ndarray,
    train_bounds: Tuple[int, ...],
    train_cfg: Optional[TrainConfig] = None,
    custom_seq_len: Optional[int] = None,
    train_fraction: float = 1.0,
    db_path: Optional[str] = None,
    save_checkpoint: bool = True,
    custom_ckpt_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    统一训练与测试流程函数
    """
    if train_cfg is None:
        train_cfg = TrainConfig()

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.device == "cuda" else "cpu")

    model = create_model(model_name, dataset_cfg, custom_seq_len=custom_seq_len)
    model.to(device)
    num_params = count_parameters(model)

    is_baseline = model_name in ["Persistence", "DailySeasonalNaive"]
    actual_epochs = 0
    train_time_sec = 0.0

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict()) if not is_baseline else None
    if custom_ckpt_path is not None:
        ckpt_path = custom_ckpt_path
    else:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_cfg.name}_{model_name}_best.pt")

    # 训练循环 (基线直接跳过训练)
    if not is_baseline:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        patience_counter = 0

        start_train_t = time.time()
        for epoch in range(1, train_cfg.max_epochs + 1):
            model.train()
            train_loss_list = []
            for x_batch, y_batch in loaders["train"]:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())

            # 验证集评估
            model.eval()
            val_loss_list = []
            with torch.no_grad():
                for x_val, y_val in loaders["val"]:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    val_pred = model(x_val)
                    val_loss = criterion(val_pred, y_val)
                    val_loss_list.append(val_loss.item())

            avg_val_loss = np.mean(val_loss_list)
            actual_epochs = epoch

            # 早停与最佳模型保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
                if save_checkpoint and ckpt_path:
                    tmp_ckpt = f"{ckpt_path}.tmp.{os.getpid()}"
                    torch.save(best_model_weights, tmp_ckpt)
                    os.replace(tmp_ckpt, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.patience:
                    break

        train_time_sec = time.time() - start_train_t
        # 加载最佳验证 checkpoint
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

    # 测试集评测与推理测速
    model.eval()
    all_preds = []
    all_trues = []

    # 预热推理测速
    warmup_batch = next(iter(loaders["test"]))[0].to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_infer_t = time.time()

    with torch.no_grad():
        for x_test, y_test in loaders["test"]:
            x_test = x_test.to(device)
            preds = model(x_test)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(y_test.numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_infer_time = time.time() - start_infer_t
    num_test_batches = len(loaders["test"])
    infer_time_ms = (total_infer_time / max(1, num_test_batches)) * 1000.0

    y_pred_norm = np.concatenate(all_preds, axis=0)  # [N, H, C]
    y_true_norm = np.concatenate(all_trues, axis=0)  # [N, H, C]

    # 逆标准化回原始真实尺度
    y_pred_raw = scaler.inverse_transform(y_pred_norm)
    y_true_raw = scaler.inverse_transform(y_true_norm)

    # 仅从实际可用的训练目标区间计算季节性分母
    if len(train_bounds) == 3:
        train_start_b, b1, _ = train_bounds
    else:
        train_start_b, b1 = 0, train_bounds[0]
    train_target_slice = raw_data[train_start_b:b1, :]
    denominators = compute_seasonal_naive_denominators(train_target_slice, dataset_cfg.seasonal_m)

    # 1. 完整 24 小时指标
    metrics_24h = calculate_metrics_raw_scale(y_pred_raw, y_true_raw, denominators)

    # 2. 前 4 小时指标
    short_h = dataset_cfg.short_horizon
    metrics_4h = calculate_metrics_raw_scale(
        y_pred_raw[:, :short_h, :],
        y_true_raw[:, :short_h, :],
        denominators
    )

    # 3. 逐 lead-time 误差曲线
    lead_time_errors = compute_lead_time_errors(y_pred_raw, y_true_raw)

    # 4. 业务切片指标 (峰值负荷 / 活跃光伏)
    slice_metrics = compute_slice_metrics(y_pred_raw, y_true_raw, dataset_cfg.name, train_target_slice)

    result_record = {
        "dataset": dataset_cfg.name,
        "model_name": model_name,
        "seed": train_cfg.seed,
        "seq_len": custom_seq_len if custom_seq_len is not None else dataset_cfg.seq_len,
        "pred_len": dataset_cfg.pred_len,
        "train_fraction": train_fraction,
        "num_params": num_params,
        "actual_epochs": actual_epochs,
        "train_time_sec": round(train_time_sec, 2),
        "infer_time_ms": round(infer_time_ms, 2),
        "mae_24h": round(metrics_24h["mae"], 4),
        "rmse_24h": round(metrics_24h["rmse"], 4),
        "mase_24h": round(metrics_24h["mase"], 4),
        "mae_4h": round(metrics_4h["mae"], 4),
        "rmse_4h": round(metrics_4h["rmse"], 4),
        "mase_4h": round(metrics_4h["mase"], 4),
        "slice_mae": round(slice_metrics.get("peak_load_mae", slice_metrics.get("active_solar_mae", 0.0)), 4),
        "slice_rmse": round(slice_metrics.get("peak_load_rmse", slice_metrics.get("active_solar_rmse", 0.0)), 4),
        "lead_time_mae": lead_time_errors["mae_curve"].tolist(),
        "lead_time_rmse": lead_time_errors["rmse_curve"].tolist(),
        "config": {
            "lr": train_cfg.lr,
            "batch_size": train_cfg.batch_size,
            "patience": train_cfg.patience,
            "device": device.type
        }
    }

    # 持久化到 SQLite 数据库
    if db_path is not None:
        log_experiment_run(result_record, db_path=db_path)
    else:
        log_experiment_run(result_record)

    return result_record
