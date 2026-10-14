# -*- coding: utf-8 -*-
"""Evaluate predictions after inverse scaling. Training-defined seasonal references and output masks keep evaluation conditions independent of model predictions."""

import numpy as np

try:
    import torch
except ImportError:
    torch = None
from typing import Dict, Tuple, Optional, List, Any


def compute_seasonal_naive_denominators(
    train_data: np.ndarray, seasonal_m: int, eps: float = 1e-8
) -> np.ndarray:
    """Use training-only daily differences as the per-channel MASE reference, rather than a test-dependent denominator."""
    if train_data.ndim == 1:
        train_data = train_data[:, np.newaxis]
    T, C = train_data.shape
    if T <= seasonal_m:
        raise ValueError(
            f"Training length {T} cannot support seasonal period {seasonal_m} for differencing."
        )

    diff = np.abs(train_data[seasonal_m:, :] - train_data[:-seasonal_m, :])
    denominators = np.mean(diff, axis=0)  # [C]
    # A floor makes MASE defined for nearly constant training series.
    if np.any(denominators <= eps):
        print(f"[Warning] Near-zero seasonal differences; using denominator floor {eps}.")
    denominators = np.maximum(denominators, eps)
    return denominators


def calculate_metrics_raw_scale(
    y_pred: np.ndarray, y_true: np.ndarray, denominators: np.ndarray
) -> Dict[str, Any]:
    """Report global MAE/RMSE and channel-averaged MASE; averaging channel RMSEs would change the intended metric."""
    denominators = np.asarray(denominators, dtype=np.float64).reshape(-1)
    # Retain channel metrics because aggregate errors can hide scale-dependent differences.
    mae_per_channel = np.mean(np.abs(y_pred - y_true), axis=(0, 1))  # [C]
    rmse_per_channel = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(0, 1)))  # [C]
    mase_per_channel = mae_per_channel / denominators  # [C]

    # Take one square root after averaging all squared errors, not an average of channel roots.
    overall_mae = float(np.mean(np.abs(y_pred - y_true)))
    overall_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    overall_mase = float(np.mean(mase_per_channel))

    return {
        "mae": overall_mae,
        "rmse": overall_rmse,
        "mase": overall_mase,
        "mae_per_channel": mae_per_channel.tolist(),
        "rmse_per_channel": rmse_per_channel.tolist(),
        "mase_per_channel": mase_per_channel.tolist(),
    }


def compute_lead_time_errors(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """Average over origins and channels separately at each lead, so short and long forecast distances can be compared."""
    # Leave the horizon axis intact to expose changes with forecast distance.
    mae_curve = np.mean(np.abs(y_pred - y_true), axis=(0, 2))  # [H]
    rmse_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(0, 2)))  # [H]
    return {"mae_curve": mae_curve, "rmse_curve": rmse_curve}


def compute_slice_metrics(
    y_pred: np.ndarray, y_true: np.ndarray, dataset_name: str, train_data: np.ndarray
) -> Dict[str, float]:
    """Use true-output masks defined by training thresholds to inspect high-load or active-output errors consistently across models."""
    if dataset_name == "Electricity":
        # Define high-load conditions from training, never from model errors.
        thresholds = np.percentile(train_data, 85, axis=0, keepdims=True)  # [1, C]
        thresholds = np.expand_dims(thresholds, axis=0)  # [1, 1, C]
        mask = y_true >= thresholds
        slice_name = "peak_load"
    elif dataset_name == "Solar":
        # Define active output from training; do not imply verified daylight labels.
        thresholds = 0.10 * np.max(train_data, axis=0, keepdims=True)
        thresholds = np.expand_dims(thresholds, axis=0)
        mask = y_true >= thresholds
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
        "slice_sample_ratio": slice_ratio,
    }
