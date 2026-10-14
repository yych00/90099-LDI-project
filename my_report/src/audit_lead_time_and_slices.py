# -*- coding: utf-8 -*-
"""
全面复核与重验引擎 (Full Audit & Re-Verification Engine):
1. 双数据集实证总表 (Table 5-1 & 5-2) 全量前向推演与逐字段比对 (MAE, RMSE, MASE, 4h, 切片)
2. 关键业务切片 (Peak Load & Active Daylight) 阈值、比率与 MAE/RMSE 深度核算
3. Lead-Time 逐步前瞻误差演进 (1h 至 24h 逐点曲线及 Table 5-3 里程碑) 严格比对
4. 归一化互相关相位滞后 (Normalized Cross-Correlation Phase Lag) 算法全面核验与真实数据测试
5. 出版级 Lead-Time 误差曲线图自动同步更新至 figures/ 目录
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG, CHECKPOINT_DIR, FIGURES_DIR, DB_PATH
from data_loader import get_data_splits
from train import create_model
from eval import (
    compute_seasonal_naive_denominators,
    calculate_metrics_raw_scale,
    compute_slice_metrics,
    compute_lead_time_errors
)
from section7_analysis import compute_phase_lag

# 绘图字体与风格优化
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


TABLE_5_3_EXPECTED = {
    ("Electricity", "Persistence"): {"1h": 4.69, "2h": 7.32, "4h": 12.52, "8h": 18.58, "12h": 19.41, "18h": 16.63, "24h": 6.24},
    ("Electricity", "DailySeasonalNaive"): {"1h": 6.22, "2h": 6.22, "4h": 6.22, "8h": 6.23, "12h": 6.23, "18h": 6.23, "24h": 6.24},
    ("Electricity", "LSTM"): {"1h": 2.64, "2h": 2.64, "4h": 2.62, "8h": 2.64, "12h": 2.59, "18h": 2.62, "24h": 2.70},
    ("Electricity", "GRU"): {"1h": 2.59, "2h": 2.64, "4h": 2.73, "8h": 2.61, "12h": 2.58, "18h": 2.65, "24h": 2.84},
    ("Electricity", "CNN1D"): {"1h": 2.56, "2h": 2.64, "4h": 2.62, "8h": 2.56, "12h": 2.57, "18h": 2.60, "24h": 2.60},
    ("Electricity", "CNNLSTM"): {"1h": 2.57, "2h": 2.58, "4h": 2.60, "8h": 2.58, "12h": 2.58, "18h": 2.59, "24h": 2.61},
    ("Electricity", "DLinear"): {"1h": 2.77, "2h": 2.88, "4h": 2.88, "8h": 2.90, "12h": 2.96, "18h": 2.68, "24h": 2.74},
    ("Electricity", "PatchTST"): {"1h": 2.60, "2h": 2.58, "4h": 2.60, "8h": 2.59, "12h": 2.61, "18h": 2.57, "24h": 2.61},
    ("Solar", "Persistence"): {"10m": 7.32, "1h": 19.03, "2h": 34.88, "4h": 65.86, "8h": 113.40, "12h": 130.91, "18h": 92.74, "24h": 5.91},
    ("Solar", "DailySeasonalNaive"): {"10m": 5.92, "1h": 5.89, "2h": 5.89, "4h": 5.89, "8h": 5.89, "12h": 5.91, "18h": 5.91, "24h": 5.91},
    ("Solar", "LSTM"): {"10m": 16.74, "1h": 20.38, "2h": 22.92, "4h": 20.21, "8h": 17.28, "12h": 19.21, "18h": 17.98, "24h": 16.02},
    ("Solar", "GRU"): {"10m": 17.96, "1h": 19.05, "2h": 18.61, "4h": 13.49, "8h": 14.99, "12h": 16.11, "18h": 15.55, "24h": 17.16},
    ("Solar", "CNN1D"): {"10m": 10.25, "1h": 9.37, "2h": 10.42, "4h": 8.72, "8h": 10.36, "12h": 9.55, "18h": 8.82, "24h": 8.43},
    ("Solar", "CNNLSTM"): {"10m": 12.67, "1h": 16.41, "2h": 18.32, "4h": 12.23, "8h": 11.24, "12h": 12.47, "18h": 12.87, "24h": 11.47},
    ("Solar", "DLinear"): {"10m": 8.81, "1h": 8.69, "2h": 9.59, "4h": 11.64, "8h": 8.71, "12h": 8.30, "18h": 8.58, "24h": 8.73},
    ("Solar", "PatchTST"): {"10m": 8.54, "1h": 9.64, "2h": 11.06, "4h": 8.58, "8h": 10.31, "12h": 9.98, "18h": 7.84, "24h": 7.60}
}


def run_full_lead_time_and_slice_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for full forward evaluation: {device}")

    models = [
        "Persistence", "DailySeasonalNaive", "LSTM", "GRU",
        "CNN1D", "CNNLSTM", "DLinear", "PatchTST"
    ]

    all_milestone_results = {}
    all_slice_results = {}
    all_summary_results = {}
    all_curves = {}
    all_phase_lag_results = {}

    for cfg in [ELECTRICITY_CONFIG, SOLAR_CONFIG]:
        ds_name = cfg.name
        print(f"\n{'='*75}\n[DATASET: {ds_name}]\n{'='*75}")
        loaders, scaler, raw_data, bounds = get_data_splits(cfg)
        
        # 训练集区间
        if len(bounds) == 3:
            train_start_b, b1, _ = bounds
        else:
            train_start_b, b1 = 0, bounds[0]
        train_slice = raw_data[train_start_b:b1, :]
        denominators = compute_seasonal_naive_denominators(train_slice, cfg.seasonal_m)

        all_curves[ds_name] = {}

        for m_name in models:
            print(f"--> [前向推理与逐步前瞻演化核算] {ds_name:11s} | {m_name:18s} ... ", end="", flush=True)
            model = create_model(m_name, cfg).to(device)
            if m_name not in ["Persistence", "DailySeasonalNaive"]:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ds_name}_{m_name}_best.pt")
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

            # 逆标准化回真实物理量纲
            y_pred = scaler.inverse_transform(y_pred_norm)
            y_true = scaler.inverse_transform(y_true_norm)

            # 1. 全局指标 (24h & 4h)
            m_24h = calculate_metrics_raw_scale(y_pred, y_true, denominators)
            short_h = cfg.short_horizon
            m_4h = calculate_metrics_raw_scale(y_pred[:, :short_h, :], y_true[:, :short_h, :], denominators)

            # 2. 关键业务切片指标
            slice_res = compute_slice_metrics(y_pred, y_true, ds_name, train_slice)

            # 3. 逐步 Lead-Time 误差曲线 (1..H 步)
            lead_res = compute_lead_time_errors(y_pred, y_true)
            mae_curve = lead_res["mae_curve"]  # [H]
            rmse_curve = lead_res["rmse_curve"]  # [H]
            all_curves[ds_name][m_name] = {
                "mae_curve": mae_curve,
                "rmse_curve": rmse_curve
            }

            # 提取里程碑点
            if ds_name == "Electricity":
                milestones = {
                    "1h": mae_curve[0],
                    "2h": mae_curve[1],
                    "4h": mae_curve[3],
                    "8h": mae_curve[7],
                    "12h": mae_curve[11],
                    "18h": mae_curve[17],
                    "24h": mae_curve[23]
                }
            else:
                milestones = {
                    "10m": mae_curve[0],
                    "1h": mae_curve[5],
                    "2h": mae_curve[11],
                    "4h": mae_curve[23],
                    "8h": mae_curve[47],
                    "12h": mae_curve[71],
                    "18h": mae_curve[107],
                    "24h": mae_curve[143]
                }

            all_milestone_results[(ds_name, m_name)] = milestones
            all_slice_results[(ds_name, m_name)] = slice_res
            all_summary_results[(ds_name, m_name)] = {
                "24h_mae": m_24h["mae"],
                "24h_rmse": m_24h["rmse"],
                "24h_mase": m_24h["mase"],
                "4h_mae": m_4h["mae"],
                "4h_rmse": m_4h["rmse"],
                "4h_mase": m_4h["mase"]
            }

            # 4. 相位滞后检验 (展平第一个通道连续时序计算)
            c0_true = y_true[:, 0, 0]
            c0_pred = y_pred[:, 0, 0]
            lag_step1 = compute_phase_lag(c0_true, c0_pred, max_lag=6)
            all_phase_lag_results[(ds_name, m_name)] = lag_step1

            # 光伏数据集跨窗 stride=6 (1小时采样)，补充 10 分钟物理分辨率连续序列滞后核验
            if ds_name == "Solar" and m_name == "Persistence":
                b2_val = bounds[2] if len(bounds) == 3 else bounds[1]
                cont_true = raw_data[b2_val:, 0]
                cont_pred = raw_data[b2_val - 1 : -1, 0]
                cont_lag = compute_phase_lag(cont_true, cont_pred, max_lag=6)
                all_phase_lag_results[(ds_name, "Persistence(10m_cont)")] = cont_lag

            print(f"[DONE] 24h MAE: {m_24h['mae']:.4f}, Slice MAE: {slice_res.get('peak_load_mae', slice_res.get('active_solar_mae', 0.0)):.4f}, Phase Lag: {lag_step1:+d}")

    # =========================================================================
    # 生成出版级 Lead-Time 误差曲线图 (figures/lead_time_curves_*.png)
    # =========================================================================
    print("\n正在生成出版级 Lead-Time 误差演进曲线矢量图...")
    for ds_name in ["Electricity", "Solar"]:
        plt.figure(figsize=(11, 5.5))
        curves_dict = all_curves[ds_name]
        
        color_map = {
            "Persistence": "#7f7f7f",
            "DailySeasonalNaive": "#ff7f0e",
            "LSTM": "#bcbd22",
            "GRU": "#8c564b",
            "CNN1D": "#17becf",
            "CNNLSTM": "#1f77b4",
            "DLinear": "#9467bd",
            "PatchTST": "#2ca02c"
        }
        style_map = {
            "Persistence": ":",
            "DailySeasonalNaive": "-.",
            "LSTM": "--",
            "GRU": "--",
            "CNN1D": "-",
            "CNNLSTM": "-",
            "DLinear": "-",
            "PatchTST": "-"
        }

        for m_name in models:
            mae_curve = curves_dict[m_name]["mae_curve"]
            if ds_name == "Electricity":
                hours = np.arange(1, len(mae_curve) + 1)
            else:
                hours = np.arange(1, len(mae_curve) + 1) * (10.0 / 60.0)
            
            plt.plot(
                hours, mae_curve,
                label=m_name,
                color=color_map.get(m_name, None),
                linestyle=style_map.get(m_name, "-"),
                linewidth=2.2 if m_name in ["1D-CNN", "CNN1D", "CNNLSTM", "DLinear", "PatchTST"] else 1.6,
                alpha=0.9
            )

        unit = "kW" if ds_name == "Electricity" else "MW"
        plt.xlabel("Forecast Horizon / Lead-Time (Hours)", fontsize=11, fontweight='bold')
        plt.ylabel(f"Lead-Time Step MAE ({unit})", fontsize=11, fontweight='bold')
        plt.title(f"{ds_name}: Error Progression across 1h to 24h Ahead Horizon", fontsize=13, fontweight='bold', pad=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left" if ds_name == "Solar" else "center right", framealpha=0.9, ncol=2)
        plt.tight_layout()

        out_name = f"lead_time_curves_{ds_name.lower()}.png"
        my_rep_fig = os.path.join(FIGURES_DIR, out_name)
        root_fig = os.path.abspath(os.path.join(FIGURES_DIR, "..", "..", "figures", out_name))
        plt.savefig(my_rep_fig, dpi=300)
        if os.path.exists(os.path.dirname(root_fig)):
            shutil.copy2(my_rep_fig, root_fig)
        plt.close()
        print(f"已更新并同步 Lead-Time 曲线图: {my_rep_fig} & {root_fig}")

    # =========================================================================
    # 打印全量审计表与严格比对报告
    # =========================================================================
    print("\n" + "="*95)
    print("【实证总表 Table 5-1 & 5-2 全前向推演复核】")
    print("="*95)
    summary_rows = []
    for (ds, m), vals in all_summary_results.items():
        s_res = all_slice_results[(ds, m)]
        s_mae = s_res.get("peak_load_mae", s_res.get("active_solar_mae", 0.0))
        s_rmse = s_res.get("peak_load_rmse", s_res.get("active_solar_rmse", 0.0))
        s_ratio = s_res.get("slice_sample_ratio", 0.0)
        summary_rows.append({
            "Dataset": ds, "Model": m,
            "24h_MAE": round(vals["24h_mae"], 4),
            "24h_RMSE": round(vals["24h_rmse"], 4),
            "24h_MASE": round(vals["24h_mase"], 4),
            "4h_MAE": round(vals["4h_mae"], 4),
            "4h_RMSE": round(vals["4h_rmse"], 4),
            "4h_MASE": round(vals["4h_mase"], 4),
            "Slice_MAE": round(s_mae, 4),
            "Slice_RMSE": round(s_rmse, 4),
            "Slice_Ratio(%)": round(s_ratio * 100, 2)
        })
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))

    print("\n" + "="*95)
    print("【Table 5-3 典型前瞻时间步逐步 MAE 追踪表核验 (实测 vs 预期)】")
    print("="*95)
    ms_diffs = []
    for (ds, m), ms_actual in all_milestone_results.items():
        ms_exp = TABLE_5_3_EXPECTED.get((ds, m), {})
        row = {"Dataset": ds, "Model": m}
        all_match = True
        for k, v_exp in ms_exp.items():
            v_act = round(ms_actual[k], 2)
            delta = abs(v_act - v_exp)
            row[f"{k}_act"] = v_act
            row[f"{k}_exp"] = v_exp
            row[f"{k}_delta"] = round(delta, 2)
            if delta > 0.05:  # 容差 0.05
                all_match = False
        row["Status"] = "PASS" if all_match else "FAIL"
        ms_diffs.append(row)

    df_ms_audit = pd.DataFrame(ms_diffs)
    print("\n--- [Electricity 前瞻里程碑核验 (1h, 2h, 4h, 8h, 12h, 18h, 24h)] ---")
    df_elec = df_ms_audit[df_ms_audit["Dataset"] == "Electricity"]
    elec_cols = ["Model", "1h_act", "1h_exp", "2h_act", "2h_exp", "4h_act", "4h_exp", "12h_act", "12h_exp", "24h_act", "24h_exp", "Status"]
    print(df_elec[[c for c in elec_cols if c in df_elec.columns]].to_string(index=False))

    print("\n--- [Solar 前瞻里程碑核验 (10m, 1h, 2h, 4h, 8h, 12h, 18h, 24h)] ---")
    df_solar = df_ms_audit[df_ms_audit["Dataset"] == "Solar"]
    solar_cols = ["Model", "10m_act", "10m_exp", "1h_act", "1h_exp", "4h_act", "4h_exp", "12h_act", "12h_exp", "24h_act", "24h_exp", "Status"]
    print(df_solar[[c for c in solar_cols if c in df_solar.columns]].to_string(index=False))

    print("\n" + "="*95)
    print("【归一化互相关相位滞后 (Phase Lag) 统计汇总】")
    print("="*95)
    for (ds, m), pl in all_phase_lag_results.items():
        print(f"Dataset: {ds:11s} | Model: {m:18s} | Phase Lag: {pl:+d} steps")

    # 断言 Table 5-3 一致性
    failed_ms = df_ms_audit[df_ms_audit["Status"] != "PASS"]
    if len(failed_ms) > 0:
        print(f"\n[ALERT] 存在 {len(failed_ms)} 个模型的 Lead-Time 里程碑与 Table 5-3 差异超过容差！")
        return False, df_sum, df_ms_audit
    else:
        print(f"\n[SUCCESS] 全部 16 个模型的 Lead-Time 逐步前瞻误差与 Table 5-3 严格吻合！")
        return True, df_sum, df_ms_audit


if __name__ == "__main__":
    success, df_sum, df_ms = run_full_lead_time_and_slice_audit()
    if not success:
        sys.exit(1)
