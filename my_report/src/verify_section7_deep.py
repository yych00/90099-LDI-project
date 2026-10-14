# -*- coding: utf-8 -*-
"""
Section 7 深度复核与全方位实验比对脚本
针对：
1. CNN 模块消融受控实验 (Electricity & Solar)
2. 光伏夜间零值振荡特性与物理非负截断 (9.75% 增益等)
3. 测试集残差高阶统计矩全量测算 (偏度、峰度、Shapiro-Wilk 正态性检验)
4. 真实尖峰负荷样本 (#76 样本，峰值 206.6 kW) 及极端案例精确追踪
5. 出版级图表重新渲染与数据闭环校验
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG, CHECKPOINT_DIR, FIGURES_DIR, DB_PATH
from data_loader import get_data_splits
from train import create_model
from cross_verify_experiments import evaluate_model_on_test

def get_test_predictions(model_name: str, dataset_cfg, test_loader, scaler, device="cpu"):
    """
    加载模型并在测试集上计算全量预测和真值（原物理量纲）
    """
    all_x = []
    all_y = []
    for x_b, y_b in test_loader:
        all_x.append(x_b)
        all_y.append(y_b)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    if model_name == "Persistence":
        last_val = all_x[:, -1:, :]
        preds = last_val.repeat(1, dataset_cfg.pred_len, 1).numpy()
    elif model_name == "DailySeasonalNaive":
        m = dataset_cfg.seasonal_m
        past_season = all_x[:, -m:, :].numpy()
        preds = past_season
    else:
        model = create_model(model_name, dataset_cfg)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_cfg.name}_{model_name}_best.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"[Warning] Checkpoint not found: {ckpt_path}")
        model.to(device)
        model.eval()

        preds_list = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, all_x.size(0), batch_size):
                bx = all_x[i:i+batch_size].to(device)
                by_pred = model(bx)
                preds_list.append(by_pred.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)

    preds_raw = scaler.inverse_transform(preds)
    true_raw = scaler.inverse_transform(all_y.numpy())
    return preds_raw, true_raw

def main():
    print("=" * 80)
    print("ELEN90099 Section 7 深度量化复核与闭环实验验证")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/5] 加载 Electricity 与 Solar 测试集...")
    elec_loaders, elec_scaler, elec_raw, elec_bounds = get_data_splits(ELECTRICITY_CONFIG)
    solar_loaders, solar_scaler, solar_raw, solar_bounds = get_data_splits(SOLAR_CONFIG)

    test_loader_e = elec_loaders["test"]
    test_loader_s = solar_loaders["test"]

    models = ["Persistence", "DailySeasonalNaive", "LSTM", "GRU", "CNN1D", "CNNLSTM", "DLinear", "PatchTST"]

    # 2. 推理预测与指标
    print("\n[2/5] 执行全量模型测试集推理...")
    preds_e = {}
    preds_s = {}
    true_e = None
    true_s = None

    for m in models:
        p_e, t_e = get_test_predictions(m, ELECTRICITY_CONFIG, test_loader_e, elec_scaler)
        preds_e[m] = p_e
        if true_e is None:
            true_e = t_e

        p_s, t_s = get_test_predictions(m, SOLAR_CONFIG, test_loader_s, solar_scaler)
        preds_s[m] = p_s
        if true_s is None:
            true_s = t_s

    # 3. CNN 模块受控消融复核
    print("\n[3/5] 复核 CNN 模块消融受控实验 (LSTM vs CNN1D vs CNNLSTM)...")
    ablation_models = ["LSTM", "CNN1D", "CNNLSTM"]
    
    print("\n--- Electricity (L=48, H=24) 消融对比 ---")
    for m in ablation_models:
        mae = float(np.mean(np.abs(true_e - preds_e[m])))
        rmse = float(np.sqrt(np.mean((true_e - preds_e[m])**2)))
        model_inst = create_model(m, ELECTRICITY_CONFIG)
        params = sum(p.numel() for p in model_inst.parameters() if p.requires_grad)
        print(f"Model: {m:10s} | MAE: {mae:.4f} kW | RMSE: {rmse:.4f} kW | Params: {params:,}")

    print("\n--- Solar (L=288, H=144) 消融对比 ---")
    for m in ablation_models:
        mae = float(np.mean(np.abs(true_s - preds_s[m])))
        rmse = float(np.sqrt(np.mean((true_s - preds_s[m])**2)))
        model_inst = create_model(m, SOLAR_CONFIG)
        params = sum(p.numel() for p in model_inst.parameters() if p.requires_grad)
        print(f"Model: {m:10s} | MAE: {mae:.4f} MW | RMSE: {rmse:.4f} MW | Params: {params:,}")

    # 4. 光伏夜间零值振荡与物理非负截断验证
    print("\n[4/5] 复核光伏夜间零值振荡与物理非负截断 (9.75% 增益)...")
    night_mask = (true_s <= 0.01)
    active_thresh = 0.10 * np.max(solar_raw, axis=0, keepdims=True)
    active_mask = (true_s > active_thresh)
    print(f"总测试样本点数: {true_s.size:,} | 夜间零值点: {np.sum(night_mask):,} ({np.mean(night_mask)*100:.2f}%) | 白天活跃点: {np.sum(active_mask):,} ({np.mean(active_mask)*100:.2f}%)")

    clamp_summary = []
    for m in models:
        p = preds_s[m]
        res = true_s - p
        res_night = res[night_mask]
        res_active = res[active_mask]
        
        neg_mask = (p < 0.0)
        neg_count = int(np.sum(neg_mask))
        neg_ratio = float(neg_count / p.size * 100.0)
        min_neg = float(np.min(p)) if neg_count > 0 else 0.0
        
        p_clamped = np.maximum(p, 0.0)
        mae_raw = float(np.mean(np.abs(true_s - p)))
        rmse_raw = float(np.sqrt(np.mean((true_s - p)**2)))
        mae_clamped = float(np.mean(np.abs(true_s - p_clamped)))
        rmse_clamped = float(np.sqrt(np.mean((true_s - p_clamped)**2)))
        gain = float((mae_raw - mae_clamped) / mae_raw * 100.0) if mae_raw > 0 else 0.0
        
        # 夜间负值与正向微小漂移
        night_preds = p[night_mask]
        night_neg_ratio = float(np.mean(night_preds < 0.0) * 100.0)

        clamp_summary.append({
            "Model": m,
            "Raw MAE": mae_raw,
            "Clamped MAE": mae_clamped,
            "Gain (%)": gain,
            "Raw RMSE": rmse_raw,
            "Clamped RMSE": rmse_clamped,
            "Neg Count": neg_count,
            "Neg Ratio (%)": neg_ratio,
            "Min Neg (MW)": min_neg,
            "Night Std (MW)": float(np.std(res_night)),
            "Active Std (MW)": float(np.std(res_active)),
            "Night Neg Ratio (%)": night_neg_ratio
        })

    df_clamp = pd.DataFrame(clamp_summary)
    print("\n【光伏物理截断与夜间振荡详细对照表】")
    print(df_clamp.to_string(index=False))

    # 5. 残差高阶统计矩测算 (偏度、峰度、Shapiro-Wilk)
    print("\n[5/5] 重新测算测试集残差高阶统计矩 (偏度、峰度、正态性检验)...")
    def calc_moments(residuals, name):
        flat = residuals.flatten()
        mbe = float(np.mean(flat))
        std = float(np.std(flat))
        skew = float(stats.skew(flat))
        kurt = float(stats.kurtosis(flat))
        # Shapiro-Wilk (subsample 5000)
        sample_sub = flat[:5000] if len(flat) > 5000 else flat
        sw_stat, sw_p = stats.shapiro(sample_sub)
        return {
            "Model": name,
            "MBE (Mean)": mbe,
            "Std Dev": std,
            "Skewness": skew,
            "Kurtosis": kurt,
            "SW Stat": float(sw_stat),
            "SW p-val": float(sw_p)
        }

    elec_moments = [calc_moments(true_e - preds_e[m], m) for m in models]
    solar_moments = [calc_moments(true_s - preds_s[m], m) for m in models]

    print("\n【Electricity 残差高阶统计矩】")
    print(pd.DataFrame(elec_moments).to_string(index=False))

    print("\n【Solar 残差高阶统计矩】")
    print(pd.DataFrame(solar_moments).to_string(index=False))

    # 6. 验证尖峰负荷样本 (#76 样本，峰值 206.6 kW)
    print("\n[6/6] 验证极端尖峰负荷样本 (#76 样本，真实峰值 206.6 kW)...")
    # 查找最大峰值样本
    max_per_sample = np.max(true_e[:, :, 0], axis=1)
    peak_sample_idx = int(np.argmax(max_per_sample))
    peak_h = int(np.argmax(true_e[peak_sample_idx, :, 0]))
    actual_peak = float(true_e[peak_sample_idx, peak_h, 0])
    
    print(f"最大尖峰样本索引: #{peak_sample_idx}, 通道: 0 (T5), 峰值时间步: h={peak_h+1}, 真实峰值: {actual_peak:.2f} kW")
    
    # 检查特定 #76 样本
    if peak_sample_idx != 76:
        print(f"注意：最大尖峰样本索引为 #{peak_sample_idx}，同时检查 #{76} 样本：")
        peak_76 = float(np.max(true_e[76, :, 0]))
        peak_76_h = int(np.argmax(true_e[76, :, 0]))
        print(f"样本 #76 峰值: {peak_76:.2f} kW (h={peak_76_h+1})")
        sample_to_use = 76 if abs(peak_76 - 206.6) < 1.0 else peak_sample_idx
    else:
        sample_to_use = peak_sample_idx

    print(f"\n样本 #{sample_to_use} 峰值各模型跟踪详情：")
    for m in models:
        pred_val = float(preds_e[m][sample_to_use, peak_h, 0])
        error = actual_peak - pred_val
        rel_under = error / actual_peak * 100.0
        print(f"  {m:20s}: 预测 = {pred_val:6.1f} kW | 低估 = {error:+6.1f} kW | 相对误差 = {rel_under:+5.2f}%")

    return {
        "df_clamp": df_clamp,
        "elec_moments": elec_moments,
        "solar_moments": solar_moments,
        "sample_to_use": sample_to_use,
        "actual_peak": actual_peak
    }

if __name__ == "__main__":
    main()
