# -*- coding: utf-8 -*-
"""
第 7 节分析逻辑与统计度量深度单元测试套件：
覆盖消融数据库提取、自相关函数 (ACF) 边界条件、残差统计矩计算、
多维物理非负截断算子、相位滞后 (Phase Lag) 互相关识别。
"""

import unittest
import numpy as np
import sys
import os

try:
    import pandas as pd
    import torch
    HAS_DEPS = True
except ImportError:
    pd = None
    torch = None
    HAS_DEPS = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if HAS_DEPS:
    try:
        from section7_analysis import (
            get_ablation_dataframe_from_db,
            compute_acf,
            calculate_residual_moments,
            compute_phase_lag,
            apply_physical_clamping,
            get_cached_predictions
        )
        from config import ELECTRICITY_CONFIG, SOLAR_CONFIG
    except Exception:
        HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "pandas/torch 或分析依赖在当前轻量级环境中未完全安装，跳过 Section 7 测试")
class TestSection7Analysis(unittest.TestCase):

    def test_ablation_dataframe_schema_and_values(self):
        """测试消融实验数据框格式、必要列名与关键数值合理性"""
        df = get_ablation_dataframe_from_db()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 6)
        
        required_cols = ["Dataset", "Model", "MAE", "RMSE", "MASE", "Params", "TrainTime", "InferLat"]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # 验证 Electricity 数据集 CNN-LSTM 的 MAE 低于 Standalone LSTM
        elec_lstm = df[(df["Dataset"] == "Electricity") & (df["Model"] == "Standalone LSTM")]["MAE"].values[0]
        elec_cnnlstm = df[(df["Dataset"] == "Electricity") & (df["Model"] == "Hybrid CNN-LSTM")]["MAE"].values[0]
        self.assertLess(elec_cnnlstm, elec_lstm)

        # 验证 Solar 数据集 CNN-LSTM 相较于 LSTM 的 MAE 大幅下降 (> 20%)
        solar_lstm = df[(df["Dataset"] == "Solar") & (df["Model"] == "Standalone LSTM")]["MAE"].values[0]
        solar_cnnlstm = df[(df["Dataset"] == "Solar") & (df["Model"] == "Hybrid CNN-LSTM")]["MAE"].values[0]
        self.assertLess(solar_cnnlstm, solar_lstm)
        self.assertGreater((solar_lstm - solar_cnnlstm) / solar_lstm, 0.20)

    def test_compute_acf_constant_series(self):
        """测试自相关函数在常数序列（零方差）上的鲁棒性"""
        const_series = np.ones(50)
        acf_res = compute_acf(const_series, max_lag=10)
        self.assertEqual(len(acf_res), 11)
        self.assertEqual(acf_res[0], 1.0)
        # 其余滞后阶数返回 0.0，杜绝除以零崩溃
        np.testing.assert_array_equal(acf_res[1:], np.zeros(10))

    def test_compute_acf_white_noise(self):
        """测试自相关函数在标准高斯白噪声上的统计收敛性"""
        np.random.seed(42)
        white_noise = np.random.randn(3000)
        acf_res = compute_acf(white_noise, max_lag=20)
        self.assertEqual(len(acf_res), 21)
        self.assertAlmostEqual(acf_res[0], 1.0, places=5)
        # 高斯白噪声在 lag >= 1 的样本相关度应受限于 3 / sqrt(N)
        bound = 3.0 / np.sqrt(len(white_noise))
        for lag in range(1, 21):
            self.assertLess(abs(acf_res[lag]), bound)

    def test_compute_acf_short_series_boundary(self):
        """测试自相关函数在输入序列长度小于 max_lag 时的边界自适应截断补齐"""
        short_series = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        acf_res = compute_acf(short_series, max_lag=10)
        self.assertEqual(len(acf_res), 11)
        self.assertEqual(acf_res[0], 1.0)

    def test_calculate_residual_moments(self):
        """测试残差高阶统计矩计算 (MBE, Std, Skewness, Kurtosis, Shapiro-Wilk)"""
        np.random.seed(42)
        # 构造零均值标准正态残差
        norm_res = np.random.normal(loc=0.0, scale=2.5, size=2000)
        moments = calculate_residual_moments(norm_res, "SyntheticNormal")
        
        self.assertEqual(moments["Model"], "SyntheticNormal")
        self.assertAlmostEqual(moments["MBE (Mean)"], 0.0, delta=0.15)
        self.assertAlmostEqual(moments["Std Dev"], 2.5, delta=0.15)
        self.assertAlmostEqual(moments["Skewness"], 0.0, delta=0.20)
        self.assertAlmostEqual(moments["Kurtosis"], 0.0, delta=0.30)
        self.assertGreater(moments["Normality p-val"], 0.01)

    def test_physical_clamping_multidimensional(self):
        """测试物理非负截断算子在多维张量上的非负性与 MAE 单调减少性"""
        # 构造含微弱负值的模拟多维预测 [N=4, H=6, C=2]
        np.random.seed(42)
        y_true = np.random.uniform(0.0, 100.0, size=(4, 6, 2))
        noise = np.random.normal(0.0, 10.0, size=(4, 6, 2))
        y_pred = y_true + noise
        # 强制部分日落夜间时步为负
        y_pred[0, :3, 0] = -5.0
        y_pred[1, 4:, 1] = -2.5

        res = apply_physical_clamping(y_pred, y_true)
        self.assertGreater(res["neg_count"], 0)
        self.assertGreater(res["neg_ratio"], 0.0)
        self.assertLess(res["min_neg_val"], 0.0)
        
        # 截断后预测全为非负
        self.assertTrue(np.all(res["preds_clamped"] >= 0.0))
        # 对非负真值，截断后 MAE 必须严格小于等于截断前
        self.assertLessEqual(res["mae_clamped"], res["mae_raw"])
        self.assertGreaterEqual(res["reduction_gain_percent"], 0.0)

    def test_phase_lag_cross_correlation(self):
        """测试互相关峰值相位滞后量化算法及其边界鲁棒性"""
        t = np.linspace(0, 4 * np.pi, 100)
        wave = np.sin(t)
        
        # 1. 零相位差
        lag_zero = compute_phase_lag(wave, wave, max_lag=5)
        self.assertEqual(lag_zero, 0)
        
        # 2. 预测落后 2 个时间步 (y_pred[t] = wave[t - 2])
        wave_delayed = np.roll(wave, 2)
        lag_delayed = compute_phase_lag(wave[5:-5], wave_delayed[5:-5], max_lag=5)
        self.assertEqual(lag_delayed, 2)

        # 3. 预测超前 2 个时间步 (y_pred[t] = wave[t + 2])
        wave_advanced = np.roll(wave, -2)
        lag_advanced = compute_phase_lag(wave[5:-5], wave_advanced[5:-5], max_lag=5)
        self.assertEqual(lag_advanced, -2)

        # 4. 常数全零极端边界（夜间零值无相关性，安全返回 0）
        lag_const = compute_phase_lag(np.zeros(50), np.zeros(50), max_lag=5)
        self.assertEqual(lag_const, 0)

        # 5. 超短序列边界防御
        lag_short = compute_phase_lag(np.array([1.0, 2.0]), np.array([1.0, 2.0]), max_lag=5)
        self.assertEqual(lag_short, 0)

        # 6. 线性斜坡序列自身对比 (防多重峰值平局伪判负相位差)
        ramp = np.arange(10.0)
        self.assertEqual(compute_phase_lag(ramp, ramp, max_lag=3), 0)
        self.assertEqual(compute_phase_lag(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), max_lag=3), 0)

        # 7. NaN / Inf 异常输入防御
        self.assertEqual(compute_phase_lag(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])), 0)
        self.assertEqual(compute_phase_lag(np.array([1.0, np.inf, 3.0]), np.array([1.0, 2.0, 3.0])), 0)

        # 8. 非正 max_lag 防御
        self.assertEqual(compute_phase_lag(wave, wave, max_lag=0), 0)
        self.assertEqual(compute_phase_lag(wave, wave, max_lag=-2), 0)

    def test_actual_peak_load_tracking_sample2325(self):
        """测试真实测试集尖峰负荷样本 (#2325, 峰值 206.62 kW) 追踪精度闭环"""
        cached_e = get_cached_predictions(ELECTRICITY_CONFIG)
        true_e = cached_e["true_raw"]
        preds_e = cached_e["preds_dict"]

        # 验证通道 0 (T5) 最大尖峰样本为 #2325，时间步 h=24，峰值 206.62 kW
        max_per_sample = np.max(true_e[:, :, 0], axis=1)
        peak_idx = int(np.argmax(max_per_sample))
        self.assertEqual(peak_idx, 2325)

        peak_h = int(np.argmax(true_e[peak_idx, :, 0]))
        self.assertEqual(peak_h, 23)  # h=24 (0-indexed 23)
        actual_peak = float(true_e[peak_idx, peak_h, 0])
        self.assertAlmostEqual(actual_peak, 206.62, places=1)

        # 验证核心模型预测追踪精度及削峰幅度
        cnn_peak = float(preds_e["CNNLSTM"][peak_idx, peak_h, 0])
        patch_peak = float(preds_e["PatchTST"][peak_idx, peak_h, 0])
        dlin_peak = float(preds_e["DLinear"][peak_idx, peak_h, 0])
        naive_peak = float(preds_e["DailySeasonalNaive"][peak_idx, peak_h, 0])

        self.assertAlmostEqual(cnn_peak, 195.21, delta=0.2)
        self.assertAlmostEqual(patch_peak, 198.73, delta=0.2)
        self.assertAlmostEqual(dlin_peak, 197.82, delta=0.2)
        self.assertAlmostEqual(naive_peak, 195.02, delta=0.2)

        # 削峰低估相对误差验证 (负值表示低估)
        self.assertAlmostEqual((actual_peak - cnn_peak) / actual_peak * 100.0, 5.52, delta=0.1)
        self.assertAlmostEqual((actual_peak - patch_peak) / actual_peak * 100.0, 3.82, delta=0.1)

    def test_solar_physical_clamping_exact_gain(self):
        """测试光伏测试集物理非负截断增益精确闭环 (9.75% 误差削减)"""
        cached_s = get_cached_predictions(SOLAR_CONFIG)
        true_s = cached_s["true_raw"]
        preds_s_cnnlstm = cached_s["preds_dict"]["CNNLSTM"]

        clamp_res = apply_physical_clamping(preds_s_cnnlstm, true_s)
        self.assertAlmostEqual(clamp_res["mae_raw"], 12.9477, delta=0.01)
        self.assertAlmostEqual(clamp_res["mae_clamped"], 11.6852, delta=0.01)
        self.assertAlmostEqual(clamp_res["reduction_gain_percent"], 9.75, delta=0.05)
        self.assertAlmostEqual(clamp_res["neg_ratio"], 21.18, delta=0.1)

        # 夜间零值点占比应超过 50%
        night_ratio = float(np.mean(true_s <= 0.01) * 100.0)
        self.assertAlmostEqual(night_ratio, 50.69, delta=0.1)

    def test_solar_cloud_shock_intermittency_step35(self):
        """测试光伏多云突变极端案例 (样本 Origin #76, step 35 突跌 -109.4 MW)"""
        cached_s = get_cached_predictions(SOLAR_CONFIG)
        true_s = cached_s["true_raw"]

        day_var = np.var(true_s[:, 36:108, 0], axis=1)
        cloudy_idx = int(np.argmax(day_var))
        self.assertEqual(cloudy_idx, 76)

        s_gt = true_s[cloudy_idx, :, 0]
        # 验证单步 10 分钟突跌发生在 step 34 -> step 35
        drop_34_35 = float(s_gt[34] - s_gt[35])
        self.assertAlmostEqual(drop_34_35, 109.4, delta=0.5)


if __name__ == "__main__":
    unittest.main()
