# -*- coding: utf-8 -*-
"""
单元测试：评估指标计算、Seasonal MASE 零保护与切片指标。
覆盖：
1. Seasonal MASE 分母基准计算与零分母常数通道防护。
2. 原尺度 MAE、RMSE 与 MASE 的全局均值与逐通道一致性检验。
3. Lead-time 逐步误差统计。
4. Electricity 峰值负荷与 Solar 活跃出力切片指标。
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval import (
    compute_seasonal_naive_denominators,
    calculate_metrics_raw_scale,
    compute_lead_time_errors,
    compute_slice_metrics
)


class TestMetrics(unittest.TestCase):
    def test_seasonal_mase_denominators(self):
        # 构造简单正弦时序
        t = np.arange(100)
        series = np.sin(2 * np.pi * t / 24).reshape(-1, 1)
        denoms = compute_seasonal_naive_denominators(series, seasonal_m=24)
        self.assertEqual(len(denoms), 1)
        # 正弦周期差分应接近 0，但有 eps 保护
        self.assertGreaterEqual(float(denoms[0]), 1e-8)

        # 常数序列保护
        const_series = np.ones((100, 2), dtype=np.float32) * 42.0
        const_denoms = compute_seasonal_naive_denominators(const_series, seasonal_m=24)
        self.assertEqual(len(const_denoms), 2)
        self.assertAlmostEqual(float(const_denoms[0]), 1e-8, places=7)

    def test_raw_scale_metrics(self):
        N, H, C = 10, 24, 16
        y_true = np.ones((N, H, C), dtype=np.float32) * 10.0
        # 预测误差恒为 2.0
        y_pred = np.ones((N, H, C), dtype=np.float32) * 12.0
        denominators = np.ones(C, dtype=np.float32) * 4.0

        metrics = calculate_metrics_raw_scale(y_pred, y_true, denominators)
        self.assertAlmostEqual(metrics["mae"], 2.0, places=4)
        self.assertAlmostEqual(metrics["rmse"], 2.0, places=4)
        # MASE = MAE / denom = 2.0 / 4.0 = 0.5
        self.assertAlmostEqual(metrics["mase"], 0.5, places=4)

    def test_lead_time_and_slice(self):
        N, H, C = 5, 24, 16
        y_true = np.ones((N, H, C), dtype=np.float32) * 10.0
        y_pred = np.ones((N, H, C), dtype=np.float32) * 11.0
        
        lt_res = compute_lead_time_errors(y_pred, y_true)
        self.assertEqual(len(lt_res["mae_curve"]), H)
        self.assertAlmostEqual(float(lt_res["mae_curve"][0]), 1.0, places=4)

        # 1. Electricity 峰值切片
        train_data_elec = np.ones((100, C), dtype=np.float32) * 5.0
        slice_elec = compute_slice_metrics(y_pred, y_true, "Electricity", train_data_elec)
        self.assertIn("peak_load_mae", slice_elec)
        self.assertAlmostEqual(slice_elec["peak_load_mae"], 1.0, places=4)

        # 2. Solar 活跃出力切片
        train_data_solar = np.ones((100, C), dtype=np.float32) * 20.0
        slice_solar = compute_slice_metrics(y_pred, y_true, "Solar", train_data_solar)
        self.assertIn("active_solar_mae", slice_solar)
        self.assertAlmostEqual(slice_solar["active_solar_mae"], 1.0, places=4)

    def test_seasonal_mase_insufficient_length(self):
        """测试训练集样本长度不足以支持季节周期差分时抛出 ValueError"""
        short_series = np.ones((10, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_seasonal_naive_denominators(short_series, seasonal_m=24)

    def test_empty_slice_protection(self):
        """测试当无任何样本满足切分门限时的零除保护与安全返回"""
        N, H, C = 5, 24, 16
        # 真实值全为 1.0
        y_true = np.ones((N, H, C), dtype=np.float32) * 1.0
        y_pred = np.ones((N, H, C), dtype=np.float32) * 2.0
        # 训练集均在 50.0 以上，因此 85 分位数门限为 50.0，y_true(1.0) 绝无命中
        train_data_high = np.ones((100, C), dtype=np.float32) * 50.0
        
        # 1. Electricity 空切片
        slice_elec = compute_slice_metrics(y_pred, y_true, "Electricity", train_data_high)
        self.assertEqual(slice_elec["slice_sample_ratio"], 0.0)
        self.assertEqual(slice_elec["peak_load_mae"], 0.0)
        self.assertEqual(slice_elec["peak_load_rmse"], 0.0)

        # 2. Solar 空切片 (10% 最大出力门限为 5.0，y_true(1.0) 绝无命中)
        slice_solar = compute_slice_metrics(y_pred, y_true, "Solar", train_data_high)
        self.assertEqual(slice_solar["slice_sample_ratio"], 0.0)
        self.assertEqual(slice_solar["active_solar_mae"], 0.0)
        self.assertEqual(slice_solar["active_solar_rmse"], 0.0)

        # 3. 未知数据集类型安全回退
        slice_unknown = compute_slice_metrics(y_pred, y_true, "UnknownGrid", train_data_high)
        self.assertEqual(slice_unknown, {})

    def test_jensen_inequality_defense(self):
        """数学验证：证明跨批次平均 RMSE 必在数学上严格小于或等于全局真实 RMSE (Jensen 不等式)"""
        # 模拟两个方差显著不同的批次
        b1_err = np.ones((10, 24, 16)) * 1.0  # MSE_1 = 1.0, RMSE_1 = 1.0
        b2_err = np.ones((10, 24, 16)) * 5.0  # MSE_2 = 25.0, RMSE_2 = 5.0
        
        # 跨批次先取 RMSE 再平均 (错误做法)
        batch_avg_rmse = (1.0 + 5.0) / 2.0  # 3.0
        
        # 全局均方误差后开方 (正确做法)
        global_mse = (1.0**2 + 5.0**2) / 2.0  # (1 + 25) / 2 = 13.0
        global_rmse = np.sqrt(global_mse)     # sqrt(13) ≈ 3.6055
        
        self.assertLess(batch_avg_rmse, global_rmse)
        self.assertAlmostEqual(global_rmse, np.sqrt(13.0), places=4)


if __name__ == "__main__":
    unittest.main()

