# -*- coding: utf-8 -*-
"""
单元测试：数据加载、划分与标准化无泄漏校验。
覆盖：
1. StandardScaler 在零方差通道下的稳定性与逆变换精度。
2. Electricity 与 Solar 数据集的严格时序切分范围与时间泄漏隔离。
3. 训练数据量缩减 (train_data_fraction=0.25) 与共同起点 (common_start_idx) 边界测试。
"""

import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG
from data_loader import StandardScaler, get_data_splits, generate_benchmark_synthetic_data


class TestDataLoader(unittest.TestCase):
    def test_standard_scaler(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(50.0, 10.0, size=(1000, 16)).astype(np.float32)
        # 添加一个常数通道测试零方差保护
        raw[:, 0] = 5.0
        
        scaler = StandardScaler()
        scaler.fit(raw)
        self.assertAlmostEqual(float(scaler.std[0, 0]), 1e-8, places=8)

        scaled = scaler.transform(raw)
        # 验证缩放后均值接近 0，标准差接近 1 (非零通道)
        self.assertAlmostEqual(float(np.mean(scaled[:, 1])), 0.0, delta=0.1)
        self.assertAlmostEqual(float(np.std(scaled[:, 1])), 1.0, delta=0.1)

        # 逆变换精确还原
        inversed = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(inversed, raw, rtol=1e-5, atol=1e-5)

    def test_electricity_data_splits_and_leakage(self):
        loaders, scaler, raw_data, bounds = get_data_splits(ELECTRICITY_CONFIG)
        train_start_b, b1, b2 = bounds

        # 检查划分范围
        self.assertEqual(train_start_b, 0)
        self.assertEqual(b1, int(np.floor(0.70 * raw_data.shape[0])))
        self.assertEqual(b2, int(np.floor(0.80 * raw_data.shape[0])))

        # 检查训练集起点都在 b1 内，绝无泄漏
        train_starts = loaders["train_starts"]
        self.assertGreaterEqual(train_starts[0], ELECTRICITY_CONFIG.seq_len)
        self.assertLessEqual(train_starts[-1] + ELECTRICITY_CONFIG.pred_len, b1)

        # 检查验证集起点在 b1 到 b2 内
        val_starts = loaders["val_starts"]
        self.assertGreaterEqual(val_starts[0], b1)
        self.assertLessEqual(val_starts[-1] + ELECTRICITY_CONFIG.pred_len, b2)

        # 检查测试集起点在 b2 之后
        test_starts = loaders["test_starts"]
        self.assertGreaterEqual(test_starts[0], b2)
        self.assertLessEqual(test_starts[-1] + ELECTRICITY_CONFIG.pred_len, raw_data.shape[0])

        # 检查 Batch 维度 (若环境存在 torch)
        if torch is not None and loaders["train"] is not None:
            for x_b, y_b in loaders["train"]:
                self.assertEqual(x_b.shape[1], ELECTRICITY_CONFIG.seq_len)
                self.assertEqual(x_b.shape[2], 16)
                self.assertEqual(y_b.shape[1], ELECTRICITY_CONFIG.pred_len)
                self.assertEqual(y_b.shape[2], 16)
                break

    def test_solar_data_splits(self):
        """测试光伏数据集时间切分与窗口"""
        loaders, scaler, raw_data, bounds = get_data_splits(SOLAR_CONFIG)
        train_start_b, b1, b2 = bounds
        self.assertEqual(b1, int(np.floor(0.70 * raw_data.shape[0])))
        self.assertEqual(b2, int(np.floor(0.80 * raw_data.shape[0])))

        train_starts = loaders["train_starts"]
        self.assertGreaterEqual(train_starts[0], SOLAR_CONFIG.seq_len)
        self.assertLessEqual(train_starts[-1] + SOLAR_CONFIG.pred_len, b1)

    def test_train_fraction_and_common_start(self):
        """测试数据量缩减与共同起点偏移"""
        loaders_full, _, _, _ = get_data_splits(ELECTRICITY_CONFIG, custom_seq_len=24, common_start_idx=168)
        loaders_168, _, _, _ = get_data_splits(ELECTRICITY_CONFIG, custom_seq_len=168, common_start_idx=168)
        # 起始点集合应完全一致
        np.testing.assert_array_equal(loaders_full["train_starts"], loaders_168["train_starts"])

        # 25% 数据集缩减
        loaders_sub, _, raw_data, bounds = get_data_splits(ELECTRICITY_CONFIG, train_data_fraction=0.25)
        train_start_b, b1, _ = bounds
        self.assertEqual(train_start_b, b1 - int(b1 * 0.25))
        self.assertGreaterEqual(loaders_sub["train_starts"][0], train_start_b + ELECTRICITY_CONFIG.seq_len)

    def test_scaler_unfitted_raises_runtime_error(self):
        """测试 StandardScaler 未进行 fit 时调用 transform 或 inverse_transform 严格抛出异常"""
        fresh_scaler = StandardScaler()
        dummy = np.zeros((10, 16), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            fresh_scaler.transform(dummy)
        with self.assertRaises(RuntimeError):
            fresh_scaler.inverse_transform(dummy)

    def test_all_windows_strictly_within_target_intervals(self):
        """全量遍历检验：断言所有训练、验证与测试起点的预测目标区间完全不重叠且严格隔离"""
        for cfg in [ELECTRICITY_CONFIG, SOLAR_CONFIG]:
            loaders, _, raw_data, bounds = get_data_splits(cfg)
            train_start_b, b1, b2 = bounds
            T = raw_data.shape[0]
            H = cfg.pred_len
            
            # 训练集目标区间必须严格位于 [train_start_b, b1)
            for t in loaders["train_starts"]:
                self.assertGreaterEqual(t, train_start_b + cfg.seq_len)
                self.assertLessEqual(t + H, b1)
            
            # 验证集目标区间必须严格位于 [b1, b2)
            for t in loaders["val_starts"]:
                self.assertGreaterEqual(t, b1)
                self.assertLessEqual(t + H, b2)
            
            # 测试集目标区间必须严格位于 [b2, T)
            for t in loaders["test_starts"]:
                self.assertGreaterEqual(t, b2)
                self.assertLessEqual(t + H, T)


if __name__ == "__main__":
    unittest.main()

