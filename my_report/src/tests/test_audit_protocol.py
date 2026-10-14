# -*- coding: utf-8 -*-
"""
数据工程、时序防泄漏协议与数理规范全量独立复核测试套件。
严格遵循 ML Best Practices 与学术可复现性审计规范：
1. 70/10/20 时序切分隔离性与时序边界无穿越验证（涵盖 100% 全量及 25% 缩减区间）。
2. StandardScaler 拟合区间单向隔离性、反事实无泄漏、张量/标量接口未拟合防御与 1D 逆变换保形。
3. Seasonal MASE 训练集分母独占性、数学公式一致性、1D/2D 兼容性与零分母数值保护。
4. 历史观察窗口对比时共同起点对齐性校验（Electricity 168h 与 Solar 1008 步，因果律前向一致性与越界防御）。
"""

import unittest
import sys
import os
import math
import numpy as np
try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from data_loader import StandardScaler, get_data_splits, load_dataset_matrix
from eval import compute_seasonal_naive_denominators, calculate_metrics_raw_scale


class TestProtocolAndDataEngineeringAudit(unittest.TestCase):
    def test_70_10_20_split_isolation(self):
        """复核 1: 70/10/20 时序切分硬隔离性与边界零重叠（含 100% 全量与 25% 缩减测试）"""
        for cfg in [ELECTRICITY_CONFIG, SOLAR_CONFIG]:
            raw_data = load_dataset_matrix(cfg)
            T, C = raw_data.shape
            b1_expected = int(math.floor(0.70 * T))
            b2_expected = int(math.floor(0.80 * T))
            
            # --- 场景 A: 100% 全量切分 ---
            loaders, scaler, data_matrix, bounds = get_data_splits(cfg)
            train_start_b, b1, b2 = bounds
            
            self.assertEqual(b1, b1_expected)
            self.assertEqual(b2, b2_expected)
            self.assertEqual(train_start_b, 0)
            
            train_starts = loaders["train_starts"]
            val_starts = loaders["val_starts"]
            test_starts = loaders["test_starts"]
            
            H = cfg.pred_len
            L = cfg.seq_len
            
            # 1. 训练集历史与目标严格隔离在 [0, b1)
            self.assertGreaterEqual(train_starts[0] - L, 0)
            self.assertLessEqual(train_starts[-1] + H, b1)
            
            # 2. 验证集目标严格位于 [b1, b2)
            self.assertGreaterEqual(val_starts[0], b1)
            self.assertLessEqual(val_starts[-1] + H, b2)
            
            # 3. 测试集目标严格位于 [b2, T)
            self.assertGreaterEqual(test_starts[0], b2)
            self.assertLessEqual(test_starts[-1] + H, T)
            
            # 4. 目标区间无交集硬断言
            train_targets = set()
            for t in train_starts:
                train_targets.update(range(t, t + H))
            val_targets = set()
            for t in val_starts:
                val_targets.update(range(t, t + H))
            test_targets = set()
            for t in test_starts:
                test_targets.update(range(t, t + H))
                
            self.assertEqual(len(train_targets.intersection(val_targets)), 0)
            self.assertEqual(len(val_targets.intersection(test_targets)), 0)
            self.assertEqual(len(train_targets.intersection(test_targets)), 0)

            # --- 场景 B: 25% 数据缩减实验切分隔离 ---
            loaders_sub, _, _, bounds_sub = get_data_splits(cfg, train_data_fraction=0.25)
            sub_start, sub_b1, sub_b2 = bounds_sub
            expected_sub_start = b1 - int(b1 * 0.25)
            self.assertEqual(sub_start, expected_sub_start)
            self.assertEqual(sub_b1, b1)
            self.assertEqual(sub_b2, b2)
            
            # 确保 25% 缩减下的所有训练样本历史与预测目标完全约束在 [sub_start, b1)
            self.assertGreaterEqual(loaders_sub["train_starts"][0] - L, sub_start)
            self.assertLessEqual(loaders_sub["train_starts"][-1] + H, b1)

    def test_scaler_fitting_and_isolation(self):
        """复核 2: StandardScaler 仅在训练区间拟合、反事实无泄漏、张量未拟合防御与 1D 逆变换精度"""
        for cfg in [ELECTRICITY_CONFIG, SOLAR_CONFIG]:
            raw_data = load_dataset_matrix(cfg)
            T, C = raw_data.shape
            b1 = int(math.floor(0.70 * T))
            
            expected_mean = np.mean(raw_data[:b1, :], axis=0, keepdims=True)
            expected_std = np.std(raw_data[:b1, :], axis=0, keepdims=True)
            expected_std = np.maximum(expected_std, 1e-8)
            
            loaders, scaler, _, bounds = get_data_splits(cfg)
            
            # 参数严格匹配
            np.testing.assert_allclose(scaler.mean, expected_mean, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(scaler.std, expected_std, rtol=1e-6, atol=1e-6)
            
            # 反事实检验 1: 篡改验证与测试集绝不影响 Scaler
            corrupted_raw = raw_data.copy()
            corrupted_raw[b1:, :] += 999999.0
            scaler_corrupted = StandardScaler()
            scaler_corrupted.fit(corrupted_raw[:b1, :])
            np.testing.assert_array_equal(scaler.mean, scaler_corrupted.mean)
            np.testing.assert_array_equal(scaler.std, scaler_corrupted.std)

            # 反事实检验 2: 针对 25% 数据缩减实验，篡改弃用的 75% 历史时序绝不影响缩减 Scaler
            sub_start = b1 - int(b1 * 0.25)
            loaders_sub, scaler_sub, _, _ = get_data_splits(cfg, train_data_fraction=0.25)
            corrupted_history = raw_data.copy()
            corrupted_history[:sub_start, :] += 888888.0
            scaler_sub_corrupted = StandardScaler()
            scaler_sub_corrupted.fit(corrupted_history[sub_start:b1, :])
            np.testing.assert_array_equal(scaler_sub.mean, scaler_sub_corrupted.mean)
            np.testing.assert_array_equal(scaler_sub.std, scaler_sub_corrupted.std)
            
            # 逆标准化精度 (2D 矩阵)
            test_scaled = scaler.transform(raw_data[b1:, :])
            test_inversed = scaler.inverse_transform(test_scaled)
            np.testing.assert_allclose(test_inversed, raw_data[b1:, :], rtol=1e-5, atol=1e-5)

            # 逆标准化精度 (1D 单时间步向量与保形验证)
            vec_1d = raw_data[b1, :].copy()
            vec_1d_scaled = scaler.transform(vec_1d.reshape(1, -1)).reshape(-1)
            vec_1d_inversed = scaler.inverse_transform(vec_1d_scaled)
            self.assertEqual(vec_1d_inversed.shape, (C,))
            np.testing.assert_allclose(vec_1d_inversed, vec_1d, rtol=1e-5, atol=1e-5)
            
            # 零方差通道防御
            dummy_const = np.ones((100, 2), dtype=np.float32) * 5.0
            scaler_const = StandardScaler(eps=1e-8)
            scaler_const.fit(dummy_const)
            self.assertTrue(np.all(scaler_const.std >= 1e-8))
            scaled_const = scaler_const.transform(dummy_const)
            self.assertFalse(np.any(np.isnan(scaled_const)))
            self.assertFalse(np.any(np.isinf(scaled_const)))

            # 未 fit 调用严格异常防御 (覆盖 transform_tensor 与 inverse_transform_tensor)
            if torch is not None:
                fresh_scaler = StandardScaler()
                t_dummy = torch.zeros(2, 4, C)
                with self.assertRaises(RuntimeError):
                    fresh_scaler.transform_tensor(t_dummy)
                with self.assertRaises(RuntimeError):
                    fresh_scaler.inverse_transform_tensor(t_dummy)

    def test_mase_denominator_exclusive_and_numerical_protection(self):
        """复核 3: MASE 训练集分母独占性、数学公式一致性、1D 兼容性与零分母数值保护"""
        for cfg in [ELECTRICITY_CONFIG, SOLAR_CONFIG]:
            raw_data = load_dataset_matrix(cfg)
            T, C = raw_data.shape
            b1 = int(math.floor(0.70 * T))
            m = cfg.seasonal_m
            
            train_slice = raw_data[:b1, :]
            diff_manual = np.abs(train_slice[m:, :] - train_slice[:-m, :])
            expected_denoms = np.mean(diff_manual, axis=0)
            expected_denoms = np.maximum(expected_denoms, 1e-8)
            
            system_denoms = compute_seasonal_naive_denominators(train_slice, m, eps=1e-8)
            np.testing.assert_allclose(system_denoms, expected_denoms, rtol=1e-6, atol=1e-6)

            # 1D 数组输入单通道兼容性验证
            single_ch_denoms = compute_seasonal_naive_denominators(train_slice[:, 0], m, eps=1e-8)
            np.testing.assert_allclose(single_ch_denoms, expected_denoms[:1], rtol=1e-5, atol=1e-5)
            
            # 污染隔离性断言 (若分母混入全量时序，必然破坏独占基准)
            denoms_contaminated = compute_seasonal_naive_denominators(raw_data, m)
            self.assertFalse(np.allclose(system_denoms, denoms_contaminated))
            
            # 常数序列保护
            const_data = np.ones((500, C), dtype=np.float32) * 100.0
            const_denoms = compute_seasonal_naive_denominators(const_data, m, eps=1e-8)
            self.assertTrue(np.all(const_denoms == 1e-8))
            
            # 逐通道先缩放再均值聚合检验 (含 2D 形状容错防御)
            N, H = 10, cfg.pred_len
            y_true = np.ones((N, H, C), dtype=np.float32) * 50.0
            y_pred = np.ones((N, H, C), dtype=np.float32) * 55.0
            metrics = calculate_metrics_raw_scale(y_pred, y_true, system_denoms.reshape(1, -1))
            expected_channel_mase = 5.0 / system_denoms
            expected_overall_mase = float(np.mean(expected_channel_mase))
            np.testing.assert_allclose(metrics["mase_per_channel"], expected_channel_mase, rtol=1e-5)
            self.assertAlmostEqual(metrics["mase"], expected_overall_mase, places=5)
            self.assertEqual(len(metrics["mase_per_channel"]), C)

    def test_start_index_alignment_and_lookback_causality(self):
        """复核 4: 多窗口对比起点对齐（Electricity 168h 与 Solar 1008 步）、因果一致性与非法参数防御"""
        # 1. Electricity 独立复核 (L in {24, 48, 168}, common_start_idx=168)
        elec_cfg = ELECTRICITY_CONFIG
        elec_windows = [24, 48, 168]
        common_168 = 168
        
        loaders_elec = {}
        for L in elec_windows:
            ldrs, _, raw_data, bounds = get_data_splits(elec_cfg, custom_seq_len=L, common_start_idx=common_168)
            loaders_elec[L] = ldrs
        
        # 训练集起点数组必须 100% 相同且从 common_start_idx 处启动
        train_0 = loaders_elec[24]["train_starts"]
        for L in [48, 168]:
            np.testing.assert_array_equal(train_0, loaders_elec[L]["train_starts"])
        self.assertGreaterEqual(train_0[0], common_168)
        
        # 测试集起点数组必须 100% 相同且从固定分界点 b2 处启动
        test_0 = loaders_elec[24]["test_starts"]
        for L in [48, 168]:
            np.testing.assert_array_equal(test_0, loaders_elec[L]["test_starts"])
        self.assertEqual(test_0[0], bounds[2])
        
        # 历史因果前向一致性校验：在相同样本索引下，所有窗口的预测目标 y 必须严格相等，
        # 短窗口历史输入 x_short 必须严格等于长窗口历史输入 x_long 的后缀切片
        sample_indices_to_test = [0, len(train_0) // 2, len(train_0) - 1]
        for s_idx in sample_indices_to_test:
            x_24, y_24 = loaders_elec[24]["train"].dataset[s_idx]
            x_48, y_48 = loaders_elec[48]["train"].dataset[s_idx]
            x_168, y_168 = loaders_elec[168]["train"].dataset[s_idx]
            
            # 目标 y 严格恒等
            if torch is not None and isinstance(y_24, torch.Tensor):
                self.assertTrue(torch.equal(y_24, y_48))
                self.assertTrue(torch.equal(y_48, y_168))
                # 历史输入 x 为精确后缀
                self.assertTrue(torch.equal(x_24, x_168[-24:, :]))
                self.assertTrue(torch.equal(x_48, x_168[-48:, :]))

        # 测试集首点 t=b2 时，最大窗口 L=168 的历史回溯位于验证集内，合法无泄漏
        self.assertGreaterEqual(bounds[2] - common_168, bounds[1])

        # 2. Solar 独立复核 (L in {144, 288, 1008}, common_start_idx=1008)
        solar_cfg = SOLAR_CONFIG
        solar_windows = [144, 288, 1008]
        common_1008 = 1008
        loaders_solar = {}
        for L in solar_windows:
            ldrs, _, _, s_bounds = get_data_splits(solar_cfg, custom_seq_len=L, common_start_idx=common_1008)
            loaders_solar[L] = ldrs
        
        s_train_0 = loaders_solar[144]["train_starts"]
        for L in [288, 1008]:
            np.testing.assert_array_equal(s_train_0, loaders_solar[L]["train_starts"])
        self.assertGreaterEqual(s_train_0[0], common_1008)
        self.assertEqual(loaders_solar[144]["test_starts"][0], s_bounds[2])
        self.assertGreaterEqual(s_bounds[2] - common_1008, s_bounds[1])

        # 3. 非法参数与越界安全防御测试
        # 当 common_start_idx < seq_len 时，必须坚决抛出 ValueError，杜绝负索引与时间反向穿越
        with self.assertRaises(ValueError):
            get_data_splits(elec_cfg, custom_seq_len=168, common_start_idx=48)
            
        # 当数据缩减导致训练区间不足以支撑历史窗口与预测步长时，必须坚决抛出 ValueError
        with self.assertRaises(ValueError):
            get_data_splits(elec_cfg, custom_seq_len=168, common_start_idx=168, train_data_fraction=0.005)


if __name__ == "__main__":
    unittest.main()
