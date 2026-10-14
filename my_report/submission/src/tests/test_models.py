# -*- coding: utf-8 -*-
"""
单元测试：所有 8 种预测模型的输入输出维度、正向传播、反向传播及边界异常校验。
包含：
1. Persistence 与 Daily Seasonal Naive 基线检验。
2. 6 种深度学习模型在 Electricity (L=48, H=24) 与 Solar (L=288, H=144) 尺寸下的前向与反向梯度检验。
3. PatchTST 在非整除历史窗口 (例如 seq_len=50) 下的 padding 机制与末尾关键时间步梯度传导检验。
4. 异常输入边界保护检验 (如历史窗口不足以支持日周期基线)。
"""

import unittest
import sys
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if HAS_TORCH:
    from models import (
        PersistenceModel, DailySeasonalNaiveModel,
        LSTMModel, GRUModel, CNN1DModel,
        CNNLSTMModel, DLinearModel, PatchTSTModel
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch 未安装在当前环境中，跳过模型前向与梯度测试")
class TestModels(unittest.TestCase):

    def setUp(self):
        self.B = 4
        self.L = 48
        self.H = 24
        self.C = 16
        self.dummy_input = torch.randn(self.B, self.L, self.C)

    def test_baselines(self):
        # 1. Persistence
        pers_model = PersistenceModel(pred_len=self.H)
        out = pers_model(self.dummy_input)
        self.assertEqual(out.shape, (self.B, self.H, self.C))
        self.assertTrue(torch.allclose(out[:, 0, :], self.dummy_input[:, -1, :]))
        self.assertTrue(torch.allclose(out[:, -1, :], self.dummy_input[:, -1, :]))

        # 2. Daily Seasonal Naive (Electricity)
        dsn_model = DailySeasonalNaiveModel(pred_len=self.H, seasonal_m=24)
        out_dsn = dsn_model(self.dummy_input)
        self.assertEqual(out_dsn.shape, (self.B, self.H, self.C))

        # 3. Daily Seasonal Naive 边界异常测试 (L < m 抛出 ValueError)
        short_input = torch.randn(self.B, 12, self.C)
        with self.assertRaises(ValueError):
            dsn_model(short_input)

    def test_deep_learning_models_forward_and_backward(self):
        models = [
            ("LSTM", LSTMModel(num_channels=self.C, seq_len=self.L, pred_len=self.H)),
            ("GRU", GRUModel(num_channels=self.C, seq_len=self.L, pred_len=self.H)),
            ("CNN1D", CNN1DModel(num_channels=self.C, seq_len=self.L, pred_len=self.H)),
            ("CNNLSTM", CNNLSTMModel(num_channels=self.C, seq_len=self.L, pred_len=self.H)),
            ("DLinear", DLinearModel(num_channels=self.C, seq_len=self.L, pred_len=self.H)),
            ("PatchTST", PatchTSTModel(num_channels=self.C, seq_len=self.L, pred_len=self.H, patch_len=16, stride=8))
        ]

        for name, model in models:
            out = model(self.dummy_input)
            self.assertEqual(out.shape, (self.B, self.H, self.C), f"{name} 输出形状不匹配！")
            
            # 测试反向传播梯度计算
            loss = out.sum()
            loss.backward()
            
            # 校验有可更新梯度的参数
            has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
            self.assertTrue(has_grad, f"{name} 反向传播未产生有效梯度！")

    def test_solar_dimensions(self):
        """测试高频光伏数据集规格 (L=288, H=144) 包含所有基线与基础模型"""
        solar_input = torch.randn(2, 288, 16)
        
        # 1. Persistence on Solar
        pers_solar = PersistenceModel(pred_len=144)
        out_pers = pers_solar(solar_input)
        self.assertEqual(out_pers.shape, (2, 144, 16))
        self.assertTrue(torch.allclose(out_pers[:, 0, :], solar_input[:, -1, :]))

        # 2. Daily Seasonal Naive on Solar (m=144)
        dsn_solar = DailySeasonalNaiveModel(pred_len=144, seasonal_m=144)
        out_dsn = dsn_solar(solar_input)
        self.assertEqual(out_dsn.shape, (2, 144, 16))
        self.assertTrue(torch.allclose(out_dsn, solar_input[:, -144:, :]))

        # 3. LSTM on Solar
        lstm_solar = LSTMModel(num_channels=16, seq_len=288, pred_len=144)
        out_lstm = lstm_solar(solar_input)
        self.assertEqual(out_lstm.shape, (2, 144, 16))

        # 4. GRU on Solar
        gru_solar = GRUModel(num_channels=16, seq_len=288, pred_len=144)
        out_gru = gru_solar(solar_input)
        self.assertEqual(out_gru.shape, (2, 144, 16))

        # 5. CNN1D on Solar
        cnn_solar = CNN1DModel(num_channels=16, seq_len=288, pred_len=144)
        out_cnn = cnn_solar(solar_input)
        self.assertEqual(out_cnn.shape, (2, 144, 16))

        # 6. CNNLSTM on Solar
        cnnlstm_solar = CNNLSTMModel(num_channels=16, seq_len=288, pred_len=144)
        out_cnnlstm = cnnlstm_solar(solar_input)
        self.assertEqual(out_cnnlstm.shape, (2, 144, 16))

        # 7. DLinear & PatchTST
        dl_model = DLinearModel(num_channels=16, seq_len=288, pred_len=144)
        out_dl = dl_model(solar_input)
        self.assertEqual(out_dl.shape, (2, 144, 16))

        patch_model = PatchTSTModel(num_channels=16, seq_len=288, pred_len=144, patch_len=16, stride=8)
        out_patch = patch_model(solar_input)
        self.assertEqual(out_patch.shape, (2, 144, 16))

    def test_parameter_counts_exact_match(self):
        """测试模型实际参数量与报告推导理论公式及数据库记录的 100% 精确一致性"""
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Electricity (L=48, H=24, C=16)
        self.assertEqual(count_params(PersistenceModel(pred_len=24)), 0)
        self.assertEqual(count_params(DailySeasonalNaiveModel(pred_len=24, seasonal_m=24)), 0)
        self.assertEqual(count_params(LSTMModel(num_channels=16, seq_len=48, pred_len=24)), 45952)
        self.assertEqual(count_params(GRUModel(num_channels=16, seq_len=48, pred_len=24)), 40704)
        self.assertEqual(count_params(CNN1DModel(num_channels=16, seq_len=48, pred_len=24)), 401568)
        self.assertEqual(count_params(CNNLSTMModel(num_channels=16, seq_len=48, pred_len=24)), 61504)
        self.assertEqual(count_params(DLinearModel(num_channels=16, seq_len=48, pred_len=24)), 2352)
        self.assertEqual(count_params(PatchTSTModel(num_channels=16, seq_len=48, pred_len=24, patch_len=16, stride=8)), 76056)
        # PatchTST 敏感性长窗口 (L=168, H=24) -> 20 个 tokens
        self.assertEqual(count_params(PatchTSTModel(num_channels=16, seq_len=168, pred_len=24, patch_len=16, stride=8)), 100056)

        # Solar (L=288, H=144, C=16)
        self.assertEqual(count_params(PersistenceModel(pred_len=144)), 0)
        self.assertEqual(count_params(DailySeasonalNaiveModel(pred_len=144, seasonal_m=144)), 0)
        self.assertEqual(count_params(LSTMModel(num_channels=16, seq_len=288, pred_len=144)), 170752)
        self.assertEqual(count_params(GRUModel(num_channels=16, seq_len=288, pred_len=144)), 165504)
        self.assertEqual(count_params(CNN1DModel(num_channels=16, seq_len=288, pred_len=144)), 2369568)
        self.assertEqual(count_params(CNNLSTMModel(num_channels=16, seq_len=288, pred_len=144)), 186304)
        self.assertEqual(count_params(DLinearModel(num_channels=16, seq_len=288, pred_len=144)), 83232)
        self.assertEqual(count_params(PatchTSTModel(num_channels=16, seq_len=288, pred_len=144, patch_len=16, stride=8)), 392976)

    def test_daily_seasonal_naive_tiling_and_invalid_dim(self):
        """测试 DailySeasonalNaive 在 H > m 时的周期平铺逻辑以及非 3 维输入的异常防御"""
        # 1. H > m 平铺: pred_len=48, seasonal_m=24
        dsn = DailySeasonalNaiveModel(pred_len=48, seasonal_m=24)
        x = torch.randn(2, 48, 16)
        out = dsn(x)
        self.assertEqual(out.shape, (2, 48, 16))
        self.assertTrue(torch.allclose(out[:, :24, :], x[:, -24:, :]))
        self.assertTrue(torch.allclose(out[:, 24:, :], x[:, -24:, :]))

        # 2. 非 3 维输入防御测试
        with self.assertRaises(ValueError):
            dsn(torch.randn(48, 16))
        pers = PersistenceModel(pred_len=24)
        with self.assertRaises(ValueError):
            pers(torch.randn(48, 16))
        lstm = LSTMModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            lstm(torch.randn(48, 16))
        cnn1d = CNN1DModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            cnn1d(torch.randn(48, 16))
        cnnlstm = CNNLSTMModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            cnnlstm(torch.randn(48, 16))
        dlinear = DLinearModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            dlinear(torch.randn(48, 16))
        patchtst = PatchTSTModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            patchtst(torch.randn(48, 16))

    def test_patchtst_non_aligned_window_gradients(self):
        """测试 PatchTST 当历史窗口非 stride 整除时，末尾关键时间步是否得到梯度覆盖"""
        patch_model = PatchTSTModel(num_channels=1, seq_len=50, pred_len=10, patch_len=16, stride=8)
        x = torch.randn(1, 50, 1, requires_grad=True)
        out = patch_model(x)
        self.assertEqual(out.shape, (1, 10, 1))
        out.sum().backward()
        # 确保最后两个时间步（历史最邻近观测）梯度不为 0
        last_grads = x.grad[0, 48:, 0]
        self.assertFalse(torch.all(last_grads == 0), "PatchTST 丢弃了最后时间步的有效时序信息！")

    def test_cnn1d_adaptive_pool_edge_cases_and_gradients(self):
        """测试 1D-CNN 在 L < 16 极端短输入下的自适应上采样与反向梯度健康度"""
        cnn_model = CNN1DModel(num_channels=16, seq_len=48, pred_len=24)
        for short_l in [8, 12, 16]:
            x = torch.randn(2, short_l, 16, requires_grad=True)
            out = cnn_model(x)
            self.assertEqual(out.shape, (2, 24, 16), f"L={short_l} 输出形状异常！")
            loss = out.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertFalse(torch.isnan(x.grad).any(), f"L={short_l} 出现梯度 NaN！")
            self.assertGreater(x.grad.norm().item(), 0.0, f"L={short_l} 梯度为 0！")

    def test_cnn1d_capacity_locked_across_windows(self):
        """测试 1D-CNN 跨任意输入时序长度 L 的模型参数锁定与结构稳定性"""
        cnn_model = CNN1DModel(num_channels=16, seq_len=48, pred_len=24)
        base_params = sum(p.numel() for p in cnn_model.parameters())
        for test_l in [24, 48, 96, 168, 288]:
            x = torch.randn(2, test_l, 16)
            out = cnn_model(x)
            self.assertEqual(out.shape, (2, 24, 16))
            cur_params = sum(p.numel() for p in cnn_model.parameters())
            self.assertEqual(cur_params, base_params)


if __name__ == "__main__":
    unittest.main()
