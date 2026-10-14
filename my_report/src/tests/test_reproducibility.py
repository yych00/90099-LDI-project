# -*- coding: utf-8 -*-
"""
单元测试：实验可复现性与数据库跨验证断言 (Reproducibility & Cross-Verification Tests)
覆盖：
1. 校验 Baseline 与轻量级深度模型在测试集上的前向重验指标与 experiments.db 吻合度。
2. 校验测试集逆标准化反变换后的物理尺度一致性。
3. 校验模型参数量解析闭式公式与实际可训练参数量严格一致。
"""

import unittest
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG, CHECKPOINT_DIR, DB_PATH
from data_loader import get_data_splits
from train import create_model, count_parameters
from cross_verify_experiments import evaluate_model_on_test, fetch_db_benchmark_metrics


class TestReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 预载 Electricity 数据划分
        cls.elec_loaders, cls.elec_scaler, cls.elec_raw, cls.elec_bounds = get_data_splits(ELECTRICITY_CONFIG)

    def test_persistence_reproducibility(self):
        """测试 Persistence 基线重验与数据库 100% 吻合"""
        re_metrics = evaluate_model_on_test(
            "Persistence", ELECTRICITY_CONFIG,
            self.elec_loaders, self.elec_scaler, self.elec_raw, self.elec_bounds
        )
        db_metrics = fetch_db_benchmark_metrics(DB_PATH, "Electricity", "Persistence")
        self.assertAlmostEqual(re_metrics["mae_24h"], db_metrics["mae_24h"], places=3)
        self.assertAlmostEqual(re_metrics["mase_24h"], db_metrics["mase_24h"], places=3)

    def test_daily_seasonal_naive_reproducibility(self):
        """测试 DailySeasonalNaive 基线重验与数据库 100% 吻合"""
        re_metrics = evaluate_model_on_test(
            "DailySeasonalNaive", ELECTRICITY_CONFIG,
            self.elec_loaders, self.elec_scaler, self.elec_raw, self.elec_bounds
        )
        db_metrics = fetch_db_benchmark_metrics(DB_PATH, "Electricity", "DailySeasonalNaive")
        self.assertAlmostEqual(re_metrics["mae_24h"], db_metrics["mae_24h"], places=3)
        self.assertAlmostEqual(re_metrics["mase_24h"], db_metrics["mase_24h"], places=3)

    def test_dlinear_checkpoint_reproducibility(self):
        """测试 DLinear 训练权重重验与数据库 100% 吻合"""
        re_metrics = evaluate_model_on_test(
            "DLinear", ELECTRICITY_CONFIG,
            self.elec_loaders, self.elec_scaler, self.elec_raw, self.elec_bounds
        )
        db_metrics = fetch_db_benchmark_metrics(DB_PATH, "Electricity", "DLinear")
        self.assertAlmostEqual(re_metrics["mae_24h"], db_metrics["mae_24h"], places=3)
        self.assertAlmostEqual(re_metrics["mase_24h"], db_metrics["mase_24h"], places=3)

    def test_parameter_counts_exact_match(self):
        """测试模型实际参数量与报告闭式公式严格匹配"""
        dlinear = create_model("DLinear", ELECTRICITY_CONFIG)
        # DLinear 公式: 2 * (L * H + H) = 2 * (48 * 24 + 24) = 2 * (1152 + 24) = 2352
        self.assertEqual(count_parameters(dlinear), 2352)

        lstm = create_model("LSTM", ELECTRICITY_CONFIG)
        # PyTorch nn.LSTM(input_size=16, hidden_size=64) 包含 bias_ih 和 bias_hh:
        # LSTM layer = 4 * (input_size*d_h + d_h^2 + 2*d_h) = 4 * (1024 + 4096 + 128) = 20,992
        # Head: 64 * (24*16) + 24*16 = 24,576 + 384 = 24,960
        # Total = 20,992 + 24,960 = 45,952
        self.assertEqual(count_parameters(lstm), 45952)

        gru = create_model("GRU", ELECTRICITY_CONFIG)
        # PyTorch nn.GRU(input_size=16, hidden_size=64):
        # GRU layer = 3 * (input_size*d_h + d_h^2 + 2*d_h) = 3 * (1024 + 4096 + 128) = 15,744
        # Head = 24,960
        # Total = 15,744 + 24,960 = 40,704
        self.assertEqual(count_parameters(gru), 40704)

    def test_dlinear_retrain_reproducibility(self):
        """测试 DLinear 在 retrain 重新训练模式下收敛结果与数据库完全吻合"""
        retrain_metrics = evaluate_model_on_test(
            "DLinear", ELECTRICITY_CONFIG,
            self.elec_loaders, self.elec_scaler, self.elec_raw, self.elec_bounds,
            retrain=True
        )
        db_metrics = fetch_db_benchmark_metrics(DB_PATH, "Electricity", "DLinear")
        self.assertAlmostEqual(retrain_metrics["mae_24h"], db_metrics["mae_24h"], places=3)
        self.assertAlmostEqual(retrain_metrics["mase_24h"], db_metrics["mase_24h"], places=3)

    def test_classical_models_checkpoint_reproducibility(self):
        """测试 LSTM, GRU, CNN1D 在 Electricity 检查点重验与数据库完全吻合"""
        for m_name in ["LSTM", "GRU", "CNN1D"]:
            re_metrics = evaluate_model_on_test(
                m_name, ELECTRICITY_CONFIG,
                self.elec_loaders, self.elec_scaler, self.elec_raw, self.elec_bounds
            )
            db_metrics = fetch_db_benchmark_metrics(DB_PATH, "Electricity", m_name)
            self.assertAlmostEqual(re_metrics["mae_24h"], db_metrics["mae_24h"], places=3)
            self.assertAlmostEqual(re_metrics["rmse_24h"], db_metrics["rmse_24h"], places=3)
            self.assertAlmostEqual(re_metrics["mase_24h"], db_metrics["mase_24h"], places=3)
            self.assertAlmostEqual(re_metrics["mae_4h"], db_metrics["mae_4h"], places=3)

    def test_modern_and_hybrid_checkpoint_reproducibility(self):
        """测试现代模型与混合模型 (CNNLSTM, PatchTST) 在 Electricity 检查点重验与数据库完全吻合"""
        for m_name in ["CNNLSTM", "PatchTST"]:
            re_metrics = evaluate_model_on_test(
                m_name, ELECTRICITY_CONFIG,
                self.elec_loaders, self.elec_scaler, self.elec_raw, self.elec_bounds
            )
            db_metrics = fetch_db_benchmark_metrics(DB_PATH, "Electricity", m_name)
            self.assertAlmostEqual(re_metrics["mae_24h"], db_metrics["mae_24h"], places=3)
            self.assertAlmostEqual(re_metrics["rmse_24h"], db_metrics["rmse_24h"], places=3)
            self.assertAlmostEqual(re_metrics["mase_24h"], db_metrics["mase_24h"], places=3)
            self.assertAlmostEqual(re_metrics["mae_4h"], db_metrics["mae_4h"], places=3)
            self.assertAlmostEqual(re_metrics["slice_mae"], db_metrics["slice_mae"], places=3)

    def test_solar_all_models_reproducibility(self):
        """测试 Solar 数据集下基线、经典、现代与混合模型与数据库 100% 吻合度"""
        solar_loaders, solar_scaler, solar_raw, solar_bounds = get_data_splits(SOLAR_CONFIG)
        for m_name in ["Persistence", "DailySeasonalNaive", "CNN1D", "CNNLSTM", "DLinear", "PatchTST"]:
            re_metrics = evaluate_model_on_test(
                m_name, SOLAR_CONFIG,
                solar_loaders, solar_scaler, solar_raw, solar_bounds
            )
            db_metrics = fetch_db_benchmark_metrics(DB_PATH, "Solar", m_name)
            self.assertAlmostEqual(re_metrics["mae_24h"], db_metrics["mae_24h"], places=3)
            self.assertAlmostEqual(re_metrics["rmse_24h"], db_metrics["rmse_24h"], delta=0.005)
            self.assertAlmostEqual(re_metrics["mase_24h"], db_metrics["mase_24h"], places=3)


if __name__ == "__main__":
    unittest.main()
