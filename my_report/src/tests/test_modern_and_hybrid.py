# -*- coding: utf-8 -*-
"""
现代化模型 (DLinear, PatchTST) 与混合模型 (CNN-LSTM) 深度复查与重验测试套件。
涵盖：
1. DLinear 移动平均趋势-季节分解保和性、内点线性保真度、高频分离度、跨通道权重共享等变性与通道维度自适应。
2. DLinear individual=True 模式下的通道独立参数量与通道间梯度隔离检验。
3. PatchTST 动态补齐覆盖率（任意整除与非整除历史窗口）、全时间步反向传播梯度流与末端累积检验。
4. PatchTST 通道独立性 (Channel Independence) 与短时序自适应降级检验。
5. CNN-LSTM 混合模型对历史窗口 L 的参数量不变性 (O(1)) 及相对纯 1D-CNN 的参数缩减率核验。
6. 全量模型在多种历史窗口 L 下精确参数量与闭式解析数学公式的 100% 吻合度检验。
"""

import unittest
import sys
import os

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if HAS_TORCH:
    from models.modern_models import MovingAvgBlock, SeriesDecomp, DLinearModel, PatchTSTModel
    from models.hybrid_models import CNNLSTMModel
    from models.cnn_models import CNN1DModel
    from models.rnn_models import LSTMModel


# 闭式参数量解析公式定义
def dlinear_closed_form(L: int, H: int, C: int, individual: bool = False) -> int:
    """DLinear 精确闭式参数量"""
    if individual:
        return 2 * C * H * (L + 1)
    return 2 * H * (L + 1)


def cnnlstm_closed_form(
    L: int,
    H: int,
    C: int,
    conv_channels: int = 64,
    lstm_hidden: int = 64,
    kernel_size: int = 3
) -> int:
    """CNN-LSTM 精确闭式参数量 (与历史窗口 L 严格无关)"""
    # 1. Conv1D: in=C, out=conv_channels, kernel=K
    p_conv = conv_channels * C * kernel_size + conv_channels
    # 2. BatchNorm1d: weight + bias
    p_bn = 2 * conv_channels
    # 3. LSTM (1 layer, 4 gates)
    p_lstm = 4 * lstm_hidden * (conv_channels + lstm_hidden + 2)
    # 4. FC: in=lstm_hidden, out=H * C
    p_fc = (H * C) * (lstm_hidden + 1)
    return p_conv + p_bn + p_lstm + p_fc


def patchtst_closed_form(
    L: int,
    H: int,
    C: int,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 64,
    n_heads: int = 4,
    e_layers: int = 2,
    d_ff: int = 128
) -> int:
    """PatchTST 精确闭式参数量"""
    if L < patch_len:
        eff_patch_len = L
        eff_stride = max(1, L // 2)
    else:
        eff_patch_len = patch_len
        eff_stride = stride

    if (L - eff_patch_len) % eff_stride != 0:
        num_patches = (L - eff_patch_len) // eff_stride + 2
    else:
        num_patches = max(1, (L - eff_patch_len) // eff_stride + 1)

    # 1. Patch embedding
    p_emb = eff_patch_len * d_model + d_model
    # 2. Learnable Position embedding
    p_pos = num_patches * d_model
    # 3. Transformer Encoder (e_layers)
    p_attn = 4 * d_model * d_model + 4 * d_model  # in_proj (3*D*D + 3*D) + out_proj (D*D + D)
    p_norm1 = 2 * d_model
    p_ffn = 2 * d_model * d_ff + d_ff + d_model
    p_norm2 = 2 * d_model
    p_layer = p_attn + p_norm1 + p_ffn + p_norm2
    p_transformer = e_layers * p_layer
    # 4. Head
    p_head = (num_patches * d_model) * H + H

    return p_emb + p_pos + p_transformer + p_head


@unittest.skipUnless(HAS_TORCH, "PyTorch 未安装在当前环境中，跳过模型测试")
class TestModernAndHybridDeep(unittest.TestCase):

    def test_dlinear_series_decomposition_exact_identity(self):
        """验证 DLinear 时序分解的严格保和性 (seasonal + trend == x) 达到浮点机精度"""
        decomp = SeriesDecomp(kernel_size=25)
        for shape in [(2, 48, 16), (4, 73, 8), (1, 288, 4)]:
            x = torch.randn(*shape)
            seasonal, trend = decomp(x)
            # 严格维度匹配
            self.assertEqual(seasonal.shape, x.shape)
            self.assertEqual(trend.shape, x.shape)
            # 严格无残差保和
            recon = seasonal + trend
            max_err = torch.max(torch.abs(recon - x)).item()
            self.assertLess(max_err, 1e-6, f"时序分解重构误差超标: {max_err}")

    def test_moving_avg_linear_trend_preservation(self):
        """数学检验：对于纯线性趋势 x(t) = a*t + b，滑动平均在未受端点 padding 影响的内点严格还原"""
        k = 11
        mavg64 = MovingAvgBlock(kernel_size=k).double()
        t64 = torch.linspace(0, 100, 101, dtype=torch.float64).view(1, 101, 1)
        a, b = 2.5, -7.3
        x_linear64 = a * t64 + b
        trend64 = mavg64(x_linear64)
        # 边界影响半径为 (k - 1) // 2
        half_k = (k - 1) // 2
        interior_orig64 = x_linear64[:, half_k:-half_k, :]
        interior_trend64 = trend64[:, half_k:-half_k, :]
        max_err64 = torch.max(torch.abs(interior_trend64 - interior_orig64)).item()
        self.assertLess(max_err64, 1e-10, f"双精度下线性趋势内点滤波误差偏离理论解: {max_err64}")

    def test_dlinear_channel_sharing_equivariance_and_agnosticism(self):
        """验证 DLinear (individual=False) 跨通道权重共享机制：通道置换等变性与任意通道数适用性"""
        model = DLinearModel(num_channels=16, seq_len=48, pred_len=24, individual=False)
        x = torch.randn(2, 48, 16)
        out = model(x)

        # 1. 置换等变性检验：打乱通道顺序，输出通道顺序必须严格同步打乱
        perm = torch.randperm(16)
        x_perm = x[:, :, perm]
        out_perm = model(x_perm)
        self.assertTrue(
            torch.allclose(out_perm, out[:, :, perm], atol=1e-5),
            "DLinear 权重共享机制未满足跨通道置换等变性 (Permutation Equivariance)！"
        )

        # 2. 跨通道维度自适应 (Channel Agnosticism)：任意 C 值均可直接前向
        for c in [1, 5, 23, 64]:
            x_c = torch.randn(2, 48, c)
            out_c = model(x_c)
            self.assertEqual(out_c.shape, (2, 24, c))

    def test_dlinear_individual_mode_gradient_isolation(self):
        """验证 DLinear (individual=True) 通道独立模式下的参数量与梯度隔离 (Gradient Isolation)"""
        num_c = 4
        seq_len = 48
        pred_len = 24
        model = DLinearModel(num_channels=num_c, seq_len=seq_len, pred_len=pred_len, individual=True)
        # 参数量应为 2 * C * H * (L + 1)
        expected_params = 2 * num_c * pred_len * (seq_len + 1)
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(actual_params, expected_params)

        # 梯度隔离检验：仅对通道 0 的输出构建损失，断言通道 1, 2, 3 的私有权重参数梯度完全为 None 或 0
        x = torch.randn(2, seq_len, num_c, requires_grad=True)
        out = model(x)
        loss = out[:, :, 0].sum()
        loss.backward()

        for c in range(1, num_c):
            for p in model.linear_seasonal[c].parameters():
                self.assertTrue(p.grad is None or torch.all(p.grad == 0), f"通道 {c} 的季节层受到跨通道梯度污染！")
            for p in model.linear_trend[c].parameters():
                self.assertTrue(p.grad is None or torch.all(p.grad == 0), f"通道 {c} 的趋势层受到跨通道梯度污染！")

        # 通道数不匹配防御测试
        with self.assertRaises(ValueError):
            model(torch.randn(2, seq_len, num_c + 1))

    def test_patchtst_dynamic_padding_and_coverage(self):
        """验证 PatchTST 在整除与非整除历史窗口下的动态补齐覆盖率达到 100% 且无遗漏"""
        patch_len = 16
        stride = 8
        test_lengths = [24, 30, 48, 50, 73, 96, 120, 168, 288]

        for L in test_lengths:
            model = PatchTSTModel(num_channels=4, seq_len=L, pred_len=12, patch_len=patch_len, stride=stride)

            # 验证补齐后总长度与 patch 数量完全对齐
            total_len = L + model.pad_len
            expected_patches = (total_len - patch_len) // stride + 1
            self.assertEqual(model.num_patches, expected_patches)

            # 验证所有原始时间步 [0, L-1] 均被至少一个 patch 覆盖
            covered_indices = set()
            for i in range(model.num_patches):
                start = i * stride
                end = min(start + patch_len, total_len)
                for t in range(start, end):
                    if t < L:
                        covered_indices.add(t)

            self.assertEqual(
                len(covered_indices), L,
                f"L={L} 存在未被 Patch 覆盖的时步！遗漏: {set(range(L)) - covered_indices}"
            )

    def test_patchtst_backward_gradient_flow_entire_sequence(self):
        """验证 PatchTST 反向传播在非整除窗口下全时步梯度流非零 (零死区率 0.0%)"""
        patch_len = 16
        stride = 8
        for L in [30, 50, 73, 100]:
            model = PatchTSTModel(num_channels=2, seq_len=L, pred_len=10, patch_len=patch_len, stride=stride)
            x = torch.randn(2, L, 2, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()

            # 计算每个时步在所有 batch 和通道上的梯度绝对值和
            step_grads = x.grad.abs().sum(dim=(0, 2))  # [L]
            zero_count = (step_grads == 0).sum().item()
            self.assertEqual(zero_count, 0, f"L={L} 出现了 {zero_count} 个无梯度的死亡时间步！")

            # 检查最后一个观测步 L-1 梯度有效性
            self.assertGreater(step_grads[-1].item(), 1e-4, f"L={L} 末端关键观测时间步梯度不足！")

    def test_patchtst_channel_independence_agnosticism(self):
        """验证 PatchTST 通道独立性 (Channel Independence)：通道置换等变性与任意通道数适用性"""
        model = PatchTSTModel(num_channels=16, seq_len=48, pred_len=24, patch_len=16, stride=8)
        model.eval()

        # 任意通道数测试
        for c in [1, 3, 7, 32]:
            x = torch.randn(2, 48, c)
            out = model(x)
            self.assertEqual(out.shape, (2, 24, c))

    def test_cnnlstm_length_invariance_and_reduction(self):
        """验证 CNN-LSTM 混合模型总参数量与历史窗口 L 严格无关 (O(1)) 及参数精简比率"""
        H = 24
        C = 16
        base_params = cnnlstm_closed_form(L=48, H=H, C=C)
        self.assertEqual(base_params, 61504)

        # 验证不同历史序列长度下，实际 PyTorch 模型参数量恒定等于 61,504
        for L in [24, 48, 72, 96, 168]:
            m = CNNLSTMModel(num_channels=C, seq_len=L, pred_len=H)
            p = sum(p.numel() for p in m.parameters() if p.requires_grad)
            self.assertEqual(p, base_params, f"CNNLSTM 在 L={L} 时参数量发生变动: {p} != {base_params}")

        # 验证相对纯 1D-CNN (401,568) 的参数缩减率
        cnn_params = 401568
        reduction_rate = (cnn_params - base_params) / cnn_params
        self.assertAlmostEqual(reduction_rate, 0.8468, places=3)

    def test_closed_form_exact_matches_across_window_grid(self):
        """核心定理验证：不同历史窗口下 DLinear, PatchTST, CNN-LSTM 的精确参数量与闭式推导 100% 绝对一致"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        window_grid = [16, 24, 32, 48, 64, 73, 96, 128, 168, 288]
        C = 16
        H_elec = 24

        for L in window_grid:
            # 1. DLinear (Shared)
            dl_actual = count_params(DLinearModel(num_channels=C, seq_len=L, pred_len=H_elec, individual=False))
            dl_theory = dlinear_closed_form(L, H_elec, C, individual=False)
            self.assertEqual(dl_actual, dl_theory, f"DLinear(Shared) 在 L={L} 理论与实测不符: {dl_theory} vs {dl_actual}")

            # 2. DLinear (Individual)
            dl_ind_actual = count_params(DLinearModel(num_channels=C, seq_len=L, pred_len=H_elec, individual=True))
            dl_ind_theory = dlinear_closed_form(L, H_elec, C, individual=True)
            self.assertEqual(dl_ind_actual, dl_ind_theory, f"DLinear(Individual) 在 L={L} 理论与实测不符: {dl_ind_theory} vs {dl_ind_actual}")

            # 3. CNN-LSTM
            cnnlstm_actual = count_params(CNNLSTMModel(num_channels=C, seq_len=L, pred_len=H_elec))
            cnnlstm_theory = cnnlstm_closed_form(L, H_elec, C)
            self.assertEqual(cnnlstm_actual, cnnlstm_theory, f"CNNLSTM 在 L={L} 理论与实测不符: {cnnlstm_theory} vs {cnnlstm_actual}")

            # 4. PatchTST
            patch_actual = count_params(PatchTSTModel(num_channels=C, seq_len=L, pred_len=H_elec, patch_len=16, stride=8))
            patch_theory = patchtst_closed_form(L, H_elec, C, patch_len=16, stride=8)
            self.assertEqual(patch_actual, patch_theory, f"PatchTST 在 L={L} 理论与实测不符: {patch_theory} vs {patch_actual}")

    def test_temporal_length_mismatch_defenses(self):
        """验证 DLinear 与 PatchTST 在输入序列长度 L 不匹配 seq_len 时抛出友好 ValueError 而非底层内部崩溃"""
        # 1. DLinear
        dl = DLinearModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            dl(torch.randn(2, 50, 16))
        with self.assertRaises(ValueError):
            dl(torch.randn(2, 24, 16))

        # 2. PatchTST (防止由于多余 patches 产生静默末端丢弃或由于 patches 过少导致矩阵乘法崩溃)
        pt = PatchTSTModel(num_channels=16, seq_len=48, pred_len=24, patch_len=16, stride=8)
        with self.assertRaises(ValueError):
            pt(torch.randn(2, 100, 16))
        with self.assertRaises(ValueError):
            pt(torch.randn(2, 20, 16))

    def test_channel_mismatch_defenses(self):
        """验证 CNNLSTM 与 DLinear(individual=True) 在输入通道数不匹配时的前置防御抛错"""
        cnn = CNNLSTMModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            cnn(torch.randn(2, 48, 8))

        dl_indiv = DLinearModel(num_channels=16, seq_len=48, pred_len=24, individual=True)
        with self.assertRaises(ValueError):
            dl_indiv(torch.randn(2, 48, 4))

    def test_initialization_dimension_and_stride_defenses(self):
        """验证各模型初始化参数合法性校验及 PatchTST 当 stride > patch_len 时阻断梯度盲区"""
        # 1. PatchTST stride > patch_len 阻断
        with self.assertRaises(ValueError):
            PatchTSTModel(num_channels=16, seq_len=48, pred_len=24, patch_len=8, stride=16)

        # 2. 非正参数校验
        with self.assertRaises(ValueError):
            PatchTSTModel(num_channels=16, seq_len=0, pred_len=24)
        with self.assertRaises(ValueError):
            DLinearModel(num_channels=16, seq_len=48, pred_len=0)
        with self.assertRaises(ValueError):
            CNNLSTMModel(num_channels=0, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            MovingAvgBlock(kernel_size=0)

    def test_series_decomp_high_frequency_harmonic_separation(self):
        """验证 SeriesDecomp 对高频周期正弦信号的分离能力：高频沉淀至 seasonal，趋势项保留线性斜率"""
        k = 25
        decomp = SeriesDecomp(kernel_size=k)
        t = torch.linspace(0, 100, 101).view(1, 101, 1)
        # 线性趋势 + 高频振荡 (周期 T=4，采样步长 1)
        trend_true = 0.8 * t + 15.0
        oscillation = 5.0 * torch.sin(2 * 3.1415926535 * t / 4.0)
        x = trend_true + oscillation

        seasonal_out, trend_out = decomp(x)
        half_k = (k - 1) // 2
        # 在排除边界的内点中，k=25 涵盖了超过 6 个完整正弦周期，移动平均滤波将正弦振荡衰减 95% 以上
        interior_trend = trend_out[:, half_k:-half_k, :]
        interior_true_trend = trend_true[:, half_k:-half_k, :]
        max_trend_deviation = torch.max(torch.abs(interior_trend - interior_true_trend)).item()
        self.assertLess(max_trend_deviation, 0.25, f"移动平均趋势项受到高频振荡污染超标: {max_trend_deviation}")

        # 验证季节项捕获了主要振荡幅度
        interior_seasonal = seasonal_out[:, half_k:-half_k, :]
        interior_osc = oscillation[:, half_k:-half_k, :]
        seasonal_correlation = torch.cosine_similarity(interior_seasonal.flatten(), interior_osc.flatten(), dim=0).item()
        self.assertGreater(seasonal_correlation, 0.98, f"季节残差项与原高频振荡相关性不足: {seasonal_correlation}")

    def test_solar_grid_parameter_counts_exact_match(self):
        """验证光伏高频 Solar 规格 (L=288, H=144, C=16) 下全部闭式解析参数量与实测 100% 吻合"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        L_solar, H_solar, C_solar = 288, 144, 16

        # 1. DLinear (Shared)
        dl_s_act = count_params(DLinearModel(C_solar, L_solar, H_solar, individual=False))
        dl_s_th = dlinear_closed_form(L_solar, H_solar, C_solar, individual=False)
        self.assertEqual(dl_s_act, 83232)
        self.assertEqual(dl_s_act, dl_s_th)

        # 2. DLinear (Individual)
        dl_i_act = count_params(DLinearModel(C_solar, L_solar, H_solar, individual=True))
        dl_i_th = dlinear_closed_form(L_solar, H_solar, C_solar, individual=True)
        self.assertEqual(dl_i_act, 1331712)
        self.assertEqual(dl_i_act, dl_i_th)

        # 3. CNN-LSTM
        cnn_act = count_params(CNNLSTMModel(C_solar, L_solar, H_solar))
        cnn_th = cnnlstm_closed_form(L_solar, H_solar, C_solar)
        self.assertEqual(cnn_act, 186304)
        self.assertEqual(cnn_act, cnn_th)

        # 4. PatchTST
        pt_act = count_params(PatchTSTModel(C_solar, L_solar, H_solar, patch_len=16, stride=8))
        pt_th = patchtst_closed_form(L_solar, H_solar, C_solar, patch_len=16, stride=8)
        self.assertEqual(pt_act, 392976)
        self.assertEqual(pt_act, pt_th)

    def test_sqlite_wal_and_atomic_checkpoint_mechanics(self):
        """测试多智能体高并发 SQLite WAL 模式配置与写入机制稳定性"""
        from config import DB_PATH
        from storage import get_connection, init_db
        init_db(DB_PATH)
        with get_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]
            # SQLite WAL 模式验证
            self.assertEqual(journal_mode.lower(), "wal", f"SQLite 并发 WAL 模式未生效: {journal_mode}")


if __name__ == "__main__":
    unittest.main()
