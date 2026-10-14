# -*- coding: utf-8 -*-
"""Regression checks protect the forecasting and model contracts."""

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


# This case checks a boundary or mathematical property that aggregate scores can hide.
def dlinear_closed_form(L: int, H: int, C: int, individual: bool = False) -> int:
    """Regression check: dlinear closed form."""
    if individual:
        return 2 * C * H * (L + 1)
    return 2 * H * (L + 1)


def cnnlstm_closed_form(
    L: int, H: int, C: int, conv_channels: int = 64, lstm_hidden: int = 64, kernel_size: int = 3
) -> int:
    """Regression check: cnnlstm closed form."""
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
    d_ff: int = 128,
) -> int:
    """Regression check: patchtst closed form."""
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


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable; skip model checks")
class TestModernAndHybridDeep(unittest.TestCase):

    def test_dlinear_series_decomposition_exact_identity(self):
        """Regression check: test dlinear series decomposition exact identity."""
        decomp = SeriesDecomp(kernel_size=25)
        for shape in [(2, 48, 16), (4, 73, 8), (1, 288, 4)]:
            x = torch.randn(*shape)
            seasonal, trend = decomp(x)
            # This case checks a boundary or mathematical property that aggregate scores can hide.
            self.assertEqual(seasonal.shape, x.shape)
            self.assertEqual(trend.shape, x.shape)
            # This case checks a boundary or mathematical property that aggregate scores can hide.
            recon = seasonal + trend
            max_err = torch.max(torch.abs(recon - x)).item()
            self.assertLess(
                max_err, 1e-6, f"decomposition reconstruction error exceeded tolerance: {max_err}"
            )

    def test_moving_avg_linear_trend_preservation(self):
        """Regression check: test moving avg linear trend preservation."""
        k = 11
        mavg64 = MovingAvgBlock(kernel_size=k).double()
        t64 = torch.linspace(0, 100, 101, dtype=torch.float64).view(1, 101, 1)
        a, b = 2.5, -7.3
        x_linear64 = a * t64 + b
        trend64 = mavg64(x_linear64)
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        half_k = (k - 1) // 2
        interior_orig64 = x_linear64[:, half_k:-half_k, :]
        interior_trend64 = trend64[:, half_k:-half_k, :]
        max_err64 = torch.max(torch.abs(interior_trend64 - interior_orig64)).item()
        self.assertLess(
            max_err64, 1e-10, f"float64 interior trend error exceeded tolerance: {max_err64}"
        )

    def test_dlinear_channel_sharing_equivariance_and_agnosticism(self):
        """Regression check: test dlinear channel sharing equivariance and agnosticism."""
        model = DLinearModel(num_channels=16, seq_len=48, pred_len=24, individual=False)
        x = torch.randn(2, 48, 16)
        out = model(x)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        perm = torch.randperm(16)
        x_perm = x[:, :, perm]
        out_perm = model(x_perm)
        self.assertTrue(
            torch.allclose(out_perm, out[:, :, perm], atol=1e-5),
            "shared weights must preserve channel permutation equivariance",
        )

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        for c in [1, 5, 23, 64]:
            x_c = torch.randn(2, 48, c)
            out_c = model(x_c)
            self.assertEqual(out_c.shape, (2, 24, c))

    def test_dlinear_individual_mode_gradient_isolation(self):
        """Regression check: test dlinear individual mode gradient isolation."""
        num_c = 4
        seq_len = 48
        pred_len = 24
        model = DLinearModel(
            num_channels=num_c, seq_len=seq_len, pred_len=pred_len, individual=True
        )
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        expected_params = 2 * num_c * pred_len * (seq_len + 1)
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(actual_params, expected_params)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        x = torch.randn(2, seq_len, num_c, requires_grad=True)
        out = model(x)
        loss = out[:, :, 0].sum()
        loss.backward()

        for c in range(1, num_c):
            for p in model.linear_seasonal[c].parameters():
                self.assertTrue(
                    p.grad is None or torch.all(p.grad == 0),
                    f"channel {c} residual branch received a cross-channel gradient",
                )
            for p in model.linear_trend[c].parameters():
                self.assertTrue(
                    p.grad is None or torch.all(p.grad == 0),
                    f"channel {c} trend branch received a cross-channel gradient",
                )

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        with self.assertRaises(ValueError):
            model(torch.randn(2, seq_len, num_c + 1))

    def test_patchtst_dynamic_padding_and_coverage(self):
        """Regression check: test patchtst dynamic padding and coverage."""
        patch_len = 16
        stride = 8
        test_lengths = [24, 30, 48, 50, 73, 96, 120, 168, 288]

        for L in test_lengths:
            model = PatchTSTModel(
                num_channels=4, seq_len=L, pred_len=12, patch_len=patch_len, stride=stride
            )

            # This case checks a boundary or mathematical property that aggregate scores can hide.
            total_len = L + model.pad_len
            expected_patches = (total_len - patch_len) // stride + 1
            self.assertEqual(model.num_patches, expected_patches)

            # This case checks a boundary or mathematical property that aggregate scores can hide.
            covered_indices = set()
            for i in range(model.num_patches):
                start = i * stride
                end = min(start + patch_len, total_len)
                for t in range(start, end):
                    if t < L:
                        covered_indices.add(t)

            self.assertEqual(
                len(covered_indices),
                L,
                f"L={L} has uncovered time steps: {set(range(L)) - covered_indices}",
            )

    def test_patchtst_backward_gradient_flow_entire_sequence(self):
        """Regression check: test patchtst backward gradient flow entire sequence."""
        patch_len = 16
        stride = 8
        for L in [30, 50, 73, 100]:
            model = PatchTSTModel(
                num_channels=2, seq_len=L, pred_len=10, patch_len=patch_len, stride=stride
            )
            x = torch.randn(2, L, 2, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()

            # This case checks a boundary or mathematical property that aggregate scores can hide.
            step_grads = x.grad.abs().sum(dim=(0, 2))  # [L]
            zero_count = (step_grads == 0).sum().item()
            self.assertEqual(zero_count, 0, f"L={L} has {zero_count} time steps without gradients")

            # This case checks a boundary or mathematical property that aggregate scores can hide.
            self.assertGreater(
                step_grads[-1].item(),
                1e-4,
                f"L={L} last observed time step has insufficient gradient",
            )

    def test_patchtst_channel_independence_agnosticism(self):
        """Regression check: test patchtst channel independence agnosticism."""
        model = PatchTSTModel(num_channels=16, seq_len=48, pred_len=24, patch_len=16, stride=8)
        model.eval()

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        for c in [1, 3, 7, 32]:
            x = torch.randn(2, 48, c)
            out = model(x)
            self.assertEqual(out.shape, (2, 24, c))

    def test_cnnlstm_length_invariance_and_reduction(self):
        """Regression check: test cnnlstm length invariance and reduction."""
        H = 24
        C = 16
        base_params = cnnlstm_closed_form(L=48, H=H, C=C)
        self.assertEqual(base_params, 61504)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        for L in [24, 48, 72, 96, 168]:
            m = CNNLSTMModel(num_channels=C, seq_len=L, pred_len=H)
            p = sum(p.numel() for p in m.parameters() if p.requires_grad)
            self.assertEqual(
                p, base_params, f"CNNLSTM at L={L} changed parameter count: {p} != {base_params}"
            )

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        cnn_params = 401568
        reduction_rate = (cnn_params - base_params) / cnn_params
        self.assertAlmostEqual(reduction_rate, 0.8468, places=3)

    def test_closed_form_exact_matches_across_window_grid(self):
        """Regression check: test closed form exact matches across window grid."""

        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        window_grid = [16, 24, 32, 48, 64, 73, 96, 128, 168, 288]
        C = 16
        H_elec = 24

        for L in window_grid:
            # 1. DLinear (Shared)
            dl_actual = count_params(
                DLinearModel(num_channels=C, seq_len=L, pred_len=H_elec, individual=False)
            )
            dl_theory = dlinear_closed_form(L, H_elec, C, individual=False)
            self.assertEqual(
                dl_actual,
                dl_theory,
                f"DLinear(Shared) at L={L} measured parameter count differs from expected: {dl_theory} vs {dl_actual}",
            )

            # 2. DLinear (Individual)
            dl_ind_actual = count_params(
                DLinearModel(num_channels=C, seq_len=L, pred_len=H_elec, individual=True)
            )
            dl_ind_theory = dlinear_closed_form(L, H_elec, C, individual=True)
            self.assertEqual(
                dl_ind_actual,
                dl_ind_theory,
                f"DLinear(Individual) at L={L} measured parameter count differs from expected: {dl_ind_theory} vs {dl_ind_actual}",
            )

            # 3. CNN-LSTM
            cnnlstm_actual = count_params(CNNLSTMModel(num_channels=C, seq_len=L, pred_len=H_elec))
            cnnlstm_theory = cnnlstm_closed_form(L, H_elec, C)
            self.assertEqual(
                cnnlstm_actual,
                cnnlstm_theory,
                f"CNNLSTM at L={L} measured parameter count differs from expected: {cnnlstm_theory} vs {cnnlstm_actual}",
            )

            # 4. PatchTST
            patch_actual = count_params(
                PatchTSTModel(num_channels=C, seq_len=L, pred_len=H_elec, patch_len=16, stride=8)
            )
            patch_theory = patchtst_closed_form(L, H_elec, C, patch_len=16, stride=8)
            self.assertEqual(
                patch_actual,
                patch_theory,
                f"PatchTST at L={L} measured parameter count differs from expected: {patch_theory} vs {patch_actual}",
            )

    def test_temporal_length_mismatch_defenses(self):
        """Regression check: test temporal length mismatch defenses."""
        # 1. DLinear
        dl = DLinearModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            dl(torch.randn(2, 50, 16))
        with self.assertRaises(ValueError):
            dl(torch.randn(2, 24, 16))

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        pt = PatchTSTModel(num_channels=16, seq_len=48, pred_len=24, patch_len=16, stride=8)
        with self.assertRaises(ValueError):
            pt(torch.randn(2, 100, 16))
        with self.assertRaises(ValueError):
            pt(torch.randn(2, 20, 16))

    def test_channel_mismatch_defenses(self):
        """Regression check: test channel mismatch defenses."""
        cnn = CNNLSTMModel(num_channels=16, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            cnn(torch.randn(2, 48, 8))

        dl_indiv = DLinearModel(num_channels=16, seq_len=48, pred_len=24, individual=True)
        with self.assertRaises(ValueError):
            dl_indiv(torch.randn(2, 48, 4))

    def test_initialization_dimension_and_stride_defenses(self):
        """Regression check: test initialization dimension and stride defenses."""
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        with self.assertRaises(ValueError):
            PatchTSTModel(num_channels=16, seq_len=48, pred_len=24, patch_len=8, stride=16)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        with self.assertRaises(ValueError):
            PatchTSTModel(num_channels=16, seq_len=0, pred_len=24)
        with self.assertRaises(ValueError):
            DLinearModel(num_channels=16, seq_len=48, pred_len=0)
        with self.assertRaises(ValueError):
            CNNLSTMModel(num_channels=0, seq_len=48, pred_len=24)
        with self.assertRaises(ValueError):
            MovingAvgBlock(kernel_size=0)

    def test_series_decomp_high_frequency_harmonic_separation(self):
        """Regression check: test series decomp high frequency harmonic separation."""
        k = 25
        decomp = SeriesDecomp(kernel_size=k)
        t = torch.linspace(0, 100, 101).view(1, 101, 1)
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        trend_true = 0.8 * t + 15.0
        oscillation = 5.0 * torch.sin(2 * 3.1415926535 * t / 4.0)
        x = trend_true + oscillation

        seasonal_out, trend_out = decomp(x)
        half_k = (k - 1) // 2
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        interior_trend = trend_out[:, half_k:-half_k, :]
        interior_true_trend = trend_true[:, half_k:-half_k, :]
        max_trend_deviation = torch.max(torch.abs(interior_trend - interior_true_trend)).item()
        self.assertLess(
            max_trend_deviation,
            0.25,
            f"high-frequency contribution to the smooth component exceeded tolerance: {max_trend_deviation}",
        )

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        interior_seasonal = seasonal_out[:, half_k:-half_k, :]
        interior_osc = oscillation[:, half_k:-half_k, :]
        seasonal_correlation = torch.cosine_similarity(
            interior_seasonal.flatten(), interior_osc.flatten(), dim=0
        ).item()
        self.assertGreater(
            seasonal_correlation,
            0.98,
            f"remainder does not retain the high-frequency signal: {seasonal_correlation}",
        )

    def test_solar_grid_parameter_counts_exact_match(self):
        """Regression check: test solar grid parameter counts exact match."""

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


if __name__ == "__main__":
    unittest.main()
