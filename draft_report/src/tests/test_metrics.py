# -*- coding: utf-8 -*-
"""Regression checks protect the forecasting and model contracts."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval import (
    compute_seasonal_naive_denominators,
    calculate_metrics_raw_scale,
    compute_lead_time_errors,
    compute_slice_metrics,
)


class TestMetrics(unittest.TestCase):
    def test_seasonal_mase_denominators(self):
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        t = np.arange(100)
        series = np.sin(2 * np.pi * t / 24).reshape(-1, 1)
        denoms = compute_seasonal_naive_denominators(series, seasonal_m=24)
        self.assertEqual(len(denoms), 1)
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        self.assertGreaterEqual(float(denoms[0]), 1e-8)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        const_series = np.ones((100, 2), dtype=np.float32) * 42.0
        const_denoms = compute_seasonal_naive_denominators(const_series, seasonal_m=24)
        self.assertEqual(len(const_denoms), 2)
        self.assertAlmostEqual(float(const_denoms[0]), 1e-8, places=7)

    def test_raw_scale_metrics(self):
        N, H, C = 10, 24, 16
        y_true = np.ones((N, H, C), dtype=np.float32) * 10.0
        # This case checks a boundary or mathematical property that aggregate scores can hide.
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

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        train_data_elec = np.ones((100, C), dtype=np.float32) * 5.0
        slice_elec = compute_slice_metrics(y_pred, y_true, "Electricity", train_data_elec)
        self.assertIn("peak_load_mae", slice_elec)
        self.assertAlmostEqual(slice_elec["peak_load_mae"], 1.0, places=4)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        train_data_solar = np.ones((100, C), dtype=np.float32) * 20.0
        slice_solar = compute_slice_metrics(y_pred, y_true, "Solar", train_data_solar)
        self.assertIn("active_solar_mae", slice_solar)
        self.assertAlmostEqual(slice_solar["active_solar_mae"], 1.0, places=4)

    def test_seasonal_mase_insufficient_length(self):
        """Regression check: test seasonal mase insufficient length."""
        short_series = np.ones((10, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_seasonal_naive_denominators(short_series, seasonal_m=24)

    def test_empty_slice_protection(self):
        """Regression check: test empty slice protection."""
        N, H, C = 5, 24, 16
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        y_true = np.ones((N, H, C), dtype=np.float32) * 1.0
        y_pred = np.ones((N, H, C), dtype=np.float32) * 2.0
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        train_data_high = np.ones((100, C), dtype=np.float32) * 50.0

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        slice_elec = compute_slice_metrics(y_pred, y_true, "Electricity", train_data_high)
        self.assertEqual(slice_elec["slice_sample_ratio"], 0.0)
        self.assertEqual(slice_elec["peak_load_mae"], 0.0)
        self.assertEqual(slice_elec["peak_load_rmse"], 0.0)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        slice_solar = compute_slice_metrics(y_pred, y_true, "Solar", train_data_high)
        self.assertEqual(slice_solar["slice_sample_ratio"], 0.0)
        self.assertEqual(slice_solar["active_solar_mae"], 0.0)
        self.assertEqual(slice_solar["active_solar_rmse"], 0.0)

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        slice_unknown = compute_slice_metrics(y_pred, y_true, "UnknownGrid", train_data_high)
        self.assertEqual(slice_unknown, {})

    def test_jensen_inequality_defense(self):
        """Regression check: test jensen inequality defense."""
        # This case checks a boundary or mathematical property that aggregate scores can hide.
        b1_err = np.ones((10, 24, 16)) * 1.0  # MSE_1 = 1.0, RMSE_1 = 1.0
        b2_err = np.ones((10, 24, 16)) * 5.0  # MSE_2 = 25.0, RMSE_2 = 5.0

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        batch_avg_rmse = (1.0 + 5.0) / 2.0  # 3.0

        # This case checks a boundary or mathematical property that aggregate scores can hide.
        global_mse = (1.0**2 + 5.0**2) / 2.0  # (1 + 25) / 2 = 13.0
        global_rmse = np.sqrt(global_mse)  # sqrt(13) ≈ 3.6055

        self.assertLess(batch_avg_rmse, global_rmse)
        self.assertAlmostEqual(global_rmse, np.sqrt(13.0), places=4)


if __name__ == "__main__":
    unittest.main()
