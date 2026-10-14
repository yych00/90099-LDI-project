"""Exercise future-study controls without launching a real benchmark campaign."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG
from data_loader import StandardScaler
from error_analysis import moving_block_interval
from summarize_runs import aggregate
import validation_search


class FutureWorkTests(unittest.TestCase):
    def test_capacity_control_is_closest_integer_width_and_reloadable(self):
        for cfg in (ELECTRICITY_CONFIG, SOLAR_CONFIG):
            model = experiment.create_model("LSTMCapacityMatched", cfg)
            target = sum(p.numel() for p in experiment.create_model("CNNLSTM", cfg).parameters())
            count = sum(p.numel() for p in model.parameters())
            for width in (model.hidden_size - 1, model.hidden_size + 1):
                alternative = experiment.create_model(
                    "LSTM", cfg, model_options={"hidden_size": width}
                )
                self.assertLessEqual(
                    abs(count - target),
                    abs(sum(p.numel() for p in alternative.parameters()) - target),
                )
            reload = experiment.create_model(
                "LSTMCapacityMatched", cfg, model_options={"hidden_size": model.hidden_size}
            )
            reload.load_state_dict(model.state_dict())
            self.assertEqual(
                reload(torch.zeros(1, cfg.seq_len, cfg.num_channels)).shape,
                (1, cfg.pred_len, cfg.num_channels),
            )

    def test_validation_only_run_never_evaluates_test_and_obeys_update_ceiling(self):
        # Tiny deterministic fixtures test orchestration, not forecasting accuracy.
        cfg = ELECTRICITY_CONFIG
        raw = np.arange(128 * 16, dtype=np.float32).reshape(128, 16)
        scaler = StandardScaler().fit(raw[:96])
        inputs = torch.zeros(2, cfg.seq_len, cfg.num_channels)
        targets = torch.ones(2, cfg.pred_len, cfg.num_channels)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
        loaders = {key: loader for key in ("train", "val", "test")}
        loaders.update({key + "_starts": np.array([48, 51]) for key in ("train", "val", "test")})
        with tempfile.TemporaryDirectory() as directory, patch.object(
            experiment, "ROOT", Path(directory)
        ), patch.object(
            experiment, "get_data_splits", return_value=(loaders, scaler, raw, (0, 96, 110))
        ), patch.object(
            experiment, "evaluate", side_effect=AssertionError("Test was accessed")
        ), patch.object(
            experiment, "benchmark", return_value={"median_ms": 0}
        ), redirect_stdout(
            io.StringIO()
        ):
            row = experiment.run(
                cfg,
                "DLinear",
                "tuning",
                42,
                3,
                device_name="cpu",
                learning_rate=3e-4,
                max_updates=1,
                evaluate_test=False,
            )
            self.assertEqual(row["optimizer_steps"], 1)
            self.assertEqual(row["hyperparameters"]["lr"], 3e-4)
            self.assertFalse(row["test_evaluated"])
            self.assertNotIn("metrics_24h", row)
            self.assertFalse(
                (Path(directory) / row["artifact_dir"] / "test_predictions.npz").exists()
            )
            self.assertTrue(np.isfinite(row["best_validation_mse"]))

    def test_invalid_update_budget_rejected_before_loading_data(self):
        with patch.object(
            experiment, "get_data_splits", side_effect=AssertionError("Data was accessed")
        ):
            with self.assertRaises(ValueError):
                experiment.run(ELECTRICITY_CONFIG, "LSTM", "main", 42, 5, max_updates=0)

    def test_seed_summary_does_not_pool_different_optimisers(self):
        record = dict(
            stage="main",
            dataset="Electricity",
            model="LSTM",
            seq_len=48,
            pred_len=24,
            train_fraction=1,
            common_start_idx=None,
            seed=42,
            hyperparameters={"lr": 1e-3},
            num_params=45952,
            metrics_24h={"mae": 2.0, "rmse": 3.0, "mase": 1.0},
        )
        second = copy.deepcopy(record)
        second["seed"] = 43
        changed = copy.deepcopy(record)
        changed["hyperparameters"]["lr"] = 3e-4
        summaries = aggregate([record, second, changed])
        self.assertEqual(sorted(row["n"] for row in summaries), [1, 2])
        with self.assertRaises(ValueError):
            aggregate([record, copy.deepcopy(record)])

    def test_block_resampling_preserves_constant_errors_and_rejects_invalid_blocks(self):
        interval = moving_block_interval(np.full(12, 3.0), block_length=4, repetitions=10)
        self.assertEqual((interval["lower"], interval["upper"]), (3.0, 3.0))
        with self.assertRaises(ValueError):
            moving_block_interval(np.ones(4), block_length=5)

    def test_search_selects_validation_only_and_records_all_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"

            def fake_run(*args, **kwargs):
                self.assertIs(kwargs["evaluate_test"], False)
                return {
                    "best_validation_mse": kwargs["learning_rate"],
                    "run_id": str(kwargs["learning_rate"]),
                }

            argv = [
                "validation_search.py",
                "--dataset",
                "Electricity",
                "--model",
                "LSTM",
                "--output-root",
                str(output),
                "--learning-rates",
                ".001",
                ".0003",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                experiment, "ROOT", Path(directory)
            ), patch.object(experiment, "run", side_effect=fake_run), redirect_stdout(
                io.StringIO()
            ):
                validation_search.main()
            result = json.loads((output / "selection.json").read_text())
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(result["selected"]["learning_rate"], 0.0003)
            self.assertFalse(result["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
