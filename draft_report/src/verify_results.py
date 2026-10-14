"""Recompute metrics from saved predictions, validate SQLite and reload checkpoints."""

import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
import torch
from analyze import ROOT, read_records
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG
from data_loader import get_data_splits
from eval import (
    calculate_metrics_raw_scale,
    compute_seasonal_naive_denominators,
    compute_slice_metrics,
)
from experiment import create_model
from prepare_data import sha256


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    torch.set_num_threads(4)
    records = read_records(ROOT)
    checked = []
    reload_deltas = []
    with sqlite3.connect(f'file:{ROOT / "experiments.db"}?mode=ro', uri=True) as c:
        db = {
            run_id: json.loads(body)
            for run_id, body in c.execute("SELECT run_id,record_json FROM runs")
        }
    for r in records:
        if r.get("test_evaluated", True) is False:
            continue
        cfg = ELECTRICITY_CONFIG if r["dataset"] == "Electricity" else SOLAR_CONFIG
        loaders, scaler, raw, bounds = get_data_splits(
            cfg,
            custom_seq_len=r["seq_len"],
            train_data_fraction=r["train_fraction"],
            common_start_idx=r["common_start_idx"],
            batch_size=r["hyperparameters"]["batch_size"],
            seed=r["seed"],
        )
        z = np.load(ROOT / r["artifact_dir"] / "test_predictions.npz")
        np.testing.assert_array_equal(z["origins"], loaders["test_starts"])
        true = np.stack([raw[t : t + cfg.pred_len] for t in z["origins"]])
        np.testing.assert_allclose(z["target"], true, rtol=3e-6, atol=0.02)
        denominator = compute_seasonal_naive_denominators(
            raw[bounds[0] : bounds[1]].astype(np.float64), cfg.seasonal_m
        )
        m = calculate_metrics_raw_scale(
            z["prediction"].astype(np.float64), z["target"].astype(np.float64), denominator
        )
        for k in ("mae", "rmse", "mase"):
            np.testing.assert_allclose(m[k], r["metrics_24h"][k], rtol=3e-6, atol=1e-4)
        assert r == db[r["run_id"]], f'SQLite mismatch: {r["run_id"]}'
        if r["num_params"]:
            checkpoint = ROOT / r["artifact_dir"] / "best.pt"
            assert sha256(checkpoint) == r["checkpoint_sha256"]
        if args.reload:
            # CPU reload verifies portability and a deterministic sampled forecast.
            model = create_model(r["model"], cfg, r["seq_len"], r.get("model_options")).eval()
            if r["num_params"]:
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["state_dict"])
                np.testing.assert_array_equal(ckpt["scaler_mean"], scaler.mean)
            x, _ = loaders["test"].dataset[0]
            with torch.inference_mode():
                pred = scaler.inverse_transform(model(x.unsqueeze(0)).numpy())[0]
            # Express portability tolerance in training-standard-deviation units:
            # TF32 CUDA convolution and CPU FP32 can differ on large raw channels.
            scaled_difference = (pred - z["prediction"][0]) / scaler.std
            np.testing.assert_allclose(scaled_difference, 0.0, rtol=0.0, atol=0.005)
            native_max = None
            if torch.cuda.is_available():
                # Strong same-device check with the original evaluation batch size.
                model = model.to("cuda")
                xbatch, _ = next(iter(loaders["test"]))
                with torch.inference_mode():
                    native = scaler.inverse_transform(model(xbatch.to("cuda")).cpu().numpy())
                expected = z["prediction"][: len(xbatch)]
                np.testing.assert_allclose(native, expected, rtol=1e-6, atol=1e-4)
                native_max = float(np.max(np.abs(native - expected)))
            reload_deltas.append(
                dict(
                    run_id=r["run_id"],
                    cpu_max_standardised_difference=float(np.max(np.abs(scaled_difference))),
                    native_max_raw_difference=native_max,
                )
            )
        checked.append(r["run_id"])
        print("PASS", r["run_id"], flush=True)
    report = dict(
        checked_runs=len(checked),
        checkpoint_reload=args.reload,
        all_passed=True,
        run_ids=checked,
        reload_deltas=reload_deltas,
        cpu_portability_atol_standardised=0.005,
        native_reload_rtol=1e-6,
        native_reload_atol=1e-4,
    )
    (ROOT / "audit").mkdir(exist_ok=True)
    (ROOT / "audit" / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
