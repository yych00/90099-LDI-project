"""Compare raw/clipped predictions and export channel, lead-time and fixed-case errors."""

import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from analyze import read_records
from data_loader import load_dataset_matrix
from eval import (
    calculate_metrics_raw_scale,
    compute_seasonal_naive_denominators,
    compute_slice_metrics,
)
from experiment import CFG


def moving_block_interval(errors, block_length=24, repetitions=200, seed=42):
    """Resample consecutive origins rather than falsely treating overlap as independence.

    The interval is exploratory and depends on the chosen block length.
    It is not a significance test comparing two architectures.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or len(errors) == 0 or not np.isfinite(errors).all():
        raise ValueError("Expected a finite nonempty per-origin error series")
    if not 1 <= block_length <= len(errors) or repetitions < 2:
        raise ValueError("Invalid block length or bootstrap repetition count")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(
            0, len(errors) - block_length + 1, size=math.ceil(len(errors) / block_length)
        )
        sample = np.concatenate([errors[s : s + block_length] for s in starts])[: len(errors)]
        estimates.append(float(sample.mean()))
    return dict(
        lower=float(np.quantile(estimates, 0.025)),
        upper=float(np.quantile(estimates, 0.975)),
        block_length=block_length,
        repetitions=repetitions,
        seed=seed,
    )


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=0)
    parser.add_argument("--block-hours", type=int, default=24)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    evidence = Path(__file__).resolve().parent / "evidence"
    if output == evidence or evidence in output.parents:
        parser.error("Write extended analysis outside frozen evidence.")
    output.mkdir(parents=True, exist_ok=True)
    channels, leads, overall, cases, intervals = [], [], [], [], []
    for row in read_records(root):
        if row.get("test_evaluated", True) is False:
            continue
        cfg = CFG[row["dataset"]]
        path = root / row["artifact_dir"] / "test_predictions.npz"
        if not path.exists():
            parser.error(f"Full forecast artifacts are required: {path}")
        raw = load_dataset_matrix(cfg)
        start, end, _ = row["bounds"]
        available = raw[start:end].astype(np.float64)
        denom = compute_seasonal_naive_denominators(available, cfg.seasonal_m)
        with np.load(path, allow_pickle=False) as data:
            true, prediction, origins = (
                data["target"].astype(float),
                data["prediction"].astype(float),
                data["origins"],
            )
        for mode, pred in (("raw", prediction), ("nonnegative_clip", np.maximum(prediction, 0))):
            # Apply the same post-processing rule to every model; never choose it by test score.
            metrics = calculate_metrics_raw_scale(pred, true, denom)
            conditional = compute_slice_metrics(pred, true, cfg.name, available)
            overall.append(
                dict(
                    run_id=row["run_id"],
                    dataset=cfg.name,
                    model=row["model"],
                    mode=mode,
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    mase=metrics["mase"],
                    conditional_mae=conditional.get(
                        "peak_load_mae", conditional.get("active_solar_mae")
                    ),
                )
            )
            error = pred - true
            for c, name in enumerate(cfg.channel_ids):
                channels.append(
                    dict(
                        run_id=row["run_id"],
                        mode=mode,
                        channel=name,
                        mae=metrics["mae_per_channel"][c],
                        rmse=metrics["rmse_per_channel"][c],
                        mase=metrics["mase_per_channel"][c],
                        bias=float(error[:, :, c].mean()),
                    )
                )
            for lead in range(cfg.pred_len):
                leads.append(
                    dict(
                        run_id=row["run_id"],
                        mode=mode,
                        lead_hours=(lead + 1) * 24 / cfg.pred_len,
                        mae=float(np.abs(error[:, lead]).mean()),
                        rmse=float(np.sqrt(np.square(error[:, lead]).mean())),
                    )
                )
        # A fixed first origin and a target-defined maximum expose both ordinary and hard scenes.
        for index in sorted({0, int(np.argmax(true[:, :, 0].max(axis=1)))}):
            cases.append(
                dict(
                    run_id=row["run_id"],
                    channel=cfg.channel_ids[0],
                    origin=int(origins[index]),
                    selection=(
                        "first_test_origin" if index == 0 else "largest_true_target_earliest_tie"
                    ),
                    target=true[index, :, 0].tolist(),
                    prediction=prediction[index, :, 0].tolist(),
                )
            )
        if args.bootstrap_repetitions:
            interval = moving_block_interval(
                np.abs(prediction - true).mean(axis=(1, 2)),
                args.block_hours,
                args.bootstrap_repetitions,
            )
            intervals.append(dict(run_id=row["run_id"], **interval))
    write_csv(output / "postprocessing.csv", overall)
    write_csv(output / "channels.csv", channels)
    write_csv(output / "lead_times.csv", leads)
    (output / "fixed_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (output / "moving_block_intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )
    print(f"Analysed {len(overall)//2} saved runs without training.")


if __name__ == "__main__":
    main()
