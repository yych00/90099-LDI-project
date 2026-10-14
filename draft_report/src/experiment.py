"""Unified, auditable pilot/final experiments. No automatic synthetic fallback.

Examples: python experiment.py --stage main --epochs 5 --seeds 42
          python experiment.py --stage window --datasets Electricity --epochs 5
          python experiment.py --stage main --epochs 20 --seeds 42 43 44
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sqlite3
import time
import uuid
import numpy as np
import torch
from torch import nn
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG, REPORT_DIR, DATA_DIR
from data_loader import get_data_splits
from eval import (
    calculate_metrics_raw_scale,
    compute_seasonal_naive_denominators,
    compute_lead_time_errors,
    compute_slice_metrics,
)
from models import (
    PersistenceModel,
    DailySeasonalNaiveModel,
    LSTMModel,
    GRUModel,
    CNN1DModel,
    CNNLSTMModel,
    DLinearModel,
    PatchTSTModel,
)
from prepare_data import sha256

MODELS = [
    "Persistence",
    "DailySeasonalNaive",
    "LSTM",
    "GRU",
    "CNN1D",
    "CNNLSTM",
    "DLinear",
    "PatchTST",
]
ABLATION_MODELS = ["LSTM", "CNNLSTM", "LSTMCapacityMatched"]
CFG = {"Electricity": ELECTRICITY_CONFIG, "Solar": SOLAR_CONFIG}
ROOT = Path(REPORT_DIR)


def set_seed(seed):
    # Fix stochastic initialisation and avoid cuDNN's runtime kernel search so
    # seed comparisons measure training variability under one declared policy.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(name, cfg, length=None, model_options=None):
    """Keep historical state-dict keys compatible; add controls as named variants."""
    kw = dict(num_channels=cfg.num_channels, seq_len=length or cfg.seq_len, pred_len=cfg.pred_len)
    options = dict(model_options or {})
    if name == "Persistence":
        return PersistenceModel(cfg.pred_len)
    if name == "DailySeasonalNaive":
        return DailySeasonalNaiveModel(cfg.pred_len, cfg.seasonal_m)
    classes = dict(
        LSTM=LSTMModel,
        GRU=GRUModel,
        CNN1D=CNN1DModel,
        CNNLSTM=CNNLSTMModel,
        DLinear=DLinearModel,
        PatchTST=PatchTSTModel,
    )
    if name == "LSTMCapacityMatched":
        # The closest integer hidden width controls parameter count without
        # changing the direct-output protocol or adding a convolutional frontend.
        target = sum(p.numel() for p in CNNLSTMModel(**kw).parameters())
        output = cfg.pred_len * cfg.num_channels
        width = min(
            range(1, 513),
            key=lambda h: abs(4 * h * (cfg.num_channels + h + 2) + output * (h + 1) - target),
        )
        options.setdefault("hidden_size", width)
        return LSTMModel(**kw, **options)
    return classes[name](**kw, **options)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, loader, scaler, cfg, raw, bounds, device):
    model.eval()
    preds, truths = [], []
    with torch.inference_mode():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            truths.append(y.numpy())
    pred = scaler.inverse_transform(np.concatenate(preds)).astype(np.float64)
    true = scaler.inverse_transform(np.concatenate(truths)).astype(np.float64)
    available = raw[bounds[0] : bounds[1]].astype(np.float64)
    denom = compute_seasonal_naive_denominators(available, cfg.seasonal_m)
    metrics = calculate_metrics_raw_scale(pred, true, denom)
    short = calculate_metrics_raw_scale(
        pred[:, : cfg.short_horizon], true[:, : cfg.short_horizon], denom
    )
    leads = compute_lead_time_errors(pred, true)
    slices = compute_slice_metrics(pred, true, cfg.name, available)
    return (
        dict(
            metrics_24h=metrics,
            metrics_4h=short,
            slices=slices,
            lead_time={k: v.tolist() for k, v in leads.items()},
            negative_prediction_ratio=float((pred < 0).mean()),
        ),
        pred,
        true,
    )


def benchmark(model, loader, device):
    """Model-only wall latency, input already on device; 10 warmups, 30 repeats."""
    x = next(iter(loader))[0].to(device)
    samples = []
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        sync(device)
        for _ in range(30):
            sync(device)
            t = time.perf_counter()
            model(x)
            sync(device)
            samples.append(1000 * (time.perf_counter() - t))
    return dict(
        batch_size=len(x),
        mean_ms=float(np.mean(samples)),
        median_ms=float(np.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
        repeats=30,
        warmups=10,
        scope="model-only; input resident on device",
    )


def save_record(record):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{record['run_id']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with sqlite3.connect(ROOT / "experiments.db", timeout=30) as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, stage TEXT, dataset TEXT, model TEXT, seed INTEGER, record_json TEXT)"
        )
        c.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?)",
            (
                record["run_id"],
                record["stage"],
                record["dataset"],
                record["model"],
                record["seed"],
                json.dumps(record),
            ),
        )


def run(
    cfg,
    name,
    stage,
    seed,
    epochs,
    length=None,
    fraction=1.0,
    common=None,
    batch_size=32,
    device_name="auto",
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=3,
    max_updates=None,
    model_options=None,
    evaluate_test=True,
):
    """Fit on training and select by validation; tuning can explicitly skip test.

    max_updates provides a matched optimiser budget for later data-fraction
    studies. It is a ceiling: early stopping or the epoch ceiling can stop sooner.
    """
    if (
        epochs < 1
        or learning_rate <= 0
        or weight_decay < 0
        or patience < 1
        or (max_updates is not None and max_updates < 1)
    ):
        raise ValueError("Invalid training budget or optimiser settings")
    set_seed(seed)
    length = length or cfg.seq_len
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    )
    loaders, scaler, raw, bounds = get_data_splits(
        cfg,
        custom_seq_len=length,
        train_data_fraction=fraction,
        common_start_idx=common,
        batch_size=batch_size,
        seed=seed,
    )
    model = create_model(name, cfg, length, model_options).to(device)
    model_options = dict(model_options or {})
    if name == "LSTMCapacityMatched":
        model_options["hidden_size"] = model.hidden_size
    params = sum(p.numel() for p in model.parameters())
    run_id = f"{stage}_{cfg.name}_{name}_L{length}_f{fraction:g}_s{seed}_{uuid.uuid4().hex[:8]}"
    out = ROOT / "artifacts" / run_id
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "best.pt"
    history = []
    steps = 0
    best = float("inf")
    best_epoch = None
    stale = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t0 = time.perf_counter()
    if params:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        best_state = None
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            elements = 0
            for x, y in loaders["train"]:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                # Standardised loss balances differently scaled channels. Scores
                # are later inverse-scaled so reported errors remain interpretable.
                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                steps += 1
                total += float(loss.detach()) * y.numel()
                elements += y.numel()
                if max_updates is not None and steps >= max_updates:
                    break
            model.eval()
            val_total = 0.0
            val_elements = 0
            with torch.inference_mode():
                for x, y in loaders["val"]:
                    # Sum errors by element count; a short final batch must not
                    # receive the same weight as a full batch.
                    y = y.to(device)
                    pred = model(x.to(device))
                    val_total += float(nn.functional.mse_loss(pred, y, reduction="sum"))
                    val_elements += y.numel()
            val_loss = val_total / val_elements
            history.append(dict(epoch=epoch, train_mse=total / elements, val_mse=val_loss))
            print(run_id, epoch, f"train={total/elements:.5f} val={val_loss:.5f}", flush=True)
            if val_loss < best:
                best = val_loss
                best_epoch = epoch
                stale = 0
                # state_dict tensors share storage with the live model. Cloning
                # prevents later optimiser steps from changing the selected state.
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
            if max_updates is not None and steps >= max_updates:
                break
        if best_state is None:
            raise RuntimeError("No finite best checkpoint")
        model.load_state_dict(best_state)
        torch.save(
            dict(
                state_dict=best_state,
                model=name,
                dataset=cfg.name,
                seq_len=length,
                scaler_mean=scaler.mean,
                scaler_std=scaler.std,
                model_options=model_options,
            ),
            checkpoint,
        )
    sync(device)
    training_seconds = time.perf_counter() - t0 if params else 0.0
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    metrics = {}
    if evaluate_test:
        metrics, pred, true = evaluate(model, loaders["test"], scaler, cfg, raw, bounds, device)
        # Full saved outputs allow independent scoring without another training run.
        np.savez_compressed(
            out / "test_predictions.npz",
            prediction=pred.astype(np.float32),
            target=true.astype(np.float32),
            origins=loaders["test_starts"],
        )
    np.savez_compressed(out / "scaler.npz", mean=scaler.mean, std=scaler.std)
    np.savez_compressed(
        out / "origins.npz", **{s: loaders[s + "_starts"] for s in ["train", "val", "test"]}
    )
    manifest = json.loads((Path(DATA_DIR) / f"{cfg.name}_manifest.json").read_text())
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if "__pycache__" not in str(path):
            code_hash.update(path.name.encode() + path.read_bytes())
    record = dict(
        run_id=run_id,
        stage=stage,
        dataset=cfg.name,
        model=name,
        seed=seed,
        seq_len=length,
        pred_len=cfg.pred_len,
        train_fraction=fraction,
        common_start_idx=common,
        bounds=list(bounds),
        origin_counts={s: len(loaders[s + "_starts"]) for s in ["train", "val", "test"]},
        origin_sha256={
            s: hashlib.sha256(loaders[s + "_starts"].tobytes()).hexdigest()
            for s in ["train", "val", "test"]
        },
        dataset_config=asdict(cfg),
        data_manifest=manifest,
        code_sha256=code_hash.hexdigest(),
        hyperparameters=dict(
            lr=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=epochs,
            patience=patience,
            max_updates=max_updates,
        ),
        model_options=model_options,
        best_validation_mse=best if params else None,
        test_evaluated=evaluate_test,
        actual_epochs=len(history),
        best_epoch=best_epoch,
        optimizer_steps=steps,
        history=history,
        num_params=params,
        training_seconds=training_seconds,
        peak_allocated_mib=peak_memory,
        latency=benchmark(model, loaders["test"] if evaluate_test else loaders["val"], device),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device=str(device),
            device_name=(
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            platform=platform.platform(),
            cudnn_deterministic=True,
        ),
        artifact_dir=out.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha256(checkpoint) if params else None,
        **metrics,
    )
    save_record(record)
    print("DONE", run_id, metrics.get("metrics_24h", {}).get("mae", "validation-only"), flush=True)
    return record


def main():
    global ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["main", "window", "fraction", "ablation"], default="main")
    p.add_argument("--datasets", nargs="+", choices=list(CFG), default=list(CFG))
    p.add_argument("--models", nargs="+", choices=MODELS + ["LSTMCapacityMatched"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-root", type=Path, default=ROOT)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-updates", type=int)
    args = p.parse_args()
    torch.set_num_threads(4)
    ROOT = args.output_root.resolve()
    # Frozen results are read-only evidence, regardless of the working directory.
    evidence = Path(__file__).resolve().parent / "evidence"
    if ROOT == evidence or evidence in ROOT.parents:
        p.error("Output root cannot be inside frozen evidence")
    if args.epochs < 1:
        p.error("--epochs must be positive")
    for ds in args.datasets:
        cfg = CFG[ds]
        names = args.models or (
            MODELS
            if args.stage == "main"
            else ABLATION_MODELS if args.stage == "ablation" else ["LSTM", "PatchTST"]
        )
        lengths = (
            [cfg.seq_len]
            if args.stage != "window"
            else [cfg.seasonal_m, 2 * cfg.seasonal_m, 7 * cfg.seasonal_m]
        )
        fractions = [0.25, 0.5, 1.0] if args.stage == "fraction" else [1.0]
        for length in lengths:
            for fraction in fractions:
                for name in names:
                    for seed in args.seeds:
                        run(
                            cfg,
                            name,
                            args.stage,
                            seed,
                            args.epochs,
                            length,
                            fraction,
                            max(lengths) if args.stage == "window" else None,
                            args.batch_size,
                            args.device,
                            args.learning_rate,
                            args.weight_decay,
                            args.patience,
                            args.max_updates,
                        )


if __name__ == "__main__":
    main()
