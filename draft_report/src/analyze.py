"""Generate tables and figures from identified runs; never query a 'best test run'."""

import csv
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import REPORT_DIR, ELECTRICITY_CONFIG, SOLAR_CONFIG
from data_loader import load_dataset_matrix
from experiment import MODELS

ROOT = Path(REPORT_DIR)
LABEL = {
    "Persistence": "Persistence",
    "DailySeasonalNaive": "Daily naive",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "CNN1D": "1D-CNN",
    "CNNLSTM": "CNN-LSTM",
    "DLinear": "DLinear",
    "PatchTST": "PatchTST-style",
}
COLORS = dict(
    zip(
        MODELS,
        ["#9ca3af", "#111827", "#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981", "#ef4444"],
    )
)


def read_records(root=None):
    root = Path(root) if root is not None else ROOT
    records = [
        json.loads(p.read_text(encoding="utf-8-sig"))
        for p in sorted((root / "results").glob("*.json"))
        if p.name != "summary.json"
    ]
    if not records:
        raise ValueError(f'No run records in {root / "results"}')
    # Repeated invocations must be made in fresh result roots; ambiguous runs are rejected.
    keys = [
        (
            r["stage"],
            r["dataset"],
            r["model"],
            r["seq_len"],
            r["train_fraction"],
            r["seed"],
            r.get("common_start_idx"),
            json.dumps(r["hyperparameters"], sort_keys=True),
            json.dumps(r.get("model_options", {}), sort_keys=True),
        )
        for r in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate experiment identity; use a fresh output folder")
    return records


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    evidence = Path(__file__).resolve().parent / "evidence"
    if ROOT == evidence or evidence in ROOT.parents:
        parser.error(
            "Figure reconstruction writes outputs; use an artifact-complete copy outside frozen evidence"
        )
    records = read_records()
    out = ROOT / "report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summary = []
    for cfg in (ELECTRICITY_CONFIG, SOLAR_CONFIG):
        for model in MODELS:
            rs = [
                r
                for r in records
                if r["stage"] == "main" and r["dataset"] == cfg.name and r["model"] == model
            ]
            if not rs:
                continue
            signatures = {
                (
                    r["seq_len"],
                    r["train_fraction"],
                    r.get("common_start_idx"),
                    json.dumps(r["hyperparameters"], sort_keys=True),
                    json.dumps(r.get("model_options", {}), sort_keys=True),
                )
                for r in rs
            }
            if len(signatures) > 1:
                raise ValueError(
                    "Main figure mixes training protocols; use separate roots or summarize_runs.py"
                )
            row = dict(
                dataset=cfg.name,
                model=model,
                seeds=",".join(str(r["seed"]) for r in rs),
                n=len(rs),
                params=rs[0]["num_params"],
            )
            for name, fn in [
                ("mae", lambda r: r["metrics_24h"]["mae"]),
                ("rmse", lambda r: r["metrics_24h"]["rmse"]),
                ("mase", lambda r: r["metrics_24h"]["mase"]),
                ("mae4", lambda r: r["metrics_4h"]["mae"]),
                (
                    "slice_mae",
                    lambda r: r["slices"].get("peak_load_mae", r["slices"].get("active_solar_mae")),
                ),
                ("seconds", lambda r: r["training_seconds"]),
                ("latency_ms", lambda r: r["latency"]["median_ms"]),
                ("memory_mib", lambda r: r["peak_allocated_mib"]),
                ("epochs", lambda r: r["actual_epochs"]),
            ]:
                a = np.array([fn(r) for r in rs], dtype=float)
                row[name] = float(a.mean())
                row[name + "_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            summary.append(row)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # summary.json is a derived file, outside the run identity reader below.
    with open(ROOT / "report" / "main_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    fig, axes = plt.subplots(2, 2, figsize=(8, 3.2))
    for i, cfg in enumerate((ELECTRICITY_CONFIG, SOLAR_CONFIG)):
        raw = load_dataset_matrix(cfg)
        b1 = int(0.7 * len(raw))
        c = 0
        steps = 7 * cfg.seasonal_m
        x = np.arange(steps) / cfg.seasonal_m
        axes[i, 0].plot(x, raw[:steps, c], lw=0.9, color="#2563eb")
        axes[i, 0].set(
            title=f"{cfg.name}: first 7 days, {cfg.channel_ids[c]}",
            xlabel="Days from source start",
            ylabel="Value",
        )
        corr = np.corrcoef(raw[:b1].T)
        im = axes[i, 1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        axes[i, 1].set(title="Train correlation", xlabel="Channel", ylabel="Channel")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "data_overview.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.2))
    for ax, cfg in zip(axes, (ELECTRICITY_CONFIG, SOLAR_CONFIG)):
        rows = [s for s in summary if s["dataset"] == cfg.name]
        xs = np.arange(len(rows))
        ax.bar(
            xs,
            [r["mase"] for r in rows],
            yerr=[r["mase_std"] for r in rows],
            capsize=2,
            color=[COLORS[r["model"]] for r in rows],
        )
        ax.set_xticks(xs, [LABEL[r["model"]] for r in rows], rotation=35, ha="right")
        ax.set(title=cfg.name, ylabel="MASE")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "comparison.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.2))
    for ax, cfg in zip(axes, (ELECTRICITY_CONFIG, SOLAR_CONFIG)):
        for model in ["DailySeasonalNaive", "LSTM", "DLinear", "PatchTST"]:
            rs = [
                r
                for r in records
                if r["stage"] == "main" and r["dataset"] == cfg.name and r["model"] == model
            ]
            curves = np.array([r["lead_time"]["mae_curve"] for r in rs])
            ax.plot(
                np.arange(1, cfg.pred_len + 1) * 24 / cfg.pred_len,
                curves.mean(0),
                label=LABEL[model],
                color=COLORS[model],
                lw=1.3,
            )
        ax.set(title=cfg.name, xlabel="Lead time (hours)", ylabel="MAE")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "lead_time.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 3.2))
    for i, cfg in enumerate((ELECTRICITY_CONFIG, SOLAR_CONFIG)):
        for j, stage in enumerate(["window", "fraction"]):
            for model in ["LSTM", "PatchTST"]:
                rs = sorted(
                    [
                        r
                        for r in records
                        if r["stage"] == stage and r["dataset"] == cfg.name and r["model"] == model
                    ],
                    key=lambda r: r["seq_len"] if stage == "window" else r["train_fraction"],
                )
                if rs:
                    # Repeated seeds belong to one setting, not consecutive points
                    # along a curve. Aggregate them before drawing error bars.
                    settings = sorted(
                        {r["seq_len"] if stage == "window" else r["train_fraction"] for r in rs}
                    )
                    means = []
                    deviations = []
                    for setting in settings:
                        values = [
                            r["metrics_24h"]["mase"]
                            for r in rs
                            if (r["seq_len"] if stage == "window" else r["train_fraction"])
                            == setting
                        ]
                        means.append(float(np.mean(values)))
                        deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
                    xs = [
                        setting / cfg.seasonal_m * 24 if stage == "window" else setting * 100
                        for setting in settings
                    ]
                    axes[i, j].errorbar(
                        xs,
                        means,
                        yerr=deviations,
                        marker="o",
                        capsize=3,
                        label=LABEL[model],
                        color=COLORS[model],
                    )
            axes[i, j].set(
                title=f"{cfg.name}: {stage}",
                xlabel=(
                    "History (hours)" if stage == "window" else "Available training interval (%)"
                ),
                ylabel="MASE",
            )
            axes[i, j].grid(alpha=0.2)
            if i == 0 and j == 0:
                axes[i, j].legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "sensitivity.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    # Pre-specified hard scene: largest true future value on the first frozen channel.
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.2))
    cases = []
    for ax, cfg in zip(axes, (ELECTRICITY_CONFIG, SOLAR_CONFIG)):
        reference = next(
            r
            for r in records
            if r["stage"] == "main"
            and r["dataset"] == cfg.name
            and r["model"] == "DailySeasonalNaive"
            and r["seed"] == 42
        )
        z = np.load(ROOT / reference["artifact_dir"] / "test_predictions.npz")
        index = int(np.argmax(z["target"][:, :, 0].max(1)))
        hours = np.arange(1, cfg.pred_len + 1) * 24 / cfg.pred_len
        ax.plot(hours, z["target"][index, :, 0], label="Target", color="black", lw=1.5)
        for model in ["DailySeasonalNaive", "DLinear", "PatchTST"]:
            r = next(
                r
                for r in records
                if r["stage"] == "main"
                and r["dataset"] == cfg.name
                and r["model"] == model
                and r["seed"] == 42
            )
            a = np.load(ROOT / r["artifact_dir"] / "test_predictions.npz")
            ax.plot(
                hours, a["prediction"][index, :, 0], label=LABEL[model], color=COLORS[model], lw=1.0
            )
        ax.set(
            title=f'{cfg.name}, {cfg.channel_ids[0]}, origin {int(z["origins"][index])}',
            xlabel="Lead time (hours)",
            ylabel="Value",
        )
        ax.grid(alpha=0.2)
        cases.append(
            dict(
                dataset=cfg.name,
                index=index,
                origin=int(z["origins"][index]),
                channel=cfg.channel_ids[0],
            )
        )
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "hard_cases.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    (ROOT / "report" / "case_selection.json").write_text(json.dumps(cases, indent=2))
    print("Wrote", len(summary), "summary rows and 5 figure PDFs")


if __name__ == "__main__":
    main()
