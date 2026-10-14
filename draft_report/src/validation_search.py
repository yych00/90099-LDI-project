"""Select a small hyperparameter grid by validation loss without evaluating test."""

import argparse
import itertools
import json
from pathlib import Path
import statistics
import torch
import experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(experiment.CFG), required=True)
    parser.add_argument("--model", choices=experiment.MODELS[2:], required=True)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 3e-4])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    root = args.output_root.resolve()
    evidence = Path(__file__).resolve().parent / "evidence"
    if root == evidence or evidence in root.parents:
        parser.error("Search output cannot be inside frozen evidence.")
    if root.exists() and any(root.iterdir()):
        parser.error("Use a fresh search directory to retain the full attempt history.")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    trials = []
    for index, (lr, decay) in enumerate(itertools.product(args.learning_rates, args.weight_decays)):
        experiment.ROOT = root / f"trial_{index:03d}"
        records = [
            experiment.run(
                experiment.CFG[args.dataset],
                args.model,
                "tuning",
                seed,
                args.epochs,
                device_name=args.device,
                learning_rate=lr,
                weight_decay=decay,
                evaluate_test=False,
            )
            for seed in args.seeds
        ]
        trials.append(
            dict(
                learning_rate=lr,
                weight_decay=decay,
                seeds=args.seeds,
                mean_validation_mse=statistics.mean(r["best_validation_mse"] for r in records),
                run_ids=[r["run_id"] for r in records],
            )
        )
    # The selection key contains no test metrics; stable ties use grid order.
    selected = min(trials, key=lambda row: row["mean_validation_mse"])
    result = dict(
        dataset=args.dataset,
        model=args.model,
        selected=selected,
        attempts=trials,
        test_evaluated=False,
    )
    (root / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
