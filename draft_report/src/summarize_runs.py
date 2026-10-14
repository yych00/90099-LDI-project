"""Aggregate seeds only when the task, architecture and training protocol match."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
from analyze import read_records


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        if row.get("test_evaluated", True) is False:
            continue
        # Capacity, optimisation and available training interval all affect scores.
        # Pooling runs with different settings would produce a misleading mean.
        key = (
            row["stage"],
            row["dataset"],
            row["model"],
            row["seq_len"],
            row["pred_len"],
            row["train_fraction"],
            row.get("common_start_idx"),
            json.dumps(row["hyperparameters"], sort_keys=True),
            json.dumps(row.get("model_options", {}), sort_keys=True),
        )
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda pair: str(pair[0])):
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("Duplicate seed within a comparison group")
        summary = dict(
            stage=key[0],
            dataset=key[1],
            model=key[2],
            seq_len=key[3],
            pred_len=key[4],
            train_fraction=key[5],
            common_start_idx=key[6],
            hyperparameters=key[7],
            model_options=key[8],
            seeds=",".join(map(str, sorted(r["seed"] for r in rows))),
            n=len(rows),
            params=rows[0]["num_params"],
        )
        for metric in ("mae", "rmse", "mase"):
            values = [r["metrics_24h"][metric] for r in rows]
            summary[metric] = statistics.mean(values)
            summary[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent / "evidence"
    if args.output.resolve() == evidence or evidence in args.output.resolve().parents:
        parser.error("Write summaries outside frozen evidence.")
    output = aggregate(read_records(args.root.resolve()))
    if not output:
        parser.error("No test-evaluated runs to aggregate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} comparison groups to {args.output}")


if __name__ == "__main__":
    main()
