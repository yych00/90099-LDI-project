"""Verify the small frozen evidence bundle without claiming to reload omitted checkpoints."""

import csv
import json
import sqlite3
import statistics
from pathlib import Path
from data_loader import load_dataset_matrix
from experiment import CFG


def main():
    root = Path(__file__).resolve().parent / "evidence"
    records = [
        json.loads(p.read_text(encoding="utf-8-sig"))
        for p in (root / "results").glob("*.json")
        if p.name != "summary.json"
    ]
    assert len(records) == 72, "The frozen pilot contains 72 identified records"
    for cfg in CFG.values():
        load_dataset_matrix(cfg)
    with sqlite3.connect((root / "experiments.db").as_uri() + "?mode=ro", uri=True) as connection:
        database = {
            key: json.loads(body)
            for key, body in connection.execute("SELECT run_id, record_json FROM runs")
        }
    assert len(database) == len(records)
    for row in records:
        assert row == database[row["run_id"]], row["run_id"]
        assert row["hyperparameters"]["max_epochs"] == 5
    with (root / "report" / "main_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 16
    for summary in summaries:
        selected = [
            r
            for r in records
            if r["stage"] == "main"
            and r["dataset"] == summary["dataset"]
            and r["model"] == summary["model"]
        ]
        assert sorted(r["seed"] for r in selected) == [42, 43, 44]
        for metric in ("mae", "rmse", "mase"):
            values = [r["metrics_24h"][metric] for r in selected]
            assert abs(statistics.mean(values) - float(summary[metric])) < 1e-8
            assert abs(statistics.stdev(values) - float(summary[metric + "_std"])) < 1e-8
    print("PASS: original-data hashes, 72 JSON/SQLite records and 96 main metric means/SDs.")
    print(
        "Full saved-prediction and checkpoint reload checks require the separate artifact directory."
    )


if __name__ == "__main__":
    main()
