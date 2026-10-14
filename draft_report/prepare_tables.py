"""Prepare draft-report assets from existing records; no training."""
import csv
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "my_report" / "submission"
MODELS = ["Persistence", "DailySeasonalNaive", "LSTM", "GRU", "CNN1D", "CNNLSTM", "DLinear", "PatchTST"]
NAMES = {"Persistence": "Persistence", "DailySeasonalNaive": "Daily naive", "CNN1D": "1D-CNN", "CNNLSTM": "CNN--LSTM", "PatchTST": "PatchTST-style"}
def main():
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    for name in ("data_overview.pdf", "hard_cases.pdf"):
        shutil.copy2(SOURCE / "report" / "figures" / name, ROOT / "figures" / name)
    with (SOURCE / "report" / "main_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 16
    rows = {(r["dataset"], r["model"]): r for r in records}
    assert len(rows) == 16
    result = [r"% Generated from my_report/submission/report/main_results.csv.", r"\begin{table}[htbp]", r"\centering\small\linespread{1.1}\selectfont", r"\setlength{\tabcolsep}{4pt}", r"\ifdefined\ChineseReport", r"\caption{初步测试结果。可训练模型为三个种子的均值 $\pm$ 样本标准差；确定性基线仅列数值。各指标越低越好。}", r"\else", r"\caption{Initial test results. Trainable models: mean $\pm$ sample standard deviation over three seeds; deterministic baselines: single values. Lower is better.}", r"\fi", r"\label{tab:results}", r"\begin{tabular}{lrrr}", r"\toprule", r"Model & MAE & RMSE & MASE\\", r"\midrule"]
    for dataset in ("Electricity", "Solar"):
        result.append(r"\multicolumn{4}{l}{\textbf{" + dataset + r"}}\\")
        for model in MODELS:
            row = rows[dataset, model]
            assert row["seeds"] == "42,43,44" and int(row["n"]) == 3
            vals = []
            for metric in ("mae", "rmse", "mase"):
                value = f"{float(row[metric]):,.3f}"
                if model not in ("Persistence", "DailySeasonalNaive"):
                    value += r" $\pm$ " + f"{float(row[metric + '_std']):.3f}"
                vals.append(value)
            result.append(NAMES.get(model, model) + " & " + " & ".join(vals) + r"\\")
        result.append(r"\midrule" if dataset == "Electricity" else r"\bottomrule")
    result.extend([r"\end{tabular}", r"\end{table}"])
    (ROOT / "tables" / "results.tex").write_text("\n".join(result) + "\n", encoding="utf-8")
    print("Prepared two figures and a shared 16-row results table from existing records.")
if __name__ == "__main__":
    main()
