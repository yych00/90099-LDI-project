# ELEN90099 LDI: verified preliminary forecasting experiments

This is the submission code folder. The accompanying English and Chinese reports
use original Monash Electricity Hourly / Solar 10 Minutes files, fixed 16-channel
subsets, and a 48-hour history to predict 24 hours. The preliminary budget is five
maximum epochs. Main results use seeds 42/43/44; sensitivity uses seed 42.

Both student names, IDs and actual section responsibility must be filled in the
report before upload. They are intentionally not invented by this code package.

## Layout

```text
src/                 implementation, tests, notebook, report builder
data/                frozen selected arrays and their source/hash manifests
results/             one JSON per uniquely identified run, plus derived summary
artifacts/<run_id>/   best.pt, scaler.npz, origins.npz, test_predictions.npz
experiments.db       SQLite runs table; complete JSON and relative artifact paths
report/              matching report sources, generated figures and result CSV
audit/               verification, tests, and run logs
```

Paths are resolved from this file's location, so `src/` must remain adjacent to
the other directories. No absolute author-specific paths are required at runtime.
The code ZIP includes the small selected arrays, manifests, and result metadata.
The complete `artifacts/` (about 712 MiB, including 72 runs' predictions and weights)
is retained in the local submission working folder, and excluded from the lightweight
code ZIP. Saved-output verification and figure reconstruction require that folder;
the code ZIP alone supports the tests and fresh training to regenerate artifacts.
The full
downloaded TSF/ZIP files are retained in the working folder but excluded from the
submission ZIP. `prepare_data.py` downloads them again when reconstruction is needed.

## Environment and initial checks

Tested with Python 3.10.21, PyTorch 2.5.1+cu121, NumPy 2.2.6, Matplotlib 3.10.9
on Windows and an NVIDIA GeForce RTX 3060 Laptop GPU. The exact observed package
versions are listed in `audit/environment.json`. CPU fallback is supported;
CPU training latency will differ. Colab T4 is a supported intended runtime,
but this submission does **not** report T4 measurements.

From this `src` folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
# With the full local artifacts folder, or after fresh training:
python verify_results.py --reload
```

If choosing a CUDA wheel manually, use the wheel compatible with the host GPU;
the tested version is `torch==2.5.1` with the CUDA 12.1 build. The requirements
allow other compatible releases, so exact bitwise reproduction across software
versions/hardware is not promised. CPU checkpoint portability is checked in
training-standard-deviation units; native-device reload uses a tighter tolerance.

## Inspect and reproduce the submitted numbers

`verify_results.py --reload` reads saved predictions, checks their true targets
against original selected arrays and exact evaluation origins, recomputes MAE,
RMSE and MASE, checks the JSON against SQLite, validates checkpoint checksums,
and reloads checkpoints on CPU. With CUDA available it also checks the original
evaluation batch on the native device. It writes `audit/verification.json`.
No training is needed to inspect or verify the submitted outputs.

```bash
python analyze.py
python build_report.py
```

These commands regenerate the CSV, summary JSON, five figure PDFs, and both LaTeX
report sources using recorded values. `analyze.py` rejects duplicate run identity;
it never picks the smallest test error or latest row from a mixed database.

## Download and independently reconstruct the data

```bash
python prepare_data.py
```

The script downloads Zenodo records 4656140 and 4656144, checks all series, uses
training-only mean stratification to reproduce the outline's frozen IDs, and
saves SHA256 for archive, TSF and selected cache. The source loader requires a
matching manifest and rejects missing or modified arrays. There is no synthetic
fallback and no random split of forecast windows. Source values are not converted
to an assumed kW/MW unit. Solar's published source includes simulated series.

## Run new training experiments

To preserve submitted evidence, copy **src + data** to a fresh sibling working
directory before retraining. A repeated identity in one results directory is
deliberately rejected by the analysis stage rather than silently selected.

```bash
# The recorded preliminary protocol (all eight models, both datasets)
python experiment.py --stage main --epochs 5 --seeds 42 43 44
python experiment.py --stage window --epochs 5 --seeds 42
python experiment.py --stage fraction --epochs 5 --seeds 42

# A narrower run, e.g. when checking a new host
python experiment.py --stage main --datasets Electricity --models LSTM DLinear --epochs 1 --seeds 42

# Planned fuller training; not claimed as completed in the preliminary report
python experiment.py --stage main --epochs 20 --seeds 42 43 44
```

Every trained model saves the validation-best state rather than the final epoch.
Validation MSE is element weighted; all errors are inverse-scaled. Scaler fitting,
seasonal MASE denominators and conditional thresholds use only available training
data. Train origins use 3-hour stride, validation/test use 1 hour. All future
targets fit inside their own chronological interval. Past context can cross a
validation/test boundary, because it is already observed at that rolling origin.

History experiments use common training origins for the 168-hour maximum window.
Data-size experiments retain the most recent 25/50/100% of training, refit scaling,
and retain the fixed channels and validation/test intervals. Changing data amount
also changes season coverage, training steps and the MASE reference denominator.

## Implementation and attribution

- `config.py`: frozen datasets, IDs, strides and task dimensions.
- `prepare_data.py`: official releases, TSF parser, subset reproduction, manifests.
- `data_loader.py`: training-only StandardScaler, verified loading and windows.
- `models/`: persistence/daily controls, LSTM, GRU, CNN, hybrid, DLinear, patch model.
- `experiment.py`: model factory, unified training/evaluation, resource recording,
  unique artifacts and SQLite storage.
- `eval.py`: original-scale and per-channel metrics, lead errors and fixed slices.
- `analyze.py`: seed aggregation, fixed-rule cases, comparisons and sensitivities.
- `verify_results.py`: independent reconstruction and checkpoint reload checks.
- `build_report.py`: matching English/Chinese nine-page body narratives and tables.

The modern classes are **self-written implementations**, not vendored or imported
Time-Series-Library classes. DLinear uses a shared two-map moving-average
decomposition with random initialisation. `PatchTSTModel` is labelled
**PatchTST-style** in the report: channel-independent shared encoder, patches
16/stride8, width64, 4 heads, 2 layers, FF128, dropout0.1, learnable positions,
PyTorch LayerNorm blocks, and no RevIN. It is not an exact published replication.
The accompanying report cites the original papers and data releases. No upstream
code commit is claimed because no upstream model source is imported.

## Build the PDFs

Install a working XeLaTeX distribution with the packages named by the report
preamble. The Chinese version uses `ctex` with Windows fonts; on Linux/Colab change
`fontset=windows` to `fontset=fandol` if Windows CJK fonts are unavailable.

```bash
cd ../report
xelatex -interaction=nonstopmode -halt-on-error LDI_Preliminary_Report_EN.tex
xelatex -interaction=nonstopmode -halt-on-error LDI_Preliminary_Report_EN.tex
xelatex -interaction=nonstopmode -halt-on-error LDI_Preliminary_Report_CN.tex
xelatex -interaction=nonstopmode -halt-on-error LDI_Preliminary_Report_CN.tex
```

Reports have nine body pages plus title and references. Fill authors in
`build_report.py` (or the generated `.tex` files if no further regeneration is
needed), including actual authorship under section headers. PDF and code ZIP are
separate uploads, as the supplied preliminary instructions require. This package
is prepared for review; no Canvas upload or submission is performed.
