# ELEN90099 LDI: draft and final-study code

**Authors:** Yuchen Yang (1587576) and Zhengnan Cao (1711701).

Yuchen Yang is responsible for data preparation and experimental analysis.
Zhengnan Cao is responsible for models, theory, the introduction and conclusion.
The accompanying draft identifies section authors individually; Section 2.2 belongs
to Yuchen Yang, while Sections 2.1 and 2.3–2.4 belong to Zhengnan Cao.

This folder contains the current forecasting implementation **and executable
entry points for the agreed final-report studies**. More complete training,
validation-based tuning, repeated sensitivity, capacity controls, and extended
error analysis are implemented as tools. Their availability is not a claim that
those research experiments have already been completed.

## 1. Quick start

Use Python 3.10 or later. The recorded pilot environment was Python 3.10.21,
PyTorch 2.5.1+cu121, NumPy 2.2.6, Matplotlib 3.10.9 and an RTX 3060 Laptop GPU.
CPU execution is supported. Laptop GPU timings in the report are not Colab T4
measurements.

From **this src directory**:

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python verify_metadata.py
~~~

Install an appropriate PyTorch CPU/CUDA wheel for the host; the requirements do
not force one GPU driver or claim bitwise agreement across devices. The package
includes both selected data arrays and their source manifests, so tests and
metadata checks require neither a download nor new benchmark training.

## 2. Directory guide

~~~text
src/
  config.py                    frozen tasks, channels, paths and defaults
  prepare_data.py               official release download and TSF parsing
  data_loader.py                train-only scaling and causal windows
  models/                      all eight pilot models
  experiment.py                fitting, validation selection, recording and ablations
  validation_search.py         small validation-only hyperparameter grid
  summarize_runs.py            seed summaries grouped by the complete protocol
  error_analysis.py            clipping, channels, lead times, fixed cases and block intervals
  eval.py                      original-scale metrics and training-defined slices
  analyze.py                   full pilot-style figure reconstruction
  verify_metadata.py           checks supported by the small bundled evidence
  verify_results.py            full predictions/checkpoint verification
  reporting/                   table preparation, report compilation, QA and ZIP packaging
  tests/                       model, metric, protocol and future-control regression checks
  notebooks/                   Colab execution entry point
  data/                        two fixed 16-channel arrays and provenance manifests
  evidence/                    frozen 72-run pilot metadata, SQLite and original summary figures
  outputs/                     new experiments; excluded from the submitted code ZIP
~~~

The report sources live one level above src. They are intentionally authored
LaTeX rather than a generated narrative; build_reports.py updates numerical tables
and compiles them without discarding revisions, author details or the agreed outline.

The small evidence bundle contains JSON records, a matching SQLite database,
the 16-row main summary, protocol ranges, fixed-case selections, and figures.
It excludes the approximately 712 MiB of full pilot forecasts and checkpoints.
This keeps the code ZIP small without making a false promise that omitted artifacts
can be reloaded. The historical verification.json records earlier full checks;
verify_metadata.py performs only the checks it can independently support now.
Recorded code hashes identify the historical pilot source, before the current
English documentation and future-study extensions.

## 3. Data and causal evaluation

Electricity uses 16 of 321 hourly series (2012–2014), with L=48 and H=24.
Solar uses 16 of 137 ten-minute series (2006), with L=288 and H=144.
Both represent 48 hours of history and a 24-hour forecast; models are fitted
separately. Solar's published source includes simulated PV series.

The exact channel IDs are frozen in config.py. Subset reconstruction uses only
training-period means, four scale groups and seed 42. TSF parsing matches series
names, not assumed numeric column positions. Provenance manifests preserve hashes
for the original archive, TSF and selected array.

~~~bash
# Optional: independently reconstruct the bundled arrays from official releases.
python prepare_data.py
~~~

This command downloads Monash-linked Zenodo records 4656140 and 4656144.
Missing or changed provenance causes loading to fail; there is no synthetic fallback.
Values stay in original release units, with no assumed kW/MW conversion or local timezone.

Target intervals are chronological 70%/10%/20%. Every full target belongs to one
split; earlier observations may supply context at a later rolling origin.
Scaling, MASE denominators and conditional thresholds use available training data
only. Training origins advance every three hours; evaluation origins every hour.
Training uses standardised MSE. Test MAE/RMSE use the inverse-scaled outputs;
MASE averages per-channel ratios rather than dividing a global MAE.

History studies share origins admissible for the maximum history. Data-fraction
studies use the newest 25%/50%/100% of training, refitting scaling on each available
segment. This also changes recency, seasonal coverage, optimiser steps and the MASE
denominator; sample count alone is not an isolated cause.

## 4. Reproduce the pilot protocol

Use a **fresh output directory for each campaign**. Existing draft evidence remains
under evidence; it is never a training output. Unique run IDs protect artifacts,
while analysis rejects repeated identities rather than selecting a favourable run.

~~~bash
python experiment.py --stage main --epochs 5 --seeds 42 43 44 --output-root outputs/pilot_main
python experiment.py --stage window --epochs 5 --seeds 42 --output-root outputs/pilot_window
python experiment.py --stage fraction --epochs 5 --seeds 42 --output-root outputs/pilot_fraction

# Optional narrow local smoke experiment; it produces new results, not report evidence.
python experiment.py --stage main --datasets Electricity --models DLinear --epochs 1 --seeds 42 --device cpu --output-root outputs/smoke
~~~

The main stage fits LSTM, GRU, 1D-CNN, CNN–LSTM, DLinear and PatchTST-style,
and evaluates persistence/daily naive. Defaults match the pilot: batch 32, Adam,
learning rate 0.001, weight decay 0.0001 and patience 3. Early stopping uses
validation loss; the validation-best state, not the last epoch, is evaluated.

Every test-evaluated run saves a JSON record and SQLite row, its best checkpoint
when trainable, scaler, origin arrays and test_predictions.npz. Configuration,
environment, source hashes, history, optimiser steps and measured resources remain
associated with that unique run. Model-only latency uses ten warmups and 30
synchronised forward passes with input already on the device.

## 5. Agreed final-report studies

These commands are ready for later use. They were **not run as new benchmark
campaigns when preparing this code package**.

### Fuller training and validation-only search

~~~bash
python experiment.py --stage main --epochs 20 --seeds 42 43 44 --output-root outputs/final_main

python validation_search.py --dataset Electricity --model LSTM --epochs 20 --learning-rates 0.001 0.0003 --weight-decays 0.0001 --seeds 42 43 44 --output-root outputs/tuning_electricity_lstm
~~~

The search writes every attempted configuration and selection.json. Selection is
by mean best validation MSE across the requested seeds. Test evaluation is disabled
during these trials: no test forecast file or test score is produced. Apply the
selected settings in a separate fresh main run via --learning-rate and
--weight-decay; do not choose them using the test ranking.

### Replicated sensitivities and an update-budget control

~~~bash
python experiment.py --stage window --epochs 20 --seeds 42 43 44 --output-root outputs/final_window
python experiment.py --stage fraction --epochs 20 --seeds 42 43 44 --output-root outputs/final_fraction

# Additional control: same update ceiling across data fractions.
python experiment.py --stage fraction --epochs 100 --max-updates 1000 --patience 100 --seeds 42 43 44 --output-root outputs/fraction_update_control
~~~

History settings are 24/48/168 physical hours. The latter example increases the
epoch/patience ceilings so the 1000-update ceiling can become the stopping limit.
Check recorded optimizer_steps before describing the comparison as matched:
max-updates is a ceiling, and an epoch limit or early stopping can still stop sooner.

### Frontend removal and capacity matching

~~~bash
python experiment.py --stage ablation --epochs 20 --seeds 42 43 44 --output-root outputs/final_ablation
~~~

The ablation stage compares ordinary 64-unit LSTM (no convolution), CNN–LSTM and
LSTMCapacityMatched. The last control chooses the integer hidden width with the
closest parameter count to CNN–LSTM for the current output size, and records that
width in model_options. It is approximate capacity matching, not an assertion of
identical FLOPs, optimisation behaviour or representations. All three use the
same data, origins and direct multi-step targets.

### Per-channel errors, lead times, clipping and fixed cases

~~~bash
python error_analysis.py --root outputs/final_main --output outputs/final_main/extended
# Optional exploratory interval; one origin corresponds to one hour.
python error_analysis.py --root outputs/final_main --output outputs/final_main/extended_blocks --bootstrap-repetitions 200 --block-hours 24
~~~

Outputs are postprocessing.csv (raw versus uniformly clipped), channels.csv,
lead_times.csv, fixed_cases.json and moving_block_intervals.json. The first test
origin and earliest window containing the first channel's true maximum provide
fixed case rules. They are not chosen by a model's error. Bootstrap resamples
consecutive origins; the interval is exploratory, depends on block length, and
is not an independent-window significance test or a comparison of models.

A pretrained foundation model is an optional research extension, not an implemented
or completed experiment in this package. The agreed mandatory final-study tools
above do not depend on one. Weather/calendar extensions and weekday analysis also
require additional verified information and are not claimed here.

## 6. Summaries, figures and full verification

~~~bash
# Works with any test-evaluated stage, including ablations and repeated sensitivities.
python summarize_runs.py --root outputs/final_main --output outputs/final_main/summary.csv
python summarize_runs.py --root outputs/final_ablation --output outputs/final_ablation/summary.csv

# Requires each run's saved prediction/checkpoint artifacts:
python verify_results.py --root outputs/final_main --reload
~~~

Summaries combine seeds only when stage, task, history, fraction, common origins,
optimiser settings and model options agree. One seed has no sample SD estimate.
Parameter-free baseline replicas do not become independent forecasting trials.

analyze.py reconstructs the original five-figure pilot layout from a **complete**
root with both datasets, all main models and seed 42. If sensitivity records are
also present, repeated seeds are aggregated per setting with SD error bars.
It needs full saved predictions for fixed-case plots. For partial runs or
sensitivity-only roots, use summarize_runs.py and error_analysis.py instead.
Do not run analysis tools against frozen evidence as a writable output directory.

verify_results.py recomputes target alignment, origins, MAE/RMSE/MASE and SQLite
agreement. With --reload it also checks checkpoint hashes and CPU forecasts;
CUDA reload is checked when available. It writes audit/verification.json in the
chosen run root. It is distinct from verify_metadata.py, which needs no artifacts.

## 7. Report compilation and packaging

Install TeX Live or MiKTeX with pdflatex, xelatex and ctex. Chinese fonts must be
available to ctex (Windows font set locally; Fandol on a suitable Linux installation).

~~~bash
python -m pip install -r requirements-report.txt
python reporting/build_reports.py --check
python reporting/package_submission.py
~~~

The second command preserves the authored prose and names, prepares the shared
table with best means **and their SDs** in bold, compiles each report twice,
and checks identities, all references, values, pagination and approximate chapter
allocation. PDF page images should additionally be inspected visually.

The code ZIP contains src (including small data/evidence), report source files,
figures, tables and the root README; caches, new experiment outputs and heavy
checkpoints are excluded. Submit report.pdf separately from code.zip. report_cn.pdf
is the Chinese companion. These tools do not upload or submit anything.

## 8. Design comments and implementation attribution

Comments explain why boundaries and choices exist: training-only statistics
prevent leakage; complete target intervals enforce chronological isolation;
common origins avoid sample drift; element-weighted validation does not over-weight
a short final batch; cloned best states preserve the selected checkpoint; tail
padding keeps recent observations; independent run roots protect submitted evidence.

The model classes are self-written. DLinear shares two linear maps after a
25-step moving average. PatchTST-style uses patches 16/stride 8, width 64, four heads,
two encoder layers, feed-forward width 128, dropout 0.1, learnable positions and
PyTorch LayerNorm blocks. It has no RevIN and is not an exact upstream reproduction.
Original LSTM/GRU, DLinear, PatchTST, Monash and MASE papers are cited in the report.

The supplied rubric assigns 15 points to structure/style, 25 to problem/task
formulation, 30 to design/initial experiments, 10 to literature/references and 20
to initial code. The report's four chapters address the narrative criteria; the
modules, rationale comments, tests and this README address reproducibility and
code readability. This mapping is not a prediction of the awarded grade.

## 9. Troubleshooting and limits

- Missing data/manifest: run prepare_data.py; do not generate substitute signals.
- Hash/ID mismatch: reconstruct the original releases and keep the frozen channels.
- CUDA unavailable: use --device cpu. This changes runtime, not the task definition.
- Duplicate identities or mixed tuning protocols: use a fresh output root and
  retain all attempts; do not keep whichever test score looks best.
- Missing test_predictions.npz or best.pt: use an artifact-complete run directory
  or rerun the documented campaign. Small frozen metadata cannot replace them.
- Missing TeX/fonts: install the documented report tools before compilation;
  numerical experiment code does not require TeX or PDF libraries.
- Report authorship is already confirmed. Do not use a narrative generator that
  overwrites the manually revised four-chapter draft.
