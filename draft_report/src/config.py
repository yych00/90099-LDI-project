# -*- coding: utf-8 -*-
"""Fixed task definitions keep channel identity, physical horizons and splits comparable across experiments."""

import os
from dataclasses import dataclass, field
from typing import List

# Resolve paths from the source file so commands work outside the original author's machine.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
# Keep frozen pilot evidence outside new output roots, so fuller training cannot
# overwrite the records supporting the submitted draft.
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "pilot")
CHECKPOINT_DIR = os.path.join(REPORT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
DB_PATH = os.path.join(REPORT_DIR, "experiments.db")

# Importing task definitions must not create experiment outputs.


@dataclass
class DatasetConfig:
    name: str
    freq: str  # 'hourly' or '10_minutes'
    total_length: int
    num_channels: int
    channel_ids: List[str]
    seq_len: int  # History steps; both datasets represent the same 48 hours.
    pred_len: int  # Output steps; preserve the same 24-hour task at each frequency.
    seasonal_m: int  # Daily period for the baseline and training-only MASE reference.
    train_stride: int  # Subsample training origins to keep the pilot computationally manageable.
    val_stride: int  # Evaluate on the same fixed hourly grid for every model.
    test_stride: int  # Hold evaluation origins constant so errors are comparable.
    short_horizon: int  # First four hours provide a short-horizon view of the same forecasts.
    unit_name: str = "original release units"


# Keep the frozen Electricity IDs; do not pick users by test performance.
ELECTRICITY_CONFIG = DatasetConfig(
    name="Electricity",
    freq="hourly",
    total_length=26304,
    num_channels=16,
    channel_ids=[
        "T5",
        "T8",
        "T21",
        "T30",
        "T99",
        "T133",
        "T144",
        "T146",
        "T198",
        "T219",
        "T231",
        "T251",
        "T285",
        "T291",
        "T304",
        "T309",
    ],
    seq_len=48,  # 48 hours of observed context.
    pred_len=24,  # 24 hours of future targets.
    seasonal_m=24,  # One daily cycle at hourly frequency.
    train_stride=3,  # Three-hour stride reduces overlapping training examples.
    val_stride=1,  # Hourly evaluation origins permit fair rolling comparisons.
    test_stride=1,  # Hourly evaluation origins permit fair rolling comparisons.
    short_horizon=4,  # Four-hour prefix at hourly frequency.
    unit_name="original release units",
)

# Convert physical hours to ten-minute steps without changing the task.
SOLAR_CONFIG = DatasetConfig(
    name="Solar",
    freq="10_minutes",
    total_length=52560,
    num_channels=16,
    channel_ids=[
        "T7",
        "T9",
        "T27",
        "T28",
        "T41",
        "T58",
        "T60",
        "T64",
        "T69",
        "T77",
        "T82",
        "T97",
        "T115",
        "T123",
        "T127",
        "T131",
    ],
    seq_len=288,  # 48 hours at six samples per hour.
    pred_len=144,  # 24 hours at six samples per hour.
    seasonal_m=144,  # One daily cycle at ten-minute frequency.
    train_stride=18,  # Three hours at six samples per hour.
    val_stride=6,  # The same hourly evaluation grid in ten-minute steps.
    test_stride=6,  # The same hourly evaluation grid in ten-minute steps.
    short_horizon=24,  # The same four-hour prefix in ten-minute steps.
    unit_name="original release units",
)

# Chronological target partitions prevent future labels leaking into fitting.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


# Defaults can be overridden for later validation-based studies.
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 3  # Stop after this many validation epochs without improvement.
    device: str = "cuda"  # Use CPU when no CUDA device is available.
