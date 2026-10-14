# -*- coding: utf-8 -*-
"""Expose the same forecasting interface for classical controls, recurrent, convolutional, hybrid and modern designs."""

from .baselines import PersistenceModel, DailySeasonalNaiveModel
from .rnn_models import LSTMModel, GRUModel
from .cnn_models import CNN1DModel
from .hybrid_models import CNNLSTMModel
from .modern_models import DLinearModel, PatchTSTModel

__all__ = [
    "PersistenceModel",
    "DailySeasonalNaiveModel",
    "LSTMModel",
    "GRUModel",
    "CNN1DModel",
    "CNNLSTMModel",
    "DLinearModel",
    "PatchTSTModel",
]
