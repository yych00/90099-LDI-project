# -*- coding: utf-8 -*-
"""
配置文件：定义数据规格、切分协议、模型超参数及路径。
严格遵照 ELEN90099 LDI 项目大纲及 ML Best Practices 规范。
"""

import os
from dataclasses import dataclass, field
from typing import List

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(REPORT_DIR, "data")
CHECKPOINT_DIR = os.path.join(REPORT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
DB_PATH = os.path.join(REPORT_DIR, "experiments.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


@dataclass
class DatasetConfig:
    name: str
    freq: str  # 'hourly' or '10_minutes'
    total_length: int
    num_channels: int
    channel_ids: List[str]
    seq_len: int  # 历史观察步数 (48小时)
    pred_len: int  # 预测未来步数 (24小时)
    seasonal_m: int  # 季节性周期 (24小时对应的步数)
    train_stride: int  # 训练起点步长
    val_stride: int  # 验证起点步长
    test_stride: int  # 测试起点步长
    short_horizon: int  # 短期评估跨度 (前4小时)
    unit_name: str = "原始数据单位"


# 1. Electricity Hourly 配置 (大纲 1.2 - 1.4)
ELECTRICITY_CONFIG = DatasetConfig(
    name="Electricity",
    freq="hourly",
    total_length=26304,
    num_channels=16,
    channel_ids=[
        "T5", "T8", "T21", "T30", "T99", "T133", "T144", "T146",
        "T198", "T219", "T231", "T251", "T285", "T291", "T304", "T309"
    ],
    seq_len=48,       # 48小时
    pred_len=24,      # 24小时
    seasonal_m=24,    # 日周期24小时 = 24步
    train_stride=3,   # 3小时起点步长
    val_stride=1,     # 1小时
    test_stride=1,    # 1小时
    short_horizon=4,  # 前4小时 = 4步
    unit_name="kW"
)

# 2. Solar 10 Minutes 配置 (大纲 1.2 - 1.4)
SOLAR_CONFIG = DatasetConfig(
    name="Solar",
    freq="10_minutes",
    total_length=52560,
    num_channels=16,
    channel_ids=[
        "T7", "T9", "T27", "T28", "T41", "T58", "T60", "T64",
        "T69", "T77", "T82", "T97", "T115", "T123", "T127", "T131"
    ],
    seq_len=288,      # 48小时 = 288步
    pred_len=144,     # 24小时 = 144步
    seasonal_m=144,   # 日周期24小时 = 144步
    train_stride=18,  # 3小时起点步长 = 18步
    val_stride=6,     # 1小时 = 6步
    test_stride=6,    # 1小时 = 6步
    short_horizon=24, # 前4小时 = 24步
    unit_name="MW"
)

# 统一时间分割比例 (70% Train, 10% Val, 20% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# 训练起始配置 (大纲 2.4)
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 3  # 早停轮数
    device: str = "cuda"  # 运行时动态回退至 cpu
