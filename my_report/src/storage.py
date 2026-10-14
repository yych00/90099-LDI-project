# -*- coding: utf-8 -*-
"""
实验结果记录与 SQLite 数据库持久化模块。
记录每次实验的超参数、训练资源（用时、参数量、推理延迟）以及完整评估指标。
"""

import sqlite3
import json
import time
from typing import Dict, Any, Optional, List
import pandas as pd
from config import DB_PATH


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def init_db(db_path: str = DB_PATH):
    """
    初始化 SQLite 数据库与实验记录表（启用 WAL 模式支持多智能体并发安全写入）
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                timestamp TEXT,
                dataset TEXT,
                model_name TEXT,
                seed INTEGER,
                seq_len INTEGER,
                pred_len INTEGER,
                train_fraction REAL,
                num_params INTEGER,
                actual_epochs INTEGER,
                train_time_sec REAL,
                infer_time_ms REAL,
                mae_24h REAL,
                rmse_24h REAL,
                mase_24h REAL,
                mae_4h REAL,
                rmse_4h REAL,
                mase_4h REAL,
                slice_mae REAL,
                slice_rmse REAL,
                config_json TEXT
            )
        """)
        conn.commit()


def log_experiment_run(record: Dict[str, Any], db_path: str = DB_PATH):
    """
    记录单次实验运行结果
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (
                experiment_id, timestamp, dataset, model_name, seed,
                seq_len, pred_len, train_fraction, num_params, actual_epochs,
                train_time_sec, infer_time_ms,
                mae_24h, rmse_24h, mase_24h,
                mae_4h, rmse_4h, mase_4h,
                slice_mae, slice_rmse, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("experiment_id", f"exp_{int(time.time())}"),
            record.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            record.get("dataset", ""),
            record.get("model_name", ""),
            record.get("seed", 42),
            record.get("seq_len", 48),
            record.get("pred_len", 24),
            record.get("train_fraction", 1.0),
            record.get("num_params", 0),
            record.get("actual_epochs", 0),
            record.get("train_time_sec", 0.0),
            record.get("infer_time_ms", 0.0),
            record.get("mae_24h", 0.0),
            record.get("rmse_24h", 0.0),
            record.get("mase_24h", 0.0),
            record.get("mae_4h", 0.0),
            record.get("rmse_4h", 0.0),
            record.get("mase_4h", 0.0),
            record.get("slice_mae", 0.0),
            record.get("slice_rmse", 0.0),
            json.dumps(record.get("config", {}), ensure_ascii=False)
        ))
        conn.commit()


def get_experiments_dataframe(dataset: Optional[str] = None, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    读取实验记录为 Pandas DataFrame
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        query = "SELECT * FROM experiments"
        params = []
        if dataset:
            query += " WHERE dataset = ?"
            params.append(dataset)
        df = pd.read_sql_query(query, conn, params=params)
    return df
