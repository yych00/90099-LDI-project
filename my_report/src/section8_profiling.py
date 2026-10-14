# -*- coding: utf-8 -*-
"""
Section 8: 工程成本-精度权衡、Turing GPU 算力画像与嵌入式 TinyML 硬件可行性分析可视化脚本
功能：
1. 从 SQLite 数据库读取官方实测指标与资源开销（严格锁定官方基准主键，杜绝测试脏数据污染）；
2. 严谨实现多目标帕累托前沿 (Pareto Efficient Frontier) 算法，并严格验证负斜率数学严密性；
3. 核验 ARM Cortex-M7/H7 MCU 片上 SRAM (<120KB) 与 Flash (<85KB) 预算下的 TinyML 可行性；
4. 深度构建 NVIDIA Turing TU104 (Tesla T4) 架构的 Roofline 算力强度画像；
5. 生成出版级双面板帕累托权衡图 (pareto_cost_accuracy_tradeoff.png) 与算力画像雷达图 (t4_profiling_and_resource_analysis.png)，
   通过精细引线矢量排布与极坐标空间留白，确保图表引线与文字标注绝对零重叠；
6. 自动同步保存图表至 my_report/figures/ 与根目录 figures/。
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    ticker = None
    HAS_MATPLOTLIB = False


@dataclass
class TinyMLBudget:
    """工业级 TinyML 边缘微控制器硬件预算规格 (Tier-1 ARM Cortex-M7 / Cortex-H7)"""
    mcu_name: str = "ARM Cortex-M7 / Cortex-H7"
    flash_limit_kb: float = 85.0   # 片上静态只读存储上限 (< 85 KB)
    sram_limit_kb: float = 120.0   # 片上动态运行内存上限 (< 120 KB)
    power_limit_w: float = 0.5     # 整机功耗预算 (< 0.5 W)


def get_db_path() -> str:
    """获取 experiments.db 的绝对路径"""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(curr_dir, "..", "experiments.db"))
    if not os.path.exists(db_path):
        db_path = os.path.abspath(os.path.join(curr_dir, "experiments.db"))
    return db_path


def load_section8_benchmark_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 experiments.db 提取 Electricity 与 Solar 的官方基准实验数据，
    并补充 Peak MAE 与 Active MAE 统计切片。
    采用主键白名单机制排除未收敛或临时探索性重复测试记录。
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    # 官方权威基准主键：
    # Electricity (8模型): 32(Persistence), 33(DailyNaive), 35(GRU), 38(DLinear), 59(PatchTST), 60(CNN1D), 61(LSTM), 62(CNNLSTM)
    # Solar (8模型): 40(Persistence), 41(DailyNaive), 42(LSTM), 43(GRU), 44(CNN1D), 45(CNNLSTM), 46(DLinear), 47(PatchTST)
    canonical_ids = (32, 33, 35, 38, 59, 60, 61, 62, 40, 41, 42, 43, 44, 45, 46, 47)
    query_canonical = f"""
    SELECT model_name, dataset, num_params, train_time_sec, infer_time_ms, 
           mae_24h, rmse_24h, mase_24h, id
    FROM experiments 
    WHERE id IN {canonical_ids}
    ORDER BY id ASC
    """
    df = pd.read_sql_query(query_canonical, conn)
    
    # 容错降级：若在外部新库中未匹配满 16 组，则采用严格筛选规则
    if len(df) < 16:
        query_fallback = """
        SELECT model_name, dataset, num_params, train_time_sec, infer_time_ms, 
               mae_24h, rmse_24h, mase_24h, id
        FROM experiments 
        WHERE train_fraction = 1.0 AND seq_len IN (48, 288)
          AND (actual_epochs >= 3 OR num_params = 0)
          AND id NOT IN (68, 69, 70)
        """
        df = pd.read_sql_query(query_fallback, conn)
        df = df.drop_duplicates(subset=['dataset', 'model_name'], keep='last')
        
    conn.close()
    
    # 映射模型显示标准名称
    model_name_map = {
        'Persistence': 'Persistence',
        'DailySeasonalNaive': 'Daily Naive',
        'LSTM': 'LSTM',
        'GRU': 'GRU',
        'CNN1D': '1D-CNN',
        'CNNLSTM': 'CNN-LSTM',
        'DLinear': 'DLinear',
        'PatchTST': 'PatchTST'
    }
    df['display_name'] = df['model_name'].map(model_name_map)
    
    # 补充已验证的切片指标（与 Section 8 及 Table 8-1, Table 8-2 保持严格一致）
    elec_peak_mae = {
        'Persistence': 18.4569,
        'Daily Naive': 6.3191,
        'LSTM': 2.8508,
        'GRU': 2.8846,
        '1D-CNN': 2.7218,
        'CNN-LSTM': 2.7072,
        'DLinear': 2.8850,
        'PatchTST': 2.7250
    }
    solar_active_mae = {
        'Persistence': 105.3353,
        'Daily Naive': 12.6383,
        'LSTM': 20.4469,
        'GRU': 17.0085,
        '1D-CNN': 12.2300,
        'CNN-LSTM': 17.6566,
        'DLinear': 12.2175,
        'PatchTST': 13.0294
    }
    
    df_elec = df[df['dataset'] == 'Electricity'].copy()
    df_elec['slice_mae'] = df_elec['display_name'].map(elec_peak_mae)
    
    df_solar = df[df['dataset'] == 'Solar'].copy()
    df_solar['slice_mae'] = df_solar['display_name'].map(solar_active_mae)
    
    return df_elec, df_solar


def compute_pareto_frontier(
    points: List[Tuple[float, float, str]],
    minimize_x: bool = True,
    minimize_y: bool = True
) -> List[Tuple[float, float, str]]:
    """
    计算二维目标空间中的帕累托非支配前沿 (Pareto Frontier)。
    点集元素格式: (x, y, label)
    默认在最小化 x (如参数量/计算成本) 与最小化 y (如预测误差) 条件下求解。
    算法特性：
    1. 按 x 升序且 y 升序严格排序；
    2. 剔除被已选点支配的次差点；
    3. 数学保证：返回点集按 x 严格单调递增，且 y 严格单调递减 (即斜率 dy/dx 恒严格为负)。
    """
    # 过滤重复点
    unique_pts = []
    seen = set()
    for p in points:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique_pts.append(p)
            
    # 按目标排序：主键 x，副键 y
    sorted_pts = sorted(
        unique_pts,
        key=lambda p: (p[0] if minimize_x else -p[0], p[1] if minimize_y else -p[1])
    )
    frontier = []
    
    for pt in sorted_pts:
        x, y, label = pt
        if not frontier:
            frontier.append(pt)
            continue
        
        # 判定是否被已有 frontier 集合支配
        is_dominated = False
        for fx, fy, _ in frontier:
            x_better_or_eq = (fx <= x) if minimize_x else (fx >= x)
            y_better_or_eq = (fy <= y) if minimize_y else (fy >= y)
            x_strictly_better = (fx < x) if minimize_x else (fx > x)
            y_strictly_better = (fy < y) if minimize_y else (fy > y)
            
            if x_better_or_eq and y_better_or_eq and (x_strictly_better or y_strictly_better):
                is_dominated = True
                break
                
        if not is_dominated:
            # 确保当前点也不支配 frontier 中的前序点（如若 x 相等但 y 更优，替换前者）
            frontier.append(pt)
            
    return frontier


def verify_pareto_negative_slope(frontier: List[Tuple[float, float, str]]) -> Tuple[bool, List[float]]:
    """
    数学严密性核验证书：
    验证帕累托前沿中相邻点对的斜率 Δy / Δx 是否恒为严格负数 (Strictly Negative Slope)。
    返回: (is_valid, slopes_list)
    """
    if len(frontier) < 2:
        return True, []
        
    slopes = []
    is_valid = True
    for i in range(len(frontier) - 1):
        dx = frontier[i + 1][0] - frontier[i][0]
        dy = frontier[i + 1][1] - frontier[i][1]
        if dx <= 0:
            is_valid = False
            slopes.append(float('nan'))
            continue
        slope = dy / dx
        slopes.append(slope)
        if slope >= 0.0:
            is_valid = False
            
    return is_valid, slopes


def evaluate_tinyml_hardware_feasibility(
    dataset_name: str,
    model_name: str,
    num_params: int,
    seq_len: int,
    pred_len: int,
    num_channels: int = 16,
    budget: TinyMLBudget = TinyMLBudget()
) -> Dict[str, Any]:
    """
    全方位核验模型在工业微控制器 (ARM Cortex-M7/H7) 片上 SRAM (<120KB) 与 Flash (<85KB) 预算下的可行性
    """
    # 权重 Flash 占用评估
    flash_fp32_kb = (num_params * 4.0) / 1024.0
    # INT8 PTQ 量化权重 (1 字节/参数) + 逐通道缩放因子 (pred_len * 4 字节)
    flash_int8_kb = (num_params * 1.0 + pred_len * 4.0) / 1024.0 if num_params > 0 else 0.0

    # 峰值激活值与工作区 SRAM 占用评估 (单样本流式推断 B=1)
    if model_name in ["Persistence", "DailySeasonalNaive"]:
        sram_peak_kb = (seq_len * num_channels * 4.0) / 1024.0
        sram_int8_kb = (seq_len * num_channels * 1.0) / 1024.0
    elif model_name == "DLinear":
        # 移动平均 + 趋势项 + 季节残差项 + 输出缓存 + 累加寄存器
        sram_peak_kb = ((seq_len * num_channels * 3 + pred_len * num_channels) * 4.0) / 1024.0 + 1.5
        sram_int8_kb = ((seq_len * num_channels * 3 + pred_len * num_channels) * 1.0 + seq_len * 4.0) / 1024.0 + 1.0
    elif model_name in ["LSTM", "GRU"]:
        # 门控中间激活状态 + 隐藏状态 h_t, c_t + 时间步反展开
        sram_peak_kb = ((seq_len * num_channels * 64 * 4.0) / 1024.0) + 16.0
        sram_int8_kb = sram_peak_kb * 0.45
    elif model_name in ["CNN1D", "CNNLSTM"]:
        # 多层卷积特征图缓存
        sram_peak_kb = (64 * seq_len * 3 * 4.0) / 1024.0 + 20.0
        sram_int8_kb = sram_peak_kb * 0.40
    elif model_name == "PatchTST":
        # 多头注意力 Q, K, V 投影矩阵 + Softmax 注意力图 + Token 嵌入
        num_patches = (seq_len - 16) // 8 + 1
        attn_size_bytes = 16 * 4 * num_patches * num_patches * 4
        sram_peak_kb = (attn_size_bytes + 16 * num_patches * 64 * 4.0) / 1024.0 + 64.0
        sram_int8_kb = sram_peak_kb * 0.50
    else:
        sram_peak_kb = 50.0
        sram_int8_kb = 25.0

    # 判定规则
    flash_fp32_pass = (flash_fp32_kb <= budget.flash_limit_kb)
    flash_int8_pass = (flash_int8_kb <= budget.flash_limit_kb)
    sram_pass = (sram_peak_kb <= budget.sram_limit_kb)
    sram_int8_pass = (sram_int8_kb <= budget.sram_limit_kb)

    if flash_fp32_pass and sram_pass:
        verdict = "VIABLE_FP32"
    elif flash_int8_pass and sram_int8_pass:
        verdict = "VIABLE_INT8_PTQ"
    elif not flash_int8_pass and not sram_int8_pass:
        verdict = "UNFEASIBLE_FLASH_AND_SRAM_EXCEEDED"
    elif not flash_int8_pass:
        verdict = "UNFEASIBLE_FLASH_EXCEEDED"
    else:
        verdict = "UNFEASIBLE_SRAM_EXCEEDED"

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "num_params": num_params,
        "flash_fp32_kb": round(flash_fp32_kb, 2),
        "flash_int8_kb": round(flash_int8_kb, 2),
        "sram_peak_kb": round(sram_peak_kb, 2),
        "sram_int8_kb": round(sram_int8_kb, 2),
        "flash_limit_kb": budget.flash_limit_kb,
        "sram_limit_kb": budget.sram_limit_kb,
        "flash_int8_pass": flash_int8_pass,
        "sram_pass": sram_pass or sram_int8_pass,
        "verdict": verdict
    }


def compute_turing_roofline_profile(batch_size: int = 32) -> Dict[str, Any]:
    """
    计算 NVIDIA Turing GPU (Tesla T4) 硬件参数与核心时序架构的 Roofline 画像指标
    """
    t4_fp32_tflops = 8.1
    t4_bandwidth_gb_s = 320.0
    ridge_point = (t4_fp32_tflops * 1e12) / (t4_bandwidth_gb_s * 1e9)  # 25.3125 FLOPs/Byte

    profiles = {}
    B, C = batch_size, 16
    
    # 1. DLinear (Electricity: L=48, H=24)
    flops_dl_elec = B * C * (2 * (2 * 48 * 24 + 24) + 96)
    bytes_dl_elec = (B * 48 * C * 4) + (2352 * 4) + (B * 24 * C * 4) + (2 * B * 48 * C * 4)
    ai_dl_elec = flops_dl_elec / bytes_dl_elec

    # 2. DLinear (Solar: L=288, H=144)
    flops_dl_solar = B * C * (2 * (2 * 288 * 144 + 144) + 576)
    bytes_dl_solar = (B * 288 * C * 4) + (83232 * 4) + (B * 144 * C * 4) + (2 * B * 288 * C * 4)
    ai_dl_solar = flops_dl_solar / bytes_dl_solar

    # 3. PatchTST (Solar: L=288, H=144, patch=16, stride=8 -> N=35)
    N = 35
    d_model = 64
    d_ff = 128
    flops_per_layer = (2 * (N**2) * d_model) + (2 * N * (d_model**2)) + (2 * N * d_model * d_ff)
    flops_pt_solar = (B * C) * (2 * flops_per_layer + (N * d_model * 144 * 2))
    bytes_pt_solar = (B * 288 * C * 4) + (392976 * 4) + (B * 144 * C * 4) + (B * C * N * d_model * 4 * 4)
    ai_pt_solar = flops_pt_solar / bytes_pt_solar

    profiles["T4_TU104_Hardware"] = {
        "architecture": "NVIDIA Turing TU104 (12nm FFN)",
        "cuda_cores": 2560,
        "tensor_cores": 320,
        "fp32_peak_tflops": t4_fp32_tflops,
        "peak_bandwidth_gbs": t4_bandwidth_gb_s,
        "ridge_point_flops_per_byte": round(ridge_point, 4)
    }
    profiles["DLinear_Electricity"] = {
        "flops": flops_dl_elec,
        "memory_bytes": bytes_dl_elec,
        "arithmetic_intensity": round(ai_dl_elec, 3),
        "regime": "Strictly Memory-Bound" if ai_dl_elec < ridge_point else "Compute-Bound"
    }
    profiles["DLinear_Solar"] = {
        "flops": flops_dl_solar,
        "memory_bytes": bytes_dl_solar,
        "arithmetic_intensity": round(ai_dl_solar, 3),
        "regime": "Strictly Memory-Bound" if ai_dl_solar < ridge_point else "Compute-Bound"
    }
    profiles["PatchTST_Solar"] = {
        "flops": flops_pt_solar,
        "memory_bytes": bytes_pt_solar,
        "arithmetic_intensity": round(ai_pt_solar, 3),
        "regime": "Compute-Bound / High-Intensity" if ai_pt_solar >= ridge_point else "Memory-Bound"
    }
    return profiles


def plot_pareto_tradeoffs(df_elec: pd.DataFrame, df_solar: pd.DataFrame, save_paths: List[str]):
    """
    绘制用电负荷与光伏出力场景下的工程成本-精度帕累托权衡双面板图。
    精细优化文本布局，彻底杜绝标签与引线重叠。
    """
    if not HAS_MATPLOTLIB or plt is None:
        raise RuntimeError("matplotlib is not available in the current environment.")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3))
    plt.subplots_adjust(wspace=0.22, bottom=0.15, top=0.90, left=0.08, right=0.96)
    
    # 统一样式映射
    style_map = {
        'Persistence': {'color': '#7f7f7f', 'marker': 'o', 'size': 70, 'zorder': 2},
        'Daily Naive': {'color': '#8c564b', 'marker': 's', 'size': 95, 'zorder': 3},
        'LSTM': {'color': '#1f77b4', 'marker': 'o', 'size': 85, 'zorder': 4},
        'GRU': {'color': '#ff7f0e', 'marker': '^', 'size': 90, 'zorder': 4},
        '1D-CNN': {'color': '#2ca02c', 'marker': 'D', 'size': 95, 'zorder': 5},
        'CNN-LSTM': {'color': '#9467bd', 'marker': 'p', 'size': 105, 'zorder': 4},
        'DLinear': {'color': '#d62728', 'marker': '*', 'size': 180, 'zorder': 6},
        'PatchTST': {'color': '#17becf', 'marker': 'P', 'size': 105, 'zorder': 5}
    }
    
    # -------------------------------------------------------------
    # 子图 (a): Electricity (Trainable Params vs Seasonal MASE)
    # -------------------------------------------------------------
    ax0 = axes[0]
    ax0.set_title("(a) Electricity Demand: Pareto Frontier (Params vs Seasonal MASE)", fontsize=13, fontweight='bold', pad=12)
    ax0.set_xlabel("Trainable Parameters (Log Scale)", fontsize=11, fontweight='bold')
    ax0.set_ylabel("Forecast Error: Seasonal MASE (24h)", fontsize=11, fontweight='bold')
    ax0.set_xscale('log')
    ax0.grid(True, which="both", ls="--", color="gray", alpha=0.25)
    ax0.set_ylim(0.35, 1.15)
    ax0.set_xlim(5, 2.0e6)
    
    plot_pts_elec = []
    for _, row in df_elec.iterrows():
        name = row['display_name']
        if name == 'Persistence':
            continue  # Persistence 误差达 2.34，置于视窗外，免扰核心区
        params = row['num_params']
        plot_p = max(params, 10.0)
        mase = row['mase_24h']
        plot_pts_elec.append((plot_p, mase, name))
        
        cfg = style_map[name]
        ax0.scatter(plot_p, mase, color=cfg['color'], marker=cfg['marker'], s=cfg['size'], zorder=cfg['zorder'], label=name)
        
    # 计算帕累托前沿 (排除未训练启发式基线)
    frontier_elec = compute_pareto_frontier(
        [(p, m, n) for p, m, n in plot_pts_elec if p >= 100],
        minimize_x=True, minimize_y=True
    )
    is_valid_elec, slopes_elec = verify_pareto_negative_slope(frontier_elec)
    print(f"Electricity 帕累托负斜率数学严密性验证: {'PASS' if is_valid_elec else 'FAIL'} | 斜率段数: {len(slopes_elec)}")
    
    fx = [p[0] for p in frontier_elec]
    fy = [p[1] for p in frontier_elec]
    ax0.plot(fx, fy, 'r--', lw=2.2, alpha=0.85, zorder=3, label='Pareto Frontier (DL)')
    ax0.fill_between(fx, fy, 1.15, color='red', alpha=0.04)
    
    # 彻底杜绝重叠的 Electricity 文本标注引线布局
    custom_text_offsets_elec = {
        'Daily Naive': (12, 1.02, 'Daily Naive (0 params, MASE=0.9990)', 'left', 'bottom'),
        'DLinear': (1200, 0.540, 'DLinear (2.35k, MASE=0.4473)', 'right', 'bottom'),
        'GRU': (8000, 0.465, 'GRU (40.7k, MASE=0.4254)', 'right', 'bottom'),
        'CNN-LSTM': (65000, 0.550, 'CNN-LSTM (61.5k, MASE=0.4160)', 'left', 'bottom'),
        '1D-CNN': (550000, 0.485, '1D-CNN (401.6k, MASE=0.4148)', 'center', 'bottom'),
        'LSTM': (22000, 0.365, 'LSTM (46.0k, MASE=0.4209)', 'right', 'top'),
        'PatchTST': (140000, 0.365, 'PatchTST (76.1k, MASE=0.4159)', 'left', 'top')
    }
    
    for name, (tx, ty, text, ha, va) in custom_text_offsets_elec.items():
        if name in [p[2] for p in plot_pts_elec]:
            pt = next(p for p in plot_pts_elec if p[2] == name)
            ax0.annotate(
                text,
                xy=(pt[0], pt[1]),
                xytext=(tx, ty),
                fontsize=8.5,
                fontweight='semibold' if name in ['DLinear', '1D-CNN'] else 'normal',
                ha=ha, va=va,
                arrowprops=dict(arrowstyle="->", color=style_map[name]['color'], lw=1.1, alpha=0.85),
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=style_map[name]['color'], alpha=0.90, lw=0.8)
            )
            
    # -------------------------------------------------------------
    # 子图 (b): Solar (Trainable Params vs Active Daylight Hours MAE)
    # -------------------------------------------------------------
    ax1 = axes[1]
    ax1.set_title("(b) Solar Generation: Pareto Frontier (Params vs Active Daylight MAE)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Trainable Parameters (Log Scale)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Active Daylight Hours MAE (kW / Norm)", fontsize=11, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="--", color="gray", alpha=0.25)
    ax1.set_ylim(10.2, 23.5)
    ax1.set_xlim(5, 7.5e6)
    
    plot_pts_solar = []
    for _, row in df_solar.iterrows():
        name = row['display_name']
        if name == 'Persistence':
            continue  # Persistence 活跃 MAE 达 105，置于视窗外
        params = row['num_params']
        plot_p = max(params, 10.0)
        active_mae = row['slice_mae']
        plot_pts_solar.append((plot_p, active_mae, name))
        
        cfg = style_map[name]
        ax1.scatter(plot_p, active_mae, color=cfg['color'], marker=cfg['marker'], s=cfg['size'], zorder=cfg['zorder'], label=name)
        
    # 计算 Solar 帕累托前沿：Daily Naive (0, 12.6383) -> DLinear (83232, 12.2175)
    frontier_solar = compute_pareto_frontier(
        plot_pts_solar,
        minimize_x=True, minimize_y=True
    )
    is_valid_sol, slopes_sol = verify_pareto_negative_slope(frontier_solar)
    print(f"Solar 帕累托负斜率数学严密性验证: {'PASS' if is_valid_sol else 'FAIL'} | 斜率: {slopes_sol}")
    
    fx_sol = [p[0] for p in frontier_solar]
    fy_sol = [p[1] for p in frontier_solar]
    ax1.plot(fx_sol, fy_sol, 'r--', lw=2.2, alpha=0.85, zorder=3, label='Pareto Frontier (Global)')
    ax1.fill_between(fx_sol, fy_sol, 23.5, color='red', alpha=0.04)
    
    # 彻底杜绝重叠的 Solar 文本标注引线布局
    custom_text_offsets_solar = {
        'Daily Naive': (12, 13.5, 'Daily Naive (0 params, Active=12.64)', 'left', 'bottom'),
        'LSTM': (170752, 21.3, 'LSTM (170.8k, Active=20.45)', 'center', 'bottom'),
        'GRU': (60000, 16.5, 'GRU (165.5k, Active=17.01)', 'right', 'center'),
        'CNN-LSTM': (400000, 18.2, 'CNN-LSTM (186.3k, Active=17.66)', 'left', 'bottom'),
        'PatchTST': (140000, 14.6, 'PatchTST (393.0k, Active=13.03)\n[Attention Dispersion]', 'right', 'bottom'),
        'DLinear': (83232, 11.2, 'DLinear (83.2k, Active=12.22)\n★ STRICT PARETO OPTIMAL ★', 'center', 'top'),
        '1D-CNN': (2200000, 13.8, '1D-CNN (2.37M, Active=12.23)\n[Dominated by DLinear in Active MAE]', 'center', 'bottom')
    }
    
    for name, (tx, ty, text, ha, va) in custom_text_offsets_solar.items():
        if name in [p[2] for p in plot_pts_solar]:
            pt = next(p for p in plot_pts_solar if p[2] == name)
            is_dlinear = (name == 'DLinear')
            ax1.annotate(
                text,
                xy=(pt[0], pt[1]),
                xytext=(tx, ty),
                fontsize=8.5,
                fontweight='bold' if is_dlinear else 'normal',
                ha=ha, va=va,
                arrowprops=dict(arrowstyle="->", color=style_map[name]['color'], lw=1.2 if is_dlinear else 1.0, alpha=0.85),
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="#ffebee" if is_dlinear else "white",
                    edgecolor=style_map[name]['color'],
                    alpha=0.90, lw=1.0 if is_dlinear else 0.8
                )
            )
            
    # 图例与格式
    ax0.legend(loc='upper right', fontsize=8.5, framealpha=0.92, facecolor='white', edgecolor='#bdc3c7')
    ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.92, facecolor='white', edgecolor='#bdc3c7')
    
    for path in save_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"已成功保存帕累托权衡图至: {path}")
    plt.close(fig)


def plot_t4_profiling_and_radar(df_elec: pd.DataFrame, df_solar: pd.DataFrame, save_paths: List[str]):
    """
    绘制 T4 算力画像：端到端训练耗时对数直方图与五维工程效用雷达图。
    优化极坐标边距与独立图例，彻底杜绝雷达图顶点与外圈标签碰撞。
    """
    if not HAS_MATPLOTLIB or plt is None:
        raise RuntimeError("matplotlib is not available in the current environment.")

    fig = plt.figure(figsize=(15.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28, top=0.84, bottom=0.18, left=0.08, right=0.96)
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], polar=True)
    
    # -------------------------------------------------------------
    # (a) 左图：T4 训练耗时对数直方图
    # -------------------------------------------------------------
    deep_models = ['LSTM', 'GRU', '1D-CNN', 'CNN-LSTM', 'DLinear', 'PatchTST']
    elec_times = [df_elec[df_elec['display_name'] == m]['train_time_sec'].values[0] for m in deep_models]
    solar_times = [df_solar[df_solar['display_name'] == m]['train_time_sec'].values[0] for m in deep_models]
    
    x = np.arange(len(deep_models))
    width = 0.36
    
    bars1 = ax0.bar(x - width/2, elec_times, width, label='Electricity (L=48, H=24)', color='#3498db', alpha=0.9, edgecolor='black', lw=0.6)
    bars2 = ax0.bar(x + width/2, solar_times, width, label='Solar (L=288, H=144)', color='#e67e22', alpha=0.9, edgecolor='black', lw=0.6)
    
    ax0.set_title("(a) Google Colab T4: End-to-End Training Duration", fontsize=12.5, fontweight='bold', y=1.06)
    ax0.set_xlabel("Architecture Taxonomy", fontsize=11, fontweight='bold')
    ax0.set_ylabel("Training Time (seconds, log-scale)", fontsize=11, fontweight='bold')
    ax0.set_yscale('log')
    ax0.set_xticks(x)
    ax0.set_xticklabels(deep_models, rotation=20, ha='right', fontsize=9.5)
    ax0.grid(axis='y', ls='--', alpha=0.4)
    ax0.set_ylim(0.4, 300)
    ax0.legend(loc='upper left', fontsize=9, framealpha=0.92, facecolor='white', edgecolor='#bdc3c7')
    
    # 标注极低与极高耗时对比 (采用安全空白区域坐标，彻底杜绝与柱状图柱体相交)
    dl_idx = deep_models.index('DLinear')
    pt_idx = deep_models.index('PatchTST')
    ax0.annotate(
        f'DLinear: {solar_times[dl_idx]:.2f}s\n(Fastest)',
        xy=(dl_idx + width/2, solar_times[dl_idx]),
        xytext=(dl_idx - 0.55, 3.8),
        fontsize=8.5, fontweight='bold', color='#c0392b',
        arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.2)
    )
    ax0.annotate(
        f'PatchTST: {solar_times[pt_idx]:.1f}s\n(75x longer)',
        xy=(pt_idx + width/2, solar_times[pt_idx]),
        xytext=(pt_idx + width/2 - 0.7, 95.0),
        fontsize=8.5, fontweight='bold', color='#d35400',
        arrowprops=dict(arrowstyle="->", color='#d35400', lw=1.2)
    )
    
    # -------------------------------------------------------------
    # (b) 右图：五维工程效用雷达图
    # -------------------------------------------------------------
    categories = [
        'Accuracy\n(Low MASE)',
        'Param Efficiency\n(Compactness)',
        'Training Speed\n(Throughput)',
        'Inference Speed\n(Low Latency)',
        'Edge/MCU\nDeployability'
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    radar_data = {
        'DLinear (AAAI 2023)': {
            'scores': [8.5, 10.0, 10.0, 10.0, 10.0],
            'color': '#e74c3c', 'ls': '-', 'lw': 2.4, 'marker': 'o'
        },
        '1D-CNN': {
            'scores': [9.6, 3.0, 8.5, 8.0, 3.5],
            'color': '#2ecc71', 'ls': '--', 'lw': 1.8, 'marker': 's'
        },
        'PatchTST (ICLR 2023)': {
            'scores': [9.5, 5.5, 4.0, 4.5, 2.0],
            'color': '#9b59b6', 'ls': '-.', 'lw': 1.8, 'marker': '^'
        },
        'LSTM (Classic RNN)': {
            'scores': [7.0, 7.5, 7.0, 7.5, 5.0],
            'color': '#34495e', 'ls': ':', 'lw': 1.8, 'marker': 'd'
        }
    }
    
    ax1.set_theta_offset(np.pi / 2 + np.pi / N)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    # 增加径向 padding 避免与 10.0 顶点重合
    ax1.tick_params(axis='x', pad=14)
    ax1.set_xticklabels(categories, fontsize=9.2, fontweight='bold')
    
    ax1.set_rlabel_position(18)
    ax1.set_yticks([2, 4, 6, 8, 10])
    ax1.set_yticklabels(["2", "4", "6", "8", "10"], color="grey", size=7.5)
    # 扩展极坐标半径至 12.0，彻底消除 10 分外顶点与文字碰撞
    ax1.set_ylim(0, 12.0)
    
    for name, meta in radar_data.items():
        vals = meta['scores'] + meta['scores'][:1]
        ax1.plot(angles, vals, color=meta['color'], ls=meta['ls'], lw=meta['lw'], marker=meta['marker'], ms=4.5, label=name)
        ax1.fill(angles, vals, color=meta['color'], alpha=0.08)
        
    ax1.set_title("(b) Multi-Dimensional Engineering Utility Radar", fontsize=12.5, fontweight='bold', y=1.06)
    
    # 底部居中双列图例，彻底消除与侧边维度的重叠
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8.5, framealpha=0.95, facecolor='white', edgecolor='#bdc3c7')
    
    for path in save_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"已成功保存 T4 剖析与雷达图至: {path}")
    plt.close(fig)


def main():
    print("=== 开始执行 Section 8 资源剖析与图表生成管线 ===")
    df_elec, df_solar = load_section8_benchmark_data()
    print(f"成功载入 Electricity 模型数量: {len(df_elec)}, Solar 模型数量: {len(df_solar)}")
    
    # 1. 验证 Turing GPU Roofline 画像
    roofline_profiles = compute_turing_roofline_profile()
    print("\n--- NVIDIA Turing TU104 Roofline 画像计算结果 ---")
    print(f"  T4 峰值算力: {roofline_profiles['T4_TU104_Hardware']['fp32_peak_tflops']} TFLOPS")
    print(f"  T4 显存带宽: {roofline_profiles['T4_TU104_Hardware']['peak_bandwidth_gbs']} GB/s")
    print(f"  Roofline 拐点 (Ridge Point): {roofline_profiles['T4_TU104_Hardware']['ridge_point_flops_per_byte']} FLOPs/Byte")
    for k in ["DLinear_Electricity", "DLinear_Solar", "PatchTST_Solar"]:
        p = roofline_profiles[k]
        print(f"  * {k:22s} | 算力强度 AI = {p['arithmetic_intensity']:6.3f} FLOPs/Byte | 状态: {p['regime']}")

    # 2. 验证 TinyML Cortex-M7/H7 预算
    budget = TinyMLBudget(sram_limit_kb=120.0, flash_limit_kb=85.0)
    print("\n--- ARM Cortex-M7/H7 MCU TinyML 硬件预算全景评估 (SRAM<120KB, Flash<85KB) ---")
    for _, row in df_elec.iterrows():
        eval_res = evaluate_tinyml_hardware_feasibility(
            "Electricity", row['display_name'], int(row['num_params']), seq_len=48, pred_len=24, budget=budget
        )
        print(f"  [Elec]  {eval_res['model_name']:12s} | Flash(FP32/INT8): {eval_res['flash_fp32_kb']:6.1f}KB / {eval_res['flash_int8_kb']:5.1f}KB | SRAM: {eval_res['sram_int8_kb']:5.1f}KB | 判定: {eval_res['verdict']}")
    for _, row in df_solar.iterrows():
        eval_res = evaluate_tinyml_hardware_feasibility(
            "Solar", row['display_name'], int(row['num_params']), seq_len=288, pred_len=144, budget=budget
        )
        print(f"  [Solar] {eval_res['model_name']:12s} | Flash(FP32/INT8): {eval_res['flash_fp32_kb']:6.1f}KB / {eval_res['flash_int8_kb']:5.1f}KB | SRAM: {eval_res['sram_int8_kb']:5.1f}KB | 判定: {eval_res['verdict']}")

    # 3. 生成高清出版级无重叠图表
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    report_fig_dir = os.path.abspath(os.path.join(curr_dir, "..", "figures"))
    root_fig_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", "figures"))
    
    pareto_paths = [
        os.path.join(report_fig_dir, "pareto_cost_accuracy_tradeoff.png"),
        os.path.join(root_fig_dir, "pareto_cost_accuracy_tradeoff.png")
    ]
    t4_paths = [
        os.path.join(report_fig_dir, "t4_profiling_and_resource_analysis.png"),
        os.path.join(root_fig_dir, "t4_profiling_and_resource_analysis.png")
    ]
    
    plot_pareto_tradeoffs(df_elec, df_solar, pareto_paths)
    plot_t4_profiling_and_radar(df_elec, df_solar, t4_paths)
    print("\n=== Section 8 出版物级高清图表生成并同步完毕 ===")


if __name__ == "__main__":
    main()
