# -*- coding: utf-8 -*-
"""
Section 8: 自动化测试套件
测试目标：
1. 帕累托非支配前沿算法的数学严谨性（单调性、严格非支配性、严格负斜率数学证明、全套边界异常处理）；
2. 工业微控制器 (ARM Cortex-M7/H7) 片上 SRAM (<120KB) 与 Flash (<85KB) 硬件预算全景可行性核验；
3. NVIDIA Turing TU104 (Tesla T4) GPU 算力 Roofline 画像与算力强度 (AI) 瓶颈诊断；
4. 带 CUDA 硬件预热的批推理毫秒级延迟重新测试结果有效性检验；
5. INT8 PTQ 对称量化数学映射与误差边界；
6. 数据库中 Section 8 官方基准模型记录一致性。
"""

import unittest
import numpy as np
import sqlite3
import os
import sys
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# 导入待测分析模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if HAS_PANDAS:
    from section8_profiling import (
        compute_pareto_frontier,
        verify_pareto_negative_slope,
        load_section8_benchmark_data,
        get_db_path,
        TinyMLBudget,
        evaluate_tinyml_hardware_feasibility,
        compute_turing_roofline_profile
    )


@unittest.skipUnless(HAS_PANDAS, "pandas 未安装，跳过 Section 8 分析测试")
class TestSection8Analytics(unittest.TestCase):

    def test_pareto_frontier_math_and_negative_slope(self):
        """测试帕累托前沿算法的数学正确性、边界处理与严格负斜率特性"""
        # 测试用例 1: 边界情况（空列表与单点）
        self.assertEqual(compute_pareto_frontier([]), [])
        single_pt = [(10.0, 0.5, "Single")]
        self.assertEqual(compute_pareto_frontier(single_pt), single_pt)
        is_valid, _ = verify_pareto_negative_slope(single_pt)
        self.assertTrue(is_valid)

        # 测试用例 2: 重复点与被支配点剔除
        points = [
            (10.0, 1.0, "ModelA"),      # 帕累托点
            (10.0, 1.0, "ModelA_dup"),  # 重复点
            (100.0, 0.8, "ModelB"),     # 帕累托点
            (500.0, 0.9, "ModelC"),     # 被 ModelB 支配 (参数更多且误差更大)
            (1000.0, 0.5, "ModelD"),    # 帕累托点
            (2000.0, 0.6, "ModelE"),    # 被 ModelD 支配
        ]
        frontier = compute_pareto_frontier(points, minimize_x=True, minimize_y=True)
        frontier_names = [p[2] for p in frontier]
        self.assertEqual(frontier_names, ["ModelA", "ModelB", "ModelD"])

        # 核心数学检验：负斜率数学严密性 (Strictly Negative Slope: Δx > 0 且 Δy < 0)
        is_valid, slopes = verify_pareto_negative_slope(frontier)
        self.assertTrue(is_valid, "帕累托前沿相邻线段必须全部具有严格负斜率！")
        for s in slopes:
            self.assertLess(s, 0.0, "帕累托前沿任何两点间严禁出现零斜率或正斜率！")

    def test_official_pareto_frontiers_negative_slopes(self):
        """测试官方实测数据在 Electricity 与 Solar 下帕累托前沿的数学严密性"""
        df_elec, df_solar = load_section8_benchmark_data()

        # 1. 负荷预测 (Electricity)
        pts_elec = [(max(r['num_params'], 10.0), r['mase_24h'], r['display_name'])
                    for _, r in df_elec.iterrows() if r['num_params'] >= 100]
        front_elec = compute_pareto_frontier(pts_elec, minimize_x=True, minimize_y=True)
        is_valid_elec, slopes_elec = verify_pareto_negative_slope(front_elec)
        self.assertTrue(is_valid_elec, "Electricity 帕累托前沿必须具备严格负斜率")
        self.assertGreaterEqual(len(front_elec), 2)
        # 验证 DLinear 在负荷预测中占据极低参数端
        self.assertEqual(front_elec[0][2], "DLinear")
        for s in slopes_elec:
            self.assertLess(s, 0.0, f"负荷场景出现非法非负斜率: {s}")

        # 2. 光伏出力 (Solar)
        pts_solar = [(max(r['num_params'], 10.0), r['slice_mae'], r['display_name'])
                     for _, r in df_solar.iterrows() if r['display_name'] != 'Persistence']
        front_solar = compute_pareto_frontier(pts_solar, minimize_x=True, minimize_y=True)
        is_valid_solar, slopes_solar = verify_pareto_negative_slope(front_solar)
        self.assertTrue(is_valid_solar, "Solar 帕累托前沿必须具备严格负斜率")
        self.assertEqual(len(front_solar), 2, "在 Active Daylight MAE 维度，帕累托前沿严格仅由 Daily Naive 与 DLinear 构成")
        self.assertEqual(front_solar[0][2], "Daily Naive")
        self.assertEqual(front_solar[1][2], "DLinear")
        self.assertLess(slopes_solar[0], 0.0, "Solar 前沿从 Daily Naive 到 DLinear 必须为负斜率")

    def test_solar_active_mae_pareto_dominance(self):
        """验证光伏活跃出力场景下 DLinear 对 1D-CNN 和 PatchTST 的严格帕累托支配性"""
        df_elec, df_solar = load_section8_benchmark_data()

        dlinear_solar = df_solar[df_solar['display_name'] == 'DLinear'].iloc[0]
        cnn_solar = df_solar[df_solar['display_name'] == '1D-CNN'].iloc[0]
        patchtst_solar = df_solar[df_solar['display_name'] == 'PatchTST'].iloc[0]

        # 1. 验证参数量对比
        self.assertLess(dlinear_solar['num_params'], cnn_solar['num_params'])
        self.assertLess(dlinear_solar['num_params'], patchtst_solar['num_params'])

        # 2. 验证 Active Daylight MAE 对比（DLinear 必须严格优于 1D-CNN 与 PatchTST）
        self.assertLess(dlinear_solar['slice_mae'], cnn_solar['slice_mae'], "DLinear 日光活跃 MAE 必须优于 1D-CNN")
        self.assertLess(dlinear_solar['slice_mae'], patchtst_solar['slice_mae'], "DLinear 日光活跃 MAE 必须优于 PatchTST")

        # 3. 验证帕累托求解结果：1D-CNN 与 PatchTST 绝对不能出现在 Active MAE 的帕累托前沿中
        pts = [(r['num_params'], r['slice_mae'], r['display_name']) for _, r in df_solar.iterrows() if r['num_params'] > 0]
        front = compute_pareto_frontier(pts, minimize_x=True, minimize_y=True)
        front_names = [p[2] for p in front]

        self.assertIn("DLinear", front_names)
        self.assertNotIn("1D-CNN", front_names, "1D-CNN 在 Active MAE 上被 DLinear 支配，严禁入选前沿！")
        self.assertNotIn("PatchTST", front_names, "PatchTST 在 Active MAE 上被 DLinear 支配，严禁入选前沿！")

    def test_cortex_m7_h7_mcu_hardware_budget(self):
        """核验 ARM Cortex-M7/H7 MCU 片上 SRAM (<120KB) 与 Flash (<85KB) 预算评估"""
        budget = TinyMLBudget(sram_limit_kb=120.0, flash_limit_kb=85.0)

        # 1. DLinear Electricity: FP32 与 INT8 均小于 85KB Flash 与 120KB SRAM
        eval_dl_elec = evaluate_tinyml_hardware_feasibility("Electricity", "DLinear", 2352, 48, 24, budget=budget)
        self.assertLess(eval_dl_elec["flash_fp32_kb"], 85.0, "DLinear Elec FP32 必须小于 85 KB Flash")
        self.assertLess(eval_dl_elec["sram_int8_kb"], 120.0, "DLinear Elec SRAM 必须小于 120 KB")
        self.assertEqual(eval_dl_elec["verdict"], "VIABLE_FP32")

        # 2. DLinear Solar: FP32 超过 85KB (325.1KB)，但 INT8 PTQ 严格满足 < 85KB (81.8KB)，且 SRAM < 120KB
        eval_dl_sol = evaluate_tinyml_hardware_feasibility("Solar", "DLinear", 83232, 288, 144, budget=budget)
        self.assertGreater(eval_dl_sol["flash_fp32_kb"], 85.0, "DLinear Solar FP32 超过 85 KB Flash 预算")
        self.assertLess(eval_dl_sol["flash_int8_kb"], 85.0, "DLinear Solar INT8 必须严格在 85 KB Flash 预算内！")
        self.assertLess(eval_dl_sol["sram_int8_kb"], 120.0, "DLinear Solar SRAM 必须小于 120 KB！")
        self.assertEqual(eval_dl_sol["verdict"], "VIABLE_INT8_PTQ")

        # 3. 1D-CNN 权重超过 Flash 预算
        eval_cnn_elec = evaluate_tinyml_hardware_feasibility("Electricity", "1D-CNN", 401568, 48, 24, budget=budget)
        self.assertGreater(eval_cnn_elec["flash_int8_kb"], 85.0, "1D-CNN Elec INT8 权重依然超出 85 KB Flash")
        self.assertIn("UNFEASIBLE", eval_cnn_elec["verdict"])

        eval_cnn_sol = evaluate_tinyml_hardware_feasibility("Solar", "1D-CNN", 2369568, 288, 144, budget=budget)
        self.assertGreater(eval_cnn_sol["flash_int8_kb"], 85.0, "1D-CNN Solar INT8 权重严重超出 85 KB Flash")
        self.assertIn("UNFEASIBLE", eval_cnn_sol["verdict"])

        # 4. PatchTST 激活显存超出 SRAM 预算
        eval_pt_sol = evaluate_tinyml_hardware_feasibility("Solar", "PatchTST", 392976, 288, 144, budget=budget)
        self.assertGreater(eval_pt_sol["flash_int8_kb"], 85.0)
        self.assertGreater(eval_pt_sol["sram_int8_kb"], 120.0, "PatchTST Solar SRAM 严重超出 120 KB 预算")
        self.assertEqual(eval_pt_sol["verdict"], "UNFEASIBLE_FLASH_AND_SRAM_EXCEEDED")

    def test_turing_gpu_roofline_profiling(self):
        """测试 NVIDIA Turing TU104 (Tesla T4) 算力 Roofline 画像与瓶颈诊断"""
        profiles = compute_turing_roofline_profile()
        hw = profiles["T4_TU104_Hardware"]
        self.assertEqual(hw["fp32_peak_tflops"], 8.1)
        self.assertEqual(hw["peak_bandwidth_gbs"], 320.0)
        self.assertAlmostEqual(hw["ridge_point_flops_per_byte"], 25.3125, places=3)

        # 验证 DLinear 在负荷场景属于深度访存受限 (Strictly Memory-Bound)
        p_dl_elec = profiles["DLinear_Electricity"]
        self.assertLess(p_dl_elec["arithmetic_intensity"], hw["ridge_point_flops_per_byte"])
        self.assertEqual(p_dl_elec["regime"], "Strictly Memory-Bound")

        # 验证 PatchTST 算力强度高，处于算力受限区域
        p_pt_sol = profiles["PatchTST_Solar"]
        self.assertGreater(p_pt_sol["arithmetic_intensity"], hw["ridge_point_flops_per_byte"])
        self.assertEqual(p_pt_sol["regime"], "Compute-Bound / High-Intensity")

    def test_cuda_warmup_latency_benchmarks_consistency(self):
        """验证带 CUDA 硬件预热的批推理毫秒级延迟重测数据文件存在且合理"""
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.abspath(os.path.join(curr_dir, "..", "..", "benchmark_latency_results.json"))
        self.assertTrue(os.path.exists(json_path), "基准延迟重测文件 benchmark_latency_results.json 必须存在")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("benchmarks", data)
        self.assertIn("Electricity", data["benchmarks"])
        self.assertIn("Solar", data["benchmarks"])

        # 验证 DLinear 延迟在微秒/毫秒量级，且大幅优于 PatchTST
        elec_b1 = data["benchmarks"]["Electricity"]["batch_1"]
        self.assertIn("DLinear", elec_b1)
        self.assertIn("PatchTST", elec_b1)
        self.assertLess(elec_b1["DLinear"]["mean_ms"], elec_b1["PatchTST"]["mean_ms"])

        solar_b32 = data["benchmarks"]["Solar"]["batch_32"]
        self.assertLess(solar_b32["DLinear"]["mean_ms"], solar_b32["PatchTST"]["mean_ms"])

    def test_database_table_consistency(self):
        """测试 experiments.db 中官方基准模型记录完整性与指标有效性"""
        db_path = get_db_path()
        self.assertTrue(os.path.exists(db_path), "experiments.db 必须存在")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT count(*) FROM experiments;")
        total_records = cur.fetchone()[0]
        self.assertGreaterEqual(total_records, 40, "数据库中应至少有 40 条完整实验记录")

        canonical_ids = (32, 33, 35, 38, 59, 60, 61, 62, 40, 41, 42, 43, 44, 45, 46, 47)
        cur.execute(f"SELECT count(*) FROM experiments WHERE id IN {canonical_ids};")
        matched_canonical = cur.fetchone()[0]
        self.assertEqual(matched_canonical, 16, "官方 16 组核心基准实验记录必须全部完整存在")
        conn.close()

    def test_pareto_frontier_exhaustive_edge_cases(self):
        """测试帕累托前沿算法在极端边界、负坐标与多重支配下的完备性"""
        # 1. 空集与单点
        self.assertEqual(compute_pareto_frontier([]), [])
        pt1 = [(5.0, 2.0, "P1")]
        self.assertEqual(compute_pareto_frontier(pt1), pt1)

        # 2. 负值目标坐标（如最小化负向收益或增量指标）
        neg_pts = [
            (-10.0, -1.0, "NegA"),  # 优选 x 极小
            (-5.0, -4.0, "NegB"),   # 优选 y 极小
            (-2.0, -2.0, "NegC"),   # 被 NegB 支配 (-5 < -2 且 -4 < -2)
        ]
        neg_front = compute_pareto_frontier(neg_pts, minimize_x=True, minimize_y=True)
        self.assertEqual([p[2] for p in neg_front], ["NegA", "NegB"])
        is_neg_valid, neg_slopes = verify_pareto_negative_slope(neg_front)
        self.assertTrue(is_neg_valid)
        self.assertLess(neg_slopes[0], 0.0)

        # 3. 密集重复与同值点
        dense_pts = [(10.0, 5.0, f"D_{i}") for i in range(10)]
        self.assertEqual(len(compute_pareto_frontier(dense_pts)), 1)

    def test_pareto_illegal_slopes_and_edge_rejection(self):
        """测试斜率核验算法对非法非负斜率、零斜率与垂直退化线段的严格拒绝"""
        # 1. 严格水平段（零斜率 dy=0）：必须被核验器判定为非法
        flat_segment = [(10.0, 5.0, "A"), (20.0, 5.0, "B")]
        is_valid_flat, slopes_flat = verify_pareto_negative_slope(flat_segment)
        self.assertFalse(is_valid_flat, "零斜率段必须被判为非法！")
        self.assertEqual(slopes_flat[0], 0.0)

        # 2. 正斜率段（dy > 0，即更大参数对应更高误差）：必须被拒绝
        positive_segment = [(10.0, 5.0, "A"), (20.0, 6.0, "B")]
        is_valid_pos, slopes_pos = verify_pareto_negative_slope(positive_segment)
        self.assertFalse(is_valid_pos, "正斜率段必须被判为非法！")
        self.assertGreater(slopes_pos[0], 0.0)

        # 3. 垂直段（dx = 0）：必须被拒绝
        vertical_segment = [(10.0, 5.0, "A"), (10.0, 3.0, "B")]
        is_valid_vert, _ = verify_pareto_negative_slope(vertical_segment)
        self.assertFalse(is_valid_vert, "垂直段 (dx=0) 必须被判为非法！")

        # 4. 逆序段（dx < 0）：必须被拒绝
        reverse_segment = [(20.0, 3.0, "A"), (10.0, 5.0, "B")]
        is_valid_rev, _ = verify_pareto_negative_slope(reverse_segment)
        self.assertFalse(is_valid_rev, "逆序未排序段必须被判为非法！")

    def test_electricity_global_frontier_monotonicity(self):
        """测试包含 Daily Naive 时的 Electricity 全局帕累托前沿的严格单调负斜率"""
        df_elec, _ = load_section8_benchmark_data()
        # 包含全部 8 类模型（含 Daily Naive）
        pts_all = [(max(r['num_params'], 10.0), r['mase_24h'], r['display_name'])
                   for _, r in df_elec.iterrows() if r['display_name'] != 'Persistence']
        global_frontier = compute_pareto_frontier(pts_all, minimize_x=True, minimize_y=True)
        names = [p[2] for p in global_frontier]
        expected_order = ["Daily Naive", "DLinear", "GRU", "LSTM", "CNN-LSTM", "PatchTST", "1D-CNN"]
        self.assertEqual(names, expected_order, "全局帕累托前沿应由 Daily Naive 起步并包含全部 6 个深度模型")

        is_valid, slopes = verify_pareto_negative_slope(global_frontier)
        self.assertTrue(is_valid, "全局前沿所有 6 个线段必须具备严格负斜率")
        self.assertEqual(len(slopes), 6)
        for s in slopes:
            self.assertLess(s, 0.0, f"存在非负斜率: {s}")

    def test_turing_roofline_batch_size_scaling(self):
        """测试不同 Batch Size 下 Turing TU104 算力强度与受限区域演变"""
        hw_ridge = 25.3125

        # 1. 边缘实时流式推断 B=1
        prof_b1 = compute_turing_roofline_profile(batch_size=1)
        self.assertEqual(prof_b1["DLinear_Electricity"]["regime"], "Strictly Memory-Bound")
        self.assertAlmostEqual(prof_b1["DLinear_Electricity"]["arithmetic_intensity"], 3.771, places=2)

        # 2. 标准生产批推理 B=32
        prof_b32 = compute_turing_roofline_profile(batch_size=32)
        self.assertAlmostEqual(prof_b32["DLinear_Electricity"]["arithmetic_intensity"], 6.883, places=2)
        self.assertAlmostEqual(prof_b32["PatchTST_Solar"]["arithmetic_intensity"], 65.924, places=2)
        self.assertEqual(prof_b32["PatchTST_Solar"]["regime"], "Compute-Bound / High-Intensity")

        # 3. 极端高并发批推理 B=256
        prof_b256 = compute_turing_roofline_profile(batch_size=256)
        # 序列扩展下 Solar DLinear 保持处于高效计算区域
        self.assertGreater(prof_b256["DLinear_Solar"]["arithmetic_intensity"], hw_ridge)

    def test_tinyml_budget_boundary_and_zero_params(self):
        """测试 TinyML 预算在零参数基线与临界极值边界下的评估鲁棒性"""
        budget = TinyMLBudget(sram_limit_kb=120.0, flash_limit_kb=85.0)

        # 1. 零参数基线
        eval_persist = evaluate_tinyml_hardware_feasibility("Electricity", "Persistence", 0, 48, 24, budget=budget)
        self.assertEqual(eval_persist["flash_fp32_kb"], 0.0)
        self.assertEqual(eval_persist["flash_int8_kb"], 0.0)
        self.assertEqual(eval_persist["verdict"], "VIABLE_FP32")

        # 2. 临界上界测试 (刚好 85.0 KB Flash)
        # 85 KB * 1024 / 4 = 21760 params
        eval_bound = evaluate_tinyml_hardware_feasibility("Custom", "DLinear", 21760, 48, 24, budget=budget)
        self.assertEqual(eval_bound["flash_fp32_kb"], 85.0)
        self.assertEqual(eval_bound["verdict"], "VIABLE_FP32")

        # 3. 微量超标但在 INT8 下达标
        # 85.1 KB FP32 -> 超出 85.0 KB，但 INT8 为 21.3 KB
        eval_over = evaluate_tinyml_hardware_feasibility("Custom", "DLinear", 21785, 48, 24, budget=budget)
        self.assertGreater(eval_over["flash_fp32_kb"], 85.0)
        self.assertLess(eval_over["flash_int8_kb"], 85.0)
        self.assertEqual(eval_over["verdict"], "VIABLE_INT8_PTQ")


if __name__ == "__main__":
    unittest.main()
