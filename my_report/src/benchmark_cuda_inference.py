# -*- coding: utf-8 -*-
"""
Section 8: 带 CUDA 硬件预热的深度学习与基准模型批推理延迟重新评测脚本
功能：
1. 严谨实现 CUDA 硬件时钟同步与预热机制 (50 次 Warm-up 消除 JIT 与 GPU 动态升频瞬态)；
2. 使用 torch.cuda.Event(enable_timing=True) 实现微秒级 GPU 硬件计时；
3. 支持单样本流式推理 (Batch Size = 1, 针对边缘 MCU/RTU 评估) 与批推理 (Batch Size = 32, 针对网关/服务器)；
4. 覆盖全部 8 类模型在 Electricity (L=48, H=24) 与 Solar (L=288, H=144) 上的实测开销；
5. 自动统计 Mean, Median, P95, Std 与吞吐量 (samples/sec)；
6. 结构化导出至 JSON 与 Markdown 报告。
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG
from train import create_model


def measure_inference_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 50,
    num_repeats: int = 100,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """
    测量单个模型在指定设备与输入下的端到端推理毫秒级延迟
    """
    model.eval()
    is_cuda = (device.type == "cuda")

    # 1. 硬件预热 (Warm-up)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
            if is_cuda:
                torch.cuda.synchronize()

    # 2. 正式基准测量 (Repeated Benchmarking)
    latencies_ms = []
    
    if is_cuda:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_repeats)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_repeats)]
        
        with torch.no_grad():
            for i in range(num_repeats):
                start_events[i].record()
                _ = model(input_tensor)
                end_events[i].record()
                
        torch.cuda.synchronize()
        for i in range(num_repeats):
            latencies_ms.append(start_events[i].elapsed_time(end_events[i]))
    else:
        with torch.no_grad():
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                _ = model(input_tensor)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)
    batch_size = input_tensor.shape[0]
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    std_lat = float(np.std(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    throughput = float((batch_size / (mean_lat / 1000.0))) if mean_lat > 0 else 0.0

    return {
        "mean_ms": round(mean_lat, 4),
        "median_ms": round(median_lat, 4),
        "p95_ms": round(p95_lat, 4),
        "std_ms": round(std_lat, 4),
        "min_ms": round(min_lat, 4),
        "max_ms": round(max_lat, 4),
        "throughput_samples_per_sec": round(throughput, 2)
    }


def run_full_latency_benchmark():
    """
    全量执行 Electricity 与 Solar 场景下的推理延迟重测
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"【推理延迟基准重测】当前激活硬件环境: {device}")
    if device.type == "cuda":
        print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 算力等级: {torch.cuda.get_device_capability(0)}")
    print(f"============================================================\n")

    models_to_test = [
        "Persistence",
        "DailySeasonalNaive",
        "LSTM",
        "GRU",
        "CNN1D",
        "CNNLSTM",
        "DLinear",
        "PatchTST"
    ]

    configs = {
        "Electricity": ELECTRICITY_CONFIG,
        "Solar": SOLAR_CONFIG
    }

    results = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {}
    }

    for ds_name, cfg in configs.items():
        results["benchmarks"][ds_name] = {}
        print(f"--- 正在测试数据集: {ds_name} (L={cfg.seq_len}, H={cfg.pred_len}, C={cfg.num_channels}) ---")

        for bsz in [1, 32]:
            bsz_key = f"batch_{bsz}"
            results["benchmarks"][ds_name][bsz_key] = {}
            print(f"  > Batch Size = {bsz} (预热 50 轮，正式评测 100 轮)")

            # 构造虚拟测试张量
            dummy_input = torch.randn(bsz, cfg.seq_len, cfg.num_channels, device=device)

            for m_name in models_to_test:
                try:
                    model = create_model(m_name, cfg)
                    model.to(device)
                    # 加载预训练最佳权重（若存在）
                    ckpt_name = f"{ds_name}_{m_name}_best.pt"
                    curr_dir = os.path.dirname(os.path.abspath(__file__))
                    ckpt_path = os.path.join(curr_dir, "..", "checkpoints", ckpt_name)
                    if os.path.exists(ckpt_path):
                        state_dict = torch.load(ckpt_path, map_location=device)
                        model.load_state_dict(state_dict)

                    metrics = measure_inference_latency(
                        model=model,
                        input_tensor=dummy_input,
                        num_warmup=50,
                        num_repeats=100,
                        device=device
                    )
                    results["benchmarks"][ds_name][bsz_key][m_name] = metrics
                    print(f"    - {m_name:18s} | 均值: {metrics['mean_ms']:7.3f} ms | 中位数: {metrics['median_ms']:7.3f} ms | P95: {metrics['p95_ms']:7.3f} ms | 吞吐: {metrics['throughput_samples_per_sec']:8.1f} smp/s")
                except Exception as e:
                    print(f"    - {m_name:18s} | 测试异常: {e}")
                    results["benchmarks"][ds_name][bsz_key][m_name] = {"error": str(e)}

    # 同时在 CPU 上执行基准重测，以获得嵌入式 MCU/Edge CPU 的直接对标基准
    cpu_device = torch.device("cpu")
    print(f"\n============================================================")
    print(f"【CPU 推理延迟基准测试】针对边缘 MCU/Gateway CPU 架构评估")
    print(f"============================================================\n")
    results["cpu_benchmarks"] = {}

    for ds_name, cfg in configs.items():
        results["cpu_benchmarks"][ds_name] = {}
        print(f"--- CPU 正在测试数据集: {ds_name} ---")
        for bsz in [1, 32]:
            bsz_key = f"batch_{bsz}"
            results["cpu_benchmarks"][ds_name][bsz_key] = {}
            dummy_input_cpu = torch.randn(bsz, cfg.seq_len, cfg.num_channels, device=cpu_device)

            for m_name in models_to_test:
                try:
                    model_cpu = create_model(m_name, cfg)
                    model_cpu.to(cpu_device)
                    metrics_cpu = measure_inference_latency(
                        model=model_cpu,
                        input_tensor=dummy_input_cpu,
                        num_warmup=20,
                        num_repeats=50,
                        device=cpu_device
                    )
                    results["cpu_benchmarks"][ds_name][bsz_key][m_name] = metrics_cpu
                    print(f"    - [CPU] {m_name:18s} | B={bsz:2d} | 均值: {metrics_cpu['mean_ms']:7.3f} ms | 中位数: {metrics_cpu['median_ms']:7.3f} ms")
                except Exception as e:
                    results["cpu_benchmarks"][ds_name][bsz_key][m_name] = {"error": str(e)}

    # 保存 JSON 输出
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(curr_dir, "..", "benchmark_latency_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n基准重测完整指标已持久化至: {out_json}")

    return results


if __name__ == "__main__":
    run_full_latency_benchmark()
