# -*- coding: utf-8 -*-
"""
多窗口网格模型参数量与闭式解析吻合度验证脚本。
输出 Markdown 格式对比表格。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests.test_modern_and_hybrid import dlinear_closed_form, cnnlstm_closed_form, patchtst_closed_form
from models.modern_models import DLinearModel, PatchTSTModel
from models.hybrid_models import CNNLSTMModel


def count_p(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_grid():
    windows = [16, 24, 32, 48, 64, 73, 96, 128, 168, 288, 336]
    C = 16

    print("### 1. Electricity 数据集网格 (H=24, C=16)")
    print("| 历史窗口 L | DLinear 共享通道 (Theory/Act) | DLinear 独立通道 (Theory/Act) | CNN-LSTM (Theory/Act) | PatchTST (Theory/Act) | Patch Token 数量 | 吻合状态 |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    H_elec = 24
    for L in windows:
        dl_s_act = count_p(DLinearModel(C, L, H_elec, individual=False))
        dl_s_th = dlinear_closed_form(L, H_elec, C, individual=False)

        dl_i_act = count_p(DLinearModel(C, L, H_elec, individual=True))
        dl_i_th = dlinear_closed_form(L, H_elec, C, individual=True)

        cnn_act = count_p(CNNLSTMModel(C, L, H_elec))
        cnn_th = cnnlstm_closed_form(L, H_elec, C)

        pt_m = PatchTSTModel(C, L, H_elec, patch_len=16, stride=8)
        pt_act = count_p(pt_m)
        pt_th = patchtst_closed_form(L, H_elec, C, patch_len=16, stride=8)

        match = (dl_s_act == dl_s_th and dl_i_act == dl_i_th and cnn_act == cnn_th and pt_act == pt_th)
        status = "100% MATCH" if match else "MISMATCH"

        print(f"| {L:3d} | {dl_s_th:6d} / {dl_s_act:6d} | {dl_i_th:7d} / {dl_i_act:7d} | {cnn_th:6d} / {cnn_act:6d} | {pt_th:6d} / {pt_act:6d} | {pt_m.num_patches:3d} tokens | {status} |")

    print("\n### 2. Solar 数据集网格 (H=144, C=16)")
    print("| 历史窗口 L | DLinear 共享通道 (Theory/Act) | DLinear 独立通道 (Theory/Act) | CNN-LSTM (Theory/Act) | PatchTST (Theory/Act) | Patch Token 数量 | 吻合状态 |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    H_solar = 144
    for L in [48, 96, 144, 288, 336]:
        dl_s_act = count_p(DLinearModel(C, L, H_solar, individual=False))
        dl_s_th = dlinear_closed_form(L, H_solar, C, individual=False)

        dl_i_act = count_p(DLinearModel(C, L, H_solar, individual=True))
        dl_i_th = dlinear_closed_form(L, H_solar, C, individual=True)

        cnn_act = count_p(CNNLSTMModel(C, L, H_solar))
        cnn_th = cnnlstm_closed_form(L, H_solar, C)

        pt_m = PatchTSTModel(C, L, H_solar, patch_len=16, stride=8)
        pt_act = count_p(pt_m)
        pt_th = patchtst_closed_form(L, H_solar, C, patch_len=16, stride=8)

        match = (dl_s_act == dl_s_th and dl_i_act == dl_i_th and cnn_act == cnn_th and pt_act == pt_th)
        status = "100% MATCH" if match else "MISMATCH"

        print(f"| {L:3d} | {dl_s_th:7d} / {dl_s_act:7d} | {dl_i_th:8d} / {dl_i_act:8d} | {cnn_th:6d} / {cnn_act:6d} | {pt_th:6d} / {pt_act:6d} | {pt_m.num_patches:3d} tokens | {status} |")


if __name__ == "__main__":
    run_grid()
