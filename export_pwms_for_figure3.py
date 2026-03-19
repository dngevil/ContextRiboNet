#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_pwms_for_figure3.py

作用：
- 使用与你训练完全一致的 CNN_MLP_Fusion 结构，
  从 best.pt 载入模型权重，
  为指定的 kernels 导出 PWM 到 pwm_selected 目录。

依赖：
- train_single_label_cnn_mlp.py 里定义的 load_and_expand, CNN_MLP_Fusion
- extract_kernels_and_scores_add_cellstate.py 里定义的 save_selected_kernel_pwms

用法示例：
python export_pwms_for_figure3.py \
  --npz ../preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --ckpt ../result/cnn_mlp_add_cellstate/single_label_v1/best.pt \
  --outdir ../result/cnn_mlp_add_cellstate/single_label_v1/kernels/pwm_selected \
  --kernels kernel_k20_f8 kernel_k12_f17 kernel_k12_f29 kernel_k16_f11
"""

import argparse
from pathlib import Path

import numpy as np
import torch

# 直接复用你训练脚本里的实现
from train_cnn_mlp_add_cellstate import load_and_expand, CNN_MLP_Fusion  # noqa: E402
from extract_kernels_and_scores_add_cellstate import (                 # noqa: E402
    save_selected_kernel_pwms,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="与训练时相同的 model_inputs .npz")
    ap.add_argument("--ckpt", required=True,
                    help="训练好的 best.pt 路径")
    ap.add_argument("--outdir", required=True,
                    help="PWM 输出目录 (例如 kernels/pwm_selected)")
    ap.add_argument("--kernels", nargs="+", required=True,
                    help="要导出 PWM 的 kernel 名列表，如 kernel_k20_f8 kernel_k12_f17 ...")

    # 这些超参要和训练时一致；如果你训练时改过，就在这里改成对应值
    ap.add_argument("--kernel-sizes", type=int, nargs="+",
                    default=[6, 10, 12, 16, 20],
                    help="卷积核大小列表 (需与训练时一致)")
    ap.add_argument("--cnn-channels", type=int, default=64,
                    help="每种 kernel size 的通道数 (需与训练时一致)")
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--mlp-blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fusion", choices=["concat", "film", "gate"],
                    default="film",
                    help="训练时使用的 fusion 方式，默认是 film")
    ap.add_argument("--cond-hidden", type=int, default=128)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读取展开后的输入，维度和训练时完全一致
    pack = load_and_expand(args.npz)
    X_seq = pack["X_seq"]           # [M, L, 4]
    X_other = pack["X_other"]       # [M, other_in_dim]
    l_base = pack["l_base"]

    other_in_dim = X_other.shape[1]

    # 2) 构建模型结构（与训练脚本 main() 保持一致）
    model = CNN_MLP_Fusion(
        other_in_dim=other_in_dim,
        l_base=l_base,
        cnn_channels=args.cnn_channels,
        mlp_hidden=args.mlp_hidden,
        mlp_blocks=args.mlp_blocks,
        dropout=args.dropout,
        kernel_sizes=tuple(args.kernel_sizes),
        fusion=args.fusion,          # 默认 film，如训练时用 gate 就改成 gate
        cond_hidden=args.cond_hidden
    )
    model.to(device)

    # 3) 载入 ckpt 权重 —— 注意这里是 "model_state"，不是 "model"
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        # 兜底：如果你以后自己存成纯 state_dict，也兼容
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] missing keys in state_dict:", missing)
    if unexpected:
        print("[WARN] unexpected keys in state_dict:", unexpected)

    model.eval()

    # 4) 为所有样本准备索引（用于 topk 激活窗口）
    train_indices = np.arange(X_seq.shape[0], dtype=int)

    # 5) 调用你现有的 motif 提取函数，导出 PWM
    save_selected_kernel_pwms(
        model=model,
        device=device,
        X_seq=X_seq,
        train_indices=train_indices,
        top_kernel_names=args.kernels,
        outdir=outdir,
        topk_per_kernel=500,
        min_hits=100,
    )

    print("[PWM] 已为以下 kernels 导出 PWM：")
    for k in args.kernels:
        print("   ", k, "->", outdir / f"{k}.pwm.tsv")


if __name__ == "__main__":
    main()
