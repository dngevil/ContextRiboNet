#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rna_ribo_stage_r2.py

功能：
- 读取 RNA 与 Ribo 表（带列名 gene 和 *_RNA / *_Ribo）
- 按基因对齐，逐阶段计算 RNA 与 Ribo 的 R²
- 画散点图：横轴 = 发育阶段，纵轴 = R²
- 输出 R² 的 CSV 与 PNG

依赖：pandas, numpy, scikit-learn, matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

DEFAULT_STAGE_ORDER = ["GV","MI","MII","1C","2C","4C","8C","ICM","hESC"]

def infer_stages(columns, suffix):
    pat = re.compile(rf"(.+?)_{re.escape(suffix)}$", re.IGNORECASE)
    stages = []
    for c in columns:
        m = pat.match(str(c))
        if m:
            stages.append(m.group(1))
    return stages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_path", required=True,
                    help="RNA 表达矩阵（含 gene 与 *_RNA 列）")
    ap.add_argument("--ribo_path", required=True,
                    help="Ribo 表达矩阵（含 gene 与 *_Ribo 列）")
    ap.add_argument("--outdir", required=True,
                    help="输出目录（会创建）")
    ap.add_argument("--log1p", action="store_true",
                    help="对 RNA/Ribo 同时取 log1p 再计算 R²")
    ap.add_argument("--dedup", action="store_true",
                    help="按 gene 去重（均值聚合）")
    ap.add_argument("--stage_order", nargs="*", default=None,
                    help="自定义阶段顺序（默认 GV MI MII 1C 2C 4C 8C ICM hESC）")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读表
    rna = pd.read_csv(args.rna_path, sep=None, engine="python")
    ribo = pd.read_csv(args.ribo_path, sep=None, engine="python")

    # 列名清理
    rna.columns  = [str(c).strip().lstrip("\ufeff") for c in rna.columns]
    ribo.columns = [str(c).strip().lstrip("\ufeff") for c in ribo.columns]
    assert "gene" in rna.columns and "gene" in ribo.columns, "两张表都需要 gene 列"

    # 可选去重
    if args.dedup:
        rna = rna.groupby("gene", as_index=False).mean(numeric_only=True)
        ribo = ribo.groupby("gene", as_index=False).mean(numeric_only=True)

    # 提取阶段
    rna_stages  = set(infer_stages(rna.columns, "RNA"))
    ribo_stages = set(infer_stages(ribo.columns, "Ribo"))
    stages = sorted(list(rna_stages & ribo_stages),
                    key=(args.stage_order if args.stage_order else DEFAULT_STAGE_ORDER).index
                    if (args.stage_order or set(DEFAULT_STAGE_ORDER).issuperset(rna_stages|ribo_stages))
                    else None)
    if len(stages) == 0:
        raise SystemExit("RNA 与 Ribo 无共同阶段（*_RNA / *_Ribo 列名不匹配）。")

    # 基因对齐（内连接）
    rna = rna[["gene"] + [f"{s}_RNA" for s in stages]]
    ribo = ribo[["gene"] + [f"{s}_Ribo" for s in stages]]
    df = pd.merge(rna, ribo, on="gene", how="inner")
    if df.empty:
        raise SystemExit("两表按 gene 对齐后为空，检查基因名是否一致。")

    # 取值矩阵
    RNA = df[[f"{s}_RNA"  for s in stages]].to_numpy(dtype=float)
    RIB = df[[f"{s}_Ribo" for s in stages]].to_numpy(dtype=float)

    # 可选 log1p
    if args.log1p:
        RNA = np.log1p(RNA)
        RIB = np.log1p(RIB)

    # 逐阶段 R²
    r2_list = []
    for i, s in enumerate(stages):
        y_true = RIB[:, i]
        y_pred = RNA[:, i]
        # 需要两者有方差，才能定义R²
        if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
            r2 = np.nan
        else:
            r2 = r2_score(y_true, y_pred)
        r2_list.append(r2)

    # 保存 CSV
    r2_csv = outdir / "rna_ribo_stage_r2.csv"
    pd.DataFrame({"stage": stages, "r2": r2_list}).to_csv(r2_csv, index=False)

    # 画散点图（阶段在 x 轴，R² 在 y 轴）
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(stages))
    ax.scatter(x, r2_list, s=40)
    for xi, r2, s in zip(x, r2_list, stages):
        ax.annotate(f"{r2:.3f}" if r2==r2 else "nan", (xi, r2 if r2==r2 else 0),
                    textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(stages, rotation=0)
    ax.set_ylabel("R² (RNA vs Ribo)")
    ax.set_xlabel("Developmental stage")
    ax.set_title(f"Stage-wise R²  (N genes = {len(df)})" + ("  [log1p]" if args.log1p else ""))
    ax.set_ylim(-1.0, 1.0)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    out_png = outdir / "rna_ribo_stage_r2.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[OK] R² 已保存: {r2_csv}")
    print(f"[OK] 图已保存: {out_png}")

if __name__ == "__main__":
    main()
