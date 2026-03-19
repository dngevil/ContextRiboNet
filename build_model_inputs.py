#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_model_inputs.py  (RNA-as-features, max_seq_len filter, dedup options)

从 seq_features.csv + Ribo 矩阵 (+ 可选 RNA 矩阵) 构建模型输入：
- 仅保留 status=="ok" 的基因
- --max_seq_len：按所选序列模式长度过滤（仅内存中过滤，不改原文件）
- --dedup_ribo / --dedup_rna：按 gene 均值去重
- 序列 one-hot: (N × L × 4)，A/C/G/T，其他字符为0；对称补零到批内最大 L
- 其他数值特征: (N × l) = [CDS/UTR 长度、GC、Kozak、uORF] + [RNA 各阶段（可选）]
- 标签 Y: Ribo 各阶段 (N × S)，可选 --log1p
- 输出: .npz (X_seq, X_other, Y) + .json (meta)

示例：
python build_model_inputs.py \
  --seq_features ../preprocess_data/pipeline_out/seq_features.csv \
  --ribo_path   ../data/GSE197265_h_GVtohESC_28500_Ribo_merge_average_fpkm.txt \
  --rna_path    ../data/GSE197265_h_GVtohESC_28500_RNA_merge_average_fpkm.txt \
  --out_npz     ../preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --seq_mode cds \
  --max_seq_len 5000 \
  --dedup_ribo --dedup_rna \
  --log1p --log1p_rna
"""

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


SEQ_FEATURE_COLS = [
    "cds_len","utr5_len","utr3_len",
    "gc_cds","gc_utr5","gc_utr3",
    "kozak_score","uorf_atg_count","uorf_stop_in_frame_count"
]

def infer_stages(columns: List[str], suffix="Ribo"):
    stages = []
    pat = re.compile(rf"(.+?)_{re.escape(suffix)}$", re.IGNORECASE)
    for c in columns:
        m = pat.match(str(c))
        if m:
            stages.append(m.group(1))
    return stages

def get_sequence_by_mode(row: pd.Series, mode: str) -> str:
    if mode == "cds":
        return (row.get("cds_seq") or "")
    elif mode == "utr5":
        return (row.get("utr5_seq") or "")
    elif mode == "utr3":
        return (row.get("utr3_seq") or "")
    elif mode == "concat":
        # 5'UTR + CDS + 3'UTR
        return f"{row.get('utr5_seq') or ''}{row.get('cds_seq') or ''}{row.get('utr3_seq') or ''}"
    else:
        raise ValueError(f"Unsupported seq_mode: {mode}")

def seq_length_by_mode(row: pd.Series, mode: str) -> float:
    # 使用预先计算的长度列；concat 为三者之和（NaN 视作 0）
    if mode == "cds":
        return row.get("cds_len", np.nan)
    elif mode == "utr5":
        return row.get("utr5_len", np.nan)
    elif mode == "utr3":
        return row.get("utr3_len", np.nan)
    elif mode == "concat":
        v = (row.get("utr5_len", 0) or 0) + (row.get("cds_len", 0) or 0) + (row.get("utr3_len", 0) or 0)
        return v
    else:
        return np.nan

def one_hot_encode_batch(seqs, pad_to_max=True):
    """
    seqs: list[str]
    return: np.ndarray [N, L, 4] uint8
    - L = max length in batch
    - 对称补零：左 floor，右 ceil
    - 仅 A/C/G/T one-hot，其他字符（N等）为 0
    """
    mapping = {
        "A": np.array([1,0,0,0], dtype=np.uint8),
        "C": np.array([0,1,0,0], dtype=np.uint8),
        "G": np.array([0,0,1,0], dtype=np.uint8),
        "T": np.array([0,0,0,1], dtype=np.uint8),
    }
    seqs = [s.upper() for s in seqs]
    lengths = [len(s) for s in seqs]
    L = max(lengths) if pad_to_max and lengths else 0
    N = len(seqs)
    X = np.zeros((N, L, 4), dtype=np.uint8)
    for i, s in enumerate(seqs):
        if not s:
            continue
        arr = np.zeros((len(s), 4), dtype=np.uint8)
        for j, ch in enumerate(s):
            v = mapping.get(ch)
            if v is not None:
                arr[j] = v
        pad = L - len(s)
        left = pad // 2
        X[i, left:left+len(s), :] = arr
    return X, L, lengths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_features", required=True, help="seq_features.csv（需含 cds_seq/utr5_seq/utr3_seq 与 status 列）")
    ap.add_argument("--ribo_path", required=True, help="Ribo 表达矩阵（含 gene 与 *_Ribo 列）")
    ap.add_argument("--rna_path", default=None, help="RNA 表达矩阵（含 gene 与 *_RNA 列）；提供则将 RNA 作为额外特征加入")
    ap.add_argument("--out_npz", required=True, help="输出 npz 文件路径")
    ap.add_argument("--seq_mode", choices=["cds","utr5","utr3","concat"], default="cds")
    ap.add_argument("--max_seq_len", type=int, default=None, help="按所选序列模式的长度过滤（>该阈值的基因将被丢弃）")
    ap.add_argument("--dedup_ribo", action="store_true", help="对 Ribo 表按 gene 去重（均值聚合）")
    ap.add_argument("--dedup_rna", action="store_true", help="对 RNA 表按 gene 去重（均值聚合）")
    ap.add_argument("--log1p", action="store_true", help="对标签 Ribo 取 log1p")
    ap.add_argument("--log1p_rna", action="store_true", help="对 RNA 特征取 log1p（仅作用于 RNA 特征）")
    args = ap.parse_args()

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 读取并筛选 seq_features ---
    sf = pd.read_csv(args.seq_features)
    if "status" not in sf.columns:
        raise SystemExit("seq_features 缺少 status 列")
    sf = sf.loc[sf["status"]=="ok"].copy()

    # 按所选模式进行长度过滤（仅内存中过滤，不回写文件）
    if args.max_seq_len is not None:
        for col in ("cds_len","utr5_len","utr3_len"):
            if col not in sf.columns:
                sf[col] = np.nan
        lengths_series = sf.apply(lambda r: seq_length_by_mode(r, args.seq_mode), axis=1)
        keep = (lengths_series.fillna(0) <= float(args.max_seq_len))
        before, after = len(sf), int(keep.sum())
        sf = sf.loc[keep].copy()
        print(f"[filter] {args.seq_mode} length ≤ {args.max_seq_len}: {before} → {after}")

    if sf.empty:
        raise SystemExit("筛选后 seq_features 为空（没有 ok 且满足长度条件的基因）。")

    # --- 读取 Ribo ---
    ribo = pd.read_csv(args.ribo_path, sep=None, engine="python")
    ribo.columns = [str(c).strip().lstrip("\ufeff") for c in ribo.columns]
    assert "gene" in ribo.columns, "Ribo 表必须包含 'gene' 列"

    if args.dedup_ribo:
        n0 = len(ribo)
        ribo = ribo.groupby("gene", as_index=False).mean(numeric_only=True)
        print(f"[dedup] Ribo rows: {n0} → {len(ribo)}")

    # --- 读取 RNA（可选） ---
    rna = None
    if args.rna_path is not None:
        rna = pd.read_csv(args.rna_path, sep=None, engine="python")
        rna.columns = [str(c).strip().lstrip("\ufeff") for c in rna.columns]
        assert "gene" in rna.columns, "RNA 表必须包含 'gene' 列"
        if args.dedup_rna:
            n0 = len(rna)
            rna = rna.groupby("gene", as_index=False).mean(numeric_only=True)
            print(f"[dedup] RNA rows: {n0} → {len(rna)}")

    # --- 阶段集合（Ribo×RNA 交集；若无 RNA 则用 Ribo） ---
    ribo_stages = sorted(set(infer_stages(ribo.columns, "Ribo")))
    if rna is not None:
        rna_stages = sorted(set(infer_stages(rna.columns, "RNA")))
        stages = sorted(set(ribo_stages).intersection(rna_stages))
        if len(stages) == 0:
            raise SystemExit("RNA 与 Ribo 的阶段没有交集。")
    else:
        stages = ribo_stages

    # 裁剪到统一阶段
    ribo = ribo[["gene"] + [f"{s}_Ribo" for s in stages]].copy()
    if rna is not None:
        rna  = rna[["gene"] + [f"{s}_RNA"  for s in stages]].copy()

    # --- 对齐基因顺序（按 seq_features 的顺序） ---
    sf = sf.drop_duplicates(subset=["gene"]).copy()
    sf.set_index("gene", inplace=True)
    ribo.set_index("gene", inplace=True)
    if rna is not None:
        rna.set_index("gene", inplace=True)

    common_genes = [g for g in sf.index if g in ribo.index and (rna is None or g in rna.index)]
    if len(common_genes) == 0:
        raise SystemExit("筛选后基因与 Ribo/RNA 矩阵没有交集。")

    sf   = sf.loc[common_genes]
    ribo = ribo.loc[common_genes]
    if rna is not None:
        rna = rna.loc[common_genes]

    # --- 取序列并 one-hot ---
    seqs = [get_sequence_by_mode(sf.loc[g], args.seq_mode) for g in common_genes]
    X_seq, L, lengths_raw = one_hot_encode_batch(seqs, pad_to_max=True)

    # --- 其他数值特征（序列派生） ---
    for col in SEQ_FEATURE_COLS:
        if col not in sf.columns:
            sf[col] = np.nan
    X_other_list = [sf[SEQ_FEATURE_COLS].astype(np.float32).fillna(0.0).values]
    other_feature_names = list(SEQ_FEATURE_COLS)

    # --- 追加 RNA 特征（每阶段一列） ---
    if rna is not None:
        X_rna = rna[[f"{s}_RNA" for s in stages]].astype(np.float32).values
        if args.log1p_rna:
            X_rna = np.log1p(X_rna)
        RNA_FEATURE_NAMES = [f"rna_{s}" for s in stages]
        X_other_list.append(X_rna)
        other_feature_names += RNA_FEATURE_NAMES

    X_other = np.hstack(X_other_list).astype(np.float32)

    # --- 标签 Y（Ribo） ---
    Y = ribo[[f"{s}_Ribo" for s in stages]].astype(np.float32).values
    if args.log1p:
        Y = np.log1p(Y)

    # --- 形状断言 ---
    N = len(common_genes)
    assert X_seq.shape[0] == N and X_other.shape[0] == N and Y.shape[0] == N, \
        f"Shape mismatch: X_seq={X_seq.shape}, X_other={X_other.shape}, Y={Y.shape}, N={N}"

    # --- 保存 ---
    meta = {
        "genes": list(map(str, common_genes)),
        "stages": stages,
        "seq_mode": args.seq_mode,
        "seq_max_len": int(L),
        "seq_lengths": list(map(int, lengths_raw)),
        "other_feature_names": other_feature_names,
        "y_desc": f"Ribo values ({'log1p' if args.log1p else 'raw'})",
        "max_seq_len_filter": args.max_seq_len,
        "dedup_ribo": bool(args.dedup_ribo),
        "dedup_rna": bool(args.dedup_rna) if args.rna_path is not None else False,
        "log1p_rna": bool(args.log1p_rna) if args.rna_path is not None else False,
        "rna_as_features": bool(args.rna_path is not None),
    }
    np.savez_compressed(
        out_path,
        X_seq=X_seq,          # uint8 [N, L, 4]
        X_other=X_other,      # float32 [N, l]
        Y=Y,                  # float32 [N, S]
    )
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {out_path}")
    print(f"  X_seq:   {X_seq.shape} (uint8, seq_mode={args.seq_mode}, L={L})")
    print(f"  X_other: {X_other.shape} (float32, features={len(other_feature_names)})")
    print(f"  Y:       {Y.shape} (float32)")
    print(f"  Stages:  {', '.join(stages)}")

if __name__ == "__main__":
    main()
