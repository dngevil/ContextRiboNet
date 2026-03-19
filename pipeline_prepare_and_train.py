#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_prepare_and_train.py

Offline end-to-end pipeline:
- 输入: RNA & Ribo 表达矩阵 (列: gene, GV_RNA, MI_RNA, ..., hESC_RNA / GV_Ribo, ...)
- 输入: 本地 Ensembl FASTA 文件:
    * Homo_sapiens.GRCh38.cdna.all.fa(.gz)
    * Homo_sapiens.GRCh38.cds.all.fa(.gz)
- 输出: 每个基因的序列特征、RNA-Ribo 对齐长表、模型预测结果与指标。
"""

import argparse
from pathlib import Path
import gzip, io, re
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


# === IO 工具 ===
def open_maybe_gzip(path: str):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def read_table_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df

def infer_stages(columns, suffix):
    """从列名提取阶段名称 (如 GV_RNA -> GV)"""
    stages = []
    pat = re.compile(rf"(.+?)_{re.escape(suffix)}$", re.IGNORECASE)
    for c in columns:
        m = pat.match(str(c))
        if m:
            stages.append(m.group(1))
    return stages


# === FASTA 解析 + 特征构建 ===
def parse_fasta_with_meta(path: str) -> Dict[str, Dict[str, str]]:
    """解析 Ensembl FASTA (含 gene_symbol 与 gene_id)"""
    d = {}
    with open_maybe_gzip(path) as fh:
        tid, buf, meta = None, [], {}
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if tid is not None:
                    d[tid] = {"seq": "".join(buf).upper(), **meta}
                buf = []
                header = line[1:]
                first = header.split()[0]
                tid = first.split(".")[0]
                gene_id, gene_symbol = None, None
                for token in header.split():
                    if token.startswith("gene:"):
                        gene_id = token.split("gene:")[1].split(".")[0]
                    elif token.startswith("gene_symbol:"):
                        gene_symbol = token.split("gene_symbol:")[1]
                meta = {"gene_id": gene_id, "gene_symbol": gene_symbol}
            else:
                buf.append(line)
        if tid is not None:
            d[tid] = {"seq": "".join(buf).upper(), **meta}
    return d


def gc_content(seq: Optional[str]) -> float:
    if not seq: return np.nan
    s = seq.upper()
    acgt = sum(s.count(x) for x in "ACGT")
    return ((s.count("G")+s.count("C"))/acgt) if acgt else np.nan

def kozak_score(utr5: Optional[str], cds: Optional[str]) -> float:
    if not utr5 or not cds or len(cds) < 4: return np.nan
    u, c = utr5.upper(), cds.upper()
    if not c.startswith("ATG"): return np.nan
    left = u[-6:] if len(u) >= 6 else None; right = c[:4]
    if left is None or len(right) < 4: return np.nan
    score = 0.0
    for pos in range(2,6):
        if left[pos] in ("G","C"): score += 0.5
    if left[3] in ("A","G"): score += 1.0
    if right[3] == "G": score += 1.0
    return score

def count_uorf_basic(utr5: Optional[str]):
    if not utr5: return (np.nan, np.nan)
    s = utr5.upper()
    starts = [m.start() for m in re.finditer("ATG", s)]
    stop_codons = {"TAA","TAG","TGA"}
    with_stop = 0
    for st in starts:
        for i in range(st+3, len(s)-2, 3):
            if s[i:i+3] in stop_codons:
                with_stop += 1; break
    return float(len(starts)), float(with_stop)

def build_seq_features_offline(genes, cdna_fa, cds_fa, save_sequences=False) -> pd.DataFrame:
    cdna = parse_fasta_with_meta(cdna_fa)
    cds  = parse_fasta_with_meta(cds_fa)
    best = {}
    for tid, rec in cds.items():
        sym = rec.get("gene_symbol"); seq = rec.get("seq")
        if not sym or not seq: continue
        L = len(seq)
        if (sym not in best) or (L > best[sym][1]):
            best[sym] = (tid, L)
    rows = []
    for g in genes:
        if g not in best:
            rows.append({"gene": g, "status": "gene_not_found_in_FASTA"})
            continue
        tid, _ = best[g]
        cds_seq = cds.get(tid, {}).get("seq")
        gene_id = cds.get(tid, {}).get("gene_id")
        cdna_seq = cdna.get(tid, {}).get("seq")
        utr5_seq, utr3_seq = None, None
        if isinstance(cdna_seq, str) and isinstance(cds_seq, str):
            pos = cdna_seq.find(cds_seq)
            if pos != -1:
                utr5_seq = cdna_seq[:pos]
                utr3_seq = cdna_seq[pos+len(cds_seq):]
        row = {
            "gene": g, "ensembl_gene_id": gene_id, "transcript_id": tid,
            "cds_len": len(cds_seq) if isinstance(cds_seq, str) else np.nan,
            "utr5_len": len(utr5_seq) if isinstance(utr5_seq, str) else np.nan,
            "utr3_len": len(utr3_seq) if isinstance(utr3_seq, str) else np.nan,
            "gc_cds": gc_content(cds_seq), "gc_utr5": gc_content(utr5_seq), "gc_utr3": gc_content(utr3_seq),
            "kozak_score": kozak_score(utr5_seq, cds_seq),
            "uorf_atg_count": count_uorf_basic(utr5_seq)[0],
            "uorf_stop_in_frame_count": count_uorf_basic(utr5_seq)[1],
            "status": "ok" if isinstance(cds_seq, str) else "no_cds"
        }
        if save_sequences:
            row.update({"cds_seq": cds_seq, "utr5_seq": utr5_seq, "utr3_seq": utr3_seq})
        rows.append(row)
    return pd.DataFrame(rows)


# === 模型评估 ===
def evaluate_per_stage(y_true, y_pred, stages):
    rows = []
    for i, s in enumerate(stages):
        yt, yp = y_true[:, i], y_pred[:, i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 3:
            rows.append({"stage": s, "pearson_r": np.nan, "spearman_rho": np.nan, "rmse": np.nan})
            continue
        r = pearsonr(yt[mask], yp[mask])[0]
        rho = spearmanr(yt[mask], yp[mask]).correlation
        rmse = np.sqrt(mean_squared_error(yt[mask], yp[mask]))
        rows.append({"stage": s, "pearson_r": r, "spearman_rho": rho, "rmse": rmse})
    return pd.DataFrame(rows)


# === 主流程 ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_path", required=True)
    ap.add_argument("--ribo_path", required=True)
    ap.add_argument("--cdna_fa", required=True)
    ap.add_argument("--cds_fa", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", choices=["ribo","te"], default="ribo")
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--model", choices=["ridge","elasticnet"], default="ridge")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.1,1,10,100])
    ap.add_argument("--l1_ratio_grid", type=float, nargs="*", default=[0.1,0.5,0.9])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--save_sequences", action="store_true")
    ap.add_argument("--plot_example_stage", type=str, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rna = read_table_auto(args.rna_path)
    ribo = read_table_auto(args.ribo_path)
    assert "gene" in rna.columns and "gene" in ribo.columns, "Both tables must have 'gene' column."

    rna_stages = set(infer_stages(rna.columns, "RNA"))
    ribo_stages = set(infer_stages(ribo.columns, "Ribo"))
    stages = sorted(rna_stages.intersection(ribo_stages))

    rna = rna[["gene"] + [f"{s}_RNA" for s in stages]].copy()
    ribo = ribo[["gene"] + [f"{s}_Ribo" for s in stages]].copy()

    rna.set_index("gene", inplace=True)
    ribo.set_index("gene", inplace=True)
    common_genes = rna.index.intersection(ribo.index)
    rna, ribo = rna.loc[common_genes], ribo.loc[common_genes]

    genes = list(common_genes)
    seq_features = build_seq_features_offline(genes, args.cdna_fa, args.cds_fa, save_sequences=args.save_sequences)
    seq_features.to_csv(outdir / "seq_features.csv", index=False)

    if args.log1p:
        rna = np.log1p(rna)
        ribo = np.log1p(ribo)

    X = rna.values
    seq_df = seq_features.set_index("gene").loc[rna.index][[
        "cds_len","utr5_len","utr3_len","gc_cds","gc_utr5","gc_utr3",
        "kozak_score","uorf_atg_count","uorf_stop_in_frame_count"
    ]].fillna(0.0).values

    X = np.hstack([rna.values, np.repeat(seq_df, repeats=1, axis=0)])
    if args.target == "ribo":
        Y = ribo.values
    else:
        Y = ribo.values - rna.values

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=args.test_size, random_state=args.random_state, shuffle=True)

    if args.model == "ridge":
        base = RidgeCV(alphas=np.array(args.alphas))
    else:
        base = ElasticNetCV(l1_ratio=np.array(args.l1_ratio_grid), cv=5, n_jobs=-1, max_iter=10000)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", MultiOutputRegressor(base, n_jobs=-1))
    ])
    pipe.fit(Xtr, Ytr)
    Ypred = pipe.predict(Xte)

    df_metrics = evaluate_per_stage(Yte, Ypred, stages)
    df_metrics.to_csv(outdir / "metrics_per_stage.csv", index=False)

    overall = df_metrics[["pearson_r","spearman_rho","rmse"]].mean(numeric_only=True)
    with open(outdir / "metrics_overall.txt", "w") as f:
        f.write("Overall mean across stages:\n")
        for k, v in overall.items():
            f.write(f"{k}: {v:.4f}\n")

    if args.plot_example_stage and args.plot_example_stage in stages:
        idx = stages.index(args.plot_example_stage)
        plt.figure(figsize=(5,5))
        plt.scatter(Yte[:, idx], Ypred[:, idx], s=6, alpha=0.5)
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title(f"{args.plot_example_stage} ({args.target})")
        plt.tight_layout()
        plt.savefig(outdir / f"scatter_{args.plot_example_stage}.png", dpi=150)
        plt.close()

    print(f"✅ Done. Results saved to: {outdir}")

if __name__ == "__main__":
    main()
