#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rna_to_ribo_regression.py
---------------------------------
Read two tabular matrices (RNA & Ribo), align by gene, and train a simple regression model
to predict Ribo from RNA (per-stage, multi-output).

Expected input formats (TSV/CSV auto-detected by pandas):
- RNA file columns: ["gene", "<Stage>_RNA", ...]
- Ribo file columns: ["gene", "<Stage>_Ribo", ...]

Outputs:
- metrics_per_stage.csv: Pearson r, Spearman ρ, RMSE for each stage on held-out test set
- metrics_overall.txt: brief summary
- fitted_model.joblib: trained model (scikit-learn)
- aligned_long.csv (optional, if --save_long): long-format table for QC
- a simple scatter plot for one example stage (if --plot_example_stage provided)

Usage:
    python rna_to_ribo_regression.py \
        --rna_path GSE197265_h_GVtohESC_28500_RNA_merge_average_fpkm.txt.gz \
        --ribo_path GSE197265_h_GVtohESC_28500_Ribo_merge_average_fpkm.txt.gz \
        --outdir ./rna2ribo_out \
        --log1p \
        --test_size 0.2 \
        --random_state 42 \
        --model ridge \
        --alphas 0.1 1 10 100 \
        --plot_example_stage 2C

Notes:
- By default uses RidgeCV as a strong/simple baseline; switch to ElasticNetCV with --model elasticnet.
- Stage names are inferred from column prefixes before "_RNA"/"_Ribo". Only intersected stages are used.
- Rows (genes) with any NA after alignment are dropped.
"""

import argparse
from pathlib import Path
import sys
import re
import joblib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def read_table_auto(path: str) -> pd.DataFrame:
    # Try tab, comma, or whitespace
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback to tab
        df = pd.read_csv(path, sep="\t")
    # Strip BOM / spaces from header names
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def infer_stages(columns, suffix):
    stages = []
    pat = re.compile(rf"(.+?)_{re.escape(suffix)}$", re.IGNORECASE)
    for c in columns:
        m = pat.match(str(c))
        if m:
            stages.append(m.group(1))
    return stages


def build_XY(rna_df: pd.DataFrame, ribo_df: pd.DataFrame, log1p: bool):
    # Ensure 'gene' column exists and set as index
    if "gene" not in rna_df.columns or "gene" not in ribo_df.columns:
        raise ValueError("Both RNA and Ribo tables must contain a 'gene' column.")
    rna_df = rna_df.copy()
    ribo_df = ribo_df.copy()
    rna_df["gene"] = rna_df["gene"].astype(str)
    ribo_df["gene"] = ribo_df["gene"].astype(str)

    # Identify stages present in each table
    rna_stages = set(infer_stages(rna_df.columns, "RNA"))
    ribo_stages = set(infer_stages(ribo_df.columns, "Ribo"))
    stages = sorted(rna_stages.intersection(ribo_stages))

    if len(stages) == 0:
        raise ValueError("Could not find overlapping stages between RNA and Ribo columns. "
                         "Expected columns like 'GV_RNA' and 'GV_Ribo'.")

    # Subset and rename to consistent order
    rna_cols = ["gene"] + [f"{s}_RNA" for s in stages]
    ribo_cols = ["gene"] + [f"{s}_Ribo" for s in stages]

    missing_rna = [c for c in rna_cols if c not in rna_df.columns]
    missing_ribo = [c for c in ribo_cols if c not in ribo_df.columns]
    if missing_rna or missing_ribo:
        raise ValueError(f"Missing columns.\nRNA missing: {missing_rna}\nRibo missing: {missing_ribo}")

    rna_sub = rna_df[rna_cols].set_index("gene")
    ribo_sub = ribo_df[ribo_cols].set_index("gene")

    # Align genes (inner join)
    common_genes = rna_sub.index.intersection(ribo_sub.index)
    rna_sub = rna_sub.loc[common_genes]
    ribo_sub = ribo_sub.loc[common_genes]

    # Optional log1p transform
    if log1p:
        rna_vals = np.log1p(rna_sub.values.astype(np.float64))
        ribo_vals = np.log1p(ribo_sub.values.astype(np.float64))
        rna_sub = pd.DataFrame(rna_vals, index=rna_sub.index, columns=rna_sub.columns)
        ribo_sub = pd.DataFrame(ribo_vals, index=ribo_sub.index, columns=ribo_sub.columns)

    # Feature matrix X: all RNA stage columns (multi-stage features)
    X = rna_sub.values  # shape: [n_genes, n_stages]
    # Target Y: all Ribo stage columns (same stage order)
    Y = ribo_sub.values

    return X, Y, stages, rna_sub, ribo_sub


def evaluate_per_stage(y_true: np.ndarray, y_pred: np.ndarray, stages):
    rows = []
    for i, s in enumerate(stages):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        # Remove any potential NaN
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]
        if len(yt) < 3:
            r, rho = np.nan, np.nan
            rmse = np.nan
        else:
            r = pearsonr(yt, yp)[0]
            rho = spearmanr(yt, yp).correlation
            rmse = mean_squared_error(yt, yp, squared=False)
        rows.append({"stage": s, "pearson_r": r, "spearman_rho": rho, "rmse": rmse})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_path", required=True, help="Path to RNA matrix (columns like 'GV_RNA', '2C_RNA', ...)")
    ap.add_argument("--ribo_path", required=True, help="Path to Ribo matrix (columns like 'GV_Ribo', '2C_Ribo', ...)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--log1p", action="store_true", help="Apply log1p to both RNA and Ribo")
    ap.add_argument("--model", choices=["ridge", "elasticnet"], default="ridge")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.1, 1.0, 10.0, 100.0],
                    help="Alphas for RidgeCV (ignored for ElasticNet)")
    ap.add_argument("--l1_ratio_grid", type=float, nargs="*", default=[0.1, 0.5, 0.9],
                    help="ElasticNetCV l1_ratio grid (only for --model elasticnet)")
    ap.add_argument("--save_long", action="store_true", help="Save long-format aligned table for QC")
    ap.add_argument("--plot_example_stage", type=str, default=None, help="Stage prefix to plot (e.g., '2C')")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read
    rna_df = read_table_auto(args.rna_path)
    ribo_df = read_table_auto(args.ribo_path)

    # Build feature/target
    X, Y, stages, rna_sub, ribo_sub = build_XY(rna_df, ribo_df, log1p=args.log1p)

    # Train/test split on genes
    Xtr, Xte, Ytr, Yte = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )

    # Model
    if args.model == "ridge":
        base = RidgeCV(alphas=np.array(args.alphas), store_cv_values=False)
    else:
        base = ElasticNetCV(l1_ratio=args.l1_ratio_grid, cv=5, n_jobs=-1, max_iter=10000)

    # Standardize features; targets left in original scale
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", MultiOutputRegressor(base, n_jobs=-1))
    ])

    pipe.fit(Xtr, Ytr)

    # Predictions
    Ypred = pipe.predict(Xte)

    # Evaluate per-stage
    df_metrics = evaluate_per_stage(Yte, Ypred, stages)
    df_metrics.to_csv(outdir / "metrics_per_stage.csv", index=False)

    # Overall averages (ignoring NaNs)
    overall = df_metrics[["pearson_r", "spearman_rho", "rmse"]].mean(numeric_only=True)
    with open(outdir / "metrics_overall.txt", "w", encoding="utf-8") as f:
        f.write("Overall metrics (mean across stages):\n")
        for k, v in overall.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nStages used: " + ", ".join(stages) + "\n")

    # Save model
    joblib.dump({"pipeline": pipe, "stages": stages}, outdir / "fitted_model.joblib")

    # Optional: save long-format table for QC
    if args.save_long:
        rna_long = rna_sub.rename(columns=lambda c: c.replace("_RNA", ""))\
                          .reset_index().melt(id_vars="gene", var_name="stage", value_name="rna")
        ribo_long = ribo_sub.rename(columns=lambda c: c.replace("_Ribo", ""))\
                            .reset_index().melt(id_vars="gene", var_name="stage", value_name="ribo")
        long = pd.merge(rna_long, ribo_long, on=["gene","stage"], how="inner")
        long.to_csv(outdir / "aligned_long.csv", index=False)

    # Optional: quick scatter plot for an example stage
    if args.plot_example_stage is not None:
        s = args.plot_example_stage
        if s not in stages:
            print(f"[WARN] Stage '{s}' not found among stages: {stages}", file=sys.stderr)
        else:
            idx = stages.index(s)
            # Use the test split for visualization
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(Xte[:, idx], Yte[:, idx], s=5, alpha=0.5, label="Truth")
            plt.scatter(Xte[:, idx], Ypred[:, idx], s=5, alpha=0.5, label="Pred")
            plt.xlabel(f"RNA ({s})" + (" [log1p]" if args.log1p else ""))
            plt.ylabel(f"Ribo ({s})" + (" [log1p]" if args.log1p else ""))
            plt.title(f"RNA → Ribo (stage: {s})")
            plt.legend()
            plt.tight_layout()
            fig_path = outdir / f"scatter_{s}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)

    print(f"Done. Outputs are in: {outdir}")
    print("Files:\n - metrics_per_stage.csv\n - metrics_overall.txt\n - fitted_model.joblib\n"
          " - aligned_long.csv (if --save_long)\n - scatter_<stage>.png (if --plot_example_stage)")


