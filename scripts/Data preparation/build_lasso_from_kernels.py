#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_lasso_from_kernels.py

功能：
- 载入并展开原数据（与 train_single_label_cnn_mlp.py 一致）拿到 X_other / Y / stage_ids
- 读取 kernel 分数矩阵（rank_kernels_and_score(_with_pwm).py 生成的 *.npz）
- 构建特征： [kernel_scores] (+ 可选 [X_other])
- 训练 Lasso / LassoCV；在 TEST 集评估并画 QQ 图（overall + per-stage）
- 保存：预测 CSV、系数 CSV（标准化空间与原始尺度）、summary.json
- **新增**：生成特征权重图（Top-N，整体；可选分别绘制 kernel 与 other）

用法示例：
python build_lasso_from_kernels.py \
  --npz data/packed_inputs.npz \
  --kernel_npz runs/exp1/kern_rank/kernel_scores_topk.npz \
  --meta_json runs/exp1/kern_rank/meta.json \
  --outdir runs/exp1/lasso_on_kernels \
  --split_mode per_gene --seed 42 \
  --use_cv --standardize --include_other \
  --plot_top_n 50 --plot_separate
"""

import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ------------------------
# 数据展开与拆分（与训练脚本一致）
# ------------------------
def load_and_expand(npz_path):
    meta_path = Path(npz_path).with_suffix(".json")
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}
    pack = np.load(npz_path)
    X_seq = pack["X_seq"].astype(np.float32)      # [N, L, 4]
    X_other = pack["X_other"].astype(np.float32)  # [N, l_full]
    Y = pack["Y"].astype(np.float32)              # [N, S]

    genes = meta.get("genes")
    stages = meta.get("stages")
    feat_names = meta.get("other_feature_names", [])
    assert stages is not None and isinstance(stages, list) and len(stages) == Y.shape[1], \
        "meta.json 需要包含 stages 列表，且长度与 Y 的第二维一致。"

    # 划分 base vs RNA(<stage>) 列
    base_idx = [i for i, n in enumerate(feat_names) if not str(n).startswith("rna_")]
    rna_idx_map = {str(n)[4:]: i for i, n in enumerate(feat_names) if str(n).startswith("rna_")}
    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise SystemExit(f"缺少 RNA 特征列: {missing}。请确保特征包含 rna_<stage> 列。")

    base_feats = X_other[:, base_idx]  # [N, l_base]
    l_base = base_feats.shape[1]

    # 展开到 (N*S)
    X_seq_exp = np.repeat(X_seq, repeats=len(stages), axis=0)  # [N*S, L, 4]
    X_other_list, Y_single, stage_ids, gene_ids = [], [], [], []
    for j, st in enumerate(stages):
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col][:, None]
        X_other_list.append(np.hstack([base_feats, rna_vec]))  # [N, l_base+1]
        Y_single.append(Y[:, [j]])
        stage_ids.append(np.full((X_seq.shape[0], 1), j, dtype=np.int64))
        if genes:
            gene_ids.append(np.array(genes).reshape(-1,1))

    X_other_exp = np.vstack(X_other_list).astype(np.float32)   # [N*S, l_base+1]
    Y_exp       = np.vstack(Y_single).astype(np.float32)       # [N*S, 1]
    stage_ids   = np.vstack(stage_ids).astype(np.int64).reshape(-1)
    gene_ids    = np.vstack(gene_ids).reshape(-1).tolist() if genes else None

    # 给 X_other 的列命名（最后一列随样本阶段变化，这里统一叫 rna_current_stage）
    base_names = [feat_names[i] for i in base_idx]
    other_colnames = base_names + ["rna_current_stage"]

    return {
        "X_other": X_other_exp, "Y": Y_exp.reshape(-1),
        "stage_ids": stage_ids, "stages": stages, "gene_ids": gene_ids,
        "other_names": other_colnames, "l_base": l_base
    }

def split_by_gene(stage_ids, gene_ids, test_ratio=0.2, val_ratio=0.2, seed=42):
    if gene_ids is None:
        raise SystemExit("split_by_gene 需要 meta.json 中的 gene 列表")
    rng = np.random.default_rng(seed)
    gene_ids = np.asarray(gene_ids)
    uniq = np.unique(gene_ids); rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_ratio))
    test_set = set(uniq[:n_test])
    pool = np.array(list(set(uniq[n_test:])))
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * val_ratio))
    val_set = set(pool[:n_val]); train_set = set(pool[n_val:])
    idx_all = np.arange(len(gene_ids))
    test_idx  = idx_all[np.isin(gene_ids, list(test_set))]
    val_idx   = idx_all[np.isin(gene_ids, list(val_set))]
    train_idx = idx_all[np.isin(gene_ids, list(train_set))]
    return train_idx, val_idx, test_idx

def stratified_split_per_stage(stage_ids, test_ratio=0.2, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(stage_ids))
    stage_ids = np.asarray(stage_ids)
    train_pool, test_idx = [], []
    for s in np.unique(stage_ids):
        s_idx = idx_all[stage_ids == s]; rng.shuffle(s_idx)
        n_test = max(1, int(len(s_idx) * test_ratio))
        test_idx.append(s_idx[:n_test])
        train_pool.append(s_idx[n_test:])
    test_idx = np.concatenate(test_idx, axis=0)
    train_pool = np.concatenate(train_pool, axis=0)
    val_idx, train_idx = [], []
    for s in np.unique(stage_ids):
        pool_s = train_pool[stage_ids[train_pool] == s]; rng.shuffle(pool_s)
        n_val = max(1, int(len(pool_s) * val_ratio))
        val_idx.append(pool_s[:n_val]); train_idx.append(pool_s[n_val:])
    val_idx = np.concatenate(val_idx, axis=0)
    train_idx = np.concatenate(train_idx, axis=0)
    return train_idx, val_idx, test_idx

# ------------------------
# 可视化 / 实用函数
# ------------------------
def qq_plot(ax, y_true, y_pred, title=""):
    yt = y_true.flatten(); yp = y_pred.flatten()
    n = min(len(yt), len(yp)); yt, yp = yt[:n], yp[:n]
    ax.plot(yt, yp, '.', ms=2, alpha=0.6)
    lo = float(min(yt.min(), yp.min())); hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], '--', lw=1)
    ax.set_xlabel("Observed quantiles"); ax.set_ylabel("Predicted quantiles")
    ax.set_title(title)

def inverse_scale_linear(coef_scaled, intercept_scaled, scaler):
    """
    对 X 做标准化（(X-mu)/scale），模型： y_hat = a + sum w * X_std
    等价到原始 X： w' = w / scale， a' = a - sum (w * mu / scale)
    """
    scale = scaler.scale_
    mu = scaler.mean_
    w_prime = coef_scaled / scale
    a_prime = intercept_scaled - np.sum(coef_scaled * mu / scale)
    return w_prime, a_prime

def plot_feature_weights(out_path, names, coefs, top_n=50, title="Feature Weights (|coef| top-N)"):
    """
    画横向条形图：按 |coef| 选 Top-N，并按绝对值降序显示。
    names: list[str]
    coefs: 1D np.array（建议传入 coef_orig，更直观）
    """
    coefs = np.asarray(coefs).reshape(-1)
    names = list(map(str, names))
    # 排序并截断
    order = np.argsort(-np.abs(coefs))
    order = order[:min(top_n, len(order))]
    sel_names = [names[i] for i in order][::-1]          # 反转：条形图从小到大往上
    sel_coefs = coefs[order][::-1]

    # 图形
    H = max(3.5, 0.28 * len(order))   # 高度随特征数适配
    fig, ax = plt.subplots(1,1, figsize=(10, H))
    ax.barh(range(len(sel_coefs)), sel_coefs)  # 默认颜色即可
    ax.set_yticks(range(len(sel_names)))
    # 截断过长标签（不改变实际名称）
    lab = [(n if len(n)<=48 else n[:45]+"...") for n in sel_names]
    ax.set_yticklabels(lab, fontsize=9)
    ax.set_xlabel("Coefficient (original scale)")
    ax.set_title(title)
    ax.axvline(0, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ------------------------
# 主流程
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="原始 packed_inputs.npz（含 X_other/Y 等）")
    ap.add_argument("--kernel_npz", required=True, help="kernel_scores(.npz 或 _topk.npz)")
    ap.add_argument("--meta_json", required=False, help="若 kernel_npz 没含 kernel_names，则提供生成它的 meta.json")
    ap.add_argument("--outdir", required=True)

    # 拆分
    ap.add_argument("--split_mode", choices=["per_sample","per_gene"], default="per_gene")
    ap.add_argument("--test_ratio_per_stage", type=float, default=0.2)
    ap.add_argument("--val_ratio_in_train", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # 特征选择
    ap.add_argument("--include_other", action="store_true", help="把 X_other 并入 Lasso 输入")
    ap.add_argument("--drop_rna_col", action="store_true", help="仅保留 base 数值特征，丢弃 rna_current_stage")

    # 模型
    ap.add_argument("--use_cv", action="store_true", help="使用 LassoCV 选 alpha")
    ap.add_argument("--alpha", type=float, default=0.01, help="Lasso 的 alpha（use_cv=False 时生效）")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--standardize", action="store_true", help="对 X 做 z-score（建议开启）")

    # --- 新增：绘图控制 ---
    ap.add_argument("--plot_top_n", type=int, default=50, help="权重图显示的 Top-N 特征（按 |coef|）")
    ap.add_argument("--plot_separate", action="store_true", help="分别对 kernel 与 other 画权重图")
    ap.add_argument("--save_svg", action="store_true", help="额外保存 .svg 矢量图")

    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 载入 & 展开
    D = load_and_expand(args.npz)
    X_other = D["X_other"]                   # [M, l_other]
    other_names = D["other_names"][:]        # base_names + 'rna_current_stage'
    if args.drop_rna_col and other_names and other_names[-1] == "rna_current_stage":
        X_other = X_other[:, :-1]
        other_names = other_names[:-1]
    y_all = D["Y"].reshape(-1)
    stage_ids = np.asarray(D["stage_ids"])
    stages = D["stages"]
    gene_ids = D["gene_ids"]

    # 2) 拆分
    if args.split_mode == "per_sample":
        tr_idx, va_idx, te_idx = stratified_split_per_stage(
            stage_ids, test_ratio=args.test_ratio_per_stage,
            val_ratio=args.val_ratio_in_train, seed=args.seed
        )
    else:
        tr_idx, va_idx, te_idx = split_by_gene(
            stage_ids, gene_ids, test_ratio=args.test_ratio_per_stage,
            val_ratio=args.val_ratio_in_train, seed=args.seed
        )
    M = len(y_all)
    split_tag = np.array([""]*M, dtype=object)
    split_tag[tr_idx] = "train"; split_tag[va_idx] = "val"; split_tag[te_idx] = "test"

    # 3) 读取 kernel 分数矩阵
    Kpack = np.load(args.kernel_npz, allow_pickle=True)
    Xkern = Kpack["X"].astype(np.float32)       # [M, nK]
    if Xkern.shape[0] != M:
        raise SystemExit(f"kernel_npz 的样本数与展开后的 M 不一致: {Xkern.shape[0]} vs {M}")

    # kernel 名称
    if "kernel_names" in Kpack:
        kernel_names = list(map(str, Kpack["kernel_names"]))
    else:
        if not args.meta_json:
            raise SystemExit("kernel_npz 未包含 kernel_names，请通过 --meta_json 提供对应的 meta.json")
        meta = json.load(open(args.meta_json, "r", encoding="utf-8"))
        kernel_names = list(map(str, meta.get("kernel_names", [])))
        if len(kernel_names) != Xkern.shape[1]:
            print("[WARN] meta.json 中 kernel_names 数量与矩阵列数不一致，按列数生成占位名", file=sys.stderr)
            kernel_names = [f"kernel_{i}" for i in range(Xkern.shape[1])]

    nK = Xkern.shape[1]

    # 4) 组装 Lasso 的 X
    if args.include_other:
        X_all = np.concatenate([Xkern, X_other.astype(np.float32)], axis=1)
        feat_names = kernel_names + other_names
        kernel_mask = np.array([True]*nK + [False]*X_other.shape[1], dtype=bool)
    else:
        X_all = Xkern
        feat_names = kernel_names
        kernel_mask = np.array([True]*nK, dtype=bool)

    # 5) 划分集合
    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]
    Xte, yte = X_all[te_idx], y_all[te_idx]

    # 6) 标准化（仅 X）
    scaler = None
    if args.standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)
    else:
        Xtr_s, Xva_s, Xte_s = Xtr, Xva, Xte

    # 7) 训练 Lasso / LassoCV
    if args.use_cv:
        alphas = np.logspace(-3, 1, 20)
        model = LassoCV(alphas=alphas, cv=args.cv_folds, fit_intercept=True, random_state=args.seed, n_jobs=-1)
        model.fit(Xtr_s, ytr)
        alpha_used = float(model.alpha_)
    else:
        model = Lasso(alpha=args.alpha, fit_intercept=True, max_iter=10000)
        model.fit(Xtr_s, ytr)
        alpha_used = float(args.alpha)

    # 8) 预测与评估（TEST）
    Pte = model.predict(Xte_s)
    r2_overall = float(r2_score(yte, Pte))
    r2_by_stage = {}
    te_stage = stage_ids[te_idx]
    for j, st in enumerate(stages):
        m = (te_stage == j)
        if m.sum() >= 2:
            r2_by_stage[st] = float(r2_score(yte[m], Pte[m]))
        else:
            r2_by_stage[st] = float("nan")

    # 9) 反标准化系数到原始 X 尺度（便于解读）
    coef_scaled = model.coef_.copy()
    intercept_scaled = float(model.intercept_)
    if scaler is not None:
        coef_orig, intercept_orig = inverse_scale_linear(coef_scaled, intercept_scaled, scaler)
    else:
        coef_orig, intercept_orig = coef_scaled, intercept_scaled

    # 10) 保存系数
    coef_df = pd.DataFrame({
        "feature": feat_names,
        "coef_scaled": coef_scaled.astype(np.float64),
        "coef_orig": coef_orig.astype(np.float64),
        "is_kernel": kernel_mask
    })
    coef_df.sort_values("coef_orig", key=np.abs, ascending=False, inplace=True)
    coef_df.to_csv(Path(args.outdir) / "lasso_coefficients.csv", index=False)

    # === 新增：绘制特征权重图 ===
    top_n = max(1, int(args.plot_top_n))
    # (a) 整体 Top-N
    plot_feature_weights(
        out_path=Path(args.outdir) / f"feature_weights_top{top_n}.png",
        names=coef_df["feature"].values,
        coefs=coef_df["coef_orig"].values,
        top_n=top_n,
        title=f"Lasso Feature Weights (Top-{top_n}, original scale)"
    )
    if args.save_svg:
        plot_feature_weights(
            out_path=Path(args.outdir) / f"feature_weights_top{top_n}.svg",
            names=coef_df["feature"].values,
            coefs=coef_df["coef_orig"].values,
            top_n=top_n,
            title=f"Lasso Feature Weights (Top-{top_n}, original scale)"
        )
    # (b) 可选：分别画 kernel / other
    if args.plot_separate and args.include_other:
        ker_df = coef_df[coef_df["is_kernel"]].copy()
        oth_df = coef_df[~coef_df["is_kernel"]].copy()
        if len(ker_df) > 0:
            plot_feature_weights(
                out_path=Path(args.outdir) / f"feature_weights_kernel_top{top_n}.png",
                names=ker_df["feature"].values,
                coefs=ker_df["coef_orig"].values,
                top_n=top_n,
                title=f"Lasso Weights (Kernel-only, Top-{top_n})"
            )
            if args.save_svg:
                plot_feature_weights(
                    out_path=Path(args.outdir) / f"feature_weights_kernel_top{top_n}.svg",
                    names=ker_df["feature"].values,
                    coefs=ker_df["coef_orig"].values,
                    top_n=top_n,
                    title=f"Lasso Weights (Kernel-only, Top-{top_n})"
                )
        if len(oth_df) > 0:
            plot_feature_weights(
                out_path=Path(args.outdir) / f"feature_weights_other_top{top_n}.png",
                names=oth_df["feature"].values,
                coefs=oth_df["coef_orig"].values,
                top_n=min(top_n, len(oth_df)),
                title=f"Lasso Weights (Other-only, Top-{min(top_n, len(oth_df))})"
            )
            if args.save_svg:
                plot_feature_weights(
                    out_path=Path(args.outdir) / f"feature_weights_other_top{top_n}.svg",
                    names=oth_df["feature"].values,
                    coefs=oth_df["coef_orig"].values,
                    top_n=min(top_n, len(oth_df)),
                    title=f"Lasso Weights (Other-only, Top-{min(top_n, len(oth_df))})"
                )

    # 11) 保存预测（TEST）
    pred_df = pd.DataFrame({
        "y_true": yte,
        "y_pred": Pte,
        "stage_id": te_stage
    })
    pred_df["stage"] = [stages[i] for i in te_stage]
    pred_df.to_csv(Path(args.outdir) / "predictions_test.csv", index=False)

    # 12) QQ 图（overall + per-stage）
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    yt_sorted =  yte; yp_sorted = Pte
    nmin = min(len(yt_sorted), len(yp_sorted))
    ax.plot(yt_sorted[:nmin], yp_sorted[:nmin], '.', ms=2, alpha=0.6)
    lo, hi = float(min(yt_sorted.min(), yp_sorted.min())), float(max(yt_sorted.max(), yp_sorted.max()))
    ax.plot([lo,hi],[lo,hi],'--',lw=1)
    ax.set_title(f"QQ (TEST Overall)  R2={r2_overall:.3f}")
    ax.set_xlabel("Observed quantiles"); ax.set_ylabel("Predicted quantiles")
    fig.tight_layout(); fig.savefig(Path(args.outdir) / "qq_test_overall.png", dpi=150); plt.close(fig)

    ncol=3; nrow=int(math.ceil(len(stages)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol,5*nrow))
    axes = np.array(axes).reshape(-1)
    for i, st in enumerate(stages):
        ax = axes[i]
        m = (te_stage == i)
        if m.sum() < 2:
            ax.axis("off"); continue
        yt = yte[m]; yp = Pte[m]; k = min(len(yt), len(yp))
        ax.plot(yt[:k], yp[:k], '.', ms=2, alpha=0.6)
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo,hi],[lo,hi],'--',lw=1)
        ax.set_title(f"{st}  R2={r2_by_stage[st]:.3f}")
        ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    for j in range(len(stages), len(axes)): axes[j].axis("off")
    fig.tight_layout(); fig.savefig(Path(args.outdir) / "qq_test_by_stage.png", dpi=150); plt.close(fig)

    # 13) 汇总
    summary = {
        "alpha_used": alpha_used,
        "use_cv": bool(args.use_cv),
        "r2_overall_test": r2_overall,
        "r2_per_stage_test": r2_by_stage,
        "n_features": int(X_all.shape[1]),
        "n_kernel_features": int(Xkern.shape[1]),
        "include_other": bool(args.include_other),
        "drop_rna_col": bool(args.drop_rna_col),
        "standardize": bool(args.standardize),
        "split_mode": args.split_mode,
        "test_ratio_per_stage": args.test_ratio_per_stage,
        "val_ratio_in_train": args.val_ratio_in_train,
        "seed": args.seed,
        "feature_names_head": feat_names[:10],
        "plots": {
            "weights_all": f"feature_weights_top{top_n}.png",
            "weights_kernel": f"feature_weights_kernel_top{top_n}.png" if args.plot_separate and args.include_other else None,
            "weights_other": f"feature_weights_other_top{top_n}.png" if args.plot_separate and args.include_other else None,
            "qq_overall": "qq_test_overall.png",
            "qq_by_stage": "qq_test_by_stage.png"
        }
    }
    with open(Path(args.outdir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 控制台简短提示
    print(f"[OK] Lasso done. TEST R2={r2_overall:.4f}")
    print(f"[OK] Plots:")
    print(f"  - {summary['plots']['weights_all']}")
    if summary['plots']['weights_kernel']: print(f"  - {summary['plots']['weights_kernel']}")
    if summary['plots']['weights_other']:  print(f"  - {summary['plots']['weights_other']}")
    print(f"[OK] Artifacts -> {args.outdir}")

if __name__ == "__main__":
    main()
