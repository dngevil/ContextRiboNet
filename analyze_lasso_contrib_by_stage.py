#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_lasso_contrib_by_stage.py

基于 Lasso 结果，评估每个特征在 9 个细胞系（stages）的“贡献分布差异”。
贡献定义：contrib(i, j) = coef_orig(j) * X_all(i, j)

输入：
  --npz            原始 packed_inputs.npz（用于展开到 N*S）
  --kernel_npz     kernel_scores(_topk).npz（含 X: [M,nK], 可含 kernel_names）
  --coeff_csv      build_lasso_from_kernels.py 产出的 lasso_coefficients.csv（含 feature, coef_orig, is_kernel）
  --outdir         输出目录

可选：
  --meta_json      若 kernel_npz 中无 kernel_names，则需 meta.json 提供
  --include_other  当时 Lasso 是否包含 X_other（需要与训练时一致）
  --drop_rna_col   若当时丢弃了 rna_current_stage 这列，这里也要一致
  --top_k_plot     画图时挑选的特征数（按 |整体均值| 排序）
  --relative       输出一份“相对贡献”（每条样本按绝对贡献和归一化），可用于不同样本间对比
  --pdf_all        额外导出一个 PDF：每个入选特征一页小提琴图
  --seed/split     数据拆分一致性（只用于拿 stage_ids，不影响贡献计算）

统计检验：
  - 优先使用 SciPy 的 Kruskal–Wallis（非参），得到每个特征的跨 9 组 p 值
  - 提供 Benjamini–Hochberg FDR 矫正
  - 若环境没有 SciPy，则跳过统计检验并给出提示
"""

import argparse, json, os, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- 数据展开，与训练脚本一致 ----------------
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

    base_idx = [i for i, n in enumerate(feat_names) if not str(n).startswith("rna_")]
    rna_idx_map = {str(n)[4:]: i for i, n in enumerate(feat_names) if str(n).startswith("rna_")}
    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise SystemExit(f"缺少 RNA 特征列: {missing}。请确保特征包含 rna_<stage> 列。")

    base_feats = X_other[:, base_idx]  # [N, l_base]
    l_base = base_feats.shape[1]

    # 展开到 (N*S)
    N = X_seq.shape[0]; S = len(stages)
    X_seq_exp = np.repeat(X_seq, repeats=S, axis=0)  # [N*S, L, 4]
    X_other_list, Y_single, stage_ids, gene_ids = [], [], [], []
    for j, st in enumerate(stages):
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col][:, None]
        X_other_list.append(np.hstack([base_feats, rna_vec]))  # [N, l_base+1]
        Y_single.append(Y[:, [j]])
        stage_ids.append(np.full((N, 1), j, dtype=np.int64))
        if genes:
            gene_ids.append(np.array(genes).reshape(-1,1))

    X_other_exp = np.vstack(X_other_list).astype(np.float32)   # [N*S, l_base+1]
    Y_exp       = np.vstack(Y_single).astype(np.float32)       # [N*S, 1]
    stage_ids   = np.vstack(stage_ids).astype(np.int64).reshape(-1)
    gene_ids    = np.vstack(gene_ids).reshape(-1).tolist() if genes else None

    base_names = [feat_names[i] for i in base_idx]
    other_colnames = base_names + ["rna_current_stage"]

    return {
        "X_other": X_other_exp, "Y": Y_exp.reshape(-1),
        "stage_ids": stage_ids, "stages": stages, "gene_ids": gene_ids,
        "other_names": other_colnames, "l_base": l_base
    }

# ---------------- 统计与可视化 ----------------
def bh_fdr(pvals):
    """Benjamini–Hochberg FDR。返回 q-values，与 pvals 对应顺序一致。"""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(p)
    c = n / np.arange(1, n+1)
    ranked[order] = p[order] * c
    # 后向累计最小化
    q = np.minimum.accumulate(ranked[order][::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def violin_grid(data_dict, labels, title, out_path, rotate_x=False):
    """
    data_dict: stage -> 1D array-like（该 stage 的贡献值）
    labels: 有序 stage 名称列表
    """
    fig, ax = plt.subplots(1,1, figsize=(max(6, 0.6*len(labels)+3), 4))
    vdata = [np.asarray(data_dict[s], dtype=float) for s in labels]
    parts = ax.violinplot(vdata, showmeans=True, showextrema=False)
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=45 if rotate_x else 0, ha="right" if rotate_x else "center")
    ax.set_ylabel("Contribution (coef_orig * value)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def heatmap_means(mean_mat, row_names, col_names, title, out_path, vcenter=0.0):
    """
    mean_mat: [n_rows, n_cols]
    """
    import matplotlib.pyplot as plt
    fig_w = max(8, 0.35*len(col_names) + 3)
    fig_h = max(6, 0.28*len(row_names) + 1.5)
    fig, ax = plt.subplots(1,1, figsize=(fig_w, fig_h))
    im = ax.imshow(mean_mat, aspect="auto", cmap="bwr", vmin=np.min(mean_mat), vmax=np.max(mean_mat))
    if vcenter is not None:
        # 简单置中：对称范围
        vmax = np.max(np.abs(mean_mat))
        im.set_clim(-vmax, vmax)
    ax.set_xticks(np.arange(len(col_names))); ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_names))); ax.set_yticklabels(row_names, fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Mean contribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--kernel_npz", required=True)
    ap.add_argument("--coeff_csv", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--meta_json", default=None, help="若 kernel_npz 无 kernel_names，需要提供")
    ap.add_argument("--include_other", action="store_true", help="与 Lasso 训练时一致")
    ap.add_argument("--drop_rna_col", action="store_true", help="与 Lasso 训练时一致")
    ap.add_argument("--top_k_plot", type=int, default=30, help="画图挑选特征数（按 |整体均值|）")
    ap.add_argument("--relative", action="store_true", help="同时输出“相对贡献”（样本内按绝对贡献和归一化）")
    ap.add_argument("--pdf_all", action="store_true", help="为入选特征逐个导出 violin PDF（每页一个特征）")

    # 拆分参数（仅用于拿 stage_ids，不影响贡献计算）
    ap.add_argument("--split_mode", choices=["per_sample","per_gene"], default="per_gene")
    ap.add_argument("--test_ratio_per_stage", type=float, default=0.2)
    ap.add_argument("--val_ratio_in_train", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 载入 & 展开
    D = load_and_expand(args.npz)
    X_other = D["X_other"]                 # [M, l_other]
    other_names = D["other_names"][:]
    if args.drop_rna_col and other_names and other_names[-1] == "rna_current_stage":
        X_other = X_other[:, :-1]
        other_names = other_names[:-1]
    y_all = D["Y"]
    stage_ids = np.asarray(D["stage_ids"])
    stages = list(D["stages"])

    # 2) 读取 kernel 分数
    Kpack = np.load(args.kernel_npz, allow_pickle=True)
    Xkern = Kpack["X"].astype(np.float32)  # [M, nK]
    if "kernel_names" in Kpack:
        kernel_names = list(map(str, Kpack["kernel_names"]))
    else:
        if not args.meta_json:
            raise SystemExit("kernel_npz 未包含 kernel_names，请通过 --meta_json 提供")
        meta = json.load(open(args.meta_json, "r", encoding="utf-8"))
        kernel_names = list(map(str, meta.get("kernel_names", [])))
        if len(kernel_names) != Xkern.shape[1]:
            kernel_names = [f"kernel_{i}" for i in range(Xkern.shape[1])]

    # 3) 拼接与特征名（与 Lasso 输入一致）
    if args.include_other:
        X_all = np.concatenate([Xkern, X_other.astype(np.float32)], axis=1)
        feat_names = kernel_names + other_names
        is_kernel = np.array([True]*len(kernel_names) + [False]*len(other_names), dtype=bool)
    else:
        X_all = Xkern
        feat_names = kernel_names
        is_kernel = np.array([True]*len(kernel_names), dtype=bool)

    M, F = X_all.shape

    # 4) 读取 Lasso 系数（原始尺度）
    coef_df = pd.read_csv(args.coeff_csv)
    if not {"feature","coef_orig"}.issubset(coef_df.columns):
        raise SystemExit("coeff_csv 里需要包含列：feature, coef_orig")
    # 构造与 feat_names 对齐的系数向量
    coef_map = dict(zip(coef_df["feature"].astype(str), coef_df["coef_orig"].astype(float)))
    coefs = np.array([coef_map.get(n, 0.0) for n in feat_names], dtype=np.float64)  # [F]

    # 5) 逐样本贡献矩阵
    contrib = X_all.astype(np.float64) * coefs.reshape(1, -1)  # [M, F]

    # 可选：相对贡献（样本内按绝对贡献和归一化）
    if args.relative:
        denom = np.sum(np.abs(contrib), axis=1, keepdims=True) + 1e-12
        contrib_rel = contrib / denom  # [M, F]
        np.savez_compressed(outdir/"contrib_relative.npz",
                            contrib=contrib_rel.astype(np.float32),
                            feature_names=np.array(feat_names, dtype=object),
                            stage_ids=stage_ids, stages=np.array(stages, dtype=object))

    # 6) 统计：每个特征在各 stage 的分布 + 差异检验
    stats_rows = []
    try:
        from scipy.stats import kruskal
        have_scipy = True
    except Exception:
        have_scipy = False
        print("[WARN] 未检测到 SciPy，将跳过 Kruskal–Wallis 检验。", file=sys.stderr)

    for j, name in enumerate(feat_names):
        vec = contrib[:, j]
        groups = [vec[stage_ids == sid] for sid in range(len(stages))]

        # 描述性统计
        means = [float(np.mean(g)) if g.size else np.nan for g in groups]
        medians = [float(np.median(g)) if g.size else np.nan for g in groups]
        stds = [float(np.std(g)) if g.size else np.nan for g in groups]
        q25 = [float(np.percentile(g, 25)) if g.size else np.nan for g in groups]
        q75 = [float(np.percentile(g, 75)) if g.size else np.nan for g in groups]

        row = {
            "feature": name,
            "is_kernel": bool(is_kernel[j]),
            "overall_mean": float(np.mean(vec)),
            "overall_std": float(np.std(vec))
        }
        for sid, st in enumerate(stages):
            row[f"mean_{st}"] = means[sid]
            row[f"median_{st}"] = medians[sid]
            row[f"std_{st}"] = stds[sid]
            row[f"q25_{st}"] = q25[sid]
            row[f"q75_{st}"] = q75[sid]

        if have_scipy:
            try:
                # 只用非空组
                nonempty = [g for g in groups if g.size > 0]
                if len(nonempty) >= 2:
                    stat, p = kruskal(*nonempty)
                else:
                    stat, p = np.nan, np.nan
            except Exception:
                stat, p = np.nan, np.nan
            row["kw_stat"] = float(stat) if not np.isnan(stat) else np.nan
            row["kw_p"] = float(p) if not np.isnan(p) else np.nan

        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    # FDR
    if have_scipy and "kw_p" in stats_df.columns:
        pvals = stats_df["kw_p"].fillna(1.0).values
        qvals = bh_fdr(pvals)
        stats_df["kw_q"] = qvals

    stats_df.sort_values("overall_mean", key=np.abs, ascending=False, inplace=True)
    stats_df.to_csv(outdir/"feature_contrib_stats_by_stage.csv", index=False)

    # 7) 画图：Top-K（按 |overall_mean|）
    K = max(1, int(args.top_k_plot))
    top_feats = stats_df.head(K)["feature"].tolist()
    # 热图（均值）
    mean_mat = []
    for f in top_feats:
        row = stats_df[stats_df["feature"] == f].iloc[0]
        mean_mat.append([row[f"mean_{st}"] for st in stages])
    mean_mat = np.array(mean_mat, dtype=float)
    heatmap_means(mean_mat, row_names=top_feats, col_names=stages,
                  title=f"Feature mean contributions by stage (Top-{K})",
                  out_path=outdir/f"heatmap_mean_contrib_top{K}.png", vcenter=0.0)

    # 小提琴图（逐特征）
    vio_dir = outdir / f"violin_top{K}"
    vio_dir.mkdir(exist_ok=True, parents=True)
    if args.pdf_all:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = outdir / f"violin_top{K}.pdf"
        pdf = PdfPages(pdf_path)
    else:
        pdf = None

    for f in top_feats:
        j = feat_names.index(f)
        vec = contrib[:, j]
        data_dict = {st: vec[stage_ids == sid] for sid, st in enumerate(stages)}
        out_png = vio_dir / f"{f}.png"
        violin_grid(data_dict, stages, title=f, out_path=out_png, rotate_x=True)
        if pdf is not None:
            # 把刚才的图再画一遍写入 PDF
            fig, ax = plt.subplots(1,1, figsize=(max(6, 0.6*len(stages)+3), 4))
            vdata = [np.asarray(data_dict[s], dtype=float) for s in stages]
            ax.violinplot(vdata, showmeans=True, showextrema=False)
            ax.set_xticks(np.arange(1, len(stages)+1))
            ax.set_xticklabels(stages, rotation=45, ha="right")
            ax.set_ylabel("Contribution (coef_orig * value)")
            ax.set_title(f)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    if pdf is not None:
        pdf.close()
        print(f"[OK] PDF -> {pdf_path}")

    # 8) 保存原始贡献矩阵（可选体积较大）
    np.savez_compressed(outdir/"contrib_raw.npz",
                        contrib=contrib.astype(np.float32),
                        feature_names=np.array(feat_names, dtype=object),
                        stage_ids=stage_ids, stages=np.array(stages, dtype=object))

    # 摘要
    summary = {
        "n_samples": int(M),
        "n_features": int(F),
        "n_stages": len(stages),
        "top_k_plot": K,
        "files": {
            "stats_csv": "feature_contrib_stats_by_stage.csv",
            "heatmap": f"heatmap_mean_contrib_top{K}.png",
            "violin_dir": f"violin_top{K}/",
            "pdf_all": f"violin_top{K}.pdf" if args.pdf_all else None,
            "contrib_raw": "contrib_raw.npz",
            "contrib_relative": "contrib_relative.npz" if args.relative else None
        }
    }
    with open(outdir/"summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Done. Results -> {outdir}")
    print(f" - {summary['files']['stats_csv']}")
    print(f" - {summary['files']['heatmap']}")
    print(f" - {summary['files']['violin_dir']}")
    if summary['files']['pdf_all']:
        print(f" - {summary['files']['pdf_all']}")

if __name__ == "__main__":
    main()
