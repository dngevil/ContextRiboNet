#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rank_kernels_and_score_with_pwm.py

步骤：
1) 计算 kernel 贡献度（训练集），排序并筛选 Top-K
2) 计算全部样本的 kernel 分数（conv+ReLU 后的 global-max）
3) 存储：contrib_rank.csv / selected_kernels.txt / kernel_scores(.npz & 可选 .csv)
4) 对 Top-K kernels：基于指定集合（train/all）收集激活窗口，生成 PFM/PWM（可选 MEME）

要求：
- 与 train_single_label_cnn_mlp.py 的结构一致（需要传入 kernel_sizes / cnn_channels 等）
"""

import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# -------------------- 基础设置 --------------------
def set_seed(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# -------------------- 数据载入/展开/拆分 --------------------
def load_and_expand(npz_path):
    meta_path = Path(npz_path).with_suffix(".json")
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}
    pack = np.load(npz_path)
    X_seq = pack["X_seq"].astype(np.float32)      # [N, L, 4]
    X_other = pack["X_other"].astype(np.float32)  # [N, l_full]
    Y = pack["Y"].astype(np.float32)              # [N, S]
    N, L, _ = X_seq.shape
    S = Y.shape[1]
    genes = meta.get("genes")
    stages = meta.get("stages")
    feat_names = meta.get("other_feature_names", [])
    assert stages is not None and isinstance(stages, list) and len(stages) == S, \
        "meta.json 需要包含 stages 列表，且长度与 Y 的第二维一致。"
    base_idx = [i for i, n in enumerate(feat_names) if not str(n).startswith("rna_")]
    rna_idx_map = {}
    for i, n in enumerate(feat_names):
        if str(n).startswith("rna_"):
            st = str(n)[4:]; rna_idx_map[st] = i
    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise SystemExit(f"缺少 RNA 特征列: {missing}。请确保特征包含 rna_<stage> 列。")

    base_feats = X_other[:, base_idx]             # [N, l_base]
    l_base = base_feats.shape[1]

    X_seq_exp = np.repeat(X_seq, repeats=S, axis=0)  # [N*S, L, 4]
    X_other_list, Y_single, stage_ids, gene_ids = [], [], [], []
    for j, st in enumerate(stages):
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col][:, None]
        X_other_list.append(np.hstack([base_feats, rna_vec]))
        Y_single.append(Y[:, [j]])
        stage_ids.append(np.full((N, 1), j, dtype=np.int64))
        if genes: gene_ids.append(np.array(genes).reshape(-1,1))
    X_other_exp = np.vstack(X_other_list).astype(np.float32)  # [N*S, l_base+1]
    Y_exp       = np.vstack(Y_single).astype(np.float32)      # [N*S, 1]
    stage_ids   = np.vstack(stage_ids).astype(np.int64).reshape(-1)
    gene_ids    = np.vstack(gene_ids).reshape(-1).tolist() if genes else None
    return {
        "X_seq": X_seq_exp, "X_other": X_other_exp, "Y": Y_exp,
        "stage_ids": stage_ids, "stages": stages,
        "gene_ids": gene_ids, "L": X_seq.shape[1], "l_base": l_base
    }

def split_by_gene(stage_ids, gene_ids, test_ratio=0.2, val_ratio=0.2, seed=42):
    if gene_ids is None: raise SystemExit("split_by_gene 需要 meta.json 中的 gene 列表")
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
        test_idx.append(s_idx[:n_test]); train_pool.append(s_idx[n_test:])
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

# -------------------- 模型 --------------------
class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_blocks=3, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), activation(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_blocks)
        ])
        self.act = activation()
    def forward(self, x):
        x = self.input(x)
        for blk in self.blocks: x = self.act(blk(x) + x)
        return x

class SeqCNNBranch(nn.Module):
    def __init__(self, in_ch=4, kernel_sizes=(6,10,12,16,20), channels_per_kernel=64, activation=nn.ReLU):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_ch, channels_per_kernel, k, padding=0) for k in kernel_sizes])
        self.act = activation()
        self.out_dim = len(kernel_sizes) * channels_per_kernel
    def forward(self, x_seq):  # [B,L,4]
        x = x_seq.transpose(1,2).contiguous()    # [B,4,L]
        outs = []
        for conv in self.convs:
            z = self.act(conv(x))                # [B,C,L']
            z = torch.amax(z, dim=-1)            # [B,C]
            outs.append(z)
        return torch.cat(outs, dim=1)            # [B,sumC]

class CNN_MLP_Fusion(nn.Module):
    def __init__(self, other_in_dim, cnn_channels=64, mlp_hidden=128, mlp_blocks=3,
                 dropout=0.1, kernel_sizes=(6,10,12,16,20), activation=nn.ReLU):
        super().__init__()
        self.seq_branch = SeqCNNBranch(4, kernel_sizes, cnn_channels, activation)
        self.other_branch = ResidualMLP(other_in_dim, mlp_hidden, mlp_blocks, dropout, activation)
        fusion_in = self.seq_branch.out_dim + mlp_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, max(mlp_hidden,128)), activation(), nn.Dropout(dropout),
            nn.Linear(max(mlp_hidden,128), 1)
        )
    def forward(self, x_seq, x_other):
        h = torch.cat([self.seq_branch(x_seq), self.other_branch(x_other)], dim=1)
        return self.fusion(h)

# -------------------- 分数矩阵（全部 kernels） --------------------
@torch.no_grad()
def score_all_kernels(device, conv_modules, X_seq_all, batch_size=1024, act_relu=True):
    M, L, _ = X_seq_all.shape
    kernel_names = []
    for conv in conv_modules:
        K = conv.kernel_size[0]
        for c in range(conv.out_channels):
            kernel_names.append(f"kernel_k{K}_f{c}")
    nK = len(kernel_names)
    scores = np.zeros((M, nK), dtype=np.float32)

    device_convs = [conv.to(device).eval() for conv in conv_modules]
    n_batches = (M + batch_size - 1) // batch_size
    col = 0
    for conv in device_convs:
        C = conv.out_channels
        col_end = col + C
        for bi in range(n_batches):
            s = bi*batch_size; e = min(M, s+batch_size)
            xs = torch.from_numpy(X_seq_all[s:e]).to(device, non_blocking=True)
            x = xs.transpose(1,2).contiguous()
            z = F.conv1d(x, conv.weight, conv.bias, stride=1, padding=0)
            if act_relu: z = F.relu(z)
            zmax = torch.amax(z, dim=-1)  # [B,C]
            scores[s:e, col:col_end] = zmax.detach().cpu().numpy()
        col = col_end
    return scores, kernel_names

# -------------------- 贡献度（训练集） --------------------
def zscore(a, axis=0, eps=1e-8):
    mu = np.mean(a, axis=axis, keepdims=True)
    sd = np.std(a, axis=axis, keepdims=True) + eps
    return (a - mu) / sd, mu, sd

def contrib_corr(X_tr, y_tr):
    N = X_tr.shape[0]
    v = (X_tr.T @ y_tr.reshape(-1,1)).reshape(-1) / max(N, 1)
    return np.abs(v.astype(np.float64))

def contrib_ridge(X_tr, y_tr, alpha=1.0):
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X_tr, y_tr)
    return np.abs(reg.coef_.astype(np.float64))

# -------------------- PWM 相关 --------------------
def onehot_windows_to_pfm(onehots, weights=None, eps=1e-8):
    """
    onehots: [N, 4, K]  (A,C,G,T x K)
    weights: [N] (加权计数，不传则均匀)
    """
    assert onehots.ndim == 3 and onehots.shape[1] == 4
    N, _, K = onehots.shape
    if weights is None:
        weights = np.ones((N,), dtype=np.float64)
    weights = weights.astype(np.float64)
    pfm = np.zeros((4, K), dtype=np.float64)
    w = weights[:, None, None]  # [N,1,1]
    pfm = (onehots.astype(np.float64) * w).sum(axis=0)  # [4,K]
    colsum = pfm.sum(axis=0, keepdims=True) + eps
    pwm = pfm / colsum
    return pfm, pwm

def extract_subseq_windows(X_seq_all, idx_list, kernel_size):
    """
    X_seq_all: [M, L, 4]  one-hot
    idx_list : list of (sample_idx, start_pos)
    返回 [N, 4, K]
    """
    K = kernel_size
    out = np.zeros((len(idx_list), 4, K), dtype=np.float32)
    for i, (sid, p) in enumerate(idx_list):
        win = X_seq_all[sid, p:p+K, :]  # [K,4]
        if win.shape[0] != K: continue
        out[i] = np.transpose(win, (1,0))  # -> [4,K]
    return out

@torch.no_grad()
def collect_top_hits_selected_kernels(
    device, conv_modules, X_seq_all, selected_map,    # selected_map: {layer_index: [channel_indices]}
    subset_indices, topk=500, act_relu=True, batch_size=1024
):
    """
    对“被选中的 kernel”（按 conv 层与通道号）在 subset 上收集全局 Top-K 激活窗口。
    返回：hits_dict[(layer_idx, ch)] = [(score, sample_idx_global, start_pos), ...]（降序）
    """
    hits_dict = {}
    if not selected_map: return hits_dict

    # 为每个选中通道初始化缓存
    kept_scores = {}
    kept_bidx   = {}
    kept_pos    = {}

    # 分批
    n = len(subset_indices)
    n_batches = (n + batch_size - 1) // batch_size
    for li, conv in enumerate(conv_modules):
        if li not in selected_map: continue
        conv = conv.to(device).eval()
        C = conv.out_channels
        sel_channels = np.array(sorted(set([c for c in selected_map[li] if 0 <= c < C])), dtype=np.int64)
        if sel_channels.size == 0: continue
        # 初始化缓存
        for c in sel_channels:
            kept_scores[(li,c)] = torch.empty(0, device=device)
            kept_bidx[(li,c)]   = torch.empty(0, dtype=torch.int32, device=device)
            kept_pos[(li,c)]    = torch.empty(0, dtype=torch.int32, device=device)

        total_seen = 0
        for bi in range(n_batches):
            s = bi*batch_size; e = min(n, s+batch_size)
            idx = subset_indices[s:e]
            xs = torch.from_numpy(X_seq_all[idx]).to(device, non_blocking=True)  # [B,L,4]
            x = xs.transpose(1,2).contiguous()
            z = F.conv1d(x, conv.weight, conv.bias, stride=1, padding=0)  # [B,C,Lp]
            if act_relu: z = F.relu(z)
            B, C2, Lp = z.shape
            assert C2 == C
            # 压平成 [C, B*Lp]
            z_flat = z.permute(1,0,2).reshape(C, B*Lp).contiguous()
            k_each = min(topk, B*Lp)

            # 仅对选中通道做 topk & 合并
            vals, idxs = torch.topk(z_flat[sel_channels, :], k=k_each, dim=1)  # [n_sel, k_each]
            pos  = idxs % Lp
            bidx = idxs // Lp

            for j, c in enumerate(sel_channels):
                key = (li, int(c))
                new_scores = vals[j]
                new_bidx   = bidx[j] + total_seen
                new_pos    = pos[j]
                if kept_scores[key].numel() > 0:
                    new_scores = torch.cat([kept_scores[key], new_scores], dim=0)
                    new_bidx   = torch.cat([kept_bidx[key], new_bidx], dim=0)
                    new_pos    = torch.cat([kept_pos[key], new_pos], dim=0)
                k_final = min(topk, new_scores.numel())
                sel_vals, sel_idx = torch.topk(new_scores, k=k_final, dim=0)
                kept_scores[key] = sel_vals
                kept_bidx[key]   = new_bidx[sel_idx]
                kept_pos[key]    = new_pos[sel_idx]
            total_seen += B

        # 收尾：搬到 CPU，组装
        for c in sel_channels:
            key = (li, int(c))
            sc = kept_scores[key].detach().cpu().numpy()
            bi = kept_bidx[key].detach().cpu().numpy()
            ps = kept_pos[key].detach().cpu().numpy()
            # bi 是 subset 内的批全局索引 → 映射回原始样本索引
            bi = np.array([subset_indices[int(x)] for x in bi], dtype=np.int64)
            hits = [(float(sc[i]), int(bi[i]), int(ps[i])) for i in range(len(sc))]
            hits_dict[key] = hits  # 已是降序
    return hits_dict

def save_meme(meme_path, pwm_dict):
    with open(meme_path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        for name, pwm in pwm_dict.items():
            K = pwm.shape[1]
            f.write(f"MOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {K}\n")
            for pos in range(K):
                a,c,g,t = pwm[0,pos], pwm[1,pos], pwm[2,pos], pwm[3,pos]
                f.write(f"{a:.6f}\t{c:.6f}\t{g:.6f}\t{t:.6f}\n")
            f.write("\n")

# -------------------- 主程序 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)

    # 结构参数（需与训练一致）
    ap.add_argument("--cnn_channels", type=int, default=64)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--kernel_sizes", type=int, nargs="+", default=[6,10,12,16,20])

    # 拆分
    ap.add_argument("--split_mode", choices=["per_sample","per_gene"], default="per_gene")
    ap.add_argument("--test_ratio_per_stage", type=float, default=0.2)
    ap.add_argument("--val_ratio_in_train", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # 贡献度/排名
    ap.add_argument("--contrib", choices=["ridge","corr"], default="ridge")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--top_k", type=int, default=256)

    # 评分/收集批大小 & 输出 CSV
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--save_csv_full", action="store_true")
    ap.add_argument("--save_csv_topk", action="store_true")

    # PWM 相关
    ap.add_argument("--pwm_from", choices=["train","all"], default="train",
                    help="在训练集或全体样本上收集激活窗口")
    ap.add_argument("--topk_per_kernel", type=int, default=500,
                    help="每个选中 kernel 收集的激活窗口上限")
    ap.add_argument("--min_hits", type=int, default=100,
                    help="生成 motif 的最小窗口数门槛")
    ap.add_argument("--weight_mode", choices=["activation","uniform"], default="activation",
                    help="PFM 计数权重：按激活强度或等权")
    ap.add_argument("--save_meme", action="store_true",
                    help="导出 MEME 格式汇总（outdir/motifs/meme.txt）")

    args = ap.parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "motifs").mkdir(parents=True, exist_ok=True)

    # ==== 载入并展开 ====
    D = load_and_expand(args.npz)
    X_seq = D["X_seq"]                    # [M, L, 4]
    y_all = D["Y"].reshape(-1)            # [M]
    stage_ids = np.asarray(D["stage_ids"])
    stages = D["stages"]
    gene_ids = D["gene_ids"]
    other_in_dim = D["l_base"] + 1

    # ==== 拆分 ====
    if args.split_mode == "per_sample":
        tr_idx, va_idx, te_idx = stratified_split_per_stage(
            stage_ids, test_ratio=args.test_ratio_per_stage, val_ratio=args.val_ratio_in_train, seed=args.seed
        )
    else:
        tr_idx, va_idx, te_idx = split_by_gene(
            stage_ids, gene_ids, test_ratio=args.test_ratio_per_stage, val_ratio=args.val_ratio_in_train, seed=args.seed
        )
    M = X_seq.shape[0]
    split_tag = np.array([""]*M, dtype=object)
    split_tag[tr_idx] = "train"; split_tag[va_idx] = "val"; split_tag[te_idx] = "test"

    # ==== 构建模型 & 加载权重 ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_MLP_Fusion(
        other_in_dim=other_in_dim,
        cnn_channels=args.cnn_channels, mlp_hidden=args.mlp_hidden,
        mlp_blocks=args.mlp_blocks, dropout=args.dropout,
        kernel_sizes=tuple(args.kernel_sizes)
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()

    # ==== Step-2: 全部样本 kernel 分数 ====
    scores, kernel_names = score_all_kernels(
        device=device, conv_modules=list(model.seq_branch.convs),
        X_seq_all=X_seq, batch_size=args.batch_size, act_relu=True
    )  # scores: [M, nK]
    nK = scores.shape[1]

    # ==== Step-1: 贡献度与排序（基于训练集） ====
    m_train = (split_tag == "train")
    Xtr = scores[m_train]
    ytr = y_all[m_train]
    if args.standardize:
        Xtr_z, _, _ = zscore(Xtr, axis=0)
        ytr_z, _, _ = zscore(ytr, axis=0)
    else:
        Xtr_z, ytr_z = Xtr, ytr

    contrib_vals = contrib_ridge(Xtr_z, ytr_z, alpha=args.alpha) if args.contrib == "ridge" \
                   else contrib_corr(Xtr_z, ytr_z)
    order = np.argsort(-contrib_vals)
    topK = min(args.top_k, nK)
    top_idx = order[:topK]

    # 保存排名
    df_rank = pd.DataFrame({
        "kernel": np.array(kernel_names)[order],
        "contrib": contrib_vals[order],
        "rank": np.arange(1, nK+1)
    })
    df_rank.to_csv(outdir / "contrib_rank.csv", index=False)
    with open(outdir / "selected_kernels.txt", "w") as f:
        for kname in np.array(kernel_names)[top_idx]:
            f.write(kname + "\n")

    # ==== Step-3: 存储分数矩阵 ====
    meta = {
        "kernel_names": kernel_names,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "stages": stages,
        "n_samples": int(M),
        "n_kernels": int(nK),
        "top_k": int(topK),
        "contrib_method": args.contrib,
        "alpha": float(args.alpha),
        "standardize": bool(args.standardize),
        "kernel_sizes": args.kernel_sizes,
        "cnn_channels": int(args.cnn_channels),
    }
    np.savez_compressed(outdir / "kernel_scores.npz",
                        X=scores.astype(np.float32),
                        split=split_tag,
                        stage_ids=stage_ids.astype(np.int64),
                        y=y_all.astype(np.float32))
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    scores_topk = scores[:, top_idx]
    topk_names = list(np.array(kernel_names)[top_idx])
    np.savez_compressed(outdir / "kernel_scores_topk.npz",
                        X=scores_topk.astype(np.float32),
                        split=split_tag,
                        stage_ids=stage_ids.astype(np.int64),
                        y=y_all.astype(np.float32),
                        kernel_names=np.array(topk_names, dtype=object))

    if args.save_csv_full:
        df_full = pd.DataFrame(scores, columns=kernel_names)
        df_full.insert(0, "split", split_tag)
        df_full.insert(1, "stage_id", stage_ids)
        if D["gene_ids"] is not None: df_full.insert(2, "gene", D["gene_ids"])
        df_full.to_csv(outdir / "kernel_scores.csv", index=False)

    if args.save_csv_topk:
        df_top = pd.DataFrame(scores_topk, columns=topk_names)
        df_top.insert(0, "split", split_tag)
        df_top.insert(1, "stage_id", stage_ids)
        if D["gene_ids"] is not None: df_top.insert(2, "gene", D["gene_ids"])
        df_top.to_csv(outdir / "kernel_scores_topk.csv", index=False)

    # ==== Step-4: 仅 Top-K kernels 计算 PWM ====
    # 建立：kernel_names → (layer_idx, channel_idx, K)
    layer_offsets = []
    for li, conv in enumerate(model.seq_branch.convs):
        C = conv.out_channels
        layer_offsets.append((li, C, conv.kernel_size[0]))
    # 解析名称映射
    name_to_lc = {}
    cur = 0
    for li, conv in enumerate(model.seq_branch.convs):
        K = conv.kernel_size[0]; C = conv.out_channels
        for c in range(C):
            name_to_lc[f"kernel_k{K}_f{c}"] = (li, c, K)
        cur += C

    # Top-K 的 (layer -> channels) map
    selected_map = {}
    for kname in topk_names:
        li, c, K = name_to_lc[kname]
        selected_map.setdefault(li, []).append(c)

    # 选择用于收集窗口的样本索引集合
    if args.pwm_from == "train":
        subset = np.array(tr_idx, dtype=np.int64)
    else:
        subset = np.arange(M, dtype=np.int64)

    # 收集每个选中 kernel 的 TopK 激活窗口（GPU 合并）
    hits = collect_top_hits_selected_kernels(
        device=device,
        conv_modules=list(model.seq_branch.convs),
        X_seq_all=X_seq,
        selected_map=selected_map,
        subset_indices=subset,
        topk=args.topk_per_kernel,
        act_relu=True,
        batch_size=args.batch_size
    )

    pwm_bank = {}
    for kname in topk_names:
        li, c, K = name_to_lc[kname]
        key = (li, c)
        arr = hits.get(key, [])
        if len(arr) < args.min_hits:
            continue
        # arr: [(score, sample_idx, start_pos), ...]
        idx_list = [(sid, pos) for (_, sid, pos) in arr]
        windows = extract_subseq_windows(X_seq, idx_list, kernel_size=K)  # [N,4,K]
        if args.weight_mode == "activation":
            w = np.array([s for (s,_,_) in arr], dtype=np.float64)
        else:
            w = None  # 均匀
        pfm, pwm = onehot_windows_to_pfm(windows, weights=w)
        # 保存
        pfm_path = outdir / "motifs" / f"{kname}.pfm.tsv"
        pwm_path = outdir / "motifs" / f"{kname}.pwm.tsv"
        np.savetxt(pfm_path, pfm, fmt="%.6f", delimiter="\t")
        np.savetxt(pwm_path, pwm, fmt="%.6f", delimiter="\t")
        pwm_bank[kname] = pwm

    if args.save_meme and len(pwm_bank) > 0:
        save_meme(outdir / "motifs" / "meme.txt", pwm_bank)

    print(f"[OK] n_samples={M}, n_kernels={nK}, top_k={topK}")
    print(f"[OK] Saved to: {outdir}")
    if len(pwm_bank) == 0:
        print(f"[WARN] Top-K kernels 未达到 min_hits={args.min_hits} 的数量阈值，未生成 MEME/部分 PWM。")

if __name__ == "__main__":
    main()
