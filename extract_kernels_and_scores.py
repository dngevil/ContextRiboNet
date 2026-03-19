#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_kernels_and_scores.py

基于 train_single_label_cnn_mlp.py 的模型结构：
- 提取 SeqCNNBranch 的 Conv1d kernels
- 在训练集上从卷积激活图中挖掘每个 kernel 的高分窗口，生成 motif (PFM/PWM/MEME)
- 为（train+val+test）的每条序列计算每个 kernel 的最大卷积分数（ReLU 后的 max activation）
- 输出：
  outdir/
    motifs/
      kernel_<k>_f<j>.pfm.tsv         # 4 x K，A/C/G/T 频数
      kernel_<k>_f<j>.pwm.tsv         # 4 x K，概率矩阵（行和=1）
    meme.txt                          # (可选) MEME 格式汇总
    kernel_scores.npz                 # X: [M, num_kernels]，splits等元数据
    kernel_scores.csv                 # 同上（如需表格, 可大）
    meta.json                         # 辅助信息：kernel列表、K值、通道数、拆分模式等

依赖：与训练脚本一致的 PyTorch/Numpy 等。
注意：需要将 --kernel_sizes、--cnn_channels 与训练时一致，否则无法正确加载 ckpt。

用法示例：
python extract_kernels_and_scores.py \
  --npz data/packed_inputs.npz \
  --ckpt runs/exp1/best.pt \
  --outdir runs/exp1/kernels \
  --split_mode per_gene \
  --test_ratio_per_stage 0.2 \
  --val_ratio_in_train 0.2 \
  --seed 42 \
  --kernel_sizes 6 10 12 16 20 \
  --cnn_channels 64 \
  --mlp_hidden 128 --mlp_blocks 3 --dropout 0.1 \
  --topk_per_kernel 500 --min_hits 100 --save_meme
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
from collections import defaultdict
import heapq

# =========================
# 载入与拆分（与训练脚本一致）
# =========================
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
            st = str(n)[4:]
            rna_idx_map[st] = i
    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise SystemExit(f"缺少 RNA 特征列: {missing}。请确保特征包含 rna_<stage> 列。")

    base_feats = X_other[:, base_idx]  # [N, l_base]
    l_base = base_feats.shape[1]

    X_seq_exp = np.repeat(X_seq, repeats=S, axis=0)  # [N*S, L, 4]
    X_other_list, Y_single, stage_ids, gene_ids = [], [], [], []
    for j, st in enumerate(stages):
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col][:, None]  # [N,1]
        other_j = np.hstack([base_feats, rna_vec])
        X_other_list.append(other_j)
        Y_single.append(Y[:, [j]])
        stage_ids.append(np.full((N, 1), j, dtype=np.int64))
        if genes:
            gene_ids.append(np.array(genes).reshape(-1,1))
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
    if gene_ids is None:
        raise SystemExit("split_by_gene 需要 meta.json 中的 gene 列表")
    rng = np.random.default_rng(seed)
    gene_ids = np.asarray(gene_ids)
    uniq_genes = np.unique(gene_ids)
    rng.shuffle(uniq_genes)
    n_test_genes = max(1, int(len(uniq_genes) * test_ratio))
    test_genes = set(uniq_genes[:n_test_genes])
    train_pool_genes = set(uniq_genes[n_test_genes:])
    n_val_genes = max(1, int(len(train_pool_genes) * val_ratio))
    train_pool_genes = np.array(list(train_pool_genes))
    rng.shuffle(train_pool_genes)
    val_genes = set(train_pool_genes[:n_val_genes])
    train_genes = set(train_pool_genes[n_val_genes:])
    idx_all = np.arange(len(gene_ids))
    test_idx  = idx_all[np.isin(gene_ids, list(test_genes))]
    val_idx   = idx_all[np.isin(gene_ids, list(val_genes))]
    train_idx = idx_all[np.isin(gene_ids, list(train_genes))]
    return train_idx, val_idx, test_idx

def stratified_split_per_stage(stage_ids, test_ratio=0.2, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(stage_ids))
    stage_ids = np.asarray(stage_ids)
    train_pool, test_idx = [], []
    for s in np.unique(stage_ids):
        s_idx = idx_all[stage_ids == s]
        rng.shuffle(s_idx)
        n_test = max(1, int(len(s_idx) * test_ratio))
        test_idx.append(s_idx[:n_test])
        train_pool.append(s_idx[n_test:])
    test_idx = np.concatenate(test_idx, axis=0)
    train_pool = np.concatenate(train_pool, axis=0)
    val_idx, train_idx = [], []
    for s in np.unique(stage_ids):
        pool_s = train_pool[stage_ids[train_pool] == s]
        rng.shuffle(pool_s)
        n_val = max(1, int(len(pool_s) * val_ratio))
        val_idx.append(pool_s[:n_val])
        train_idx.append(pool_s[n_val:])
    val_idx = np.concatenate(val_idx, axis=0)
    train_idx = np.concatenate(train_idx, axis=0)
    return train_idx, val_idx, test_idx

# =========================
# 模型结构（与训练脚本一致）
# =========================
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
        for blk in self.blocks:
            x = self.act(blk(x) + x)
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

# =========================
# 核心：提取 motif & 打分
# =========================
def onehot_windows_to_pfm(onehots, weights=None, eps=1e-8):
    """
    onehots: [N, K, 4] 或 [N, 4, K]
    weights: [N] 可选（按激活强度加权）
    返回 PFM (counts, 4 x K) 与 PWM (probs, 4 x K)
    """
    if onehots.ndim != 3:
        raise ValueError("onehots must be 3D")
    if onehots.shape[1] == 4:  # [N,4,K] -> [N,K,4]
        onehots = np.transpose(onehots, (0,2,1))
    N, K, _ = onehots.shape
    if weights is None:
        weights = np.ones((N,), dtype=np.float64)
    weights = weights.astype(np.float64)
    # 计数：对每个位置 K，累积 A/C/G/T
    pfm = np.zeros((4, K), dtype=np.float64)
    # 展平做矩阵乘： (N,K,4) -> (N*K,4)
    X = onehots.reshape(N*K, 4).astype(np.float64)
    w = np.repeat(weights, K)
    # 加权计数
    pfm += (X * w[:, None]).reshape(N, K, 4).sum(axis=0).T  # -> 4 x K
    colsum = pfm.sum(axis=0, keepdims=True) + eps
    pwm = pfm / colsum
    return pfm, pwm

@torch.no_grad()
def collect_top_hits_for_kernel(device, conv_module, seq_loader, topk=500, act_relu=True):
    """
    对单个 Conv1d（C_out, 4, K）：在训练集上为其每个输出通道收集 topk 激活窗口。
    返回 hits: dict[channel] -> list of (score, sample_idx, start_pos)
    注：sample_idx 是相对于“该 loader 拼接顺序”的全局样本索引。
    """
    conv = conv_module.to(device)
    conv.eval()
    hits = [ [] for _ in range(conv.out_channels) ]  # 每个通道一个最大堆(min-heap, 存负号做最大堆)
    total_seen = 0
    for xs, _, _ in seq_loader:  # xs: [B, L, 4]
        xs = xs.to(device)                    # [B, L, 4]
        x = xs.transpose(1,2).contiguous()    # [B, 4, L]
        z = F.conv1d(x, conv.weight, conv.bias, stride=1, padding=0)  # [B, C, L']
        if act_relu:
            z = F.relu(z)
        B, C, Lp = z.shape
        # 对每个通道，拿到其 topk 候选（分批保留）
        # 我们将每个样本的位置展开：score, global_sample_idx, start_pos
        for c in range(C):
            # z[:, c, :] -> [B, Lp]
            scores = z[:, c, :].reshape(B, Lp)
            # 取每个样本的 topk 小堆合并（避免全局 topk 带来巨大的张量）
            local_candidates = []
            k_each = min(topk, Lp)
            top_vals, top_pos = torch.topk(scores, k=k_each, dim=1)  # [B, k_each]
            for b in range(B):
                gidx = total_seen + b
                for j in range(k_each):
                    s = float(top_vals[b, j].item())
                    p = int(top_pos[b, j].item())
                    # 使用最小堆保存 k 个最大项
                    heap = hits[c]
                    if len(heap) < topk:
                        heapq.heappush(heap, (s, gidx, p))
                    else:
                        if s > heap[0][0]:
                            heapq.heapreplace(heap, (s, gidx, p))
        total_seen += B
    # 转换每个通道的堆到按分数降序的列表
    out = []
    for c in range(conv.out_channels):
        arr = sorted(hits[c], key=lambda x: -x[0])
        out.append(arr)
    return out  # list of list[(score, sample_idx, start_pos)]

def extract_subseq_windows(X_seq_all, idx_list, kernel_size):
    """
    X_seq_all: numpy [M, L, 4] one-hot
    idx_list : list of (sample_idx, start_pos)
    返回 [N, 4, K] one-hot 片段
    """
    K = kernel_size
    out = np.zeros((len(idx_list), 4, K), dtype=np.float32)
    for i, (sid, p) in enumerate(idx_list):
        # X_seq_all 是 [M, L, 4]，窗口为 [p : p+K]
        win = X_seq_all[sid, p:p+K, :]  # [K, 4]
        if win.shape[0] != K:
            # 边界保护（理论上不会发生，因为 p 来源于卷积输出）
            continue
        out[i] = np.transpose(win, (1,0))  # -> [4, K]
    return out

@torch.no_grad()
def score_all_sequences(device, conv_modules, X_seq_all, act_relu=True, batch_size=512):
    """
    对全部样本，计算每个 kernel 的最大激活分数（ReLU 后 max）。
    conv_modules: list of (conv, kernel_size, local_channel_idx)
      我们将所有 conv 层展平成“kernel 列表”，每个输出通道视作一个 kernel
    返回：scores [M, num_kernels]
    """
    M, L, _ = X_seq_all.shape
    num_kernels = sum(conv.out_channels for conv in conv_modules)
    # 但 conv_modules 是每层；我们需要一个“平铺的 kernel 视图”
    flat_specs = []
    for conv in conv_modules:
        C, K = conv.out_channels, conv.kernel_size[0]
        for c in range(C):
            flat_specs.append((conv, c, K))
    scores = np.full((M, len(flat_specs)), fill_value=0.0, dtype=np.float32)

    # 为效率，按批跑所有 conv 层，然后在 Python 收集每个 kernel 的 max
    device_convs = [conv.to(device).eval() for conv in conv_modules]
    n_batches = (M + batch_size - 1) // batch_size
    for bi in range(n_batches):
        s = bi*batch_size; e = min(M, s+batch_size)
        xs = torch.from_numpy(X_seq_all[s:e]).to(device)  # [B, L, 4]
        x = xs.transpose(1,2).contiguous()                # [B, 4, L]
        col = 0
        for conv in device_convs:
            z = F.conv1d(x, conv.weight, conv.bias, stride=1, padding=0)  # [B, C, L']
            if act_relu:
                z = F.relu(z)
            zmax = torch.amax(z, dim=-1)  # [B, C]
            B, C = zmax.shape
            scores[s:e, col:col+C] = zmax.detach().cpu().numpy()
            col += C
    return scores, flat_specs  # flat_specs 对齐列

def save_meme(meme_path, pwm_dict):
    """
    pwm_dict: {name: (pwm: 4xK)} 行序 A,C,G,T
    写 MEME v4 文本（不严格含背景/字母表简化）
    """
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True, help="训练保存的 best.pt")
    ap.add_argument("--outdir", required=True)
    # 与训练时一致的结构参数（用于构建模型加载 ckpt）
    ap.add_argument("--cnn_channels", type=int, default=64)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--kernel_sizes", type=int, nargs="+", default=[6,10,12,16,20])
    # 拆分一致性
    ap.add_argument("--split_mode", choices=["per_sample","per_gene"], default="per_gene")
    ap.add_argument("--test_ratio_per_stage", type=float, default=0.2)
    ap.add_argument("--val_ratio_in_train", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # motif 收集与输出
    ap.add_argument("--topk_per_kernel", type=int, default=500, help="每个 kernel 收集的高分窗口数上限")
    ap.add_argument("--min_hits", type=int, default=100, help="生成 motif 的最小窗口数阈值")
    ap.add_argument("--save_meme", action="store_true", help="导出 MEME 文件")
    ap.add_argument("--batch_size", type=int, default=512, help="打分批大小")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "motifs").mkdir(exist_ok=True, parents=True)

    # 载入数据并展开
    D = load_and_expand(args.npz)
    X_seq = D["X_seq"]  # [M, L, 4]
    stage_ids = np.asarray(D["stage_ids"])
    stages = D["stages"]
    gene_ids = D["gene_ids"]
    other_in_dim = D["l_base"] + 1  # 与训练时一致：base + 当前阶段RNA

    # 拆分（复现训练）
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

    # 构建模型并加载权重
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

    # 1) 在训练集上为每个 Conv 层的每个通道收集 top hits
    #    我们只需要 X_seq（one-hot），不需要 X_other/Y
    #    这里单独做一个只含序列的 DataLoader（numpy -> torch, 但不使用真正的 DataLoader 以减少样板）
    def _iter_batches(indices, batch=512):
        for i in range(0, len(indices), batch):
            b = indices[i:i+batch]
            xs = torch.from_numpy(X_seq[b])  # [B, L, 4]
            # dummy xo, y 以复用接口
            xo = torch.zeros((len(b), 1), dtype=torch.float32)
            y  = torch.zeros((len(b), 1), dtype=torch.float32)
            yield xs, xo, y

    # 收集 motif 窗口
    convs = list(model.seq_branch.convs)  # 不同 kernel size 的多层
    # 我们要将“训练集的索引连续化”，collect 函数里会按 batch 顺序累加 global idx
    train_seq_loader = list(_iter_batches(tr_idx, batch=512))
    all_hits = []  # 与 convs 等长：每个元素是 list[channel] = [(score, sample_gidx, start_pos), ...]
    for li, conv in enumerate(convs):
        hits_per_channel = collect_top_hits_for_kernel(
            device=device, conv_module=conv, seq_loader=train_seq_loader,
            topk=args.topk_per_kernel, act_relu=True
        )
        all_hits.append(hits_per_channel)

    # 2) 生成 motif（PFM、PWM、MEME）
    pwm_bank = {}  # name -> pwm(4 x K)
    for li, conv in enumerate(convs):
        K = conv.kernel_size[0]
        C = conv.out_channels
        for c in range(C):
            hits = all_hits[li][c]  # [(score, gidx, pos), ...] 降序
            if len(hits) < args.min_hits:
                continue
            # 映射回原始样本索引：gidx 是“训练批中的顺序”
            # 我们将 train_idx 排序后，train_global[gidx] -> 原始 M 维索引
            train_global = np.array(tr_idx, dtype=np.int64)
            idx_list = [( int(train_global[gidx]), int(pos) ) for (score, gidx, pos) in hits]
            # 提取窗口 one-hot 并按 score 加权汇总
            windows = extract_subseq_windows(X_seq, idx_list, kernel_size=K)  # [N,4,K]
            scores  = np.array([s for (s,_,_) in hits], dtype=np.float64)
            pfm, pwm = onehot_windows_to_pfm(windows, weights=scores)
            name = f"kernel_k{K}_f{c}"
            # 保存到文件
            pfm_path = outdir / "motifs" / f"{name}.pfm.tsv"
            pwm_path = outdir / "motifs" / f"{name}.pwm.tsv"
            np.savetxt(pfm_path, pfm, fmt="%.6f", delimiter="\t")
            np.savetxt(pwm_path, pwm, fmt="%.6f", delimiter="\t")
            pwm_bank[name] = pwm
    if args.save_meme and len(pwm_bank) > 0:
        save_meme(outdir / "meme.txt", pwm_bank)

    # 3) 为全部（train/val/test）样本计算每个 kernel 的最大分数
    scores, flat_specs = score_all_sequences(
        device=device, conv_modules=convs, X_seq_all=X_seq, act_relu=True, batch_size=args.batch_size
    )
    # flat_specs 是以层为单位展开后的 kernel 列表：[(conv, channel, K), ...]
    kernel_names = []
    for conv in convs:
        K = conv.kernel_size[0]
        for c in range(conv.out_channels):
            kernel_names.append(f"kernel_k{K}_f{c}")
    assert scores.shape[1] == len(kernel_names)

    # 4) 保存数据集
    # 4.1 npz（紧凑）+ meta
    meta = {
        "kernel_names": kernel_names,
        "kernel_sizes": [int(n.split("_k")[1].split("_")[0]) for n in kernel_names],
        "split_mode": args.split_mode,
        "test_ratio_per_stage": args.test_ratio_per_stage,
        "val_ratio_in_train": args.val_ratio_in_train,
        "seed": args.seed,
        "stages": stages,
    }
    np.savez_compressed(outdir / "kernel_scores.npz",
                        X=scores.astype(np.float32),
                        split=split_tag,
                        stage_ids=stage_ids.astype(np.int64))
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 4.2 CSV（若体积可接受）
    try:
        df = pd.DataFrame(scores, columns=kernel_names)
        df.insert(0, "split", split_tag)
        df.insert(1, "stage_id", stage_ids)
        if gene_ids is not None:
            df.insert(2, "gene", gene_ids)
        df.to_csv(outdir / "kernel_scores.csv", index=False)
    except Exception as e:
        print(f"[warn] 保存 CSV 失败：{e}", file=sys.stderr)

    print(f"[OK] Motifs written to: {outdir/'motifs'}")
    if args.save_meme:
        print(f"[OK] MEME: {outdir/'meme.txt'}")
    print(f"[OK] Kernel score dataset: {outdir/'kernel_scores.npz'} (and maybe CSV)")
    print(f"[n_samples={M}, n_kernels={scores.shape[1]}]")

if __name__ == "__main__":
    main()
