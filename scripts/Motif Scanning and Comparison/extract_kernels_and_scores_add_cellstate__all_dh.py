#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_kernels_and_scores.py  (ALL-IN-ONE, 2025-11-01)

升级要点
--------
1) 数据读取/展开：
   - 读取 .npz + 同名 .json(meta)
   - 直接构造展开后的:
       X_seq_exp: [N*S, L, 4]
       Y_exp    : [N*S, 1]
       stage_ids: [N*S]
       gene_ids : [N*S] (如 meta 有 genes)
       RNA_exp  : [N*S]  (来自 meta.other_feature_names 的各阶段 rna_<stage> 列)
   - 无需依赖展开后的 X_other；更稳健，也避免与训练时是否加入全局状态列（G）产生耦合

2) 模型构建/加载：
   - 从 ckpt 的 state_dict **自动推断** other_in_dim = weight("other_branch.input.weight").shape[1]
     -> 无惧你训练时对 X_other 维度（base+rna+globals）的改动
   - 对 seq 分支，仍以命令行的 --kernel_sizes / --cnn_channels 为准，并做形状一致性校验

3) Kernel motif 挖掘与打分：
   - 训练划分可复现 (per_gene / per_sample)
   - 按 conv 通道收集 topK 激活窗口 -> 导出 PFM/PWM (tsv) + 可选 MEME
   - 计算全样本的每 kernel 最大激活，保存 npz/csv

4) “在哪个阶段起作用”分析（核心新增）：
   - 活跃度差异：Kruskal–Wallis (跨阶段)
   - 阶段特异：one-vs-rest 的 AUC 近似 + mean_z_vs_all
   - 相关性：阶段内 Spearman ρ(kernel, Ribo)
   - **Partial**：阶段内对 Ribo 先做 y ~ RNA 线性回归残差，再与 kernel 分数 Spearman ρ
   - 结果保存为 kernel_stage_activity.csv，并绘制热图 kernel_stage_activity_heatmap.png

用法示例
--------
python extract_kernels_and_scores_add_cellstate.py \
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

import argparse, json, math, sys, heapq, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1) 载入与展开
# =========================
def _read_packed(npz_path: str):
    meta_path = Path(npz_path).with_suffix(".json")
    if not meta_path.exists():
        raise SystemExit(f"[FATAL] meta json 不存在: {meta_path}")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    pack = np.load(npz_path)
    X_seq = pack["X_seq"].astype(np.float32)      # [N, L, 4]
    X_other = pack["X_other"].astype(np.float32)  # [N, l_full]
    Y = pack["Y"].astype(np.float32)              # [N, S]
    return X_seq, X_other, Y, meta


def _expand_to_single_label(X_seq, X_other, Y, meta) -> Dict[str, np.ndarray]:
    """
    输出:
      X_seq_exp: [N*S, L, 4]
      Y_exp    : [N*S, 1]
      stage_ids: [N*S]
      gene_ids : [N*S]或None
      stages   : list[str]长度 S
      RNA_exp  : [N*S]  (展开后的 "本阶段 RNA" 标量)
      L, l_base
    """
    N, L, _ = X_seq.shape
    S = Y.shape[1]

    stages = meta.get("stages")
    if not (isinstance(stages, list) and len(stages) == S):
        raise SystemExit("meta['stages'] 与 Y 第二维不一致。")

    feat_names = meta.get("other_feature_names", [])
    if not (isinstance(feat_names, list) and len(feat_names) == X_other.shape[1]):
        raise SystemExit("meta['other_feature_names'] 与 X_other 列数不一致。")

    genes = meta.get("genes")
    genes = list(map(str, genes)) if isinstance(genes, list) else None

    # 定位 base 与 rna_<stage> 列
    base_idx = [i for i, n in enumerate(feat_names) if not str(n).startswith("rna_")]
    rna_idx_map = {}
    for i, n in enumerate(feat_names):
        if str(n).startswith("rna_"):
            st = str(n)[4:]
            rna_idx_map[st] = i
    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise SystemExit(f"缺少 RNA 特征列: {missing}。请确保生成了 rna_<stage> 列。")

    base_feats = X_other[:, base_idx]  # [N, l_base]
    l_base = base_feats.shape[1]

    # 展开
    X_seq_exp = np.repeat(X_seq, repeats=S, axis=0)  # [N*S, L, 4]
    Y_exp_list, stage_ids_list, gene_ids_list, RNA_exp_list = [], [], [], []

    for j, st in enumerate(stages):
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col]                 # [N]
        Y_exp_list.append(Y[:, [j]])                  # [N,1]
        stage_ids_list.append(np.full((N, 1), j, dtype=np.int64))
        RNA_exp_list.append(rna_vec.reshape(-1, 1))   # [N,1]
        if genes is not None:
            gene_ids_list.append(np.array(genes).reshape(-1, 1))

    Y_exp = np.vstack(Y_exp_list).astype(np.float32)                # [N*S,1]
    stage_ids = np.vstack(stage_ids_list).astype(np.int64).ravel()  # [N*S]
    RNA_exp = np.vstack(RNA_exp_list).astype(np.float32).ravel()    # [N*S]
    gene_ids = np.vstack(gene_ids_list).ravel().tolist() if genes is not None else None

    return dict(
        X_seq=X_seq_exp, Y=Y_exp, stage_ids=stage_ids, RNA=RNA_exp,
        stages=stages, gene_ids=gene_ids, L=L, l_base=l_base
    )


def _stage_global_features_from_rna(rna_vec: np.ndarray, topk_list=(50, 100)) -> dict:
    """
    rna_vec: [N]，该阶段所有基因的 RNA（TPM或已log1p，请与构建时一致）。
    仅用 RNA 计算，不会泄漏 Ribo。
    """
    x = np.asarray(rna_vec, dtype=np.float64)
    x = np.clip(x, 0.0, None)  # 稳健
    tot = float(x.sum()) + 1e-12
    p = x / tot

    # 规模
    log1p_total_tpm = float(np.log1p(tot))

    # 分布形状
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    hhi = float((p ** 2).sum())
    effective_genes = float(1.0 / (hhi + 1e-12))

    # 头部占比
    x_sorted = np.sort(x)[::-1]
    top_shares = {}
    for k in topk_list:
        k_eff = min(k, len(x_sorted))
        top_shares[f"top{k}_share"] = float(x_sorted[:k_eff].sum() / tot)

    return {
        "log1p_total_tpm": log1p_total_tpm,
        "entropy": entropy,
        "hhi": hhi,
        "effective_genes": effective_genes,
        **top_shares
    }

def _zscore_over_stages(stage_feat_table: dict) -> dict:
    """
    输入: {feat_name: [S 阶段的值]}
    输出: {feat_name: z-scored [S]}，按阶段维度标准化，便于跨阶段可比。
    """
    out = {}
    for k, arr in stage_feat_table.items():
        v = np.asarray(arr, dtype=np.float64)
        mu, sd = float(v.mean()), float(v.std() + 1e-12)
        out[k] = ((v - mu) / sd).astype(np.float32)
    return out

# ===== 替换你的函数：完善“阶段上下文”拼接逻辑 =====
def load_and_expand(npz_path):
    """
    读取由 build_model_inputs.py 生成的 .npz/.json，
    展开为 (N*S) 的单标签样本，并额外在 X_other 末尾加入
    “阶段级全局RNA特征”（按阶段z-score后的标量，复制到该阶段所有样本）。
    """
    meta_path = Path(npz_path).with_suffix(".json")
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}
    pack = np.load(npz_path)
    X_seq = pack["X_seq"].astype(np.float32)      # [N, L, 4]
    X_other = pack["X_other"].astype(np.float32)  # [N, l_full]
    Y = pack["Y"].astype(np.float32)              # [N, S]
    N, L, _ = X_seq.shape
    S = Y.shape[1]

    # ---- meta 基本字段 ----
    genes = meta.get("genes")
    stages = meta.get("stages")
    feat_names = meta.get("other_feature_names", [])
    rna_as_features = bool(meta.get("rna_as_features", False))

    assert stages is not None and isinstance(stages, list) and len(stages) == S, \
        "meta.json 需要包含 stages 列表，且长度与 Y 的第二维一致。"
    assert isinstance(feat_names, list) and len(feat_names) == X_other.shape[1], \
        "meta.other_feature_names 与 X_other 列数不一致。"

    # ---- 解析 RNA 列并检查 ----
    base_idx = [i for i, n in enumerate(feat_names) if not str(n).startswith("rna_")]
    rna_idx_map = {}  # stage -> col index
    for i, n in enumerate(feat_names):
        if str(n).startswith("rna_"):
            st = str(n)[4:]
            rna_idx_map[st] = i

    if not rna_as_features or len(rna_idx_map) == 0:
        raise SystemExit(
            "未找到 RNA 作为特征：meta.rna_as_features=False 或 other_feature_names 中无任何 'rna_<stage>' 列。"
            "请用 build_model_inputs.py 提供 --rna_path 生成包含 RNA 列的特征。"
        )
    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise SystemExit(f"缺少 RNA 特征列: {missing}。请确保所有阶段都有 rna_<stage> 列。")

    # ---- 基础（与阶段无关的）基因级特征 ----
    base_feats = X_other[:, base_idx]  # [N, l_base]
    l_base = base_feats.shape[1]

    # ---- 先基于每个阶段的 RNA 计算“阶段全局特征” ----
    #     注意：这里完全不使用 Ribo，避免任何目标泄漏。
    stage_glob_raw = {}  # {feat_name: [S]}
    for st in stages:
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col]  # [N]
        feats = _stage_global_features_from_rna(rna_vec, topk_list=(50, 100))
        for k, v in feats.items():
            stage_glob_raw.setdefault(k, []).append(v)

    # ---- 在阶段维度做 z-score，得到每个特征每个阶段的值 ----
    stage_glob_z = _zscore_over_stages(stage_glob_raw)  # {feat: [S]}
    added_glob_names = [f"g_{k}" for k in stage_glob_z.keys()]  # 记录新增列名（带前缀 g_）

    # ---- 展开到 (N*S) ----
    X_seq_exp = np.repeat(X_seq, repeats=S, axis=0)  # [N*S, L, 4]
    X_other_list = []
    Y_single = []
    stage_ids = []
    gene_ids = []

    for j, st in enumerate(stages):
        # 当前阶段的“本基因 RNA”列
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col][:, None].astype(np.float32)  # [N, 1]

        # 当前阶段的“全局上下文”列（复制到 N 行）
        glob_cols = []
        for feat_name, arr_S in stage_glob_z.items():
            val = np.full((N, 1), arr_S[j], dtype=np.float32)  # [N,1]
            glob_cols.append(val)
        glob_mat = np.hstack(glob_cols) if len(glob_cols) > 0 else None  # [N, G]

        # 拼接：基因级基础特征 + 本阶段RNA + 阶段全局特征
        if glob_mat is not None:
            other_j = np.hstack([base_feats, rna_vec, glob_mat])  # [N, l_base+1+G]
        else:
            other_j = np.hstack([base_feats, rna_vec])

        X_other_list.append(other_j)
        Y_single.append(Y[:, [j]])  # [N,1]
        stage_ids.append(np.full((N, 1), j, dtype=np.int64))
        if genes:
            gene_ids.append(np.array(genes).reshape(-1, 1))

    X_other_exp = np.vstack(X_other_list).astype(np.float32)     # [N*S, l_base+1+G]
    Y_exp       = np.vstack(Y_single).astype(np.float32)         # [N*S, 1]
    stage_ids   = np.vstack(stage_ids).astype(np.int64).reshape(-1)
    gene_ids    = np.vstack(gene_ids).reshape(-1).tolist() if genes else None

    # ---- 返回时把新增特征名也带出去（便于写入 summary.json）----
    # 原始 other_feature_names 展开后的顺序：按阶段重复 [base_feats + rna_<st> (+ g_*)]
    # 若你需要精确的列名顺序，可在外层使用 added_glob_names 重建。
    return {
        "X_seq": X_seq_exp,
        "X_other": X_other_exp,
        "Y": Y_exp,
        "stage_ids": stage_ids,
        "stages": stages,
        "gene_ids": gene_ids,
        "L": L,
        "l_base": l_base,
        "added_global_feature_names": added_glob_names  # 例如: ["g_log1p_total_tpm","g_entropy",...]
    }



# =========================
# 2) 拆分
# =========================
def split_by_gene(stage_ids, gene_ids, test_ratio=0.2, val_ratio=0.2, seed=42):
    if gene_ids is None:
        raise SystemExit("split_by_gene 需要 meta['genes']。")
    rng = np.random.default_rng(seed)
    gene_ids = np.asarray(gene_ids)
    uniq_genes = np.unique(gene_ids)
    rng.shuffle(uniq_genes)

    n_test_genes = max(1, int(len(uniq_genes) * test_ratio))
    test_genes = set(uniq_genes[:n_test_genes])
    pool = np.array(list(set(uniq_genes[n_test_genes:])))
    rng.shuffle(pool)
    n_val_genes = max(1, int(len(pool) * val_ratio))
    val_genes = set(pool[:n_val_genes])
    train_genes = set(pool[n_val_genes:])

    idx_all = np.arange(len(gene_ids))
    te_idx = idx_all[np.isin(gene_ids, list(test_genes))]
    va_idx = idx_all[np.isin(gene_ids, list(val_genes))]
    tr_idx = idx_all[np.isin(gene_ids, list(train_genes))]
    return tr_idx, va_idx, te_idx


def stratified_split_per_stage(stage_ids, test_ratio=0.2, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(stage_ids))
    stage_ids = np.asarray(stage_ids)
    test_idx_list, pool_list = [], []
    for s in np.unique(stage_ids):
        s_idx = idx_all[stage_ids == s]
        rng.shuffle(s_idx)
        n_test = max(1, int(len(s_idx) * test_ratio))
        test_idx_list.append(s_idx[:n_test])
        pool_list.append(s_idx[n_test:])
    te_idx = np.concatenate(test_idx_list, axis=0)
    pool = np.concatenate(pool_list, axis=0)

    va_idx_list, tr_idx_list = [], []
    for s in np.unique(stage_ids):
        s_pool = pool[stage_ids[pool] == s]
        rng.shuffle(s_pool)
        n_val = max(1, int(len(s_pool) * val_ratio))
        va_idx_list.append(s_pool[:n_val])
        tr_idx_list.append(s_pool[n_val:])
    va_idx = np.concatenate(va_idx_list, axis=0)
    tr_idx = np.concatenate(tr_idx_list, axis=0)
    return tr_idx, va_idx, te_idx


# =========================
# 3) 模型结构（与训练脚本一致）
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
    def __init__(self, other_in_dim, l_base, cnn_channels=64, mlp_hidden=128, mlp_blocks=3,
                 dropout=0.1, kernel_sizes=(6,10,12,16,20), activation=nn.ReLU,
                 fusion: str = "film", cond_hidden: int = 128):
        """
        other_in_dim = l_base + 1 + G   (#base + rna + globals)
        l_base: 基因级基础特征数量，用于从 X_other 切分出 base / rna / globals
        fusion: "concat" | "film" | "gate"
        """
        super().__init__()
        assert fusion in ("concat","film","gate")
        self.fusion_mode = fusion
        self.l_base = int(l_base)

        # 分支
        self.seq_branch = SeqCNNBranch(4, kernel_sizes, cnn_channels, activation)
        self.other_branch = ResidualMLP(l_base + 1, mlp_hidden, mlp_blocks, dropout, activation)

        # 计算 globals 维度（可能为 0）
        self.other_in_dim = int(other_in_dim)
        self.g_dim = max(0, self.other_in_dim - (self.l_base + 1))

        # 条件化模块（当 g_dim>0 且使用 film/gate 时启用）
        self.state_proj = None
        self.gamma_beta = None
        self.gate_head = None

        if self.g_dim > 0:
            if self.fusion_mode == "film":
                # 用状态生成与 seq 通道同维的 (gamma, beta)
                self.state_proj = nn.Sequential(
                    nn.Linear(self.g_dim, cond_hidden), activation(), nn.Dropout(dropout),
                    nn.Linear(cond_hidden, 2 * self.seq_branch.out_dim)
                )
                # 初始化使训练初期接近恒等映射：gamma≈0, beta≈0
                nn.init.zeros_(self.state_proj[-1].weight)
                nn.init.zeros_(self.state_proj[-1].bias)

            elif self.fusion_mode == "gate":
                # 生成通道门（也可用单标量门，把 out_dim 改为 1）
                self.gate_head = nn.Sequential(
                    nn.Linear(self.g_dim, cond_hidden), activation(), nn.Dropout(dropout),
                    nn.Linear(cond_hidden, self.seq_branch.out_dim),
                    nn.Sigmoid()
                )

        # 融合头
        fusion_in = self.seq_branch.out_dim + mlp_hidden
        # 也可以把 globals 再投一层拼上（可选，默认不开）
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, max(mlp_hidden,128)), activation(), nn.Dropout(dropout),
            nn.Linear(max(mlp_hidden,128), 1)
        )

    def forward(self, x_seq, x_other):
        """
        x_other = [base (l_base), rna (1), globals (g_dim)]
        """
        # 切分 other
        x_base = x_other[:, :self.l_base]
        x_rna  = x_other[:, self.l_base:self.l_base+1]
        x_state = x_other[:, self.l_base+1:] if self.g_dim > 0 else None

        # 两路编码
        h_seq = self.seq_branch(x_seq)                          # [B, C_seq]
        h_other = self.other_branch(torch.cat([x_base, x_rna], dim=1))  # [B, mlp_hidden]

        # 条件化序列分支
        if self.g_dim > 0:
            if self.fusion_mode == "film":
                gb = self.state_proj(x_state)                   # [B, 2*C_seq]
                gamma, beta = gb.chunk(2, dim=1)
                # FiLM： (1+gamma)*h + beta
                h_seq = (1.0 + gamma) * h_seq + beta
            elif self.fusion_mode == "gate":
                gate = self.gate_head(x_state)                  # [B, C_seq] (或 [B,1])
                h_seq = gate * h_seq
            # concat 模式：不做条件化

        # 融合
        h = torch.cat([h_seq, h_other], dim=1)
        return self.fusion(h)


@torch.no_grad()
def score_all_sequences_conditioned(
    device,
    model,              # CNN_MLP_Fusion
    X_seq_all: np.ndarray,   # [M, L, 4]
    X_other_all: np.ndarray, # [M, other_in_dim] = [base, rna, globals]
    batch_size: int = 512
):
    """
    返回:
      scores_state: [M, n_kernels]  # 每样本、每通道的“状态作用后”的通道响应
      kernel_names: list[str]       # 与列对齐
    说明:
      - 先用 seq_branch 做 conv+ReLU+global max-pool 得到 h_seq
      - 再按 fusion 应用 FiLM 或 gate 得到 h_seq_cond
      - 不经过 other-branch / fusion 头的线性层，只取“状态作用后的序列通道向量”
    """
    import time
    t0 = time.perf_counter()

    model.eval()
    convs = list(model.seq_branch.convs)
    n_kernels = model.seq_branch.out_dim
    M = X_seq_all.shape[0]
    scores_state = np.empty((M, n_kernels), dtype=np.float32)

    # 预构 kernel 名
    kernel_names = []
    for conv in convs:
        K = conv.kernel_size[0]
        for c in range(conv.out_channels):
            kernel_names.append(f"kernel_k{K}_f{c}")

    n_batches = (M + batch_size - 1) // batch_size
    print(f"[score_state] M={M}, n_kernels={n_kernels}, batches={n_batches}, batch={batch_size}", flush=True)

    for bi in range(n_batches):
        s = bi * batch_size
        e = min(M, s + batch_size)
        xs = torch.from_numpy(X_seq_all[s:e]).to(device)              # [B, L, 4]
        xo = torch.from_numpy(X_other_all[s:e]).to(device)            # [B, other_in_dim]

        # --- 1) 序列分支: conv + ReLU + max-over-time ---
        x = xs.transpose(1, 2).contiguous()                           # [B, 4, L]
        h_seq_chunks = []
        for conv in model.seq_branch.convs:
            z = F.conv1d(x, conv.weight, conv.bias, stride=1, padding=0)  # [B, C, L']
            z = F.relu(z)
            h = torch.amax(z, dim=-1)                                     # [B, C]
            h_seq_chunks.append(h)
        h_seq = torch.cat(h_seq_chunks, dim=1)                             # [B, n_kernels]

        # --- 2) other 分支: 只用 base+rna 走 MLP 得到 h_other（不用于分数，仅与模型一致）
        l_base = model.l_base
        x_base = xo[:, :l_base]
        x_rna  = xo[:, l_base:l_base+1]
        _ = model.other_branch(torch.cat([x_base, x_rna], dim=1))          # 只是保持流程一致；结果不参与本分数

        # --- 3) 应用 cell state 条件化到 h_seq ---
        g_dim = model.g_dim
        if g_dim > 0:
            x_state = xo[:, l_base+1:]                                     # [B, g_dim]
            if model.fusion_mode == "film":
                gb = model.state_proj(x_state)                             # [B, 2*n_kernels]
                gamma, beta = gb.chunk(2, dim=1)
                h_seq = (1.0 + gamma) * h_seq + beta
            elif model.fusion_mode == "gate":
                gate = model.gate_head(x_state)                            # [B, n_kernels] 或 [B,1]
                h_seq = gate * h_seq
            # concat 模式：没有条件化，h_seq 原样

        scores_state[s:e] = h_seq.detach().cpu().numpy()

        if (bi+1) % max(1, n_batches//20) == 0 or (bi+1) == n_batches:
            print(f"[score_state] batch {bi+1}/{n_batches} (elapsed {time.perf_counter()-t0:.1f}s)", flush=True)

    return scores_state, kernel_names

# =========================
# 4) Kernel motif 收集/打分
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

    pfm = np.zeros((4, K), dtype=np.float64)
    X = onehots.reshape(N*K, 4).astype(np.float64)
    w = np.repeat(weights, K)
    pfm += (X * w[:, None]).reshape(N, K, 4).sum(axis=0).T  # -> 4 x K
    colsum = pfm.sum(axis=0, keepdims=True) + eps
    pwm = pfm / colsum
    return pfm, pwm


@torch.no_grad()
def collect_top_hits_for_kernel(device, conv_module, seq_loader, topk=500, act_relu=True,
                                train_global_idx: np.ndarray = None, log_every_batches: int = 50):
    """
    FAST版：全GPU、向量化的 per-channel top-K 收集。
    - 不使用 heap / .item() / 内层 Python 循环
    - 批内: 对每个通道一次性 topk
    - 批间: 与"全局累计"再做一次 topk 合并（每通道并行）
    返回: list[channel] -> [(score(float), sample_gidx(int), start_pos(int)), ...]  (按分数降序)
    """
    import time
    t0 = time.perf_counter()

    conv = conv_module.to(device).eval()
    C = conv.out_channels

    # 累计的全局 topK（每通道一行）
    g_vals = None      # [C, ≤topk] float32 on device
    g_sids = None      # [C, ≤topk] int64 (全局样本id) on device
    g_pos  = None      # [C, ≤topk] int64 (窗口起点)     on device

    total_seen = 0
    nb = 0

    for xs in seq_loader:
        nb += 1
        B = xs.shape[0]
        # [B,L,4] -> [B,4,L]
        x = xs.to(device, non_blocking=True).transpose(1, 2).contiguous()

        # 卷积 + ReLU
        z = F.conv1d(x, conv.weight, conv.bias, stride=1, padding=0)  # [B,C,Lp]
        if act_relu:
            z = F.relu(z)
        B, C_chk, Lp = z.shape
        assert C_chk == C

        # 批内：每通道 row-wise 取 topk
        Z2 = z.permute(1, 0, 2).reshape(C, -1)               # [C, B*Lp]
        k_cur = min(topk, Z2.size(1))
        vals_b, idx_b = torch.topk(Z2, k=k_cur, dim=1)       # [C, k_cur] (device)

        # idx_b -> (全局样本id, 位置)
        b_local = torch.div(idx_b, Lp, rounding_mode='floor') # [C, k_cur]
        pos     = idx_b - b_local * Lp                        # [C, k_cur]
        if train_global_idx is None:
            # 如果没传，就认为是 0..累加
            gidx_vec = torch.arange(total_seen, total_seen + B, device=device, dtype=torch.long)
        else:
            # 把 numpy 批次的全局索引拷到GPU一次
            gidx_vec = torch.as_tensor(train_global_idx[total_seen: total_seen + B],
                                       device=device, dtype=torch.long)
        sids_b = gidx_vec.gather(0, b_local.view(-1)).view(C, -1)  # [C, k_cur]

        # 与全局合并：拼接后再 topk
        if g_vals is None:
            g_vals = vals_b
            g_sids = sids_b
            g_pos  = pos
        else:
            Vcat = torch.cat([g_vals, vals_b], dim=1)             # [C, m]
            Scat = torch.cat([g_sids, sids_b], dim=1)
            Pcat = torch.cat([g_pos,  pos],    dim=1)
            k_keep = min(topk, Vcat.size(1))
            top_idx = torch.topk(Vcat, k=k_keep, dim=1).indices    # [C, k_keep]
            g_vals = torch.gather(Vcat, 1, top_idx)
            g_sids = torch.gather(Scat, 1, top_idx)
            g_pos  = torch.gather(Pcat,  1, top_idx)

        total_seen += B
        if nb % log_every_batches == 0:
            print(f"[collect-fast]   batches={nb}, seen={total_seen} samples, "
                  f"cur_keep={g_vals.size(1)}, elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    # 回到 CPU，按每通道转成 list[tuple]
    g_vals_cpu = g_vals.detach().cpu().numpy()
    g_sids_cpu = g_sids.detach().cpu().numpy()
    g_pos_cpu  = g_pos.detach().cpu().numpy()

    out = []
    for c in range(C):
        # 已经按值降序（topk返回有序），直接组装
        ch = [(float(g_vals_cpu[c, j]), int(g_sids_cpu[c, j]), int(g_pos_cpu[c, j]))
              for j in range(g_vals_cpu.shape[1])]
        out.append(ch)
    print(f"[collect-fast] done. batches={nb}, seen={total_seen}, "
          f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return out



def extract_subseq_windows(X_seq_all, idx_list, kernel_size):
    """
    X_seq_all: numpy [M, L, 4] one-hot
    idx_list : list of (sample_idx, start_pos)
    返回 [N, 4, K] one-hot 片段
    """
    K = kernel_size
    out = np.zeros((len(idx_list), 4, K), dtype=np.float32)
    for i, (sid, p) in enumerate(idx_list):
        win = X_seq_all[sid, p:p+K, :]  # [K, 4]
        if win.shape[0] == K:
            out[i] = np.transpose(win, (1,0))  # -> [4, K]
    return out


@torch.no_grad()
def score_all_sequences(device, conv_modules, X_seq_all, act_relu=True, batch_size=512):
    import time
    t0 = time.perf_counter()
    M, L, _ = X_seq_all.shape
    flat_specs = []
    for conv in conv_modules:
        C, K = conv.out_channels, conv.kernel_size[0]
        for c in range(C):
            flat_specs.append((conv, c, K))
    scores = np.full((M, len(flat_specs)), 0.0, dtype=np.float32)

    device_convs = [conv.to(device).eval() for conv in conv_modules]
    n_batches = (M + batch_size - 1) // batch_size
    print(f"[score_all_sequences] M={M}, kernels={len(flat_specs)}, batches={n_batches}, batch_size={batch_size}", flush=True)

    for bi in range(n_batches):
        s = bi * batch_size
        e = min(M, s + batch_size)
        xs = torch.from_numpy(X_seq_all[s:e]).to(device)  # [B,L,4]
        x = xs.transpose(1,2).contiguous()
        col = 0  # ←←← 关键：每个 batch 都要从 0 开始写列
        for conv in device_convs:
            z = F.conv1d(x, conv.weight, conv.bias, stride=1, padding=0)  # [B, C, L']
            if act_relu:
                z = F.relu(z)
            zmax = torch.amax(z, dim=-1).detach().cpu().numpy()           # [B, C]
            C = zmax.shape[1]
            scores[s:e, col:col+C] = zmax
            col += C
        if (bi+1) % max(1, n_batches//20) == 0 or (bi+1) == n_batches:
            dt = time.perf_counter() - t0
            print(f"[score_all_sequences] batch {bi+1}/{n_batches} done (elapsed {dt:.1f}s)", flush=True)
    return scores, flat_specs


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


# =========================
# 5) 辅助：从 ckpt 推断 other_in_dim，并做形状校验
# =========================
def infer_other_in_dim_from_ckpt(ckpt_state_dict: Dict[str, torch.Tensor]) -> int:
    key = "other_branch.input.weight"
    if key not in ckpt_state_dict:
        # 兼容早期命名
        for k in ckpt_state_dict:
            if k.endswith(".other_branch.input.weight") or k.endswith("other_branch.input.weight"):
                key = k
                break
    w = ckpt_state_dict.get(key, None)
    if w is None:
        raise SystemExit("[FATAL] ckpt 中未找到 other_branch.input.weight，无法推断 other_in_dim。")
    return int(w.shape[1])


def sanity_check_conv_shapes(ckpt_state_dict, kernel_sizes: List[int], cnn_channels: int):
    """简单校验 seq_branch.convs.*.weight 形状是否与命令行一致"""
    conv_keys = [k for k in ckpt_state_dict if "seq_branch.convs" in k and k.endswith(".weight")]
    if not conv_keys:
        warnings.warn("[WARN] ckpt 未找到 seq_branch.convs.*.weight；跳过形状校验。")
        return
    # 构造期望 multiset: 对每个 kernel_size 期望 [cnn_channels, 4, K]
    expected = sorted([(cnn_channels, 4, int(K)) for K in kernel_sizes])
    actual = []
    for k in conv_keys:
        W = ckpt_state_dict[k]
        if W.ndim == 3:
            actual.append(tuple(map(int, W.shape)))
    actual = sorted(actual)
    if expected != actual:
        warnings.warn(
            "[WARN] ckpt 中卷积核形状与命令行不一致：\n"
            f"  期望: {expected}\n"
            f"  实际: {actual}\n"
            "  若后续加载失败，请改用训练时的 --kernel_sizes / --cnn_channels 参数重试。"
        )


@torch.no_grad()
def compute_cellstate_effects(
    model, device,
    X_seq_np: np.ndarray,         # [M, L, 4]
    X_other_np: np.ndarray,       # [M, l_base + 1 + g_dim]
    stage_ids: np.ndarray,        # [M]
    kernel_names: List[str],
    batch_size: int = 1024,
):
    """
    计算“cell-state 对每个 kernel 的绝对增量 Δh（同量纲）”。
      - FiLM: Δh = gamma ⊙ h + beta
      - gate: Δh = (gate - 1) ⊙ h
      - concat/无状态: 0
    返回:
      DataFrame: [M x K]，列名=kernel_names，并在首列附加 'stage_id'
    """
    model.eval()
    is_film = (model.fusion_mode == "film" and model.g_dim > 0)
    is_gate = (model.fusion_mode == "gate" and model.g_dim > 0)

    M = X_seq_np.shape[0]
    K = len(kernel_names)
    effects_all = np.zeros((M, K), dtype=np.float32)

    convs = list(model.seq_branch.convs)

    def _flat_cat(mat_list: List[torch.Tensor]) -> torch.Tensor:
        # 各 conv 的 [B, C] 串接为 [B, sumC]，顺序与 kernel_names 对齐
        return torch.cat(mat_list, dim=1)

    n_batches = (M + batch_size - 1) // batch_size
    for bi in range(n_batches):
        s = bi * batch_size
        e = min(M, s + batch_size)

        # --- 准备 batch 张量 ---
        x_seq = torch.from_numpy(X_seq_np[s:e]).to(device, non_blocking=True)   # [B,L,4]
        x_oth = torch.from_numpy(X_other_np[s:e]).to(device, non_blocking=True) # [B, l_base+1+g_dim]

        # --- 序列分支：h_seq（每通道 max-pool 后的通道向量）---
        x = x_seq.transpose(1, 2).contiguous()  # [B,4,L]
        h_list = []
        for conv in convs:
            z = F.relu(conv(x))                 # [B, C, L']
            h = torch.amax(z, dim=-1)           # [B, C]
            h_list.append(h)
        h_seq = _flat_cat(h_list)               # [B, sumC]

        # --- 拆 other 并计算状态调制量 ---
        l_base = model.l_base
        x_state = x_oth[:, l_base+1:] if model.g_dim > 0 else None

        if is_film:
            gb = model.state_proj(x_state)               # [B, 2*C]
            gamma, beta = gb.chunk(2, dim=1)             # [B, C], [B, C]
            delta_h = gamma * h_seq + beta               # [B, C]
        elif is_gate:
            gate = model.gate_head(x_state)              # [B, C] in (0,1)
            delta_h = (gate - 1.0) * h_seq               # [B, C]
        else:
            delta_h = torch.zeros_like(h_seq)            # [B, C]

        effects_all[s:e, :] = delta_h.detach().cpu().numpy()

        if (bi + 1) % max(1, n_batches // 10) == 0 or (bi + 1) == n_batches:
            print(f"[cellstate-effect Δh] batch {bi+1}/{n_batches} done", flush=True)

    df = pd.DataFrame(effects_all, columns=kernel_names)
    df.insert(0, "stage_id", stage_ids.astype(int))
    return df

def summarize_and_plot_cellstate_effects(
    df_effects: pd.DataFrame,
    stages: List[str],
    outdir: Path,
    top_k: int = 16
):
    """
    - 对每个 stage 求各 kernel 的 Δh 均值
    - 选“按各 stage 绝对均值的最大值”排序的 Top-K kernel
    - 按 GV→MI→MII→1C→2C→4C→8C→ICM→ES 顺序绘制小提琴图
    - 保存汇总 CSV
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -------------------------
    # 1) 自定义阶段顺序
    stage_order = ["GV", "MI", "MII", "1C", "2C", "4C", "8C", "ICM", "hESC"]
    stage_order_lower = [s.lower() for s in stage_order]

    # 标准化 stage 名（兼容大小写）
    stage_map = {s.lower(): s for s in stage_order}
    stages_std = [stage_map.get(s.lower(), s) for s in stages]

    # -------------------------
    # 2) 聚合均值
    kernel_cols = [c for c in df_effects.columns if c != "stage_id"]
    stage_means = (df_effects
                   .groupby("stage_id")[kernel_cols]
                   .mean()
                   .rename(index=lambda i: stages_std[i]))

    # 2.5 额外保存：所有 kernel 的 Δh(stage 均值) 矩阵（不做 Top-K 筛选）
    # 说明：行是 stage，列是 kernel，值是该 stage 内所有样本的 Δh 均值
    #      这份矩阵可用于后续 Figure/统计（例如按 stage 聚类、挑选 Δh>0 的 stage-specific kernels 等）。
    stage_means_all = stage_means.copy()
    # 尽量按预定义顺序输出（若某些 stage 缺失则会是 NaN）
    stage_means_all = stage_means_all.reindex(stage_order)
    df_all = stage_means_all.copy()
    df_all.insert(0, "stage", df_all.index)
    all_csv_path = outdir / "cellstate_effect_stage_means_all.csv"
    df_all.to_csv(all_csv_path, index=False)
    print(f"[Δh] 写出 {all_csv_path}, 包含所有 kernels: {len(kernel_cols)} 个", flush=True)


    # 选Top-K

    # 1. 计算每个 kernel 在各 stage 上的 |Δh| 最大值，并排序
    abs_max = stage_means.abs().max(axis=0).sort_values(ascending=False)
    top_kernels = abs_max.index[:top_k].tolist()

    # 2. 确保 Figure 3 需要的几个 kernel 一定被包含进来
    FIG3_KERNELS = [
        "kernel_k20_f8",
        "kernel_k12_f17",
        "kernel_k12_f29",
        "kernel_k16_f11",
    ]

    # 只保留那些在 stage_means 里真实存在的列，避免写错名字直接报错
    fig3_available = [k for k in FIG3_KERNELS if k in stage_means.columns]

    # 取并集（顺序：先 topK，再补 Figure3）
    selected_kernels = []
    for k in top_kernels + fig3_available:
        if k not in selected_kernels:
            selected_kernels.append(k)

    # 3. 构建输出表：第一列是 stage，后面是各个 kernel 的 Δh
    csv_path = outdir / "cellstate_effect_stage_means_topK.csv"

    df_out = stage_means[selected_kernels].copy()
    df_out.insert(0, "stage", stage_means.index)

    df_out.to_csv(csv_path, index=False)
    print(f"[Δh] 写出 {csv_path}, 包含 kernels: {len(selected_kernels)} 个（含 Figure 3 的 4 个）")



    # -------------------------
    # 3) 绘图
    long_df = (df_effects[["stage_id"] + top_kernels]
               .melt(id_vars="stage_id", var_name="kernel", value_name="delta_h"))
    long_df["stage"] = long_df["stage_id"].map(dict(enumerate(stages_std)))
    long_df["stage"] = pd.Categorical(long_df["stage"],
                                      categories=stage_order,
                                      ordered=True)

    for name in top_kernels:
        sub = long_df[long_df["kernel"] == name].copy()
        plt.figure(figsize=(8, 4))
        sns.violinplot(data=sub, x="stage", y="delta_h",
                       cut=0, inner="box", order=stage_order)
        plt.axhline(0.0, ls="--", lw=1)
        plt.title(f"Cell-state effect per stage (Δh)\n{name}")
        plt.ylabel("Δh (Change in CNN channel activation)")
        plt.xlabel("Developmental stage")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"cellstate_effect__{name}.png", dpi=150)
        plt.close()

    print(f"[cellstate-effect Δh] saved table: {csv_path}")
    print(f"[cellstate-effect Δh] saved {len(top_kernels)} per-kernel plots to {outdir}")



from pathlib import Path
import numpy as np
import torch

@torch.no_grad()
def save_selected_kernel_pwms(
    model,
    device,
    X_seq: np.ndarray,
    train_indices: np.ndarray,
    top_kernel_names: list,
    outdir: Path,
    topk_per_kernel: int = 500,
    min_hits: int = 100,
):
    """
    只对 top_kernel_names 中的 kernel 提取 PWM，写入 outdir/pwm_selected/，
    同时生成 meme_selected.txt。
    """
    from pathlib import Path
    import numpy as np
    import torch
    from extract_kernels_and_scores_add_cellstate import (
        collect_top_hits_for_kernel, extract_subseq_windows, onehot_windows_to_pfm
    )

    convs = list(model.seq_branch.convs)
    pwm_dir = outdir / "pwm_selected"
    pwm_dir.mkdir(parents=True, exist_ok=True)
    meme_path = pwm_dir / "meme_selected.txt"
    print(f"[PWM] 输出目录: {pwm_dir}", flush=True)

    # 预计算各层的通道总数前缀和，用于“全局扁平下标”兜底解析
    layer_offsets = []
    off = 0
    for conv in convs:
        layer_offsets.append(off)
        off += conv.out_channels

    # 解析 kernel 名称的健壮函数
    def parse_kernel_name(name: str):
        # e.g. "kernel_k20_f60"
        try:
            k = int(name.split("_k")[1].split("_")[0])
            f = int(name.split("_f")[1])
        except Exception:
            raise ValueError(f"无法解析 kernel 名称: {name}")
        return k, f

    def where_is(name: str):
        k, f = parse_kernel_name(name)

        # 先按“层内通道号”解释（与你之前列名一致的语义）
        for li, conv in enumerate(convs):
            if conv.kernel_size[0] == k:
                C = conv.out_channels
                if 0 <= f < C:
                    return li, f, k  # 层号、层内通道号、kernel_size
                # 如果 f 超出层通道数，可能用户给的是“全局扁平下标”，走兜底
                break

        # 兜底：把 f 当“全局扁平下标”，找到对应层与层内通道
        # （顺序 = 你的拼接顺序：按 convs 依次拼接各自 0..C-1）
        if f >= 0:
            for li, conv in enumerate(convs):
                start = layer_offsets[li]
                end = start + conv.out_channels
                if start <= f < end:
                    ch_in_layer = f - start
                    if conv.kernel_size[0] != k:
                        # 名称的 k 与这个层的 k 不匹配，说明名字不遵循“全局扁平”语义
                        break
                    return li, ch_in_layer, conv.kernel_size[0]

        # 仍找不到，抛错并提示现有可选范围
        ks = [(i, conv.kernel_size[0], conv.out_channels) for i, conv in enumerate(convs)]
        raise ValueError(f"未找到 {name}。现有层: {ks}（每层通道 0..C-1；或使用全局扁平下标 0..{off-1}）")

    # 先按层收集一次 hits（避免对同一层重复多次跑）
    # 只对“用得到的层”跑，减少计算
    needed_by_layer = {}
    for nm in top_kernel_names:
        li, ch, K = where_is(nm)
        needed_by_layer.setdefault(li, set()).add(ch)

    # 收集 hits
    hits_per_layer = {}
    for li, use_channels in sorted(needed_by_layer.items()):
        conv = convs[li]
        print(f"[PWM] 收集层 li={li}, k={conv.kernel_size[0]}, C={conv.out_channels}, "
              f"channels={sorted(list(use_channels))}", flush=True)

        # 生成器：分批把训练集序列喂给这一层
        def _iter_seq_batches(indices, batch=1024):
            for i in range(0, len(indices), batch):
                b = indices[i:i+batch]
                yield torch.from_numpy(X_seq[b])  # CPU tensor

        all_hits_this_layer = collect_top_hits_for_kernel(
            device=device,
            conv_module=conv,
            seq_loader=_iter_seq_batches(train_indices, batch=1024),
            topk=topk_per_kernel,
            act_relu=True,
            train_global_idx=train_indices,
            log_every_batches=20
        )
        hits_per_layer[li] = all_hits_this_layer

    # 写 PWM + MEME
    meme_lines = ["MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"]

    for name in top_kernel_names:
        li, ch, K = where_is(name)
        hits = hits_per_layer[li][ch]  # [(score, sid, pos), ...]
        if len(hits) < min_hits:
            print(f"[PWM] 跳过 {name}, hits={len(hits)} < {min_hits}", flush=True)
            continue

        idx_list = [(sid, pos) for (_, sid, pos) in hits]
        windows = extract_subseq_windows(X_seq, idx_list, kernel_size=K)
        scores = np.array([s for (s, _, _) in hits], dtype=np.float64)
        pfm, pwm = onehot_windows_to_pfm(windows, weights=scores)

        np.savetxt(pwm_dir / f"{name}.pwm.tsv", pwm, fmt="%.6f", delimiter="\t")

        meme_lines.append(f"MOTIF {name}\nletter-probability matrix: alength= 4 w= {K}\n")
        for j in range(K):
            a, c, g, t = pwm[:, j]
            meme_lines.append(f"{a:.6f}\t{c:.6f}\t{g:.6f}\t{t:.6f}\n")
        meme_lines.append("\n")

    with open(meme_path, "w") as f:
        f.writelines(meme_lines)
    print(f"[PWM] 已保存至 {meme_path}", flush=True)

### NEW for Figure 3 ###
def select_kernels_for_figure3(
    df_act: pd.DataFrame,
    stages: List[str],
    metric: str = "rho_with_ribo_residual_RNA",
    top_n_global: int = 1,
    top_n_stage_specific: int = 3,
) -> Dict[str, list]:
    """
    选 Figure 3 用的代表性 kernel：
      - global: 在所有阶段 |metric| 的平均值最高且跨阶段比较均匀的 kernel
      - stage_specific: 对某个 stage 特别高的 kernel（且 top - second 差比较大）
    返回:
      {
        "global": [kernel_name1, ...],
        "stage_specific": [(kernel_name, stage_name), ...]
      }
    """
    df_use = df_act.copy()
    if metric not in df_use.columns:
        raise SystemExit(f"[Figure3] metric 列不存在: {metric}")

    # 1) 计算每个 kernel 全阶段 |metric| 的 mean / std
    grouped = df_use.groupby("kernel")[metric].agg(["mean", "std"])
    grouped["mean_abs"] = df_use.groupby("kernel")[metric].apply(lambda x: x.abs().mean())
    grouped["std_abs"] = df_use.groupby("kernel")[metric].apply(lambda x: x.abs().std(ddof=0))

    # 简单规则：mean_abs 大且 std_abs 不是特别极端
    global_order = grouped.sort_values("mean_abs", ascending=False)
    global_kernels = global_order.head(top_n_global).index.tolist()

    # 2) stage-specific：对每个 stage 找 “最偏爱该 stage” 的 kernel
    spec_candidates = []
    for st in stages:
        sub = df_use[df_use["stage"] == st][["kernel", metric]].dropna()
        if sub.empty:
            continue
        sub_sorted = sub.sort_values(metric, ascending=False)
        top = sub_sorted.iloc[0]
        if len(sub_sorted) > 1:
            second = sub_sorted.iloc[1]
            gap = float(top[metric] - second[metric])
        else:
            gap = float("nan")
        spec_candidates.append({
            "kernel": top["kernel"],
            "stage": st,
            "metric": float(top[metric]),
            "gap": gap,
        })

    if len(spec_candidates) == 0:
        stage_spec = []
    else:
        df_spec = pd.DataFrame(spec_candidates)
        # 优先 metric 高同时 gap 大的
        df_spec = df_spec.sort_values(["metric", "gap"], ascending=False)
        # 去掉已经被选成 global 的
        df_spec = df_spec[~df_spec["kernel"].isin(global_kernels)]
        stage_spec_rows = df_spec.head(top_n_stage_specific)
        stage_spec = list(zip(stage_spec_rows["kernel"], stage_spec_rows["stage"]))

    return {
        "global": global_kernels,
        "stage_specific": stage_spec,
    }


### NEW for Figure 3 ###
# 全局字体（粗体，看起来更像 seqlogo）
# 全局字体（粗体，看起来更像 seqlogo）
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

import numpy as np
import matplotlib.pyplot as plt

_SEQLOGO_FP = FontProperties(family="DejaVu Sans", weight="bold")

# 碱基配色
_BASE_COLORS = {
    "A": "#4daf4a",  # 绿
    "C": "#377eb8",  # 蓝
    "G": "#ff7f00",  # 橙
    "T": "#e41a1c",  # 红
}

def plot_seq_logo_from_pwm(
    pwm,
    title=None,
    ax=None,
    alphabet="ACGT",
    background=None,
    show_ylabel=True,
):
    """
    根据给定的 PWM 绘制 seqlogo，高度按每个位点的信息量 (information content, bits) 计算。

    参数
    ----
    pwm : np.ndarray
        形状 (4, L) 或 (L, 4)，行对应 A/C/G/T 的概率（或频数）。
    title : str
        子图标题。
    ax : matplotlib.axes.Axes
        目标 Axes；若为 None 则自动创建。
    alphabet : str
        行顺序，对应 pwm 的行；默认 "ACGT"。
    background : np.ndarray 或 None
        背景频率 q_i，用于计算信息量；若为 None，则假设均匀分布 (1/4,1/4,1/4,1/4)。
    show_ylabel : bool
        是否显示 y 轴标签（多 panel 时可以只在第一列显示）。

    说明
    ----
    对第 j 个位置：
        p = pwm[:, j] / sum(pwm[:, j])
        H(p) = -∑ p_i log2 p_i
        R_j  = log2(K) - H(p)   (K = len(alphabet))
        每个碱基的高度 = p_i * R_j
    """
    pwm = np.asarray(pwm, dtype=float)

    # 统一成 shape = (4, L)
    if pwm.shape[0] != len(alphabet) and pwm.shape[1] == len(alphabet):
        pwm = pwm.T
    if pwm.shape[0] != len(alphabet):
        raise ValueError(f"pwm shape {pwm.shape} 与 alphabet={alphabet} 不匹配")

    n_bases, L = pwm.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, L * 0.3), 3))

    # 背景频率
    if background is None:
        background = np.full((n_bases,), 1.0 / n_bases, dtype=float)
    else:
        background = np.asarray(background, dtype=float)
        background = background / background.sum()

    logK = np.log2(float(n_bases))

    # 每个 position 的总信息量 R_j
    heights_per_pos = []
    max_height = 0.0

    for pos in range(L):
        col = pwm[:, pos].astype(float)
        if col.sum() <= 0:
            heights_per_pos.append(np.zeros_like(col))
            continue

        p = col / col.sum()

        # Shannon 信息量: R = log2(K) - H(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(p * np.log2(p + 1e-12)).sum()
        R = max(0.0, logK - H)  # clamp，避免负值

        # 每个碱基高度 = p_i * R_j
        h = p * R
        heights_per_pos.append(h)
        max_height = max(max_height, float(h.sum()))

    if max_height <= 0:
        max_height = 1.0

    # 逐位画 logo
    for pos in range(L):
        h_col = heights_per_pos[pos]
        if np.all(h_col <= 0):
            continue

        # 小的在下，大的在上
        order = np.argsort(h_col)
        y_offset = 0.0
        x_center = pos + 0.5

        for idx in order:
            height = float(h_col[idx])
            if height <= 0:
                continue

            base = alphabet[idx]
            color = _BASE_COLORS.get(base, "black")

            tp = TextPath((0, 0), base, size=1, prop=_SEQLOGO_FP)
            bbox = tp.get_extents()
            letter_height = bbox.height or 1.0
            ymin = bbox.ymin

            # 缩放到指定高度
            scale = height / letter_height

            # 关键：先把字母往上平移 -ymin，使得底部对齐 0，
            # 再整体平移到 (x_center, y_offset)
            trans = (
                Affine2D()
                .scale(scale, scale)
                .translate(x_center, y_offset - ymin * scale)
            )

            patch = PathPatch(
                tp,
                transform=trans + ax.transData,
                facecolor=color,
                edgecolor="none",
            )
            ax.add_patch(patch)

            y_offset += height

    # 轴设置
    ax.set_xlim(0, L)
    ax.set_ylim(0, max_height * 1.2)

    ax.set_xticks(np.arange(L) + 0.5)
    ax.set_xticklabels(np.arange(1, L + 1))
    ax.tick_params(axis="x", labelsize=8)

    if show_ylabel:
        ax.set_ylabel("Information (bits)", fontsize=8)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize=10, pad=4)

    return ax


### NEW for Figure 3 ###
def plot_figure3_motif_grid(
    pwm_bank: Dict[str, np.ndarray],
    df_act: pd.DataFrame,
    stages: List[str],
    metric: str,
    selected: Dict[str, list],
    out_png: Path,
    out_pdf: Path,
):
    """
    Figure 3:
      - 第一列：global motif
      - 后几列：若干 stage-specific motif
      每列：
        上面：motif sequence logo
        下面：该 kernel 在各 stage 上的 metric 条形图
    """
    # 准备列的顺序和注释
    cols = []
    titles = []

    # 1) 全局 kernel
    for k in selected.get("global", []):
        cols.append(("global", k, None))
        titles.append(f"{k}\n(global)")

    # 2) stage-specific kernel
    for (k, st) in selected.get("stage_specific", []):
        cols.append(("stage_specific", k, st))
        titles.append(f"{k}\n({st}-biased)")

    if len(cols) == 0:
        print("[Figure3] 没有选出任何 kernel，跳过 Figure 3 绘图")
        return

    n_col = len(cols)
    fig, axes = plt.subplots(
        2,
        n_col,
        figsize=(2.4 * n_col, 3.8),
        sharey=False,
    )
    if n_col == 1:
        axes = np.array(axes).reshape(2, 1)

    # 把 stage 顺序固定成输入顺序
    stages_order = list(stages)

    for j, (kind, kernel_name, spec_stage) in enumerate(cols):
        ax_logo = axes[0, j]
        ax_bar = axes[1, j]

        # ---- motif logo ----
        pwm = pwm_bank.get(kernel_name, None)
        if pwm is None:
            ax_logo.set_axis_off()
        else:
            plot_seq_logo_from_pwm(
                pwm=pwm,
                ax=ax_logo,
                title=titles[j]
            )

        # ---- stage 条形图 ----
        sub = df_act[df_act["kernel"] == kernel_name][["stage", metric]].dropna()
        if sub.empty:
            ax_bar.set_axis_off()
            continue

        # 按给定顺序对 stage 排序
        stage_vals = []
        for st in stages_order:
            row = sub[sub["stage"] == st]
            if row.empty:
                stage_vals.append(np.nan)
            else:
                stage_vals.append(float(row[metric].iloc[0]))

        x = np.arange(len(stages_order))
        ax_bar.bar(x, stage_vals, width=0.7, color="#4c72b0")
        ax_bar.axhline(0.0, linestyle="--", linewidth=0.8, color="0.3")

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(stages_order, rotation=45, ha="right", fontsize=7)
        if j == 0:
            ax_bar.set_ylabel(metric, fontsize=8)
        else:
            ax_bar.set_ylabel("")
        # 高亮 stage-specific 的 stage
        if spec_stage is not None and spec_stage in stages_order:
            idx = stages_order.index(spec_stage)
            ax_bar.patches[idx].set_color("#d62728")

    plt.tight_layout()
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Figure3] motif grid saved to:\n  {out_png}\n  {out_pdf}")


# =========================
# 6) 主程序
# =========================
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

    # 与训练时一致的结构参数（用于构建模型加载 ckpt）

    # NEW: 融合策略（支持自动从 ckpt 推断）
    ap.add_argument("--fusion", choices=["auto", "concat", "film", "gate"], default="auto")
    ap.add_argument("--cond_hidden", type=int, default=-1, help="= -1 时自动从 ckpt 推断（若能推断）")

    args = ap.parse_args()


    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "motifs").mkdir(exist_ok=True, parents=True)

    # 载入数据并展开
    D = load_and_expand(args.npz)
    X_seq, X_other, Y = D["X_seq"], D["X_other"], D["Y"]
    X_seq = D["X_seq"]                    # [M, L, 4]
    Y_all = D["Y"].reshape(-1)            # [M]
    RNA_all = D["X_other"][:, D["l_base"]].reshape(-1).astype(np.float32)        # [M]
    stage_ids = np.asarray(D["stage_ids"])
    stages = D["stages"]
    gene_ids = D["gene_ids"]
    M = X_seq.shape[0]

    # 划分（复现训练）
    if args.split_mode == "per_sample":
        tr_idx, va_idx, te_idx = stratified_split_per_stage(
            stage_ids, test_ratio=args.test_ratio_per_stage, val_ratio=args.val_ratio_in_train, seed=args.seed
        )
    else:
        tr_idx, va_idx, te_idx = split_by_gene(
            stage_ids, gene_ids, test_ratio=args.test_ratio_per_stage, val_ratio=args.val_ratio_in_train, seed=args.seed
        )


    split_tag = np.array([""]*M, dtype=object)
    split_tag[tr_idx] = "train"; split_tag[va_idx] = "val"; split_tag[te_idx] = "test"

    # 构建模型并加载权重（自动推断 other_in_dim）
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    other_in_dim = infer_other_in_dim_from_ckpt(state)
    sanity_check_conv_shapes(state, args.kernel_sizes, args.cnn_channels)
    l_base = D["l_base"]

    # ---- autodetect fusion / cond_hidden from ckpt (可覆盖命令行) ----
    def _autodetect_fusion(sd: dict) -> str:
        if any(k.startswith("state_proj.") for k in sd): return "film"
        if any(k.startswith("gate_head.") for k in sd):  return "gate"
        return "concat"

    fusion_mode = _autodetect_fusion(state) if args.fusion == "auto" else args.fusion

    # 如果需要，从权重里推断 cond_hidden
    if fusion_mode == "film" and args.cond_hidden <= 0:
        # state_proj.0.weight: [cond_hidden, g_dim]
        w0 = state.get("state_proj.0.weight", None)
        if w0 is not None and w0.ndim == 2:
            args.cond_hidden = int(w0.shape[0])
        else:
            args.cond_hidden = 128
    elif fusion_mode == "gate" and args.cond_hidden <= 0:
        w0 = state.get("gate_head.0.weight", None)
        if w0 is not None and w0.ndim == 2:
            args.cond_hidden = int(w0.shape[0])
        else:
            args.cond_hidden = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_MLP_Fusion(
        other_in_dim=X_other.shape[1],
        l_base=l_base,
        cnn_channels=args.cnn_channels,
        mlp_hidden=args.mlp_hidden,
        mlp_blocks=args.mlp_blocks,
        dropout=args.dropout,
        kernel_sizes=tuple(args.kernel_sizes),
        fusion=fusion_mode,
        cond_hidden=args.cond_hidden
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.to(device).eval()


    all_hits = []

    convs = list(model.seq_branch.convs)
    train_global = np.array(tr_idx, dtype=np.int64)

    print(f"[collect] start; #convs={len(convs)}, topk_per_kernel={args.topk_per_kernel}", flush=True)
    for li, conv in enumerate(convs):
        print(f"[collect] conv {li + 1}/{len(convs)} (k={conv.kernel_size[0]}, C={conv.out_channels}) ...", flush=True)

        # 轻量生成器（不把所有批次预先materialize）
        def _iter_seq_batches(indices, batch=1024):
            for i in range(0, len(indices), batch):
                b = indices[i:i + batch]
                yield torch.from_numpy(X_seq[b])  # [B,L,4] on CPU

        hits_per_channel = collect_top_hits_for_kernel(
            device=device,
            conv_module=conv,
            seq_loader=_iter_seq_batches(tr_idx, batch=max(512, args.batch_size)),
            topk=args.topk_per_kernel,
            act_relu=True,
            train_global_idx=train_global,
            log_every_batches=20
        )
        # 映射结果（已是全局 sample id 和起点，无需再额外映射）
        all_hits.append(hits_per_channel)

    # 2) 生成 motif（PFM、PWM、MEME）
    pwm_bank = {}  # name -> pwm(4 x K)
    for li, conv in enumerate(convs):
        K = conv.kernel_size[0]
        C = conv.out_channels
        for c in range(C):
            hits = all_hits[li][c]  # [(score, sid, pos), ...] (sid 已是原始索引)
            if len(hits) < args.min_hits:
                continue
            idx_list = [(sid, pos) for (_, sid, pos) in hits]
            windows = extract_subseq_windows(X_seq, idx_list, kernel_size=K)  # [N,4,K]
            scores  = np.array([s for (s,_,_) in hits], dtype=np.float64)
            pfm, pwm = onehot_windows_to_pfm(windows, weights=scores)
            name = f"kernel_k{K}_f{c}"
            np.savetxt(outdir / "motifs" / f"{name}.pfm.tsv", pfm, fmt="%.6f", delimiter="\t")
            np.savetxt(outdir / "motifs" / f"{name}.pwm.tsv", pwm, fmt="%.6f", delimiter="\t")
            pwm_bank[name] = pwm
    if args.save_meme and len(pwm_bank) > 0:
        save_meme(outdir / "meme.txt", pwm_bank)

    # 3) 为全部样本计算每个 kernel 的最大激活
    scores, flat_specs = score_all_sequences(
        device=device, conv_modules=convs, X_seq_all=X_seq, act_relu=True, batch_size=args.batch_size
    )
    kernel_names = []
    for conv in convs:
        K = conv.kernel_size[0]
        for c in range(conv.out_channels):
            kernel_names.append(f"kernel_k{K}_f{c}")
    assert scores.shape[1] == len(kernel_names), "scores 列数与 kernel_names 不一致。"

    # 4) 保存数据集（npz + meta + csv）
    meta_out = {
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
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    try:
        df_scores = pd.DataFrame(scores, columns=kernel_names)
        df_scores.insert(0, "split", split_tag)
        df_scores.insert(1, "stage_id", stage_ids)
        if gene_ids is not None:
            df_scores.insert(2, "gene", gene_ids)
        df_scores.to_csv(outdir / "kernel_scores.csv", index=False)
    except Exception as e:
        print(f"[warn] 保存 CSV 失败：{e}", file=sys.stderr)

    print(f"[OK] Motifs written to: {outdir/'motifs'}")
    if args.save_meme:
        print(f"[OK] MEME: {outdir/'meme.txt'}")
    print(f"[OK] Kernel score dataset: {outdir/'kernel_scores.npz'}  (+CSV)")
    print(f"[n_samples={M}, n_kernels={scores.shape[1]}]")

    # 5) Kernel × Stage 统计
    from scipy.stats import spearmanr, kruskal
    import numpy.linalg as LA
    import matplotlib.pyplot as plt
    import time

    t0 = time.perf_counter()
    stages_arr = np.asarray(stages)
    stage_ids_arr = np.asarray(stage_ids)
    uniq_stages = np.unique(stage_ids_arr)
    print(f"[stage-stats] start: M={M}, K={scores.shape[1]}, S={len(uniq_stages)}", flush=True)

    # 预先缓存各 stage 的掩码、y、RNA、以及 y_res（y~RNA 的线性回归残差）
    stage_cache = {}
    for s in uniq_stages:
        m = (stage_ids_arr == s)
        y_s = Y_all[m].astype(np.float64)
        Xs = np.c_[np.ones((m.sum(),1), dtype=np.float64), RNA_all[m].reshape(-1,1)]
        try:
            beta, *_ = LA.lstsq(Xs, y_s, rcond=None)
            y_res_s = y_s - Xs.dot(beta)
        except Exception:
            y_res_s = np.full_like(y_s, np.nan, dtype=np.float64)
        stage_cache[int(s)] = dict(mask=m, y=y_s, y_res=y_res_s)

    rows = []
    vec_all_mean = scores.mean(axis=0)
    vec_all_std = scores.std(axis=0) + 1e-12

    Ktot = scores.shape[1]
    for k in range(Ktot):
        if (k+1) % max(1, Ktot//50) == 0 or (k+1) == Ktot:
            print(f"[stage-stats] kernel {k+1}/{Ktot}", flush=True)

        vec = scores[:, k]
        # 跨阶段差异
        groups = [vec[stage_cache[s]["mask"]] for s in uniq_stages]
        try:
            H, p_all = kruskal(*groups)
        except Exception:
            H, p_all = np.nan, np.nan

        mu_all, sd_all = float(vec_all_mean[k]), float(vec_all_std[k])
        # ranks 一次即可（给 AUC 近似用）
        ranks = np.argsort(np.argsort(vec)) + 1  # 1..M
        n_total = len(vec)

        for s in uniq_stages:
            m = stage_cache[s]["mask"]
            v_s = vec[m]
            if v_s.size < 3:
                rows.append({
                    "kernel": kernel_names[k],
                    "stage_id": int(s),
                    "stage": stages_arr[s],
                    "kruskal_H_allstages": float(H) if not np.isnan(H) else np.nan,
                    "kruskal_p_allstages": float(p_all) if not np.isnan(p_all) else np.nan,
                    "mean_z_vs_all": np.nan,
                    "auc_one_vs_rest": np.nan,
                    "rho_with_ribo": np.nan, "p_with_ribo": np.nan,
                    "rho_with_ribo_residual_RNA": np.nan, "p_with_ribo_residual_RNA": np.nan,
                })
                continue

            mean_z = float((v_s.mean() - mu_all) / sd_all)

            # one-vs-rest AUC 近似（秩和）
            y_bin = m.astype(np.int32)
            n_pos = y_bin.sum()
            n_neg = n_total - n_pos
            if n_pos > 0 and n_neg > 0:
                sum_ranks_pos = ranks[y_bin == 1].sum()
                auc_ovr = float((sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg))
            else:
                auc_ovr = np.nan

            # 相关（直接 & partial）
            try:
                rho_y, p_y = spearmanr(v_s, stage_cache[s]["y"])
            except Exception:
                rho_y, p_y = np.nan, np.nan
            try:
                rho_res, p_res = spearmanr(v_s, stage_cache[s]["y_res"])
            except Exception:
                rho_res, p_res = np.nan, np.nan

            rows.append({
                "kernel": kernel_names[k],
                "stage_id": int(s),
                "stage": stages_arr[s],
                "kruskal_H_allstages": float(H) if not np.isnan(H) else np.nan,
                "kruskal_p_allstages": float(p_all) if not np.isnan(p_all) else np.nan,
                "mean_z_vs_all": mean_z,
                "auc_one_vs_rest": auc_ovr,
                "rho_with_ribo": float(rho_y) if not np.isnan(rho_y) else np.nan,
                "p_with_ribo": float(p_y) if not np.isnan(p_y) else np.nan,
                "rho_with_ribo_residual_RNA": float(rho_res) if not np.isnan(rho_res) else np.nan,
                "p_with_ribo_residual_RNA": float(p_res) if not np.isnan(p_res) else np.nan,
            })

    df_act = pd.DataFrame(rows)
    out_csv = outdir / "kernel_stage_activity.csv"
    df_act.to_csv(out_csv, index=False)
    print(f"[stage-stats] table saved: {out_csv} (elapsed {time.perf_counter()-t0:.1f}s)", flush=True)

    # 热图
    pivot_metric = "rho_with_ribo_residual_RNA"
    if df_act[pivot_metric].isna().all():
        pivot_metric = "rho_with_ribo"
    heat = df_act.pivot(index="kernel", columns="stage", values=pivot_metric)
    order = heat.abs().max(axis=1).sort_values(ascending=False).index
    heat = heat.loc[order]
    plt.figure(figsize=(1.2*heat.shape[1]+4, 0.22*heat.shape[0]+4))
    im = plt.imshow(heat.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label=pivot_metric)
    plt.xticks(range(heat.shape[1]), heat.columns, rotation=45, ha="right")
    plt.yticks(range(heat.shape[0]), heat.index)
    plt.title("Kernel vs Stage association (Spearman)")
    plt.tight_layout()
    fig_path = outdir / "kernel_stage_activity_heatmap.png"
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"[stage-stats] heatmap saved: {fig_path}", flush=True)
    # =========================
    # 6') 选 Top-16 kernel & 导出/作图
    # === 评估 cell-state 对 kernel 的相对影响 ===
    df_effects = compute_cellstate_effects(
        model=model, device=device,
        X_seq_np=X_seq, X_other_np=X_other,
        stage_ids=stage_ids,
        kernel_names=kernel_names,
        batch_size=max(512, args.batch_size)
    )
    df_effects.to_csv(outdir / "cellstate_effects__rel_delta.csv", index=False)

    # 只保留“绝对效应最大的 16 个 kernel”，并画各细胞系分布
    summarize_and_plot_cellstate_effects(
        df_effects=df_effects,
        stages=stages,
        outdir=outdir,
        top_k=16
    )

    # 先拿 Top-16 名单（与绘图一致的选择逻辑）
    kernel_cols = [c for c in df_effects.columns if c != "stage_id"]
    stage_means = (df_effects.groupby("stage_id")[kernel_cols]
                   .mean()
                   .rename(index=lambda i: stages[i]))
    abs_max = stage_means.abs().max(axis=0).sort_values(ascending=False)
    top16 = abs_max.index[:16].tolist()

    # 跑 motif 提取
    save_selected_kernel_pwms(
        model=model,
        device=device,
        X_seq=X_seq,
        train_indices=tr_idx,
        top_kernel_names=top16,  # 你的 Top-16 kernel 列表
        outdir=outdir,
        topk_per_kernel=500,
        min_hits=100
    )

    # =========================
    # 7") Figure 3: 全局 & stage-specific motif 图
    # =========================
    fig3_dir = outdir / "figure3"
    fig3_dir.mkdir(parents=True, exist_ok=True)

    selected_kernels = select_kernels_for_figure3(
        df_act=df_act,
        stages=stages,
        metric=pivot_metric,
        top_n_global=1,
        top_n_stage_specific=3,
    )

    # 保存一下列表，方便手工检查 / 对照
    with open(fig3_dir / "figure3_selected_kernels.json", "w", encoding="utf-8") as f:
        json.dump(selected_kernels, f, ensure_ascii=False, indent=2)

    # 利用前面生成的 pwm_bank 画 motif + stage 条形图
    plot_figure3_motif_grid(
        pwm_bank=pwm_bank,
        df_act=df_act,
        stages=stages,
        metric=pivot_metric,
        selected=selected_kernels,
        out_png=fig3_dir / "figure3_motifs_stage_effects.png",
        out_pdf=fig3_dir / "figure3_motifs_stage_effects.pdf",
    )

if __name__ == "__main__":
    main()
