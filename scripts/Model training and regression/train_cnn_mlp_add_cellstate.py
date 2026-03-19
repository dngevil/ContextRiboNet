#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_single_label_cnn_mlp.py

单标签回归：样本=基因×阶段
输入:  X_seq[i] + [base数值特征(i), RNA(i,该阶段)]
输出:  Ribo(i,该阶段)

拆分:  每个阶段各自随机20% → TEST
      剩余样本按阶段各自随机20% → VAL
      其余 → TRAIN
"""

import argparse, json, os, math
import pdb
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ===== Nature-like 全局绘图风格 =====
plt.rcParams.update({
    "pdf.fonttype": 42,     # 矢量字体，方便 Illustrator 排版
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 7,
})

# ---------- Data reshape ----------
# 在文件顶部加：


# ===== helpers: 阶段RNA→全局特征 =====
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

def split_by_gene(stage_ids, gene_ids, test_ratio=0.2, val_ratio=0.2, seed=42):
    """
    将同一 gene 的所有样本（所有阶段）整体分到同一集合：
    - 先按基因做全局随机拆分：20% gene → TEST；其余 gene → 训练池
    - 再从训练池基因中随机取 20% gene → VAL；其余 → TRAIN
    说明：这样每个阶段的样本数不会保证“精确 20%”，但通常会非常接近；
         换来的是严格的“未见过基因”的泛化评估。
    """
    if gene_ids is None:
        raise SystemExit("split_by_gene 需要 meta.json 中的 gene 列表；请确认 build_model_inputs.py 写入了 genes。")

    rng = np.random.default_rng(seed)
    gene_ids = np.asarray(gene_ids)
    uniq_genes = np.unique(gene_ids)
    rng.shuffle(uniq_genes)

    n_test_genes = max(1, int(len(uniq_genes) * test_ratio))
    test_genes = set(uniq_genes[:n_test_genes])
    train_pool_genes = set(uniq_genes[n_test_genes:])

    # 从训练池基因里切 val
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
    """每个阶段各取 test_ratio 做测试；余下每个阶段再取 val_ratio 做验证；返回(train,val,test)索引"""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(stage_ids))
    stage_ids = np.asarray(stage_ids)
    train_pool = []
    test_idx = []

    # 先分 test
    for s in np.unique(stage_ids):
        s_idx = idx_all[stage_ids == s]
        rng.shuffle(s_idx)
        n_test = max(1, int(len(s_idx) * test_ratio))
        test_idx.append(s_idx[:n_test])
        train_pool.append(s_idx[n_test:])
    test_idx = np.concatenate(test_idx, axis=0)
    train_pool = np.concatenate(train_pool, axis=0)

    # 再从 train_pool 里按阶段取 val
    val_idx = []
    train_idx = []
    for s in np.unique(stage_ids):
        pool_s = train_pool[stage_ids[train_pool] == s]
        rng.shuffle(pool_s)
        n_val = max(1, int(len(pool_s) * val_ratio))
        val_idx.append(pool_s[:n_val])
        train_idx.append(pool_s[n_val:])
    val_idx = np.concatenate(val_idx, axis=0)
    train_idx = np.concatenate(train_idx, axis=0)

    return train_idx, val_idx, test_idx

# ---------- Dataset ----------
class SingleLabelDataset(Dataset):
    def __init__(self, X_seq, X_other, Y):
        self.X_seq = X_seq.astype(np.float32)    # [M, L, 4]
        self.X_other = X_other.astype(np.float32)# [M, l]
        self.Y = Y.astype(np.float32)            # [M, 1]
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        return self.X_seq[i], self.X_other[i], self.Y[i]

# ---------- Model ----------
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


# ---------- Utils ----------
def set_seed(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def eval_mse(model, loader, device):
    model.eval(); crit = nn.MSELoss(); tot = 0.0; n = 0
    for xs, xo, y in loader:
        xs, xo, y = xs.to(device), xo.to(device), y.to(device)
        pred = model(xs, xo)
        loss = crit(pred, y)
        tot += loss.item()*y.size(0); n += y.size(0)
    return tot/max(n,1)

def set_lr(optimizer, lr):
    for g in optimizer.param_groups: g["lr"] = lr

def plot_qq(y_true, y_pred, title="QQ (Overall)", save_path=None):
    """
    画 Predicted vs Observed 的 QQ 图
    y_true, y_pred: 1D 向量
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # R2
    r2 = r2_score(y_true, y_pred)

    # 排序后当成“分位数”
    obs = np.sort(y_true)
    pred = np.sort(y_pred)

    # 画图
    plt.figure(figsize=(7, 7))
    plt.scatter(obs, pred, s=5, alpha=0.5)

    # y = x 参考线
    vmin = min(obs.min(), pred.min())
    vmax = max(obs.max(), pred.max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")

    plt.xlabel("Observed quantiles")
    plt.ylabel("Predicted quantiles")
    plt.title(f"{title}  R2={r2:.3f}")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


def make_cosine_warmup_scheduler(optimizer, base_lr, min_lr, epochs, warmup_epochs):
    min_factor = min_lr / base_lr
    def step_fn(epoch):
        e = epoch-1
        if warmup_epochs>0 and e < warmup_epochs:
            factor = (e+1)/warmup_epochs
        else:
            t = (e - warmup_epochs) / max(1, epochs-warmup_epochs)
            factor = min_factor + 0.5*(1-min_factor)*(1+math.cos(math.pi*t))
        set_lr(optimizer, base_lr*factor)
        return optimizer.param_groups[0]['lr']
    return step_fn

def qq_plot(ax, y_true, y_pred, title="", annotate_text=None):
    yt = y_true.flatten(); yp = y_pred.flatten()
    n = min(len(yt), len(yp)); yt, yp = yt[:n], yp[:n]
    ax.plot(yt, yp, '.', ms=2, alpha=0.6)
    lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
    ax.plot([lo, hi],[lo, hi],'--',lw=1)
    ax.set_xlabel("Observed quantiles"); ax.set_ylabel("Predicted quantiles")
    ax.set_title(title)
    if annotate_text:
        ax.text(0.05, 0.95, annotate_text, transform=ax.transAxes, ha="left", va="top")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", required=True)
    # model/training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cnn_channels", type=int, default=64)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--kernel_sizes", type=int, nargs="+", default=[6,10,12,16,20])
    # split
    ap.add_argument("--test_ratio_per_stage", type=float, default=0.2)
    ap.add_argument("--val_ratio_in_train", type=float, default=0.2)
    # scheduler & early stopping
    ap.add_argument("--scheduler", choices=["plateau","cosine"], default="plateau")
    ap.add_argument("--lr_factor", type=float, default=0.5)     # plateau
    ap.add_argument("--lr_patience", type=int, default=10)      # plateau
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_epochs", type=int, default=5)     # cosine
    ap.add_argument("--early_patience", type=int, default=30)
    ap.add_argument("--fusion", choices=["concat", "film", "gate"], default="film")
    ap.add_argument("--cond_hidden", type=int, default=128)
    ap.add_argument("--split_mode", choices=["per_sample", "per_gene"], default="per_gene",
                    help="per_sample: 每个样本拆分20%作为测试集；per_gene: 同一基因的所有阶段一起划分，避免跨阶段同基因信息渗漏")

    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 载入并展开到单标签样本
    D = load_and_expand(args.npz)
    X_seq, X_other, Y = D["X_seq"], D["X_other"], D["Y"]
    stage_ids = D["stage_ids"]; stages = D["stages"]; gene_ids = D["gene_ids"]
    L = D["L"]; l_base = D["l_base"]
    # 断言其他特征最后一列是对应阶段的 RNA（我们在展开时就是这么做的）

    # 分割：每阶段20%做测试；余下每阶段20%做验证
    if args.split_mode == "per_sample":
        tr_idx, va_idx, te_idx = stratified_split_per_stage(
            stage_ids, test_ratio=args.test_ratio_per_stage, val_ratio=args.val_ratio_in_train, seed=args.seed
        )
    else:
        # per_gene
        tr_idx, va_idx, te_idx = split_by_gene(
            stage_ids, gene_ids, test_ratio=args.test_ratio_per_stage, val_ratio=args.val_ratio_in_train, seed=args.seed
        )

    # DataLoaders
    train_ds = SingleLabelDataset(X_seq[tr_idx], X_other[tr_idx], Y[tr_idx])
    val_ds   = SingleLabelDataset(X_seq[va_idx], X_other[va_idx], Y[va_idx])
    test_ds  = SingleLabelDataset(X_seq[te_idx], X_other[te_idx], Y[te_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model & Optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_MLP_Fusion(
        other_in_dim=X_other.shape[1],
        l_base=l_base,
        cnn_channels=args.cnn_channels,
        mlp_hidden=args.mlp_hidden,
        mlp_blocks=args.mlp_blocks,
        dropout=args.dropout,
        kernel_sizes=tuple(args.kernel_sizes),
        fusion=args.fusion,
        cond_hidden=args.cond_hidden
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Scheduler
    plateau = None; cosine_step = None
    if args.scheduler == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
            min_lr=args.min_lr
        )
    else:
        def set_lr(opt, lr):
            for g in opt.param_groups: g['lr'] = lr
        min_factor = args.min_lr/args.lr
        def cosine_step(epoch):
            e = epoch-1
            if args.warmup_epochs>0 and e < args.warmup_epochs:
                f = (e+1)/args.warmup_epochs
            else:
                t = (e-args.warmup_epochs)/max(1, args.epochs-args.warmup_epochs)
                f = min_factor + 0.5*(1-min_factor)*(1+math.cos(math.pi*t))
            set_lr(optimizer, args.lr*f)
            return optimizer.param_groups[0]['lr']
    best_model_path = (outdir / "best.pt").resolve()
    skip_training = best_model_path.exists()
    if skip_training:
        pass
    else:
        # Train loop
        crit = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
        best_val = float("inf"); best_epoch = -1; patience = 0

        for epoch in range(1, args.epochs+1):
            if args.scheduler == "cosine":
                cur_lr = cosine_step(epoch)
            else:
                cur_lr = optimizer.param_groups[0]['lr']

            model.train(); tot=0.0; n=0
            for xs, xo, y in train_loader:
                xs, xo, y = xs.to(device), xo.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                    pred = model(xs, xo)
                    loss = crit(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                tot += loss.item()*y.size(0); n += y.size(0)
            tr_mse = tot/max(n,1)
            va_mse = eval_mse(model, val_loader, device)
            if args.scheduler == "plateau": plateau.step(va_mse)

            print(f"[Epoch {epoch:03d}] train_mse={tr_mse:.6f}  val_mse={va_mse:.6f}  lr={cur_lr:.2e}")

            if va_mse + 1e-9 < best_val:
                best_val = va_mse; best_epoch = epoch; patience = 0
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_val_mse": best_val},
                           outdir / "best.pt")
            else:
                patience += 1
                if patience >= args.early_patience:
                    print(f"[EarlyStop] no improvement for {args.early_patience} epochs.")
                    break

    # Evaluate best on TEST, save preds & QQ plots
    # Evaluate best on TEST, save preds & QQ plots
    best_model_path = (outdir / "best.pt").resolve()
    ckpt = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"]); model.to(device); model.eval()

    Ys, Ps = [], []
    with torch.no_grad():
        for xs, xo, y in test_loader:
            xs, xo = xs.to(device), xo.to(device)
            p = model(xs, xo).cpu().numpy()
            Ps.append(p); Ys.append(y.numpy())
    Y_test = np.concatenate(Ys, 0).reshape(-1)
    P_test = np.concatenate(Ps, 0).reshape(-1)

    # 重建测试样本的 stage 标签（来自 te_idx 的顺序）
    all_stage_ids = np.asarray(stage_ids)
    St = all_stage_ids[te_idx]

    # ------- 计算 PearsonR overall & per-stage -------
    from scipy.stats import pearsonr

    # overall PearsonR
    pearson_overall = float(pearsonr(Y_test, P_test)[0])

    pearson_by_stage = {}
    for j, st in enumerate(stages):
        m = (St == j)
        if m.sum() >= 2:
            pearson_by_stage[st] = float(pearsonr(Y_test[m], P_test[m])[0])
        else:
            pearson_by_stage[st] = float("nan")

    # ------- 保存预测到 CSV，用于后续其他分析 -------
    df = pd.DataFrame({"y_true": Y_test, "y_pred": P_test, "stage_id": St})
    df["stage"] = [stages[i] for i in St]
    df.to_csv(outdir / "predictions_test.csv", index=False)

    # ===================== Figure 2(a) 样式散点图 =====================
    # ===================== Figure 2(a) 样式散点图（PearsonR） =====================
    def plot_figure2_scatter(df, stages, pearson_overall, pearson_by_stage, outdir: Path):
        """
        Figure 2(a) 风格：
          - 主图：所有 stage 混合，按 stage 上色
          - 子图：每个 stage 一个 panel
          - 相关性指标：PearsonR（不再使用 R² / r2_score）
          - 点小、风格接近 Nature
        """
        import math
        import numpy as np

        # ---- 颜色：tab20，colorblind 友好 ----
        unique_stages = list(stages)
        cmap = plt.get_cmap("tab20")
        color_map = {st: cmap(i % 20) for i, st in enumerate(unique_stages)}

        # ===================== 主图：所有 stage 混合 =====================
        fig, ax = plt.subplots(figsize=(3.5, 3.5))  # Nature 单栏宽度

        for st in unique_stages:
            sub = df[df["stage"] == st]
            if sub.empty:
                continue
            ax.scatter(
                sub["y_true"].values,
                sub["y_pred"].values,
                s=4,           # 点小
                alpha=0.6,
                edgecolors="none",
                label=st,
                c=[color_map[st]],
            )

        # y = x 参考线
        lo = min(df["y_true"].min(), df["y_pred"].min())
        hi = max(df["y_true"].max(), df["y_pred"].max())
        margin = 0.02 * (hi - lo)
        lo, hi = lo - margin, hi + margin
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.8, color="0.3")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # 外观：去掉上、右边框，保持 1:1
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Observed ribosome abundance")
        ax.set_ylabel("Predicted ribosome abundance")

        # ✅ 这里是 PearsonR
        ax.set_title(f"Test set (all ZGA stages), PearsonR = {pearson_overall:.3f}")

        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            title="Stage",
            title_fontsize=7,
        )

        ax.tick_params(axis="both", which="both", direction="out")

        fig.tight_layout()
        fig.savefig(outdir / "figure2a_scatter_overall_by_stage.png", dpi=600)
        fig.savefig(outdir / "figure2a_scatter_overall_by_stage.pdf")
        plt.close(fig)

        # ===================== 补图：按 stage 分 panel =====================
        n_stage = len(unique_stages)
        ncol = 3
        nrow = int(math.ceil(n_stage / ncol))

        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(3.0 * ncol, 3.0 * nrow),
        )
        axes = np.array(axes).reshape(-1)

        for i, st in enumerate(unique_stages):
            ax = axes[i]
            sub = df[df["stage"] == st]
            if sub.empty:
                ax.axis("off")
                continue

            yt = sub["y_true"].values
            yp = sub["y_pred"].values

            ax.scatter(
                yt,
                yp,
                s=4,
                alpha=0.6,
                edgecolors="none",
                c=[color_map[st]],
            )

            lo = min(yt.min(), yp.min())
            hi = max(yt.max(), yp.max())
            margin = 0.02 * (hi - lo)
            lo, hi = lo - margin, hi + margin
            ax.plot([lo, hi], [lo, hi], "--", linewidth=0.8, color="0.3")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(axis="both", which="both", direction="out")

            # ✅ 这里用 pearson_by_stage
            pr_stage = pearson_by_stage.get(st, float("nan"))
            ax.set_title(f"{st}  PearsonR = {pr_stage:.3f}", pad=2, fontsize=9)

            if i // ncol == nrow - 1:
                ax.set_xlabel("Observed")
            else:
                ax.set_xlabel("")
            if i % ncol == 0:
                ax.set_ylabel("Predicted")
            else:
                ax.set_ylabel("")

        for j in range(n_stage, len(axes)):
            axes[j].axis("off")

        fig.tight_layout(w_pad=0.8, h_pad=0.8)
        fig.savefig(outdir / "figure2a_scatter_by_stage_panels.png", dpi=600)
        fig.savefig(outdir / "figure2a_scatter_by_stage_panels.pdf")
        plt.close(fig)
    # 调用绘图函数，生成 Figure 2(a) 图
    plot_figure2_scatter(df, stages, pearson_overall, pearson_by_stage, outdir)

    # Summary
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_path": str(best_model_path),
                "stages": stages,
                "splits": {
                    "test_ratio_per_stage": args.test_ratio_per_stage,
                    "val_ratio_in_train": args.val_ratio_in_train,
                },
                "seed": args.seed,
                "L": int(L),
                "other_in_dim": int(X_other.shape[1]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Best model saved at: {best_model_path}")
    print(f"Artifacts -> {outdir}")


if __name__ == "__main__":
    main()
