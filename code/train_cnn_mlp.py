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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score

# ---------- Data reshape ----------
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
    # 拆分出 base特征 与 每阶段 RNA 特征列
    base_idx = [i for i, n in enumerate(feat_names) if not str(n).startswith("rna_")]
    rna_idx_map = {}  # stage -> col index
    for i, n in enumerate(feat_names):
        if str(n).startswith("rna_"):
            st = str(n)[4:]
            rna_idx_map[st] = i
    # 校验必须有对应 RNA 特征
    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise SystemExit(f"缺少 RNA 特征列: {missing}。请用 build_model_inputs.py 带 --rna_path 生成包含 rna_<stage> 的特征。")

    base_feats = X_other[:, base_idx]  # [N, l_base]
    l_base = base_feats.shape[1]

    # 展开到 (N*S)
    X_seq_exp = np.repeat(X_seq, repeats=S, axis=0)  # [N*S, L, 4]
    X_other_list = []
    Y_single = []
    stage_ids = []
    gene_ids = []

    for j, st in enumerate(stages):
        rna_col = rna_idx_map[st]
        rna_vec = X_other[:, rna_col][:, None]   # [N,1]
        other_j = np.hstack([base_feats, rna_vec])  # [N, l_base+1]
        X_other_list.append(other_j)
        Y_single.append(Y[:, [j]])  # [N,1]
        stage_ids.append(np.full((N, 1), j, dtype=np.int64))
        # 基因名跟着复制
        if genes:
            gene_ids.append(np.array(genes).reshape(-1,1))

    X_other_exp = np.vstack(X_other_list).astype(np.float32)     # [N*S, l_base+1]
    Y_exp       = np.vstack(Y_single).astype(np.float32)         # [N*S, 1]
    stage_ids   = np.vstack(stage_ids).astype(np.int64).reshape(-1)
    gene_ids    = np.vstack(gene_ids).reshape(-1).tolist() if genes else None

    return {
        "X_seq": X_seq_exp, "X_other": X_other_exp, "Y": Y_exp,
        "stage_ids": stage_ids, "stages": stages,
        "gene_ids": gene_ids, "L": L, "l_base": l_base
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
    def __init__(self, other_in_dim, cnn_channels=64, mlp_hidden=128, mlp_blocks=3,
                 dropout=0.1, kernel_sizes=(6,10,12,16,20), activation=nn.ReLU):
        super().__init__()
        self.seq_branch = SeqCNNBranch(4, kernel_sizes, cnn_channels, activation)
        self.other_branch = ResidualMLP(other_in_dim, mlp_hidden, mlp_blocks, dropout, activation)
        fusion_in = self.seq_branch.out_dim + mlp_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, max(mlp_hidden,128)), activation(), nn.Dropout(dropout),
            nn.Linear(max(mlp_hidden,128), 1)  # 单输出
        )
    def forward(self, x_seq, x_other):
        h = torch.cat([self.seq_branch(x_seq), self.other_branch(x_other)], dim=1)
        return self.fusion(h)  # [B,1]

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
    assert X_other.shape[1] == l_base + 1, "X_other 维度异常，期望=base特征+l单个RNA"

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
        cnn_channels=args.cnn_channels, mlp_hidden=args.mlp_hidden, mlp_blocks=args.mlp_blocks,
        dropout=args.dropout, kernel_sizes=tuple(args.kernel_sizes)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Scheduler
    plateau = None; cosine_step = None
    if args.scheduler == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
            min_lr=args.min_lr, verbose=True
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
    best_model_path = (outdir / "best.pt").resolve()
    ckpt = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"]); model.to(device); model.eval()

    Ys, Ps, St = [], [], []
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

    # R2
    r2_overall = r2_score(Y_test, P_test)
    r2_by_stage = {}
    for j, st in enumerate(stages):
        m = (St == j)
        if m.sum() >= 2:
            r2_by_stage[st] = float(r2_score(Y_test[m], P_test[m]))
        else:
            r2_by_stage[st] = float("nan")

    # 保存预测
    df = pd.DataFrame({"y_true": Y_test, "y_pred": P_test, "stage_id": St})
    df["stage"] = [stages[i] for i in St]
    df.to_csv(outdir / "predictions_test.csv", index=False)

    # QQ overall
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    yt_sorted = Y_test; yp_sorted = P_test
    nmin = min(len(yt_sorted), len(yp_sorted))
    ax.plot(yt_sorted[:nmin], yp_sorted[:nmin], '.', ms=2, alpha=0.6)
    lo, hi = min(yt_sorted.min(), yp_sorted.min()), max(yt_sorted.max(), yp_sorted.max())
    ax.plot([lo,hi],[lo,hi],'--',lw=1)
    ax.set_title(f"QQ (Overall)  R2={r2_overall:.3f}")
    ax.set_xlabel("Observed quantiles"); ax.set_ylabel("Predicted quantiles")
    fig.tight_layout(); fig.savefig(outdir / "qq_overall.png", dpi=150); plt.close(fig)

    # QQ per-stage
    ncol=3; nrow=int(math.ceil(len(stages)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol,5*nrow))
    axes = np.array(axes).reshape(-1)
    for i, st in enumerate(stages):
        ax = axes[i]
        m = (St == i)
        if m.sum() < 2:
            ax.axis("off"); continue
        yt = Y_test[m]; yp = P_test[m]; k = min(len(yt), len(yp))
        ax.plot(yt[:k], yp[:k], '.', ms=2, alpha=0.6)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo,hi],[lo,hi],'--',lw=1)
        ax.set_title(f"{st}  R2={r2_by_stage[st]:.3f}")
        ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    for j in range(len(stages), len(axes)):
        axes[j].axis("off")
    fig.tight_layout(); fig.savefig(outdir / "qq_by_stage.png", dpi=150); plt.close(fig)

    # Summary
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model_path": str(best_model_path),
            "r2_overall_test": float(r2_overall),
            "r2_per_stage_test": r2_by_stage,
            "stages": stages,
            "splits": {
                "test_ratio_per_stage": args.test_ratio_per_stage,
                "val_ratio_in_train": args.val_ratio_in_train
            },
            "seed": args.seed,
            "L": int(L),
            "other_in_dim": int(X_other.shape[1])
        }, f, ensure_ascii=False, indent=2)

    print(f"Best model saved at: {best_model_path}")
    print(f"Test R2 overall: {r2_overall:.4f}")
    print("R2 per stage:", r2_by_stage)
    print(f"Artifacts -> {outdir}")

if __name__ == "__main__":
    main()
