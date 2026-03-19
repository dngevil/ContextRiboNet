#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_figure3.py

基于已经计算好的结果文件，自动生成 Figure 3 相关所有图：

1. 主图：4 个代表性 motif 的 seqlogo + stage-wise partial ρ 柱状图
2. 全局分布图：
   - Kruskal–Wallis p 值分布
   - max AUC(one-vs-rest) 分布
3. 全局 heatmap：kernel × stage 的 motif 活性/相关性
4. 可选：Δh (cell-state effect) 的 stage-wise violin/boxplot

需要的输入：
- kernel_stage_activity.csv      # 你刚给我的那个
- PWM 文件目录 pwm_dir           # 每个 kernel 的 PWM，一个文件
- 可选：delta_csv (stage × kernel 的 Δh stage 均值或单细胞)

注意：
- 你需要根据自己 PWM 文件的实际格式，修改 load_pwm_for_kernel() 中的读取方式（现在提供了 .npy 和 .tsv 两种示例）。
- Δh 部分同理，只要有一个 CSV，第一列是 stage，后面每列是一个 kernel 的 Δh 值（可以是均值，也可以是单样本长表 melt）。

用法示例：
python make_figure3.py \
    --activity-csv ../result/cnn_mlp_add_cellstate/single_label_v1/kernels/kernel_stage_activity.csv \
    --pwm-dir ../result/cnn_mlp_add_cellstate/single_label_v1/kernels/pwm_selected \
    --outdir figure3_out \
    --global-kernel kernel_k20_f8 \
    --stage-kernel 4C:kernel_k12_f17 \
    --stage-kernel 8C:kernel_k12_f29 \
    --stage-kernel hESC:kernel_k16_f11 \
    --delta-csv cellstate_effect_stage_means_topK.csv

"""

import argparse
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# -----------------------
# 一些基础配置
# -----------------------

STAGE_ORDER = ["GV", "MI", "MII", "1C", "2C", "4C", "8C", "ICM", "hESC"]



# 全局字体（粗体，看起来更像 seqlogo）
_SEQLOGO_FP = FontProperties(family="DejaVu Sans", weight="bold")

# 碱基配色
_BASE_COLORS = {
    "A": "#4daf4a",  # 绿
    "C": "#377eb8",  # 蓝
    "G": "#ff7f00",  # 橙
    "T": "#e41a1c",  # 红
}

# -----------------------
# 工具函数：从 PWM 画 seqlogo
# -----------------------
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def assemble_figure3_full(outdir: Path,
                          global_kernel: str,
                          stage_kernels: dict):
    outdir = Path(outdir)

    # 年刊/期刊风格：更宽一点，右侧给 heatmap 足够空间
    fig = plt.figure(figsize=(12.5, 10.0))

    # 关键：列数拉到 10，方便“往中间挪”和“给 d 加宽”
    gs = gridspec.GridSpec(
        4, 10,
        height_ratios=[2.2, 1.0, 1.0, 1.0],  # a + (b,c) + (e,f) + (g,h)
        width_ratios=[1,1,1,1,1, 1.2,1.2,1.2,1.2,1.2],  # 右侧略宽
        hspace=0.35,
        wspace=0.25,
    )

    # ---------- (a) main motifs ----------
    # a 横跨全宽
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.imshow(mpimg.imread(outdir / "figure3_main_panels.png"))
    ax_a.axis("off")
    ax_a.text(0.005, 0.99, "a", transform=ax_a.transAxes,
              fontsize=16, fontweight="bold", va="top")

    # ---------- (b), (c) ----------
    # ✅ 往右挪：放到 col 1:3 和 3:5（整体右移一格）
    ax_b = fig.add_subplot(gs[1, 2:4])
    ax_b.imshow(mpimg.imread(outdir / "figure3_max_auc_distribution.png"))
    ax_b.axis("off")
    ax_b.text(0.02, 0.98, "b", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="top")

    ax_c = fig.add_subplot(gs[1, 4:5])
    ax_c.imshow(mpimg.imread(outdir / "figure3_kw_p_distribution.png"))
    ax_c.axis("off")
    ax_c.text(0.02, 0.98, "c", transform=ax_c.transAxes,
              fontsize=14, fontweight="bold", va="top")

    # ---------- (d) heatmap ----------
    # ✅ d 更宽：占右侧 5:10（5 列）
    ax_d = fig.add_subplot(gs[1:, 5:10])
    ax_d.imshow(mpimg.imread(outdir / "figure3_heatmap_rho_with_ribo_residual_RNA.png"))
    ax_d.axis("off")
    ax_d.text(0.01, 0.99, "d", transform=ax_d.transAxes,
              fontsize=16, fontweight="bold", va="top")

    # ---------- (e)(f)(g)(h) ----------
    # ✅ 同样往右挪：用 col 1:3、3:5
    all_kernels = [global_kernel] + [stage_kernels[s] for s in ["4C", "8C", "hESC"]]
    delta_labels = ["e", "f", "g", "h"]
    delta_positions = [(2, 2), (2, 4), (3, 2), (3, 4)]  # (row, start_col)

    for (kname, lab, (r, c0)) in zip(all_kernels, delta_labels, delta_positions):
        ax = fig.add_subplot(gs[r, c0:c0+2])  # 每个占两列
        img = mpimg.imread(outdir / f"figure3_delta_violin_{kname}.png")
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    # 防止贴边裁切（尤其是 d 的 colorbar 区域）
    fig.subplots_adjust(left=0.04, right=0.985, top=0.985, bottom=0.04)

    fig.savefig(outdir / "figure3_full.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 3] 完整拼图保存到 {outdir/'figure3_full.png'}")


def pwm_to_info_content(pwm, eps=1e-9):
    """
    pwm: 4 x L 概率矩阵
    返回：每个位点的信息量 (bits) 向量 [L]
    """
    pwm = np.asarray(pwm, dtype=float)
    L = pwm.shape[1]
    ic = np.zeros(L, dtype=float)
    for i in range(L):
        p = pwm[:, i]
        p = np.clip(p, eps, 1.0)
        ic[i] = 2.0 + np.sum(p * np.log2(p))  # 2bits - H(p)
    return ic


# 预先缓存每个字母的 TextPath 和 bbox，省得重复算
_LETTER_CACHE = {}
def _get_letter_path(base: str):
    if base not in _LETTER_CACHE:
        tp = TextPath((0, 0), base, size=1, prop=_SEQLOGO_FP)
        bbox = tp.get_extents()
        _LETTER_CACHE[base] = (tp, bbox)
    return _LETTER_CACHE[base]


def plot_seq_logo_from_pwm(
    pwm,
    title=None,
    ax=None,
    alphabet="ACGT",
    show_ylabel=True,
):
    """
    真·seqlogo：高度 = 信息量 (bits)，宽度固定，每个位点自适应，不会左右挤。

    pwm: np.ndarray, shape=(4, L) 或 (L, 4)，行顺序对应 alphabet。
    """
    import numpy as np

    pwm = np.asarray(pwm, dtype=float)

    # 统一为 (4, L)
    if pwm.shape[0] != len(alphabet) and pwm.shape[1] == len(alphabet):
        pwm = pwm.T
    if pwm.shape[0] != len(alphabet):
        raise ValueError(f"pwm shape {pwm.shape} 与 alphabet={alphabet} 不匹配")

    n_bases, L = pwm.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, L * 0.25), 3))

    logK = np.log2(float(n_bases))

    heights_per_pos = []
    max_height = 0.0

    # 先算每个位点的信息量和每个碱基的高度
    for pos in range(L):
        col = pwm[:, pos].astype(float)
        if col.sum() <= 0:
            heights_per_pos.append(np.zeros_like(col))
            continue

        p = col / col.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(p * np.log2(p + 1e-12)).sum()
        R = max(0.0, logK - H)  # 信息量 bits

        h = p * R
        heights_per_pos.append(h)
        max_height = max(max_height, float(h.sum()))

    if max_height <= 0:
        max_height = 1.0

    # 画 logo：纵向按信息量，横向宽度固定（不会因 R_j 变胖）
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

            tp, bbox = _get_letter_path(base)
            letter_height = bbox.height or 1.0
            letter_width = bbox.width or 1.0
            ymin = bbox.ymin

            # 垂直方向：按信息量缩放到“height”
            scale_y = height / letter_height

            # 水平方向：宽度固定成每个位置 0.9 个单位，避免左右挤
            target_width = 0.9  # 每个位点的宽度上限
            scale_x = target_width / letter_width

            trans = (
                Affine2D()
                .scale(scale_x, scale_y)
                .translate(x_center, y_offset - ymin * scale_y)
            )

            patch = PathPatch(
                tp,
                transform=trans + ax.transData,
                facecolor=color,
                edgecolor="none",
            )
            ax.add_patch(patch)

            y_offset += height

    # 坐标轴：x 根据 L 自适应，y 按最大信息量留一点空间
    ax.set_xlim(0, L+1)
    ax.set_ylim(0, max_height * 1.2)

    ax.set_xticks(np.arange(L) + 0.5)
    ax.set_xticklabels(np.arange(1, L + 1), fontsize=6)

    if show_ylabel:
        ax.set_ylabel("Information (bits)", fontsize=8)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize=12, pad=4)

    return ax

# -----------------------
# 数据读取相关
# -----------------------

def load_activity_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 保证 stage 是字符串
    df["stage"] = df["stage"].astype(str)
    return df


def load_pwm_for_kernel(pwm_dir: Path, kernel_name: str) -> np.ndarray:
    """
    尝试读取形如:
    pwm_dir / f"{kernel_name}.pwm.tsv"
    的文件 (4 x L)
    """
    tsv_path = pwm_dir / f"{kernel_name}.pwm.tsv"
    if tsv_path.exists():
        mat = []
        with open(tsv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                mat.append([float(x) for x in parts])
        pwm = np.array(mat, dtype=float)
        # 若是 4 x L 或 L x 4，都做一下处理
        if pwm.shape[0] == 4:
            return pwm
        elif pwm.shape[1] == 4:
            return pwm.T
        else:
            raise ValueError(f"tsv 形状奇怪: {pwm.shape}, 期待 4xL 或 Lx4")

    raise FileNotFoundError(f"找不到 PWM 文件: {tsv_path}")



# -----------------------
# motif 筛选
# -----------------------

def select_top_motifs_by_stage(df, stage, n=10,
                               auc_min=0.55, p_kw_max=1e-10) -> pd.DataFrame:
    sub = df[df["stage"] == stage].copy()
    sub = sub[sub["kruskal_p_allstages"] <= p_kw_max]
    sub = sub[sub["auc_one_vs_rest"] >= auc_min]
    if sub.empty:
        return sub
    sub["score"] = sub["auc_one_vs_rest"] * sub["rho_with_ribo_residual_RNA"].abs()
    sub = sub.sort_values("score", ascending=False)
    return sub.head(n)


# -----------------------
# 绘制 Figure 3 主图（4 个 kernel）
# -----------------------

def plot_kernel_panel(
    ax_logo,
    ax_bar,
    kernel_name,
    highlight_stage,
    df_activity,
    pwm_dir,
    show_logo_ylabel: bool,
    show_bar_ylabel: bool,
):
    # ---- logo 部分 ----
    pwm = load_pwm_for_kernel(pwm_dir, kernel_name)
    if highlight_stage is None:
        title = f"{kernel_name}\n(global)"
    else:
        title = f"{kernel_name}\n({highlight_stage}-biased)"

    plot_seq_logo_from_pwm(
        pwm,
        title=title,
        ax=ax_logo,
        show_ylabel=show_logo_ylabel,
    )

    # ---- bar 部分 ----
    sub = df_activity[df_activity["kernel"] == kernel_name].copy()
    sub["stage"] = pd.Categorical(sub["stage"], categories=STAGE_ORDER, ordered=True)
    sub = sub.sort_values("stage")

    xs = np.arange(len(STAGE_ORDER))
    rho = sub["rho_with_ribo_residual_RNA"].values

    colors = []
    for st in STAGE_ORDER:
        if highlight_stage is not None and st == highlight_stage:
            colors.append("#d62728")  # 红
        else:
            colors.append("#1f77b4")  # 蓝

    ax_bar.bar(xs, rho, color=colors, width=0.7)
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(STAGE_ORDER, rotation=45, ha="right", fontsize=8)

    if show_bar_ylabel:
        ax_bar.set_ylabel("rho_with_ribo_residual_RNA", fontsize=8)
    else:
        ax_bar.set_ylabel("")
        ax_bar.set_yticklabels([])

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)


def make_figure3_main(df_activity, pwm_dir: Path, outdir: Path,
                      global_kernel: str,
                      stage_kernels: dict):
    """
    stage_kernels: {"4C": "kernel_k12_f17", "8C": "...", "hESC": "..."}
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- NEW: 统一 y 轴范围（基于 4 个 kernel 的 rho） ----------
    all_kernels = [global_kernel] + [stage_kernels[s] for s in ["4C", "8C", "hESC"] if s in stage_kernels]
    all_rho = []
    for k in all_kernels:
        sub = df_activity[df_activity["kernel"] == k].copy()
        sub["stage"] = pd.Categorical(sub["stage"], categories=STAGE_ORDER, ordered=True)
        sub = sub.sort_values("stage")
        vals = pd.to_numeric(sub["rho_with_ribo_residual_RNA"], errors="coerce").fillna(0.0).values
        all_rho.append(vals)
    all_rho = np.concatenate(all_rho) if len(all_rho) else np.array([0.0])

    # 以 0 为中心，做对称 ylim（更适合有正有负的 rho）
    ymax = float(np.max(np.abs(all_rho))) if all_rho.size else 1.0
    ymax = max(ymax, 1e-6)
    pad = 0.08 * ymax
    y_lim = (-ymax - pad, ymax + pad)
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(
        2, 4,
        height_ratios=[2.2, 1.0],
        hspace=0.15,
        wspace=0.15
    )

    # 第 1 列：global
    ax_logo0 = fig.add_subplot(gs[0, 0])
    ax_bar0  = fig.add_subplot(gs[1, 0])
    plot_kernel_panel(
        ax_logo0, ax_bar0,
        global_kernel, None,
        df_activity, pwm_dir,
        show_logo_ylabel=True,
        show_bar_ylabel=True,
    )
    ax_bar0.set_ylim(*y_lim)  # NEW

    # 后三列：特异 motifs（bar sharey 到 ax_bar0）
    col = 1
    for stage, kname in stage_kernels.items():
        ax_logo = fig.add_subplot(gs[0, col])

        # ✅ NEW: sharey，让 y 轴尺度一致（对齐/可比）
        ax_bar = fig.add_subplot(gs[1, col], sharey=ax_bar0)

        plot_kernel_panel(
            ax_logo, ax_bar,
            kname, stage,
            df_activity, pwm_dir,
            show_logo_ylabel=False,
            show_bar_ylabel=False,
        )

        # ✅ NEW: sharey 之后最好也显式 set_ylim（更稳）
        ax_bar.set_ylim(*y_lim)

        col += 1

    fig.tight_layout()
    fig.savefig(outdir / "figure3_main_panels.png", dpi=300)
    plt.close(fig)
    print(f"[Figure 3] 主图保存到 {outdir/'figure3_main_panels.png'}")


# -----------------------
# 全局分布图 & heatmap
# -----------------------

def plot_kw_p_distribution(df_activity, outdir: Path):
    df_unique = df_activity.drop_duplicates(subset=["kernel"])
    p = df_unique["kruskal_p_allstages"].values
    # 避免 log(0)
    p = np.clip(p, 1e-300, 1.0)
    neglogp = -np.log10(p)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(neglogp, bins=50)
    ax.set_xlabel("-log10(Kruskal–Wallis p)", fontsize=10)
    ax.set_ylabel("# kernels", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "figure3_kw_p_distribution.png", dpi=300)
    plt.close(fig)


def plot_max_auc_distribution(df_activity, outdir: Path):
    df_auc = (df_activity
              .groupby("kernel")["auc_one_vs_rest"]
              .max()
              .reset_index())
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df_auc["auc_one_vs_rest"].values, bins=50)
    ax.set_xlabel("max AUC(one-vs-rest) across stages", fontsize=10)
    ax.set_ylabel("# kernels", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "figure3_max_auc_distribution.png", dpi=300)
    plt.close(fig)

def plot_global_heatmap(
    df_activity,
    outdir: Path,
    value_col="rho_with_ribo_residual_RNA",
    top_k=100,
    cluster_mode="by_peak_stage",
):
    """
    画 kernel × stage 的 heatmap，只展示 top_k 个最 stage-specific 的 kernel。

    cluster_mode:
      - "by_peak_stage": 按每个 kernel 的 peak stage(最大 abs value 所在 stage) 分块排序
      - "absmax": 仅按 max|value| 排序
    """
    # -----------------------------
    # pivot: kernel × stage
    # -----------------------------
    df_piv = (
        df_activity
        .pivot_table(
            index="kernel",
            columns="stage",
            values=value_col,
            aggfunc="mean",
        )
    )

    # stage 顺序
    cols = [s for s in STAGE_ORDER if s in df_piv.columns]
    df_piv = df_piv[cols].copy()

    # ✅ 强制数值化，避免 abs() 遇到 str
    df_piv = df_piv.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # -----------------------------
    # 选 top_k kernels
    # -----------------------------
    abs_max = df_piv.abs().max(axis=1)
    top_kernels = abs_max.sort_values(ascending=False).head(top_k).index
    df_top = df_piv.loc[top_kernels].copy()

    # -----------------------------
    # 行重排：按 peak stage 分块
    # -----------------------------
    if cluster_mode == "by_peak_stage":
        peak_stage = df_top.abs().idxmax(axis=1)
        stage_rank = {st: i for i, st in enumerate(cols)}
        peak_rank = peak_stage.map(stage_rank).fillna(10**9).astype(int)

        df_top["_peak_rank"] = peak_rank
        df_top["_absmax"] = df_top.abs().max(axis=1)

        df_top = df_top.sort_values(
            by=["_peak_rank", "_absmax"],
            ascending=[True, False],
        )

        row_labels = df_top.index.tolist()
        df_top = df_top.drop(columns=["_peak_rank", "_absmax"])

    elif cluster_mode == "absmax":
        row_labels = df_top.index.tolist()
    else:
        raise ValueError(f"Unknown cluster_mode={cluster_mode}")

    # -----------------------------
    # 画图（⬅⬅⬅ 这里拉长）
    # -----------------------------
    n_rows = df_top.shape[0]

    # 👉 每个 kernel 给 0.18–0.22 inch，高度明显更松
    height = max(6.0, 0.15 * n_rows)
    width = max(6.0, 1.1 * len(cols))

    fig, ax = plt.subplots(
        figsize=(width, height),
        constrained_layout=True,   # 比 tight_layout 稳
    )

    vmax = float(np.max(np.abs(df_top.values))) if df_top.size else 1.0
    vmax = max(vmax, 1e-6)

    im = ax.imshow(
        df_top.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    # y 轴字号随 kernel 数自适应
    if n_rows <= 40:
        y_fs = 9
    elif n_rows <= 80:
        y_fs = 7
    else:
        y_fs = 6

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=y_fs)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Stage", fontsize=11)
    ax.set_ylabel("Kernel", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label(value_col, fontsize=10)

    out_png = outdir / f"figure3_heatmap_{value_col}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------
# Δh 绘图（可选）
# -----------------------

def plot_delta_violin(delta_df: pd.DataFrame, kernels, outdir: Path):
    """
    delta_df: 长表或宽表均可。
    - 若包含列 'stage' 则视为长表：列 [stage, kernel, delta_h]
    - 若第一列是 stage，后面每列是 kernel，则先 melt 成长表。
    """
    if "stage" in delta_df.columns and "kernel" in delta_df.columns:
        long_df = delta_df.copy()
    else:
        # 默认宽表：第一列 stage，后面是 kernel 列
        long_df = delta_df.melt(id_vars=delta_df.columns[0],
                                var_name="kernel", value_name="delta_h")
        long_df = long_df.rename(columns={delta_df.columns[0]: "stage"})

    long_df["stage"] = pd.Categorical(long_df["stage"],
                                      categories=STAGE_ORDER,
                                      ordered=True)

    for k in kernels:
        sub = long_df[long_df["kernel"] == k].copy()
        if sub.empty:
            print(f"[Δh] 警告：在 delta_df 中找不到 {k}")
            continue
        fig, ax = plt.subplots(figsize=(4, 3))
        # 手写一个简单双轴 box + 点分布，避免依赖 seaborn
        stages = [st for st in STAGE_ORDER if st in sub["stage"].unique()]
        xs = np.arange(len(stages))
        data = [sub[sub["stage"] == st]["delta_h"].values for st in stages]
        ax.boxplot(data, positions=xs, widths=0.6, showfliers=False)
        # 叠加轻微 jitter 的散点
        for i, d in enumerate(data):
            if len(d) == 0:
                continue
            jitter = (np.random.rand(len(d)) - 0.5) * 0.3
            ax.scatter(xs[i] + jitter, d, s=4, alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(stages, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Δh (cell-state effect)", fontsize=10)
        ax.set_title(k, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(outdir / f"figure3_delta_violin_{k}.png", dpi=300)
        plt.close(fig)


# -----------------------
# main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity-csv", type=str, required=True,
                    help="kernel_stage_activity.csv")
    ap.add_argument("--pwm-dir", type=str, required=True,
                    help="存放各 kernel PWM 的目录")
    ap.add_argument("--outdir", type=str, required=True,
                    help="输出目录")
    ap.add_argument("--global-kernel", type=str, default="kernel_k20_f8",
                    help="Figure 3 中 global motif 的 kernel 名")
    ap.add_argument("--stage-kernel", type=str, action="append", default=[],
                    help="指定 stage 特异 motif，格式如 4C:kernel_k12_f17，可重复多次")
    ap.add_argument("--delta-csv", type=str, default=None,
                    help="可选，Δh 的 CSV 文件路径")
    ap.add_argument("--top-n", type=int, default=10,
                    help="自动筛选 top motif 时每个 stage 保留数量")
    args = ap.parse_args()

    activity_path = Path(args.activity_csv)
    pwm_dir = Path(args.pwm_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_act = load_activity_table(activity_path)

    # 1) 输出自动筛选的 top motifs 表
    stage_list_interest = ["4C", "8C", "hESC"]
    frames = []
    for st in stage_list_interest:
        sub = select_top_motifs_by_stage(df_act, st, n=args.top_n)
        if sub.empty:
            print(f"[Top motifs] stage {st} 没有满足阈值的 kernel")
        else:
            sub = sub.copy()
            # sub 里已经有 stage 列了，这里不要再插入
            # 如果你想显式标记“peak_stage”，可以额外加一列叫 peak_stage
            # sub["peak_stage"] = st
            frames.append(sub)
    if frames:
        df_top_all = pd.concat(frames, axis=0)
        df_top_all.to_csv(outdir / "top_motifs_auto_selected.csv", index=False)
        print(f"[Top motifs] 自动筛选结果已保存到 {outdir/'top_motifs_auto_selected.csv'}")

    # 2) 解析用户指定的 stage-kernel 映射
    stage_kernels = {}
    for item in args.stage_kernel:
        # e.g. "4C:kernel_k12_f17"
        if ":" not in item:
            raise SystemExit(f"--stage-kernel 格式错误: {item}")
        st, kn = item.split(":", 1)
        st = st.strip()
        kn = kn.strip()
        stage_kernels[st] = kn

    # 3) 生成 Figure 3 主图
    make_figure3_main(df_act, pwm_dir, outdir,
                      global_kernel=args.global_kernel,
                      stage_kernels=stage_kernels)

    # 4) 分布图 & heatmap
    plot_kw_p_distribution(df_act, outdir)
    plot_max_auc_distribution(df_act, outdir)
    plot_global_heatmap(df_act, outdir,
                        value_col="rho_with_ribo_residual_RNA", top_k=100)

    # 5) 可选：Δh 图
    if args.delta_csv is not None:
        delta_path = Path(args.delta_csv)
        delta_df = pd.read_csv(delta_path)
        # Δh 图只画 Figure 3 用到的 kernels（节省空间）
        kernels_for_delta = [args.global_kernel] + list(stage_kernels.values())
        plot_delta_violin(delta_df, kernels_for_delta, outdir)

    # 6) 拼成一个完整 Figure 3
    assemble_figure3_full(outdir,
                          global_kernel=args.global_kernel,
                          stage_kernels=stage_kernels)
if __name__ == "__main__":
    main()
