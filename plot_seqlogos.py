#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_seqlogos.py

将 rank_kernels_and_score_with_pwm.py 产生的 PWM/PMF 文件绘制为 seqlogo 图片，并可选合并导出为 MEME v4。

输入：
  - 目录下的 *.pwm.tsv 或 *.pfm.tsv （4 x K，行序为 A/C/G/T，列为位点）
输出：
  - 对每个输入矩阵生成 PNG（可选 SVG/PDF），文件名与输入一致（后缀改为 .png/.svg/.pdf）
  - 可选：输出到同一 outdir
  - （新增）合并所有 PWM/PMF 为一个 MEME 文件（--save_meme）

绘图模式（--mode）：
  - bits（默认）：Schneider 信息量 logo；列高 = 2 - H(p)（比特），字母高 = p_i * 列高
  - prob：列高 = 1；字母高 = p_i
  - logodds：字母高 = log2(p_i / b_i)（可正可负，在 0 轴上下分别堆叠）
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

NUCS = ["A","C","G","T"]
# 经典配色（A 绿, C 蓝, G 橙, T 红）
NUC_COLORS = {
    "A": "#4daf4a",
    "C": "#377eb8",
    "G": "#ff7f00",
    "T": "#e41a1c",
}

def load_matrix(path, wanted="auto"):
    """
    读取 4 x K 的矩阵。wanted in {"pwm","pfm","auto"}
    返回 pwm(4xK, 行序 A/C/G/T)、以及是否来自 pfm 的标记。
    """
    mat = np.loadtxt(path, delimiter="\t", dtype=float)
    if mat.shape[0] != 4:
        # 可能是 K x 4
        if mat.shape[1] == 4:
            mat = mat.T
        else:
            raise SystemExit(f"[ERR] {path} 不是 4xK 或 Kx4 矩阵")
    mat = np.asarray(mat, dtype=np.float64)

    suffix = str(path).lower()
    if wanted == "auto":
        is_pfm = suffix.endswith(".pfm.tsv") or "pfm" in suffix
    else:
        is_pfm = (wanted == "pfm")

    if is_pfm:
        # normalize to probabilities (加极小量防 0 列)
        colsum = mat.sum(axis=0, keepdims=True) + 1e-12
        pwm = mat / colsum
        return pwm, True
    else:
        # assume already probs; 再做稳健归一化
        pwm = mat / (mat.sum(axis=0, keepdims=True) + 1e-12)
        return pwm, False

def compute_heights(pwm, mode="bits", bg=None):
    """
    根据模式返回每个字母的绘制高度（4 x K）。
    - bits:  h_i = p_i * (log2(4) - H(p))，列高 <= 2 bits
    - prob:  h_i = p_i
    - logodds: h_i = log2(p_i / b_i)（允许为负；bg 长度为4）
    """
    eps = 1e-12
    if mode == "prob":
        return pwm
    elif mode == "bits":
        p = np.clip(pwm, eps, 1.0)
        H = -(p * np.log2(p)).sum(axis=0, keepdims=True)  # [1,K]
        R = np.log2(4.0) - H                              # [1,K]  <= 2
        return pwm * R
    elif mode == "logodds":
        if bg is None:
            bg = np.array([0.25,0.25,0.25,0.25], dtype=np.float64)
        bg = np.asarray(bg, dtype=np.float64).reshape(4,1)
        p = np.clip(pwm, eps, 1.0)
        b = np.clip(bg, eps, 1.0)
        return np.log2(p / b)  # 可正可负
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _add_letter(ax, tp, x_left, col_width, y_bottom, desired_height, color):
    """
    把单个字母（TextPath）画到指定列：
    - 统一用 glyph 的外接框做归一化：先平移到 (0,0)，再按目标宽/高缩放
    - 水平居中到列中
    """
    # glyph 外接框（size=1.0 下）
    bbox = tp.get_extents()
    w0 = bbox.width
    h0 = bbox.height
    minx = bbox.x0
    miny = bbox.y0
    if h0 <= 0 or w0 <= 0 or desired_height <= 0:
        return

    # 目标缩放：高度按 desired_height，宽度占列宽的 90%
    sx = (col_width * 0.90) / w0
    sy = (desired_height) / h0

    # 水平居中：列中心 - (缩放后 glyph 宽度)/2
    scaled_w = w0 * sx
    x_pos = x_left + (col_width - scaled_w) * 0.5
    # 底对齐：把 glyph 的 miny 移到 0，然后放到 y_bottom
    y_pos = y_bottom - (miny * sy)

    trans = Affine2D().translate(-minx, -miny).scale(sx, sy).translate(x_pos, y_pos)
    patch = PathPatch(tp, lw=0, fc=color, transform=trans + ax.transData)
    ax.add_patch(patch)

def draw_seqlogo(pwm, out_path, title=None, mode="bits",
                 bg=None, width_per_bp=0.6, top_margin=0.2,
                 fig_height=2.8, fmt="png", dpi=160):
    """
    绘制单个 PWM 的 seqlogo（修正基线错位）
    - pwm: 4 x K（A/C/G/T）
    """
    pwm = pwm.astype(np.float64)
    K = pwm.shape[1]

    heights = compute_heights(pwm, mode=mode, bg=bg)  # 4 x K（可能有负值）
    # 每列正负堆叠的总高度
    pos_sum = heights.clip(min=0).sum(axis=0)
    neg_sum = (-heights.clip(max=0)).sum(axis=0)

    # 画布尺寸
    xspan = K * width_per_bp
    fig_w = max(3.0, xspan + 1.0)
    fig_h = fig_height

    fig, ax = plt.subplots(1,1, figsize=(fig_w, fig_h))
    ax.set_xlim(0, xspan)
    ymax = float(np.max(pos_sum)) + top_margin
    ymin = -float(np.max(neg_sum)) - 0.1
    ax.set_ylim(ymin, ymax if ymax > 0 else 0.5)

    # 预生成字形（size=1.0），后面按外接框归一化
    tp_cache = {n: TextPath((0,0), n, size=1.0, prop=None) for n in NUCS}

    for j in range(K):
        x_left = j * width_per_bp

        # 这一列，各字母的目标高度
        col_h = {n: heights[i, j] for i, n in enumerate(NUCS)}
        pos_items = sorted([(n,h) for n,h in col_h.items() if h>0], key=lambda x:x[1])  # 小到大堆
        neg_items = sorted([(n,h) for n,h in col_h.items() if h<0], key=lambda x:x[1])  # 更负在下

        # 正向堆叠（从 0 往上）
        y = 0.0
        for n, h in pos_items:
            _add_letter(ax, tp_cache[n], x_left, width_per_bp, y_bottom=y, desired_height=h, color=NUC_COLORS[n])
            y += h

        # 负向堆叠（从 0 往下）
        y = 0.0
        for n, h in neg_items:  # h < 0
            hh = abs(h)
            # 往下堆：当前底部 = y - hh
            _add_letter(ax, tp_cache[n], x_left, width_per_bp, y_bottom=y - hh, desired_height=hh, color=NUC_COLORS[n])
            y -= hh

    ax.set_xticks(np.arange(0, xspan, width_per_bp))
    ax.set_xticklabels([str(i+1) for i in range(K)], fontsize=8)
    ax.set_xlim(0, xspan)
    ax.set_ylabel({"bits":"bits","prob":"prob.","logodds":"log2 odds"}[mode])
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)

def infer_title_from_name(name):
    # e.g., "kernel_k16_f62.pwm.tsv" -> "kernel_k16_f62"
    n = Path(name).name
    if n.endswith(".tsv"): n = n[:-4]
    return n

# ---------- NEW: 写 MEME v4 ----------
def write_meme(motifs, meme_path, bg=None, nsites=0):
    """
    motifs: list of (name, pwm) with pwm shape 4xK, rows A/C/G/T
    bg: array-like of length 4 for A,C,G,T (probabilities sum to 1)
    """
    meme_path = Path(meme_path)
    meme_path.parent.mkdir(parents=True, exist_ok=True)

    if bg is None:
        bg = np.array([0.25,0.25,0.25,0.25], dtype=np.float64)
    bg = np.asarray(bg, dtype=np.float64)
    bg = bg / (bg.sum() + 1e-12)

    with open(meme_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write(f"A {bg[0]:.3f} C {bg[1]:.3f} G {bg[2]:.3f} T {bg[3]:.3f}\n\n")

        for name, pwm in motifs:
            pwm = np.asarray(pwm, dtype=np.float64)
            if pwm.shape[0] != 4:
                raise ValueError(f"MOTIF {name}: PWM should be 4xK (rows A,C,G,T)")
            # 规范化到列和=1
            pwm = pwm / (pwm.sum(axis=0, keepdims=True) + 1e-12)
            K = pwm.shape[1]
            f.write(f"MOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {K} nsites= {int(nsites)} E= 0\n")
            for j in range(K):
                a,c,g,t = pwm[0,j], pwm[1,j], pwm[2,j], pwm[3,j]
                f.write(f"{a:.6f}\t{c:.6f}\t{g:.6f}\t{t:.6f}\n")
            f.write("\n")

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_dir", type=str, help="包含 *.pwm.tsv / *.pfm.tsv 的目录")
    g.add_argument("--file", type=str, help="单个 pwm/pfm 文件")

    ap.add_argument("--glob", type=str, default="*.pwm.tsv", help="在 input_dir 下的匹配模式，如 '*.pwm.tsv' 或 '*.pfm.tsv'")
    ap.add_argument("--outdir", type=str, default=None, help="输出目录（默认与输入相同目录）")
    ap.add_argument("--mode", choices=["bits","prob","logodds"], default="bits")
    ap.add_argument("--bg", type=str, default=None,
                    help="logodds & MEME 的背景频率，逗号分隔 4 个数（按 A,C,G,T）；默认 0.25,0.25,0.25,0.25")
    ap.add_argument("--fmt", choices=["png","svg","pdf"], default="png")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--width_per_bp", type=float, default=0.6)
    ap.add_argument("--height", type=float, default=2.8)
    ap.add_argument("--title", action="store_true", help="标题显示文件名（不含后缀）")
    ap.add_argument("--wanted", choices=["auto","pwm","pfm"], default="auto",
                    help="输入文件类型；auto 根据文件名判断")

    # NEW:
    ap.add_argument("--save_meme", action="store_true", help="将遍历到的 PWM/PMF 合并导出为一个 MEME 文件")
    ap.add_argument("--meme_path", type=str, default=None, help="MEME 输出路径（默认 <outdir>/meme.txt）")
    ap.add_argument("--nsites", type=int, default=0, help="MEME 的 nsites 字段（仅作说明性）")

    args = ap.parse_args()

    if args.bg is not None:
        try:
            bg = np.array([float(x) for x in args.bg.split(",")], dtype=np.float64)
            assert bg.size == 4
            bg = bg / (bg.sum() + 1e-12)
        except Exception as e:
            raise SystemExit(f"--bg 解析失败：{e}")
    else:
        bg = np.array([0.25,0.25,0.25,0.25], dtype=np.float64)

    # 收集输入文件
    files = []
    if args.file:
        files = [Path(args.file)]
    else:
        d = Path(args.input_dir)
        files = sorted(d.glob(args.glob))

    if not files:
        print("[WARN] 没有匹配到任何输入文件。")
        return

    # 收集用于 MEME 的 (name, pwm)
    meme_motifs = []

    for fp in files:
        pwm, from_pfm = load_matrix(fp, wanted=args.wanted)
        name = infer_title_from_name(fp)
        title = name if args.title else None
        outdir = Path(args.outdir) if args.outdir else fp.parent
        out_path = outdir / name  # 脱掉 .tsv 后缀

        # 绘图
        draw_seqlogo(
            pwm=pwm,
            out_path=out_path,
            title=title,
            mode=args.mode,
            bg=bg,
            width_per_bp=args.width_per_bp,
            top_margin=0.2,
            fig_height=args.height,
            fmt=args.fmt,
            dpi=args.dpi
        )
        print(f"[OK] {fp.name} -> {out_path.with_suffix('.'+args.fmt)}")

        if args.save_meme:
            meme_motifs.append((name, pwm))

    # 写 MEME
    if args.save_meme and meme_motifs:
        outdir = Path(args.outdir) if args.outdir else files[0].parent
        meme_path = Path(args.meme_path) if args.meme_path else (outdir / "meme.txt")
        write_meme(meme_motifs, meme_path, bg=bg, nsites=args.nsites)
        print(f"[OK] MEME written: {meme_path}")

if __name__ == "__main__":
    main()
