#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_motif_hits_figure3.py

在全数据集上扫描指定 kernel（如 kernel_k20_f8）在输入序列上的激活位置：
输出 CSV 列：
  gene, stage, kernel,
  pos_start_padded0, pos_end_padded0,
  pos_start_raw0, pos_end_raw0,
  activation, kmer,
  rna, ribo,
  seq_len_raw, seq_max_len, left_pad

说明：
- gene / stages / seq_lengths / seq_max_len / other_feature_names 全部来自 npz 同名 .json
- rna 来自 X_other 中对应 stage 的 rna_<stage> 列（build_model_inputs.py 会写入）:contentReference[oaicite:3]{index=3}
- ribo 来自 Y[:, stage_id]
- 位置 raw 坐标使用对称 padding 的 left_pad = (L - len(seq))//2（与 build_model_inputs.py 一致）:contentReference[oaicite:4]{index=4}

用法示例（Figure3 这四个 kernel）:contentReference[oaicite:5]{index=5} :
python scan_motif_hits_figure3.py \
  --npz  ./preprocess_data/pipeline_out/model_inputs_cds_lenle5000_withRNA.npz \
  --ckpt ./result/cnn_mlp_add_cellstate/single_label_v1/best.pt \
  --kernels kernel_k20_f8 kernel_k12_f17 kernel_k12_f29 kernel_k16_f11 \
  --out_csv figure3_out/motif_scan_hits.csv \
  --batch_size 256 \
  --top_hits_per_sample 3 \
  --global_quantile 0.995
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_kernel_name(name: str) -> Tuple[int, int]:
    # kernel_k20_f8 -> (20, 8)
    name = name.strip()
    if not name.startswith("kernel_k") or "_f" not in name:
        raise ValueError(f"Bad kernel name: {name}")
    k_part, f_part = name.split("_f", 1)
    K = int(k_part.replace("kernel_k", ""))
    f = int(f_part)
    return K, f


def onehot_to_seq(onehot_k4: np.ndarray) -> str:
    bases = np.array(["A", "C", "G", "T"])
    out = []
    for i in range(onehot_k4.shape[0]):
        row = onehot_k4[i]
        if np.all(row <= 0):
            out.append("N")
        else:
            out.append(bases[int(np.argmax(row))])
    return "".join(out)


class SeqCNNBranch(nn.Module):
    def __init__(self, in_ch=4, kernel_sizes=(6, 10, 12, 16, 20), channels_per_kernel=64):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_ch, channels_per_kernel, k, padding=0) for k in kernel_sizes])


def load_seq_convs_from_ckpt(ckpt_path: str, kernel_sizes: List[int], cnn_channels: int) -> SeqCNNBranch:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    seq = SeqCNNBranch(in_ch=4, kernel_sizes=tuple(kernel_sizes), channels_per_kernel=cnn_channels)

    # 只抽取 seq_branch.convs 的权重（你的训练模型里就是这个名字）:contentReference[oaicite:6]{index=6}
    sd = {}
    for k, v in state.items():
        if "seq_branch.convs" in k:
            idx = k.find("seq_branch.convs")
            kk = k[idx:]                # seq_branch.convs.0.weight
            kk = kk.replace("seq_branch.", "")  # convs.0.weight
            sd[kk] = v

    missing, _unexpected = seq.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading convs: {missing}")
    return seq


def build_single_channel_conv(seq_bank: SeqCNNBranch, K: int, f: int) -> nn.Conv1d:
    layer = None
    for conv in seq_bank.convs:
        if int(conv.kernel_size[0]) == int(K):
            layer = conv
            break
    if layer is None:
        raise ValueError(f"Cannot find conv layer for kernel size K={K}")
    if f < 0 or f >= layer.out_channels:
        raise ValueError(f"Channel f={f} out of range for K={K} (out_channels={layer.out_channels})")

    c = nn.Conv1d(4, 1, K, padding=0, bias=True)
    c.weight.data[:] = layer.weight.data[f:f+1].clone()
    c.bias.data[:] = layer.bias.data[f:f+1].clone()
    return c


def load_npz_and_meta(npz_path: str) -> Dict[str, object]:
    npz_path = str(npz_path)
    meta_path = str(Path(npz_path).with_suffix(".json"))
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"meta json not found: {meta_path}")

    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    pack = np.load(npz_path)

    X_seq = pack["X_seq"]  # uint8 [N,L,4]
    X_other = pack["X_other"].astype(np.float32)  # [N,l]
    Y = pack["Y"].astype(np.float32)              # [N,S]

    genes = meta.get("genes")
    stages = meta.get("stages")
    feat_names = meta.get("other_feature_names", [])
    seq_lengths = meta.get("seq_lengths")
    L = int(meta.get("seq_max_len"))

    if genes is None or stages is None or seq_lengths is None:
        raise ValueError("meta.json missing genes/stages/seq_lengths.")
    if len(genes) != X_seq.shape[0] or len(seq_lengths) != X_seq.shape[0]:
        raise ValueError("meta genes/seq_lengths length != X_seq N.")
    if len(stages) != Y.shape[1]:
        raise ValueError("meta stages length != Y second dim.")
    if len(feat_names) != X_other.shape[1]:
        raise ValueError("meta other_feature_names length != X_other dim.")

    # 找到每个 stage 的 RNA 列（rna_<stage>）:contentReference[oaicite:7]{index=7}
    rna_idx_map = {}
    for i, n in enumerate(feat_names):
        n = str(n)
        if n.startswith("rna_"):
            rna_idx_map[n[4:]] = i

    missing = [st for st in stages if st not in rna_idx_map]
    if missing:
        raise ValueError(f"Missing rna_<stage> columns for: {missing}")

    return {
        "X_seq": X_seq,
        "X_other": X_other,
        "Y": Y,
        "genes": list(map(str, genes)),
        "stages": list(map(str, stages)),
        "seq_lengths": np.asarray(seq_lengths, dtype=np.int64),
        "seq_max_len": L,
        "rna_idx_map": rna_idx_map,
    }


@torch.no_grad()
def estimate_threshold_by_quantile(
    device: torch.device,
    conv1: nn.Conv1d,
    X_seq: np.ndarray,
    batch_size: int,
    quantile: float,
) -> float:
    # 用每条序列的 max activation 做分位数阈值（内存友好、速度快）
    vals = []
    N = X_seq.shape[0]
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        xs = torch.from_numpy(X_seq[s:e].astype(np.float32)).to(device).transpose(1, 2).contiguous()  # [B,4,L]
        z = F.relu(F.conv1d(xs, conv1.weight, conv1.bias, stride=1, padding=0))  # [B,1,L']
        m = torch.amax(z, dim=-1).squeeze(1)  # [B]
        vals.append(m.detach().cpu().numpy())
    v = np.concatenate(vals, axis=0)
    return float(np.quantile(v, quantile))


@torch.no_grad()
def scan_kernel_hits(
    device: torch.device,
    kernel_name: str,
    conv1: nn.Conv1d,
    X_seq: np.ndarray,          # [N,L,4] uint8/float
    genes: List[str],
    stages: List[str],
    seq_lengths: np.ndarray,    # [N]
    L: int,
    X_other: np.ndarray,        # [N,l]
    Y: np.ndarray,              # [N,S]
    rna_idx_map: Dict[str, int],
    top_hits_per_sample: int,
    global_thr: Optional[float],
    batch_size: int,
    max_rows: int,
) -> pd.DataFrame:
    K = int(conv1.kernel_size[0])
    N = X_seq.shape[0]
    rows = []
    kept = 0

    # 逐 stage 扫描（输出里带 stage）
    for stage_id, stage in enumerate(stages):
        rna_col = rna_idx_map[stage]
        rna_vec = X_other[:, rna_col].astype(np.float32)   # [N]
        ribo_vec = Y[:, stage_id].astype(np.float32)       # [N]

        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)

            xs_np = X_seq[s:e].astype(np.float32)  # [B,L,4]
            xs = torch.from_numpy(xs_np).to(device).transpose(1, 2).contiguous()  # [B,4,L]
            z = F.relu(F.conv1d(xs, conv1.weight, conv1.bias, stride=1, padding=0)).squeeze(1)  # [B,L']

            k_eff = min(top_hits_per_sample, z.shape[1])
            topv, topi = torch.topk(z, k=k_eff, dim=1)  # [B,k]
            topv = topv.detach().cpu().numpy()
            topi = topi.detach().cpu().numpy()

            for bi in range(e - s):
                i = s + bi
                gene = genes[i]
                seq_len = int(seq_lengths[i])
                left_pad = int((L - seq_len) // 2)  # 与 build_model_inputs 的对称 padding 一致 :contentReference[oaicite:8]{index=8}

                for j in range(k_eff):
                    act = float(topv[bi, j])
                    if global_thr is not None and act < global_thr:
                        continue

                    pos0 = int(topi[bi, j])       # padded start (0-based)
                    pos1 = pos0 + K               # padded end (exclusive)

                    raw0 = pos0 - left_pad
                    raw1 = raw0 + K

                    # kmer（从 one-hot 反解，基于 padded 坐标）
                    kmer_oh = xs_np[bi, pos0:pos1, :]  # [K,4]
                    kmer = onehot_to_seq(kmer_oh)

                    rows.append({
                        "gene": gene,
                        "stage": stage,
                        "kernel": kernel_name,
                        "pos_start_padded0": pos0,
                        "pos_end_padded0": pos1,
                        "pos_start_raw0": raw0,
                        "pos_end_raw0": raw1,
                        "activation": act,
                        "kmer": kmer,
                        "rna": float(rna_vec[i]),
                        "ribo": float(ribo_vec[i]),
                        "seq_len_raw": seq_len,
                        "seq_max_len": L,
                        "left_pad": left_pad,
                    })
                    kept += 1
                    if kept >= max_rows:
                        return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kernels", nargs="+", required=True)
    ap.add_argument("--out_csv", required=True)

    # 与训练结构一致的 conv 参数（默认与你的训练/提取脚本一致）
    ap.add_argument("--kernel_sizes", type=int, nargs="+", default=[6, 10, 12, 16, 20])
    ap.add_argument("--cnn_channels", type=int, default=64)

    # 扫描策略
    ap.add_argument("--top_hits_per_sample", type=int, default=3)
    ap.add_argument("--global_quantile", type=float, default=0.995,
                    help="用 max-activation 的分位数做强激活阈值；设为 <0 则不做阈值过滤")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_rows_per_kernel", type=int, default=500000)

    args = ap.parse_args()

    D = load_npz_and_meta(args.npz)
    X_seq = D["X_seq"]
    X_other = D["X_other"]
    Y = D["Y"]
    genes = D["genes"]
    stages = D["stages"]
    seq_lengths = D["seq_lengths"]
    L = D["seq_max_len"]
    rna_idx_map = D["rna_idx_map"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_bank = load_seq_convs_from_ckpt(args.ckpt, args.kernel_sizes, args.cnn_channels).to(device).eval()

    out_frames = []
    for kn in args.kernels:
        K, f = parse_kernel_name(kn)
        conv1 = build_single_channel_conv(seq_bank, K, f).to(device).eval()

        thr = None
        if args.global_quantile is not None and args.global_quantile > 0:
            thr = estimate_threshold_by_quantile(
                device=device,
                conv1=conv1,
                X_seq=X_seq.astype(np.float32),
                batch_size=int(args.batch_size),
                quantile=float(args.global_quantile),
            )

        df = scan_kernel_hits(
            device=device,
            kernel_name=kn,
            conv1=conv1,
            X_seq=X_seq,
            genes=genes,
            stages=stages,
            seq_lengths=seq_lengths,
            L=L,
            X_other=X_other,
            Y=Y,
            rna_idx_map=rna_idx_map,
            top_hits_per_sample=int(args.top_hits_per_sample),
            global_thr=thr,
            batch_size=int(args.batch_size),
            max_rows=int(args.max_rows_per_kernel),
        )
        if thr is not None:
            df["global_thr"] = thr
        out_frames.append(df)

    df_all = pd.concat(out_frames, axis=0, ignore_index=True) if out_frames else pd.DataFrame()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv}  rows={len(df_all)}")


if __name__ == "__main__":
    main()
