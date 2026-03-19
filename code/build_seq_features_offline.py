#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_seq_features_offline.py

Offline sequence feature builder using local Ensembl FASTA files.
See top of file for details.
"""
import argparse, gzip, io, os, re
from typing import Dict, Tuple, Optional
import pandas as pd, numpy as np

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def parse_fasta_with_meta(path: str) -> Dict[str, Dict[str, str]]:
    d = {}
    with open_maybe_gzip(path) as fh:
        tid, buf, meta = None, [], {}
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if tid is not None:
                    d[tid] = {"seq": "".join(buf).upper(), **meta}
                buf = []
                header = line[1:]
                first = header.split()[0]
                tid = first.split(".")[0]
                gene_id, gene_symbol = None, None
                for token in header.split():
                    if token.startswith("gene:"):
                        gene_id = token.split("gene:")[1].split(".")[0]
                    elif token.startswith("gene_symbol:"):
                        gene_symbol = token.split("gene_symbol:")[1]
                meta = {"gene_id": gene_id, "gene_symbol": gene_symbol}
            else:
                buf.append(line)
        if tid is not None:
            d[tid] = {"seq": "".join(buf).upper(), **meta}
    return d

def gc_content(seq: Optional[str]) -> float:
    if not seq: return np.nan
    s = seq.upper(); acgt = sum(s.count(x) for x in "ACGT")
    return ((s.count("G")+s.count("C"))/acgt) if acgt else np.nan

def kozak_score(utr5: Optional[str], cds: Optional[str]) -> float:
    if not utr5 or not cds or len(cds) < 4: return np.nan
    u, c = utr5.upper(), cds.upper()
    if not c.startswith("ATG"): return np.nan
    left = u[-6:] if len(u) >= 6 else None; right = c[:4]
    if left is None or len(right) < 4: return np.nan
    score = 0.0
    for pos in range(2,6):
        if left[pos] in ("G","C"): score += 0.5
    if left[3] in ("A","G"): score += 1.0
    if right[3] == "G": score += 1.0
    return score

def count_uorf_basic(utr5: Optional[str]):
    if not utr5: return (np.nan, np.nan)
    s = utr5.upper()
    starts = [m.start() for m in re.finditer("ATG", s)]
    stop_codons = {"TAA","TAG","TGA"}
    with_stop = 0
    for st in starts:
        for i in range(st+3, len(s)-2, 3):
            if s[i:i+3] in stop_codons:
                with_stop += 1; break
    return float(len(starts)), float(with_stop)

def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes_txt", required=True)
    ap.add_argument("--cdna_fa", required=True)
    ap.add_argument("--cds_fa", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--save_sequences", action="store_true")
    args = ap.parse_args()

    cdna = parse_fasta_with_meta(args.cdna_fa)
    cds  = parse_fasta_with_meta(args.cds_fa)

    best = {}
    for tid, rec in cds.items():
        sym = rec.get("gene_symbol"); seq = rec.get("seq")
        if not sym or not seq: continue
        L = len(seq)
        if (sym not in best) or (L > best[sym][1]):
            best[sym] = (tid, L)

    genes = [ln.strip() for ln in open(args.genes_txt, "r", encoding="utf-8") if ln.strip()]
    genes = list(dict.fromkeys(genes))

    rows = []
    for g in genes:
        if g not in best:
            rows.append({"gene": g, "ensembl_gene_id": np.nan, "transcript_id": np.nan, "is_canonical": np.nan,
                         "cds_len": np.nan, "utr5_len": np.nan, "utr3_len": np.nan,
                         "gc_cds": np.nan, "gc_utr5": np.nan, "gc_utr3": np.nan,
                         "kozak_score": np.nan, "uorf_atg_count": np.nan, "uorf_stop_in_frame_count": np.nan,
                         **({"cds_seq": None, "utr5_seq": None, "utr3_seq": None} if args.save_sequences else {}),
                         "status": "gene_not_found_in_FASTA"})
            continue
        tid, _ = best[g]
        cds_seq = cds.get(tid, {}).get("seq")
        gene_id = cds.get(tid, {}).get("gene_id")
        cdna_seq = cdna.get(tid, {}).get("seq")

        utr5_seq, utr3_seq = None, None
        if isinstance(cdna_seq, str) and isinstance(cds_seq, str):
            pos = cdna_seq.find(cds_seq)
            if pos != -1:
                utr5_seq = cdna_seq[:pos]
                utr3_seq = cdna_seq[pos+len(cds_seq):]

        row = {
            "gene": g, "ensembl_gene_id": gene_id, "transcript_id": tid, "is_canonical": np.nan,
            "cds_len": len(cds_seq) if isinstance(cds_seq, str) else np.nan,
            "utr5_len": len(utr5_seq) if isinstance(utr5_seq, str) else np.nan,
            "utr3_len": len(utr3_seq) if isinstance(utr3_seq, str) else np.nan,
            "gc_cds": gc_content(cds_seq), "gc_utr5": gc_content(utr5_seq), "gc_utr3": gc_content(utr3_seq),
            "kozak_score": kozak_score(utr5_seq, cds_seq),
            "uorf_atg_count": count_uorf_basic(utr5_seq)[0],
            "uorf_stop_in_frame_count": count_uorf_basic(utr5_seq)[1],
            "status": "ok" if isinstance(cds_seq, str) else "no_cds"
        }
        if args.save_sequences:
            row.update({"cds_seq": cds_seq, "utr5_seq": utr5_seq, "utr3_seq": utr3_seq})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {len(df)} rows -> {args.out_csv}")
    print(df["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
