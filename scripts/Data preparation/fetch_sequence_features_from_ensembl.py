#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_sequence_features_from_ensembl_robust.py

- Maps gene symbols to Ensembl gene/transcripts (canonical preferred, else longest CDS)
- Batch fetches sequences via POST /sequence/id for cds/utr5/utr3
- Computes sequence features AND (optionally) saves raw sequences per gene
- Checkpointing & resume; verbose per-gene logs

Usage:
  python fetch_sequence_features_from_ensembl_robust.py \
    --genes_txt ../data/genes.txt \
    --out_csv   ../preprocess_data/features.csv \
    --resume_csv ../preprocess_data/features.csv \
    --timeout 60 --sleep_sec 0.15 --checkpoint_every 50 \
    --save_sequences  # add columns: cds_seq, utr5_seq, utr3_seq
"""
import argparse, time, re, os
from typing import Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd, numpy as np

ENSEMBL_REST = "https://rest.ensembl.org"

def build_session(timeout: int) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=6, connect=4, read=4, backoff_factor=0.8,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=['GET','POST'],
                  respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    s.mount("https://", adapter); s.mount("http://", adapter)
    s.request_timeout = timeout
    s.headers.update({"User-Agent": "rna2ribo/seqfeatures (contact: user)"})
    return s

def http_get_json(session: requests.Session, url: str, params=None):
    headers = {"Content-Type":"application/json", "Accept":"application/json"}
    r = session.get(url, headers=headers, params=params, timeout=session.request_timeout)
    if r.status_code == 429 and r.headers.get("Retry-After"):
        time.sleep(float(r.headers["Retry-After"])); r = session.get(url, headers=headers, params=params, timeout=session.request_timeout)
    return r.json() if r.status_code == 200 else None

def http_post_json(session: requests.Session, url: str, body: Dict):
    headers = {"Content-Type":"application/json", "Accept":"application/json"}
    r = session.post(url, headers=headers, json=body, timeout=session.request_timeout)
    if r.status_code == 429 and r.headers.get("Retry-After"):
        time.sleep(float(r.headers["Retry-After"])); r = session.post(url, headers=headers, json=body, timeout=session.request_timeout)
    return r.json() if r.status_code == 200 else None

def lookup_symbol(session: requests.Session, species: str, symbol: str):
    return http_get_json(session, f"{ENSEMBL_REST}/lookup/symbol/{species}/{requests.utils.quote(symbol)}", params={"expand":1})

def choose_transcript(gene_obj: Dict):
    txs = gene_obj.get("Transcript", []) or []
    if not txs: return None
    cid = gene_obj.get("canonical_transcript")
    if cid:
        for t in txs:
            if t.get("id") == cid: return t
    best, bestL = None, -1
    for t in txs:
        tr = t.get("Translation")
        L = tr.get("length") if isinstance(tr, dict) else None
        try: L = int(L) if L is not None else -1
        except Exception: L = -1
        if L > bestL: best, bestL = t, L
    return best or txs[0]

def fasta_to_seq(x: Optional[str]) -> Optional[str]:
    if not x: return None
    s = x.strip()
    if not s: return None
    if s.startswith(">"):
        return "".join([ln.strip() for ln in s.splitlines() if ln and not ln.startswith(">")])
    return s

def batch_sequences(session: requests.Session, enst_ids: List[str], seq_type: str) -> Dict[str, Optional[str]]:
    payload = {"ids": enst_ids, "type": seq_type}
    res = http_post_json(session, f"{ENSEMBL_REST}/sequence/id", payload)
    out = {e: None for e in enst_ids}
    if isinstance(res, list):
        for item in res:
            iid = item.get("id"); seq = item.get("seq")
            if iid in out: out[iid] = fasta_to_seq(seq)
    return out

def gc_content(seq: Optional[str]) -> float:
    if not seq: return np.nan
    s = seq.upper(); total = sum(s.count(x) for x in "ACGT")
    if total == 0: return np.nan
    return (s.count("G")+s.count("C"))/total

def kozak_score(utr5: Optional[str], cds: Optional[str]) -> float:
    if not utr5 or not cds or len(cds) < 4: return np.nan
    u = utr5.upper(); c = cds.upper()
    if not c.startswith("ATG"): return np.nan
    left = u[-6:] if len(u) >= 6 else None; right = c[:4]
    if left is None or len(right) < 4: return np.nan
    score = 0.0
    for pos in range(2,6):
        if left[pos] in ("G","C"): score += 0.5
    if left[3] in ("A","G"): score += 1.0
    if right[3] == "G": score += 1.0
    return score

def count_uorf_basic(utr5: Optional[str]) -> Tuple[float, float]:
    if not utr5: return (np.nan, np.nan)
    s = utr5.upper()
    starts = [m.start() for m in re.finditer("ATG", s)]
    stop_codons = {"TAA","TAG","TGA"}
    with_stop = 0
    for st in starts:
        for i in range(st+3, len(s)-2, 3):
            if s[i:i+3] in stop_codons: with_stop += 1; break
    return float(len(starts)), float(with_stop)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes_txt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--species", default="homo_sapiens")
    ap.add_argument("--sleep_sec", type=float, default=0.15)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--checkpoint_every", type=int, default=50)
    ap.add_argument("--resume_csv", default=None)
    ap.add_argument("--save_sequences", action="store_true")
    args = ap.parse_args()

    sess = build_session(args.timeout)
    genes = [ln.strip() for ln in open(args.genes_txt, "r", encoding="utf-8") if ln.strip()]
    genes = list(dict.fromkeys(genes))

    done = set(); rows = []
    if args.resume_csv and os.path.exists(args.resume_csv):
        try:
            prev = pd.read_csv(args.resume_csv)
            rows = prev.to_dict(orient="records"); done = set(prev["gene"].astype(str).tolist())
            print(f"[resume] loaded {len(done)} rows from {args.resume_csv}")
        except Exception as e:
            print(f"[resume] failed to read {args.resume_csv}: {e}")

    pending = []
    for g in genes:
        if g in done: continue
        obj = lookup_symbol(sess, args.species, g)
        if not obj:
            rows.append({"gene": g, "ensembl_gene_id": np.nan, "transcript_id": np.nan, "is_canonical": np.nan,
                         "cds_len": np.nan, "utr5_len": np.nan, "utr3_len": np.nan,
                         "gc_cds": np.nan, "gc_utr5": np.nan, "gc_utr3": np.nan,
                         "kozak_score": np.nan, "uorf_atg_count": np.nan, "uorf_stop_in_frame_count": np.nan,
                         **({"cds_seq": None, "utr5_seq": None, "utr3_seq": None} if args.save_sequences else {}),
                         "status": "gene_not_found"})
            continue
        t = choose_transcript(obj)
        if not t:
            rows.append({"gene": g, "ensembl_gene_id": obj.get("id"), "transcript_id": np.nan, "is_canonical": np.nan,
                         "cds_len": np.nan, "utr5_len": np.nan, "utr3_len": np.nan,
                         "gc_cds": np.nan, "gc_utr5": np.nan, "gc_utr3": np.nan,
                         "kozak_score": np.nan, "uorf_atg_count": np.nan, "uorf_stop_in_frame_count": np.nan,
                         **({"cds_seq": None, "utr5_seq": None, "utr3_seq": None} if args.save_sequences else {}),
                         "status": "no_transcript"})
            continue
        pending.append((g, obj, t))
        time.sleep(args.sleep_sec)

    if not pending:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"[done] wrote {len(rows)} rows → {args.out_csv}")
        return

    enst = [t.get("id") for _,_,t in pending]
    cds_map = batch_sequences(sess, enst, "cds"); time.sleep(args.sleep_sec)
    u5_map  = batch_sequences(sess, enst, "utr5"); time.sleep(args.sleep_sec)
    u3_map  = batch_sequences(sess, enst, "utr3"); time.sleep(args.sleep_sec)

    for i,(g, obj, t) in enumerate(pending, 1):
        ensg = obj.get("id"); enst_id = t.get("id")
        is_can = 1 if obj.get("canonical_transcript") == enst_id else 0
        cds = cds_map.get(enst_id); u5 = u5_map.get(enst_id); u3 = u3_map.get(enst_id)

        row = {
            "gene": g, "ensembl_gene_id": ensg, "transcript_id": enst_id, "is_canonical": is_can,
            "cds_len": len(cds) if isinstance(cds, str) else np.nan,
            "utr5_len": len(u5) if isinstance(u5, str) else np.nan,
            "utr3_len": len(u3) if isinstance(u3, str) else np.nan,
            "gc_cds": gc_content(cds), "gc_utr5": gc_content(u5), "gc_utr3": gc_content(u3),
            "kozak_score": kozak_score(u5, cds),
            "uorf_atg_count": count_uorf_basic(u5)[0],
            "uorf_stop_in_frame_count": count_uorf_basic(u5)[1],
            "status": "ok" if any([isinstance(cds,str), isinstance(u5,str), isinstance(u3,str)]) else "no_sequence"
        }
        if args.save_sequences:
            row.update({"cds_seq": cds, "utr5_seq": u5, "utr3_seq": u3})
        rows.append(row)

        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            print(f"[checkpoint] {i}/{len(pending)} saved → {args.out_csv}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {len(df)} rows → {args.out_csv}")
    print(df["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
