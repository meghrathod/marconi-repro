#!/usr/bin/env python3
"""
Parse Marconi mamba eviction trace from server log.

Usage:
    python scripts/parse_eviction_trace.py logs/capacity-test/server_marconi_trace.log
    python scripts/parse_eviction_trace.py logs/capacity-test/server_marconi_trace.log --max 20
"""

import sys
import re
import argparse


def parse(log_path, max_evictions=None):
    # Match the header line: "evict mamba num: N | n_cands=M ts=T"
    header_re = re.compile(
        r"evict mamba num: (\d+) \| n_cands=(\d+) ts=([\d.]+)"
    )
    # Match candidate lines: "[rank] id=X toks=Y/path=Z eff=E(n=NE) rec_n=NR util=U"
    cand_re = re.compile(
        r"\[(\d+)\] id=(\d+) toks=(\d+)/path=(\d+) eff=([\d.]+)\(n=([\d.]+)\) rec_n=([\d.]+) util=([\d.]+)"
    )

    evictions = []
    current = None

    with open(log_path) as f:
        for line in f:
            hm = header_re.search(line)
            if hm:
                if current is not None:
                    evictions.append(current)
                if max_evictions is not None and len(evictions) >= max_evictions:
                    break
                current = {
                    "need": int(hm.group(1)),
                    "n_cands": int(hm.group(2)),
                    "ts": float(hm.group(3)),
                    "candidates": [],
                }
                continue

            cm = cand_re.search(line)
            if cm and current is not None:
                current["candidates"].append({
                    "rank": int(cm.group(1)),
                    "node_id": int(cm.group(2)),
                    "node_toks": int(cm.group(3)),
                    "path_toks": int(cm.group(4)),
                    "eff": float(cm.group(5)),
                    "norm_eff": float(cm.group(6)),
                    "norm_rec": float(cm.group(7)),
                    "util": float(cm.group(8)),
                })

    if current is not None:
        evictions.append(current)

    if not evictions:
        print(f"No 'evict mamba num' trace found in {log_path}")
        print("Tip: ensure server was run with --radix-eviction-policy marconi")
        return

    print(f"Found {len(evictions)} mamba eviction events\n")
    print(f"{'#':>4}  {'need':>5}  {'cands':>6}  {'ts':>8}  "
          f"EVICTED → id/toks(node/path)/eff(norm)/rec_n/util  |  "
          f"PROTECTED (rank last) → id/toks/eff(norm)/rec_n/util")
    print("-" * 120)

    for i, ev in enumerate(evictions):
        cands = ev["candidates"]
        worst = cands[0] if cands else None
        best = cands[-1] if cands else None

        def fmt(c):
            if c is None:
                return "n/a"
            return (f"id={c['node_id']} {c['node_toks']}/{c['path_toks']}tok "
                    f"eff={c['eff']:.3f}({c['norm_eff']:.2f}) "
                    f"rec={c['norm_rec']:.2f} u={c['util']:.3f}")

        print(f"{i:>4}  {ev['need']:>5}  {ev['n_cands']:>6}  {ev['ts']:>8.0f}  "
              f"EVICT [{fmt(worst)}]  |  KEEP [{fmt(best)}]")

    print()

    # Summary stats
    evicted = [ev["candidates"][0] for ev in evictions if ev["candidates"]]
    if evicted:
        avg_toks = sum(c["node_toks"] for c in evicted) / len(evicted)
        avg_path = sum(c["path_toks"] for c in evicted) / len(evicted)
        avg_eff = sum(c["eff"] for c in evicted) / len(evicted)
        short = sum(1 for c in evicted if c["node_toks"] < 100)
        print(f"Summary: {len(evicted)} evictions")
        print(f"  avg evicted node_toks = {avg_toks:.1f}  (short<100: {short}/{len(evicted)} = {100*short//len(evicted)}%)")
        print(f"  avg evicted path_toks = {avg_path:.1f}")
        print(f"  avg evicted eff       = {avg_eff:.4f}")

        protected = [ev["candidates"][-1] for ev in evictions if len(ev["candidates"]) > 1]
        if protected:
            pavg_toks = sum(c["node_toks"] for c in protected) / len(protected)
            pavg_eff = sum(c["eff"] for c in protected) / len(protected)
            print(f"\n  avg protected node_toks = {pavg_toks:.1f}")
            print(f"  avg protected eff       = {pavg_eff:.4f}")
            print(f"\n  → Marconi evicts nodes with eff={avg_eff:.4f} and protects eff={pavg_eff:.4f}")
            if pavg_eff > avg_eff * 1.2:
                print("  Marconi strongly prefers high-eff (long-path) nodes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="Path to server log")
    ap.add_argument("--max", type=int, default=None, dest="max_evictions",
                    help="Stop after N eviction events")
    args = ap.parse_args()
    parse(args.log, max_evictions=args.max_evictions)