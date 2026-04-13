#!/usr/bin/env python3
"""
Analyze alpha sweep results and append findings to results/findings_and_next_steps.md.

Usage:
    uv run python scripts/analyze_alpha_sweep.py
    uv run python scripts/analyze_alpha_sweep.py --dataset swebench --mem 0.22
"""
import json, glob, os, statistics, argparse
from pathlib import Path

REPO = Path(__file__).parent.parent

def load_results(base, configs):
    results = {}
    for cfg in configs:
        files = glob.glob(f"{base}/{cfg}/*.jsonl")
        if not files:
            continue
        reqs = [json.loads(l) for f in files for l in open(f)]
        total_p = sum(r.get("prompt_tokens", 0) for r in reqs)
        total_c = sum(r.get("cached_tokens", 0) for r in reqs)
        errors  = sum(1 for r in reqs if r.get("error"))
        ttfts   = [r["ttft_ms"] for r in reqs if r.get("ttft_ms", 0) > 0]
        hit_pct = total_c / total_p * 100 if total_p else 0

        # Per-session breakdown
        by_sess = {}
        for r in reqs:
            s = r.get("session_id", 0)
            by_sess.setdefault(s, {"p": 0, "c": 0, "n": 0})
            by_sess[s]["p"] += r.get("prompt_tokens", 0)
            by_sess[s]["c"] += r.get("cached_tokens", 0)
            by_sess[s]["n"] += 1

        results[cfg] = {
            "hit": hit_pct,
            "ttft": statistics.mean(ttfts) if ttfts else 0,
            "n": len(reqs),
            "errors": errors,
            "sessions": {s: d["c"]/d["p"]*100 if d["p"] else 0 for s, d in by_sess.items()},
            "avg_session_len": statistics.mean([d["p"]/d["n"] for d in by_sess.values() if d["n"]]) if by_sess else 0,
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/alpha-sweep")
    ap.add_argument("--dataset", default="swebench")
    ap.add_argument("--mem", default="0.22")
    args = ap.parse_args()

    configs = ["lru", "marc_a0.3", "marc_a0.5", "marc_a0.7", "marc_a1.0", "marc_a1.5"]
    results = load_results(args.base, configs)

    if not results:
        print("No results found yet.")
        return

    lru = results.get("lru", {})
    lru_hit = lru.get("hit", 0)

    # ── Console table ──────────────────────────────────────────────────────────
    print(f"\nALPHA SWEEP RESULTS — {args.dataset}  mem={args.mem}")
    print(f"{'Config':>14}  {'Hit%':>7}  {'Δ LRU':>7}  {'AvgTTFT':>9}  {'N':>6}")
    print("  " + "-" * 52)
    for cfg in configs:
        if cfg not in results:
            continue
        r = results[cfg]
        delta = r["hit"] - lru_hit
        bar = ("▲" * int(abs(delta)/2)) if delta > 0 else ("▼" * int(abs(delta)/2))
        print(f"  {cfg:>12}  {r['hit']:>6.1f}%  {delta:>+6.1f}pp  {r['ttft']:>8.0f}ms  {r['n']:>6}  {bar}")

    # ── Per-session breakdown: which alphas help which sessions ───────────────
    all_sessions = sorted(set(
        s for r in results.values() for s in r.get("sessions", {})
    ))

    if all_sessions:
        print(f"\nPer-session hit rate (session_id → avg prompt len):")
        header = f"  {'sess':>5}"
        for cfg in configs:
            if cfg in results:
                header += f"  {cfg:>10}"
        print(header)
        print("  " + "-" * (8 + 12 * len([c for c in configs if c in results])))
        for s in all_sessions:
            row = f"  {s:>5}"
            lru_s = results.get("lru", {}).get("sessions", {}).get(s, 0)
            for cfg in configs:
                if cfg not in results:
                    continue
                val = results[cfg].get("sessions", {}).get(s, 0)
                marker = "▲" if (cfg != "lru" and val > lru_s + 5) else (
                         "▼" if (cfg != "lru" and val < lru_s - 5) else " ")
                row += f"  {val:>8.1f}%{marker}"
            print(row)

    # ── Identify crossover alpha ───────────────────────────────────────────────
    print(f"\nAlpha crossover analysis:")
    best_cfg = max((c for c in configs if c in results), key=lambda c: results[c]["hit"])
    best_hit = results[best_cfg]["hit"]
    print(f"  Best config: {best_cfg}  ({best_hit:.1f}%)")
    if lru_hit > 0:
        print(f"  LRU baseline: {lru_hit:.1f}%")
        gap = best_hit - lru_hit
        print(f"  Best Marconi vs LRU: {gap:+.1f}pp  → {'Marconi wins' if gap > 0 else 'LRU wins'}")

    # ── Write findings to markdown ─────────────────────────────────────────────
    md_path = REPO / "results" / "findings_and_next_steps.md"
    existing = md_path.read_text() if md_path.exists() else ""

    section_header = f"\n---\n\n## Alpha Sweep Results ({args.dataset}, mem={args.mem})\n"
    if section_header.strip() in existing:
        print(f"\n(Section already in {md_path} — overwriting)")
        # Remove old section
        idx = existing.find(section_header.strip())
        next_section = existing.find("\n---\n", idx + 10)
        existing = existing[:idx] + (existing[next_section:] if next_section > 0 else "")

    lines = [section_header]
    lines.append(f"```\n{'Config':>14}  {'Hit%':>7}  {'Δ LRU':>7}  {'AvgTTFT':>9}  {'N':>6}\n")
    for cfg in configs:
        if cfg not in results:
            continue
        r = results[cfg]
        delta = r["hit"] - lru_hit
        lines.append(f"{'':2}{cfg:>12}  {r['hit']:>6.1f}%  {delta:>+6.1f}pp  {r['ttft']:>8.0f}ms  {r['n']:>6}\n")
    lines.append("```\n")

    lines.append("\n**Per-session hit rates** (▲=Marconi wins this session vs LRU, ▼=LRU wins):\n\n")
    if all_sessions:
        hdr = f"| sess |"
        for cfg in configs:
            if cfg in results:
                hdr += f" {cfg} |"
        lines.append(hdr + "\n")
        lines.append("|" + "---|" * (1 + len([c for c in configs if c in results])) + "\n")
        for s in all_sessions:
            row = f"| {s} |"
            lru_s = results.get("lru", {}).get("sessions", {}).get(s, 0)
            for cfg in configs:
                if cfg not in results:
                    continue
                val = results[cfg].get("sessions", {}).get(s, 0)
                marker = "▲" if (cfg != "lru" and val > lru_s + 5) else (
                         "▼" if (cfg != "lru" and val < lru_s - 5) else "")
                row += f" {val:.0f}%{marker} |"
            lines.append(row + "\n")

    lines.append(f"\n**Key finding:** Best = `{best_cfg}` at {best_hit:.1f}% vs LRU {lru_hit:.1f}% ({best_hit-lru_hit:+.1f}pp)\n")

    lines.append("\n**Interpretation:**\n")
    marc_winners = [c for c in configs if c != "lru" and c in results and results[c]["hit"] > lru_hit + 1]
    lru_winners = [c for c in configs if c != "lru" and c in results and results[c]["hit"] < lru_hit - 1]
    if marc_winners:
        lines.append(f"- Marconi beats LRU at α ∈ {{{', '.join(a.replace('marc_a','') for a in marc_winners)}}}\n")
    if lru_winners:
        lines.append(f"- LRU beats Marconi at α ∈ {{{', '.join(a.replace('marc_a','') for a in lru_winners)}}}\n")

    # Identify sessions where high alpha helps vs hurts
    if all_sessions and "marc_a1.5" in results and "marc_a0.3" in results:
        high_alpha_wins = []
        low_alpha_wins = []
        for s in all_sessions:
            lru_s = results["lru"].get("sessions", {}).get(s, 0) if "lru" in results else 0
            high = results["marc_a1.5"].get("sessions", {}).get(s, 0)
            low = results["marc_a0.3"].get("sessions", {}).get(s, 0)
            if high > lru_s + 5:
                high_alpha_wins.append(s)
            if low < lru_s - 5:
                low_alpha_wins.append(s)
        if high_alpha_wins:
            lines.append(f"- High α (1.5) wins on sessions: {high_alpha_wins} — long/repetitive prefixes dominate\n")
        if low_alpha_wins:
            lines.append(f"- Low α (0.3) still hurts sessions: {low_alpha_wins} — short chains always lose vs recency-only\n")

    updated = existing.rstrip() + "\n" + "".join(lines)
    md_path.write_text(updated)
    print(f"\nFindings written to {md_path}")


if __name__ == "__main__":
    main()
