"""
Live Marconi experiment analysis — Nemotron-H-8B vs CPU simulation.

Five figures:
  fig1_sim_vs_live.png        — 2×3 combined panel: simulation (top) vs live (bottom)
  fig2_alpha_sweep.png        — Per-trace hit rates: LRU / Marconi α=0.3 / α=1.0
  fig3_theory_vs_practice.png — Marconi/baseline ratio: simulation predicts >1×, live <1×
  fig4_eviction_analysis.png  — LRU vs Marconi eviction characteristics
  fig5_ttft.png               — TTFT distribution by policy (v2 run)

Run from repo root: uv run python scripts/analysis/live_analysis.py
Output: figures/output/
"""

import json, re, os, glob
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
V1_DIR   = "results/live-minimal-32K"
V2_DIR   = "results/live-minimal-32K-v2"
SIM_LOGS = "marconi/logs"

COLORS = {
    "no-cache":     "#cccccc",
    "lru":          "#52B788",
    "marconi_a0.3": "#74C69D",
    "marconi_a0.7": "#2D6A4F",
    "marconi_a1.0": "#081C15",
    "vllm":         "#A8DADC",
    "sglang":       "#457B9D",
    "sim_marconi":  "#1D3557",
}

TRACE_LABELS = {
    "lmsys_sps=0.25_nums=100":         "LMSys\nsps=0.25",
    "lmsys_sps=1_nums=100":            "LMSys\nsps=1",
    "lmsys_sps=5_nums=100":            "LMSys\nsps=5",
    "sharegpt_sps=0.25_nums=100":      "ShareGPT\nsps=0.25",
    "sharegpt_sps=1_nums=100":         "ShareGPT\nsps=1",
    "sharegpt_sps=5_nums=100":         "ShareGPT\nsps=5",
    "swebench_sps=1_art=5_nums=100":   "SWEBench\nsps=1",
    "swebench_sps=5_art=5_nums=100":   "SWEBench\nsps=5\nart=5",
    "swebench_sps=5_art=7.5_nums=100": "SWEBench\nsps=5\nart=7.5",
}

DS_TRACES = {
    "lmsys":    ["lmsys_sps=0.25_nums=100", "lmsys_sps=1_nums=100", "lmsys_sps=5_nums=100"],
    "sharegpt": ["sharegpt_sps=0.25_nums=100", "sharegpt_sps=1_nums=100", "sharegpt_sps=5_nums=100"],
    "swebench": ["swebench_sps=1_art=5_nums=100", "swebench_sps=5_art=5_nums=100",
                 "swebench_sps=5_art=7.5_nums=100"],
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_dir(base, subdir):
    out = {}
    for path in sorted(glob.glob(f"{base}/{subdir}/*.jsonl")):
        trace = os.path.basename(path).replace(".jsonl", "")
        with open(path) as f:
            out[trace] = [json.loads(l) for l in f if l.strip()]
    return out

def load_live():
    return {
        "no-cache":     _load_dir(V1_DIR, "no-cache"),
        "lru_v1":       _load_dir(V1_DIR, "lru"),
        "marconi_a0.7": _load_dir(V1_DIR, "marconi"),
        "lru":          _load_dir(V2_DIR, "lru"),
        "marconi_a0.3": _load_dir(V2_DIR, "marconi_a0.3"),
        "marconi_a1.0": _load_dir(V2_DIR, "marconi_a1.0"),
    }

def load_sim():
    """
    Mirror authors' exact parsing (last occurrence per block wins).
    Returns per-dataset list of {vllm, v2, hit_rate_win} dicts.
    - vllm / v2: absolute token hit rates (%)
    - hit_rate_win: V2 relative improvement over V1 (SGLang+), same metric as authors' fig8
                    = (V2_hr - V1_hr) / V1_hr * 100
    """
    pat_hdr = re.compile(r"Cache size ([\d.e+]+) .* sessions per second: ([\d.]+)")
    pat_hit = re.compile(r"(?P<scheme>\w+[\+]?): hit rate (?P<hit_rate>[\d\.]+)%")
    pat_win = re.compile(r"V2 compared to V1: hit_rate_win (?P<win>-?[\d.]+)%")
    out = {}
    for ds in ["lmsys", "sharegpt", "swebench"]:
        rows = []
        with open(f"{SIM_LOGS}/{ds}.txt") as f:
            for entry in f.read().split("=" * 50):
                if not pat_hdr.search(entry): continue
                hits = {}
                for m in pat_hit.finditer(entry):
                    hits[m.group("scheme")] = float(m.group("hit_rate"))
                win_m = pat_win.search(entry)
                if "V1" in hits and "V2" in hits:
                    rows.append({
                        "v1":  hits["V1"],
                        "v2":  hits["V2"],
                        "hit_rate_win": float(win_m.group("win")) if win_m else None,
                    })
        out[ds] = rows
    return out

def trace_hr(rows):
    tp = sum(r["prompt_tokens"]  for r in rows if r["error"] is None)
    tc = sum(r["cached_tokens"]  for r in rows if r["error"] is None)
    return tc / tp * 100 if tp else 0.0

def req_hrs(rows):
    # cache_hit_pct is already in percentage units (0–100)
    return [r["cache_hit_pct"] for r in rows
            if r["error"] is None and r["prompt_tokens"] > 0]

def _boxplot(ax, data, colors, vert=False, ylabels=None, xlabels=None):
    bp = ax.boxplot(data, vert=vert, showfliers=False, whis=[5, 95],
                    widths=0.5, whiskerprops=dict(linewidth=1.5), patch_artist=True)
    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color); patch.set_alpha(0.85)
        bp["medians"][i].set_color("white"); bp["medians"][i].set_linewidth(2)
        for s in [2*i, 2*i+1]:
            bp["whiskers"][s].set_color(color); bp["caps"][s].set_color(color)
        if data[i]:
            if vert:
                ax.scatter(i+1, np.mean(data[i]), color="white", marker="D", s=25, zorder=5)
            else:
                ax.scatter(np.mean(data[i]), i+1, color="white", marker="D", s=25, zorder=5)
    ticks = range(1, len(data)+1)
    if ylabels:
        ax.set_yticks(ticks); ax.set_yticklabels(ylabels, fontsize=8.5)
    if xlabels:
        ax.set_xticks(ticks); ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.grid(axis="y" if vert else "x", color="lightgrey", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)

def load_lru_evictions(path):
    events, cur = [], None
    p1 = re.compile(r"evict lru num: (\d+) \| n_cands=(\d+) ts=([\d.]+)")
    p2 = re.compile(r"\[0\] id=\d+ toks=(\d+)/path=(\d+) age=([\d.]+) is_leaf=(\w+)")
    try:
        with open(path) as f:
            for line in f:
                m = p1.search(line)
                if m:
                    cur = {"num": int(m.group(1)), "n_cands": int(m.group(2)), "ts": float(m.group(3))}
                    events.append(cur)
                elif cur and "toks" not in cur:
                    m2 = p2.search(line)
                    if m2:
                        cur.update({"toks": int(m2.group(1)), "path": int(m2.group(2)),
                                    "age": float(m2.group(3)), "is_leaf": m2.group(4) == "True"})
    except FileNotFoundError:
        pass
    return events

def load_marconi_evictions(path):
    events, cur = [], None
    p1 = re.compile(r"evict mamba num: (\d+) \| n_cands=(\d+) ts=([\d.]+)")
    p2 = re.compile(r"\[0\] id=\d+ toks=(\d+)/path=(\d+) eff=[\d.]+\(n=([\d.]+)\) rec_n=([\d.]+) util=([\d.]+)")
    try:
        with open(path) as f:
            for line in f:
                m = p1.search(line)
                if m:
                    cur = {"num": int(m.group(1)), "n_cands": int(m.group(2)), "ts": float(m.group(3))}
                    events.append(cur)
                elif cur and "toks" not in cur:
                    m2 = p2.search(line)
                    if m2:
                        cur.update({"toks": int(m2.group(1)), "path": int(m2.group(2)),
                                    "norm_eff": float(m2.group(3)), "rec_n": float(m2.group(4))})
    except FileNotFoundError:
        pass
    return events


# ── Figure 1: Simulation vs Live combined (2×3 panel) ────────────────────────
# Top row: CPU simulation — SGLang+ vs Marconi (authors' Fig 7 style, no vLLM+)
# Bottom row: Live server — LRU vs Marconi α=0.3  (Fig 7 analogue)
# Both rows show absolute token hit rate distributions.

def fig1_sim_vs_live(sim, live):
    ds_info = [("lmsys", "LMSys"), ("sharegpt", "ShareGPT"), ("swebench", "SWEBench")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for col, (ds, title) in enumerate(ds_info):
        # ── top: simulation ─────────────────────────────────────────────────
        ax = axes[0][col]
        sim_sets = [[r["v1"] for r in sim[ds]], [r["v2"] for r in sim[ds]]]
        _boxplot(ax, sim_sets,
                 [COLORS["sglang"], COLORS["sim_marconi"]],
                 vert=False, ylabels=["SGLang+\n(V1)", "Marconi\n(V2)"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("CPU Simulation\n(paper traces)", fontsize=9)
        ax.set_xlabel("Token Hit Rate (%)", fontsize=8.5)
        xmax = max(max(r["v2"] for r in sim[ds]), max(r["v1"] for r in sim[ds]))
        ax.set_xlim(-2, min(xmax * 1.1 + 2, 102))

        # ── bottom: live ─────────────────────────────────────────────────────
        ax2 = axes[1][col]
        live_sets = []
        for p in ["lru", "marconi_a0.3", "marconi_a1.0"]:
            rates = []
            for t in DS_TRACES[ds]:
                rates.extend(req_hrs(live[p].get(t, [])))
            live_sets.append(rates)
        _boxplot(ax2, live_sets,
                 [COLORS["lru"], COLORS["marconi_a0.3"], COLORS["marconi_a1.0"]],
                 vert=False, ylabels=["LRU", "Marconi\nα=0.3", "Marconi\nα=1.0"])
        if col == 0:
            ax2.set_ylabel("Live Server\n(Nemotron-H-8B, 32K)", fontsize=9)
        ax2.set_xlabel("Token Hit Rate (%)", fontsize=8.5)
        ax2.set_xlim(-2, 102)

    fig.suptitle(
        "Token Hit Rate: CPU Simulation (top) vs Live Server (bottom)\n"
        "Simulation: Marconi gains over SGLang+  |  Live: LRU vs Marconi α=0.3 / α=1.0",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig1_sim_vs_live.png")


# ── Figure 2: Alpha sweep per trace (v2) ─────────────────────────────────────

def fig2_alpha_sweep(live):
    traces  = list(TRACE_LABELS.keys())
    labels  = [TRACE_LABELS[t] for t in traces]
    policies = [("lru", "LRU"), ("marconi_a0.3", "Marconi α=0.3"), ("marconi_a1.0", "Marconi α=1.0")]
    colors   = [COLORS["lru"], COLORS["marconi_a0.3"], COLORS["marconi_a1.0"]]

    hrs = {p: [trace_hr(live[p].get(t, [])) for t in traces] for p, _ in policies}

    x = np.arange(len(traces))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    for (p, plabel), off, color in zip(policies, offsets, colors):
        ax.bar(x + off, hrs[p], width, label=plabel, color=color, alpha=0.9)

    # Annotate Marconi delta vs LRU
    for i in range(len(traces)):
        for p, off in [("marconi_a0.3", 0), ("marconi_a1.0", width)]:
            delta = hrs[p][i] - hrs["lru"][i]
            ax.annotate(f"{delta:+.0f}", xy=(x[i] + off, hrs[p][i] + 1.5),
                        ha="center", fontsize=6.5,
                        color="green" if delta >= 0 else "red", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center")
    ax.set_ylabel("Token Hit Rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("v2 Run: Token Hit Rate — LRU vs Marconi α=0.3 vs α=1.0\n"
                 "(annotations = Marconi delta vs LRU; green = Marconi wins, red = LRU wins)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", color="lightgrey", linestyle="--", linewidth=0.7); ax.set_axisbelow(True)
    # Dataset separators
    for xpos in [2.5, 5.5]:
        ax.axvline(xpos, color="grey", lw=0.8, ls=":", alpha=0.6)
    for xpos, label in [(1, "LMSys"), (4, "ShareGPT"), (7, "SWEBench")]:
        ax.text(xpos, 97, label, ha="center", fontsize=9, color="grey", style="italic")
    plt.tight_layout()
    _save(fig, "fig2_alpha_sweep.png")


# ── Figure 3: Theory vs Practice — Fig 8 analogue ────────────────────────────
# Authors' Fig 8: horizontal boxplot of Marconi's % improvement over SGLang+
# Live equivalent:  % improvement of Marconi over LRU per trace
# Layout: 3 rows (one per dataset), 2 columns (simulation | live)

def fig3_theory_vs_practice(sim, live):
    ds_info = [("swebench", "SWEBench"), ("sharegpt", "ShareGPT"), ("lmsys", "LMSys")]

    # Collect sim hit_rate_win values (V2 vs V1) and live improvement values
    sim_wins  = {ds: [r["hit_rate_win"] for r in sim[ds] if r["hit_rate_win"] is not None]
                 for ds, _ in ds_info}
    live_wins = {}
    for ds, _ in ds_info:
        wins = {}
        for ak, lk in [("marconi_a0.3","lru"), ("marconi_a0.7","lru_v1"), ("marconi_a1.0","lru")]:
            w = []
            for t in DS_TRACES[ds]:
                lru_hr = trace_hr(live[lk].get(t, []))
                mar_hr = trace_hr(live[ak].get(t, []))
                if lru_hr > 0:
                    w.append((mar_hr - lru_hr) / lru_hr * 100)
            wins[ak] = w
        live_wins[ds] = wins

    # Build combined dataset: one row per dataset, boxes = [sim | α=0.3 | α=0.7 | α=1.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=False)

    for ax, (ds, title) in zip(axes, ds_info):
        data   = [sim_wins[ds],
                  live_wins[ds]["marconi_a0.3"],
                  live_wins[ds]["marconi_a0.7"],
                  live_wins[ds]["marconi_a1.0"]]
        colors = [COLORS["sim_marconi"], COLORS["marconi_a0.3"],
                  COLORS["marconi_a0.7"], COLORS["marconi_a1.0"]]
        labels = ["Sim\n(M vs SGLang+)", "Live\nα=0.3", "Live\nα=0.7", "Live\nα=1.0"]

        _boxplot(ax, data, colors, vert=True, xlabels=labels)
        ax.axhline(0, color="red", lw=1.5, ls="--", alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Hit Rate Improvement over Baseline (%)", fontsize=9)

        # Annotate median on each box
        for i, d in enumerate(data):
            if d:
                med = np.median(d)
                ax.text(i+1, med + (1 if med >= 0 else -4), f"{med:.1f}%",
                        ha="center", fontsize=7.5, fontweight="bold",
                        color="darkgreen" if med >= 0 else "darkred")

    fig.suptitle(
        "Fig 8 analogue: Marconi Hit Rate Improvement over Baseline\n"
        "Simulation baseline = SGLang+ (V1)  |  Live baseline = LRU  |  "
        "Red dashed = no improvement (0%)",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig3_theory_vs_practice.png")


# ── Figure 4: Eviction analysis (LRU vs Marconi) ──────────────────────────────

def fig4_eviction_analysis(lru_ev, marc_ev):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    C_LRU  = "#2166AC"   # blue
    C_MARC = "#D6604D"   # orange-red

    # 1. Cumulative evictions over time
    ax = axes[0]
    for events, label, color in [(lru_ev, "LRU", C_LRU),
                                   (marc_ev, "Marconi α=0.3", C_MARC)]:
        if events:
            ts  = np.array([e["ts"] for e in events])
            cum = np.cumsum([e["num"] for e in events])
            ax.plot(ts, cum, lw=1.5, color=color, label=f"{label} ({len(events):,} events)")
    ax.set_xlabel("Internal Timestamp", fontsize=9)
    ax.set_ylabel("Cumulative Evictions", fontsize=9)
    ax.set_title("Mamba State Evictions Over Time", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(color="lightgrey", linestyle="--"); ax.set_axisbelow(True)

    # 2. Eviction candidate pool size distribution
    ax = axes[1]
    for events, label, color in [(lru_ev, "LRU", C_LRU),
                                   (marc_ev, "Marconi α=0.3", C_MARC)]:
        if events:
            cands = [e["n_cands"] for e in events]
            ax.hist(cands, bins=40, alpha=0.65, color=color,
                    label=f"{label} (μ={np.mean(cands):.0f})", density=True)
    ax.set_xlabel("Eligible Candidates at Eviction (n_cands)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Pool Pressure at Each Eviction\n(higher n_cands = more pressure)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", color="lightgrey", linestyle="--"); ax.set_axisbelow(True)

    # 3. Evicted node size (segment tokens)
    ax = axes[2]
    for events, label, color in [(lru_ev, "LRU", C_LRU),
                                   (marc_ev, "Marconi α=0.3", C_MARC)]:
        toks = [e["toks"] for e in events if "toks" in e and e["toks"] > 0]
        if toks:
            ax.hist(np.log10(np.array(toks) + 1), bins=40, alpha=0.65, color=color,
                    label=f"{label} (μ={np.mean(toks):.0f} toks)", density=True)
    ax.set_xlabel("log₁₀(Evicted Segment Size + 1, tokens)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Size of Evicted Node Segments", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", color="lightgrey", linestyle="--"); ax.set_axisbelow(True)

    fig.suptitle("Eviction Characteristics: LRU vs Marconi α=0.3 (v2 run, same mamba pool cap=318)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig4_eviction_analysis.png")


# ── Figure 5: TTFT by policy ──────────────────────────────────────────────────

def fig5_ttft(live):
    ds_info = [("lmsys", "LMSys"), ("sharegpt", "ShareGPT"), ("swebench", "SWEBench")]
    v2_policies = ["lru", "marconi_a0.3", "marconi_a1.0"]
    colors      = [COLORS[p] for p in v2_policies]
    xlabels     = ["LRU", "Marconi\nα=0.3", "Marconi\nα=1.0"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, (ds, title) in zip(axes, ds_info):
        data = []
        for p in v2_policies:
            ttfts = []
            for t in DS_TRACES[ds]:
                ttfts.extend([r["ttft_ms"] for r in live[p].get(t, [])
                               if r["error"] is None and r.get("ttft_ms") is not None])
            data.append(ttfts)
        _boxplot(ax, data, colors, vert=True, xlabels=xlabels)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if ax is axes[0]: ax.set_ylabel("Time-to-First-Token (ms)", fontsize=9)
        # Annotate medians
        for i, d in enumerate(data):
            if d:
                ax.text(i+1, np.percentile(d, 95) * 1.02, f"{np.median(d):.0f}ms",
                        ha="center", fontsize=7.5, color=colors[i], fontweight="bold")

    fig.suptitle("TTFT Distribution by Policy (v2 run)\n"
                 "Marconi should reduce TTFT when cache hit rates are higher",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig5_ttft.png")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = f"{OUT_DIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def print_tables(live, sim):
    traces = list(TRACE_LABELS.keys())

    print("\n" + "=" * 95)
    print("TABLE 1  v2 Run — Token Hit Rates (TP=4, mamba cap=318)")
    print("=" * 95)
    print(f"{'Trace':<30} {'LRU':>8} {'α=0.3':>8} {'α=1.0':>8}  {'Δ α=0.3':>9}  {'Δ α=1.0':>9}  {'Winner':>8}")
    print("-" * 95)
    for t in traces:
        lr = trace_hr(live["lru"].get(t, []))
        m3 = trace_hr(live["marconi_a0.3"].get(t, []))
        m1 = trace_hr(live["marconi_a1.0"].get(t, []))
        best = "LRU" if lr >= m3 and lr >= m1 else ("α=0.3" if m3 >= m1 else "α=1.0")
        label = TRACE_LABELS[t].replace("\n", " ")
        print(f"{label:<30} {lr:>7.1f}% {m3:>7.1f}% {m1:>7.1f}%  {m3-lr:>+8.1f}%  {m1-lr:>+8.1f}%  {best:>8}")

    print("\n" + "=" * 65)
    print("TABLE 2  Alpha Sweep — Marconi/LRU Ratio (mean across traces)")
    print("  >1.0 = Marconi wins  |  <1.0 = LRU wins")
    print("  NOTE: α=0.7 baseline is v1 LRU (TP=1); others use v2 LRU (TP=4)")
    print("=" * 65)
    print(f"{'Dataset':<12}  {'α=0.3':>8}  {'α=0.7':>8}  {'α=1.0':>8}")
    print("-" * 42)
    for ds, ds_traces in [("LMSys", DS_TRACES["lmsys"]),
                           ("ShareGPT", DS_TRACES["sharegpt"]),
                           ("SWEBench", DS_TRACES["swebench"])]:
        row = {}
        for ak, lk in [("marconi_a0.3","lru"), ("marconi_a0.7","lru_v1"), ("marconi_a1.0","lru")]:
            r = [trace_hr(live[ak].get(t,[])) / trace_hr(live[lk].get(t,[]))
                 for t in ds_traces if trace_hr(live[lk].get(t,[])) > 0]
            row[ak] = np.mean(r) if r else float("nan")
        print(f"{ds:<12}  {row['marconi_a0.3']:>8.3f}×  {row['marconi_a0.7']:>8.3f}×  {row['marconi_a1.0']:>8.3f}×")

    print("\n" + "=" * 65)
    print("TABLE 3  CPU Simulation — Median Hit Rates (Marconi paper)")
    print("=" * 65)
    print(f"{'Dataset':<12}  {'N':>4}  {'SGLang+(V1)':>11}  {'Marconi(V2)':>11}  {'M/S+':>7}  {'Win%':>7}")
    print("-" * 60)
    for ds in ["lmsys", "sharegpt", "swebench"]:
        rows = sim[ds]
        v1    = np.median([r["v1"] for r in rows])
        v2    = np.median([r["v2"] for r in rows])
        ratio = np.median([r["v2"]/r["v1"] for r in rows if r["v1"] > 0])
        wins  = [r["hit_rate_win"] for r in rows if r["hit_rate_win"] is not None]
        mwin  = np.median(wins) if wins else float("nan")
        print(f"{ds:<12}  {len(rows):>4}  {v1:>10.2f}%  {v2:>10.2f}%  {ratio:>6.3f}×  {mwin:>+6.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading data...")
    live = load_live()
    sim  = load_sim()
    print(f"  v2 traces: {sum(len(v) for v in live['lru'].values())} requests per policy")
    print(f"  Simulation: {sum(len(v) for v in sim.values())} configs")

    print("Loading eviction logs...")
    lru_ev  = load_lru_evictions("logs/live-minimal-32K-v2/server_lru.log")
    marc_ev = load_marconi_evictions("logs/live-minimal-32K-v2/server_marconi_a0.3.log")
    print(f"  LRU: {len(lru_ev):,} events  |  Marconi α=0.3: {len(marc_ev):,} events")

    print("\nGenerating figures...")
    fig1_sim_vs_live(sim, live)
    fig2_alpha_sweep(live)
    fig3_theory_vs_practice(sim, live)
    fig4_eviction_analysis(lru_ev, marc_ev)
    fig5_ttft(live)

    print_tables(live, sim)
    print(f"\nDone. Figures in {OUT_DIR}/")
