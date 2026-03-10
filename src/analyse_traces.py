"""
Analyse trace replay results: TTFT comparison between cache and no-cache runs.

Generates Marconi-paper-style CDF plots (Figure 7) of per-session P95 TTFT
ratios.  Currently compares SGLang prefix-caching (cache/) vs vanilla
(no_cache/).  Will be extended with Marconi policy results later.

Usage:
    python src/analyse_traces.py [--results-dir results] [--output-dir results/figures]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Marconi paper green palette
COLORS = ["#52B788", "#40916C", "#2D6A4F", "#1B4332", "#081C15", "#95D5B2"]
LINESTYLES = ["solid", "dashed", "dotted", "dashdot", (0, (3, 1, 1, 1, 1, 1))]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(filepath: str) -> list[dict]:
    """Read a JSONL results file into a list of dicts."""
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# TTFT metrics
# ---------------------------------------------------------------------------

def compute_session_p95_ttft(results: list[dict]) -> dict[int, float]:
    """Return {session_id: P95 TTFT (ms)} computed per session."""
    by_session: dict[int, list[float]] = defaultdict(list)
    for r in results:
        if r.get("error") is not None:
            continue
        by_session[r["session_id"]].append(r["ttft_ms"])
    return {
        sid: float(np.percentile(vals, 95))
        for sid, vals in by_session.items()
    }


def compute_ttft_ratio(
    cache_p95: dict[int, float],
    no_cache_p95: dict[int, float],
) -> list[float]:
    """Compute cache/no_cache TTFT ratio for each session present in both."""
    common = sorted(set(cache_p95) & set(no_cache_p95))
    ratios = []
    for sid in common:
        if no_cache_p95[sid] > 0:
            ratios.append(cache_p95[sid] / no_cache_p95[sid])
    return ratios


# ---------------------------------------------------------------------------
# CDF helper
# ---------------------------------------------------------------------------

def values_to_cdf(values: list[float]):
    """Return (sorted_values, cdf_probs) ready for plotting."""
    s = sorted(values)
    cdf = [(i + 1) / len(s) for i in range(len(s))]
    return s, cdf


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(results_dir: str, dataset: str) -> list[dict]:
    """Find matching cache/no_cache file pairs for a dataset.

    Returns a list of dicts with keys: sps, cache_path, no_cache_path.
    """
    cache_dir = os.path.join(results_dir, "cache")
    no_cache_dir = os.path.join(results_dir, "no_cache")
    pairs = []
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.startswith(dataset + "_") or not fname.endswith(".jsonl"):
            continue
        no_cache_path = os.path.join(no_cache_dir, fname)
        if not os.path.exists(no_cache_path):
            continue
        # Extract sps and art values from filename
        parts = fname.replace(".jsonl", "").split("_")
        sps = None
        art = None
        for p in parts:
            if p.startswith("sps="):
                sps = p.split("=")[1]
            elif p.startswith("art="):
                art = p.split("=")[1]
        pairs.append({
            "sps": sps,
            "art": art,
            "label": fname.replace(".jsonl", ""),
            "cache_path": os.path.join(cache_dir, fname),
            "no_cache_path": no_cache_path,
        })
    return pairs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DATASETS = ["lmsys", "sharegpt", "swebench"]
DATASET_TITLES = {"lmsys": "LMSys", "sharegpt": "ShareGPT", "swebench": "SWEBench"}


def plot_ttft_cdf(results_dir: str, output_dir: str):
    """Generate a line plot of P95 TTFT vs arrival rate (cache vs no-cache)."""
    os.makedirs(output_dir, exist_ok=True)
    fontsize = 14

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=False)

    for fig_id, dataset in enumerate(DATASETS):
        ax = axs[fig_id]
        pairs = discover_files(results_dir, dataset)

        # Group by sps (swebench has multiple art= variants per sps)
        sps_cache: dict[float, list[float]] = defaultdict(list)
        sps_nocache: dict[float, list[float]] = defaultdict(list)

        for pair in pairs:
            sps = float(pair["sps"])
            cache_res = load_results(pair["cache_path"])
            no_cache_res = load_results(pair["no_cache_path"])

            # Compute P95 TTFT for each session independently
            cache_p95s = compute_session_p95_ttft(cache_res)
            nocache_p95s = compute_session_p95_ttft(no_cache_res)

            # To represent the test condition with a single robust number,
            # we take the median of the per-session P95 TTFTs.
            if cache_p95s:
                sps_cache[sps].append(float(np.median(list(cache_p95s.values()))))
            if nocache_p95s:
                sps_nocache[sps].append(float(np.median(list(nocache_p95s.values()))))

        # Average across art= variants, sort by sps
        sps_vals = sorted(set(sps_cache) | set(sps_nocache))
        cache_y = [np.mean(sps_cache[s]) for s in sps_vals]
        nocache_y = [np.mean(sps_nocache[s]) for s in sps_vals]

        ax.plot(sps_vals, cache_y, "o-", label="Cache", color=COLORS[0], markersize=5)
        ax.plot(sps_vals, nocache_y, "s--", label="No Cache", color=COLORS[2], markersize=5)

        title = f"({chr(97 + fig_id)}) {DATASET_TITLES[dataset]}"
        ax.set_xlabel(title, fontsize=fontsize)
        if fig_id == 0:
            ax.set_ylabel("Median of Per-Session P95 TTFT (ms)", fontsize=fontsize)
        ax.set_axisbelow(True)
        ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)

    axs[1].legend(
        loc="upper center", ncols=2, fontsize=10,
        bbox_to_anchor=(0.5, 1.3), handlelength=1.5, frameon=False,
    )

    fig.text(0.5, -0.08, "Sessions per Second (sps)", ha="center", fontsize=fontsize)
    fig.tight_layout()

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"ttft_cache_vs_nocache.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_swebench_art_curves(results_dir: str, output_dir: str):
    """Generate a line plot of P95 TTFT vs arrival rate for SWEBench, split by art= value."""
    os.makedirs(output_dir, exist_ok=True)
    fontsize = 14

    fig, ax = plt.subplots(figsize=(6, 4))
    pairs = discover_files(results_dir, "swebench")

    # Group by art, then sps
    # {art_val: {sps_val: (cache_p95_median, nocache_p95_median)}}
    art_groups = defaultdict(lambda: defaultdict(dict))

    for pair in pairs:
        if pair["art"] is None:
            continue
        art = float(pair["art"])
        sps = float(pair["sps"])
        
        cache_res = load_results(pair["cache_path"])
        no_cache_res = load_results(pair["no_cache_path"])

        cache_p95s = compute_session_p95_ttft(cache_res)
        nocache_p95s = compute_session_p95_ttft(no_cache_res)

        if cache_p95s:
            art_groups[art][sps]["cache"] = float(np.median(list(cache_p95s.values())))
        if nocache_p95s:
            art_groups[art][sps]["nocache"] = float(np.median(list(nocache_p95s.values())))

    art_vals = sorted(art_groups.keys())
    
    # We have enough colors, let's map each ART to a color
    # Cache = solid line, circle marker. No Cache = dashed line, square marker.
    for i, art in enumerate(art_vals):
        sps_data = art_groups[art]
        sps_vals = sorted(sps_data.keys())
        
        cache_y = [sps_data[s].get("cache", np.nan) for s in sps_vals]
        nocache_y = [sps_data[s].get("nocache", np.nan) for s in sps_vals]
        
        color = COLORS[i % len(COLORS)]
        
        ax.plot(sps_vals, cache_y, "o-", label=f"Cache (art={art})", color=color, markersize=5)
        ax.plot(sps_vals, nocache_y, "s--", label=f"No Cache (art={art})", color=color, markersize=5)

    ax.set_title("SWEBench: P95 TTFT by Average Response Time (art)", fontsize=fontsize)
    ax.set_xlabel("Sessions per Second (sps)", fontsize=fontsize)
    ax.set_ylabel("Median of Per-Session P95 TTFT (ms)", fontsize=fontsize)
    ax.set_axisbelow(True)
    ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.05, 1),
        fontsize=10, frameon=False
    )

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"swebench_art_breakdown.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse TTFT from trace replay results")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--output-dir", default="results/figures", help="Where to save plots")
    args = parser.parse_args()

    plot_ttft_cdf(args.results_dir, args.output_dir)
    plot_swebench_art_curves(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
