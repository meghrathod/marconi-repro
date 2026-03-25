"""
Analyse trace replay results: TTFT and cache metrics across replay policies.

Three scenarios (live layout: marconi/, lru/, no-cache/):
  (1) No prefix cache — prefix caching disabled (baseline latency).
  (2) LRU — prefix caching on, LRU eviction (SGLang-style baseline).
  (3) Marconi — prefix caching on, Marconi eviction (PR #20045).

Paper alignment ([Marconi, arXiv:2411.19379](https://arxiv.org/abs/2411.19379)):
  - Fig.~10(b)-style: empirical CDF of TTFT (ms) comparing policies on the same axes.
  - Fig.~13-style: metrics vs session arrival rate (sps) — our ttft_vs_sps plots.

Ratio plots (optional): TTFT(policy) / TTFT(scenario 1). Only scenarios (2) and (3)
  are divided by (1); they do not mean there are only two systems — the denominator
  is explicitly the no–prefix-cache run.

Usage:
    python src/analyse_traces.py --results-dir results/live --layout live \\
        --output-dir results/live/figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Marconi (green), LRU (contrast), no–prefix-cache (neutral) — consistent across figures
STYLE_MARCONI = {"color": "#1B4332", "ls": "-", "marker": None, "label": "Marconi (FLOP-aware eviction)"}
STYLE_LRU = {"color": "#2171B5", "ls": "-.", "marker": None, "label": "LRU prefix cache"}
STYLE_NO_CACHE = {"color": "#525252", "ls": "--", "marker": None, "label": "No prefix cache"}
# Legacy two-way
STYLE_CACHE_GENERIC = {"color": "#40916C", "ls": "-", "label": "Prefix cache"}

LayoutMode = Literal["auto", "legacy", "live"]


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
    policy_p95: dict[int, float],
    baseline_p95: dict[int, float],
) -> list[float]:
    """Compute policy/baseline TTFT ratio for each session present in both."""
    common = sorted(set(policy_p95) & set(baseline_p95))
    ratios = []
    for sid in common:
        if baseline_p95[sid] > 0:
            ratios.append(policy_p95[sid] / baseline_p95[sid])
    return ratios


def _parse_sps_art(fname: str) -> tuple[str | None, str | None]:
    parts = fname.replace(".jsonl", "").split("_")
    sps = art = None
    for p in parts:
        if p.startswith("sps="):
            sps = p.split("=", 1)[1]
        elif p.startswith("art="):
            art = p.split("=", 1)[1]
    return sps, art


# ---------------------------------------------------------------------------
# CDF helper
# ---------------------------------------------------------------------------


def values_to_cdf(values: list[float]) -> tuple[list[float], list[float]]:
    """Return (sorted_values, cdf_probs) ready for plotting."""
    if not values:
        return [], []
    s = sorted(values)
    cdf = [(i + 1) / len(s) for i in range(len(s))]
    return s, cdf


def collect_pooled_session_p95s(groups: list[dict], path_key: str) -> list[float]:
    """All per-session P95 TTFT values across file groups (one flat list)."""
    out: list[float] = []
    for g in groups:
        by_sess = compute_session_p95_ttft(load_results(g[path_key]))
        out.extend(by_sess.values())
    return out


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def detect_layout(results_dir: str) -> Literal["legacy", "live"]:
    """Pick live (marconi/lru/no-cache) if marconi/ has jsonl files, else legacy."""
    marconi_dir = os.path.join(results_dir, "marconi")
    if os.path.isdir(marconi_dir):
        try:
            if any(f.endswith(".jsonl") for f in os.listdir(marconi_dir)):
                return "live"
        except OSError:
            pass
    return "legacy"


def discover_groups_legacy(results_dir: str, dataset: str) -> list[dict]:
    """cache/ + no_cache/ pairs."""
    cache_dir = os.path.join(results_dir, "cache")
    no_cache_dir = os.path.join(results_dir, "no_cache")
    if not os.path.isdir(cache_dir) or not os.path.isdir(no_cache_dir):
        return []
    pairs = []
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.startswith(dataset + "_") or not fname.endswith(".jsonl"):
            continue
        no_cache_path = os.path.join(no_cache_dir, fname)
        if not os.path.exists(no_cache_path):
            continue
        sps, art = _parse_sps_art(fname)
        pairs.append({
            "sps": sps,
            "art": art,
            "label": fname.replace(".jsonl", ""),
            "cache_path": os.path.join(cache_dir, fname),
            "no_cache_path": no_cache_path,
        })
    return pairs


def discover_groups_live(results_dir: str, dataset: str) -> list[dict]:
    """marconi/ + lru/ + no-cache/ triples with matching filenames."""
    m_dir = os.path.join(results_dir, "marconi")
    l_dir = os.path.join(results_dir, "lru")
    n_dir = os.path.join(results_dir, "no-cache")
    if not all(os.path.isdir(d) for d in (m_dir, l_dir, n_dir)):
        return []
    groups = []
    for fname in sorted(os.listdir(m_dir)):
        if not fname.startswith(dataset + "_") or not fname.endswith(".jsonl"):
            continue
        l_path = os.path.join(l_dir, fname)
        n_path = os.path.join(n_dir, fname)
        if not os.path.isfile(l_path) or not os.path.isfile(n_path):
            continue
        sps, art = _parse_sps_art(fname)
        groups.append({
            "sps": sps,
            "art": art,
            "label": fname.replace(".jsonl", ""),
            "marconi_path": os.path.join(m_dir, fname),
            "lru_path": l_path,
            "no_cache_path": n_path,
        })
    return groups


def discover_groups(results_dir: str, dataset: str, layout: LayoutMode) -> tuple[list[dict], Literal["legacy", "live"]]:
    resolved: Literal["legacy", "live"]
    if layout == "auto":
        resolved = detect_layout(results_dir)
    else:
        resolved = layout  # type: ignore[assignment]
    if resolved == "live":
        return discover_groups_live(results_dir, dataset), "live"
    return discover_groups_legacy(results_dir, dataset), "legacy"


# ---------------------------------------------------------------------------
# Cache metrics
# ---------------------------------------------------------------------------


def mean_cache_hit_pct(results: list[dict]) -> float | None:
    """Mean cache_hit_pct over successful rows with prompt_tokens > 0."""
    vals = []
    for r in results:
        if r.get("error") is not None:
            continue
        pt = r.get("prompt_tokens") or 0
        if pt <= 0:
            continue
        vals.append(float(r.get("cache_hit_pct") or 0.0))
    if not vals:
        return None
    return float(np.mean(vals))


def mean_cache_hit_by_turn(results: list[dict]) -> dict[int, float]:
    """Mean cache_hit_pct per turn_id (successful rows, prompt_tokens > 0)."""
    by_turn: dict[int, list[float]] = defaultdict(list)
    for r in results:
        if r.get("error") is not None:
            continue
        pt = r.get("prompt_tokens") or 0
        if pt <= 0:
            continue
        tid = int(r.get("turn_id", 0))
        by_turn[tid].append(float(r.get("cache_hit_pct") or 0.0))
    return {t: float(np.mean(vs)) for t, vs in sorted(by_turn.items())}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DATASETS = ["lmsys", "sharegpt", "swebench"]
DATASET_TITLES = {"lmsys": "LMSys", "sharegpt": "ShareGPT", "swebench": "SWEBench"}


def plot_ttft_vs_sps(results_dir: str, output_dir: str, layout: LayoutMode) -> None:
    """Median of per-session P95 TTFT vs sps (two- or three-way)."""
    os.makedirs(output_dir, exist_ok=True)
    effective = layout if layout != "auto" else detect_layout(results_dir)
    fontsize = 14
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)

    for fig_id, dataset in enumerate(DATASETS):
        ax = axs[fig_id]
        groups, resolved = discover_groups(results_dir, dataset, layout)
        if not groups:
            ax.set_visible(False)
            continue

        if resolved == "live":
            sps_m: dict[float, list[float]] = defaultdict(list)
            sps_l: dict[float, list[float]] = defaultdict(list)
            sps_n: dict[float, list[float]] = defaultdict(list)
            for g in groups:
                sps = float(g["sps"])
                m_p95 = compute_session_p95_ttft(load_results(g["marconi_path"]))
                l_p95 = compute_session_p95_ttft(load_results(g["lru_path"]))
                n_p95 = compute_session_p95_ttft(load_results(g["no_cache_path"]))
                if m_p95:
                    sps_m[sps].append(float(np.median(list(m_p95.values()))))
                if l_p95:
                    sps_l[sps].append(float(np.median(list(l_p95.values()))))
                if n_p95:
                    sps_n[sps].append(float(np.median(list(n_p95.values()))))
            sps_vals = sorted(set(sps_m) | set(sps_l) | set(sps_n))
            y_m = [np.mean(sps_m[s]) for s in sps_vals]
            y_l = [np.mean(sps_l[s]) for s in sps_vals]
            y_n = [np.mean(sps_n[s]) for s in sps_vals]
            ax.plot(
                sps_vals, y_m, color=STYLE_MARCONI["color"], ls=STYLE_MARCONI["ls"],
                marker="o", label=STYLE_MARCONI["label"], markersize=5,
            )
            ax.plot(
                sps_vals, y_l, color=STYLE_LRU["color"], ls=STYLE_LRU["ls"],
                marker="d", label=STYLE_LRU["label"], markersize=5,
            )
            ax.plot(
                sps_vals, y_n, color=STYLE_NO_CACHE["color"], ls=STYLE_NO_CACHE["ls"],
                marker="s", label=STYLE_NO_CACHE["label"], markersize=5,
            )
        else:
            sps_c: dict[float, list[float]] = defaultdict(list)
            sps_n: dict[float, list[float]] = defaultdict(list)
            for g in groups:
                sps = float(g["sps"])
                c_p95 = compute_session_p95_ttft(load_results(g["cache_path"]))
                n_p95 = compute_session_p95_ttft(load_results(g["no_cache_path"]))
                if c_p95:
                    sps_c[sps].append(float(np.median(list(c_p95.values()))))
                if n_p95:
                    sps_n[sps].append(float(np.median(list(n_p95.values()))))
            sps_vals = sorted(set(sps_c) | set(sps_n))
            y_c = [np.mean(sps_c[s]) for s in sps_vals]
            y_n = [np.mean(sps_n[s]) for s in sps_vals]
            ax.plot(
                sps_vals, y_c, color=STYLE_CACHE_GENERIC["color"], ls="-",
                marker="o", label=STYLE_CACHE_GENERIC["label"], markersize=5,
            )
            ax.plot(
                sps_vals, y_n, color=STYLE_NO_CACHE["color"], ls=STYLE_NO_CACHE["ls"],
                marker="s", label=STYLE_NO_CACHE["label"], markersize=5,
            )

        title = f"({chr(97 + fig_id)}) {DATASET_TITLES[dataset]}"
        ax.set_xlabel(title, fontsize=fontsize)
        if fig_id == 0:
            ax.set_ylabel("Median per-session P95 TTFT (ms)", fontsize=fontsize)
        ax.set_axisbelow(True)
        ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)

    axs[1].legend(
        loc="upper center", ncols=3 if effective == "live" else 2,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.32), handlelength=1.5, frameon=False,
    )

    fig.text(
        0.5, -0.02,
        "Session arrival rate (sessions/s) — cf. Marconi Fig. 13(a)",
        ha="center", fontsize=fontsize - 1,
    )
    fig.suptitle(
        "P95 TTFT vs load: three serving configurations",
        fontsize=fontsize, y=1.02,
    )
    fig.tight_layout()

    suffix = "three_way" if effective == "live" else "cache_vs_nocache"
    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"ttft_vs_sps_{suffix}.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_swebench_art_curves(results_dir: str, output_dir: str, layout: LayoutMode) -> None:
    """SWEBench: one column per art= value; same three policy line styles (paper-clarity)."""
    os.makedirs(output_dir, exist_ok=True)
    fontsize = 12
    groups, resolved = discover_groups(results_dir, "swebench", layout)
    if not groups:
        print("Skipping swebench art breakdown: no file groups found.")
        return

    art_groups: dict[float, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for g in groups:
        if g["art"] is None:
            continue
        art = float(g["art"])
        sps = float(g["sps"])
        if resolved == "live":
            m_p95 = compute_session_p95_ttft(load_results(g["marconi_path"]))
            l_p95 = compute_session_p95_ttft(load_results(g["lru_path"]))
            n_p95 = compute_session_p95_ttft(load_results(g["no_cache_path"]))
            if m_p95:
                art_groups[art][sps]["marconi"] = float(np.median(list(m_p95.values())))
            if l_p95:
                art_groups[art][sps]["lru"] = float(np.median(list(l_p95.values())))
            if n_p95:
                art_groups[art][sps]["nocache"] = float(np.median(list(n_p95.values())))
        else:
            c_p95 = compute_session_p95_ttft(load_results(g["cache_path"]))
            n_p95 = compute_session_p95_ttft(load_results(g["no_cache_path"]))
            if c_p95:
                art_groups[art][sps]["cache"] = float(np.median(list(c_p95.values())))
            if n_p95:
                art_groups[art][sps]["nocache"] = float(np.median(list(n_p95.values())))

    art_vals = sorted(art_groups.keys())
    if not art_vals:
        print("Skipping swebench art breakdown: no art= in filenames.")
        return

    ncols = len(art_vals)
    fig, axs = plt.subplots(1, ncols, figsize=(3.8 * ncols, 3.6), sharey=True)
    if ncols == 1:
        axs = [axs]

    for col, art in enumerate(art_vals):
        ax = axs[col]
        sps_data = art_groups[art]
        sps_vals = sorted(sps_data.keys())
        if resolved == "live":
            y_m = [sps_data[s].get("marconi", np.nan) for s in sps_vals]
            y_l = [sps_data[s].get("lru", np.nan) for s in sps_vals]
            y_n = [sps_data[s].get("nocache", np.nan) for s in sps_vals]
            ax.plot(
                sps_vals, y_m, color=STYLE_MARCONI["color"], ls=STYLE_MARCONI["ls"],
                marker="o", label=STYLE_MARCONI["label"], markersize=4,
            )
            ax.plot(
                sps_vals, y_l, color=STYLE_LRU["color"], ls=STYLE_LRU["ls"],
                marker="d", label=STYLE_LRU["label"], markersize=4,
            )
            ax.plot(
                sps_vals, y_n, color=STYLE_NO_CACHE["color"], ls=STYLE_NO_CACHE["ls"],
                marker="s", label=STYLE_NO_CACHE["label"], markersize=4,
            )
        else:
            y_c = [sps_data[s].get("cache", np.nan) for s in sps_vals]
            y_n = [sps_data[s].get("nocache", np.nan) for s in sps_vals]
            ax.plot(
                sps_vals, y_c, color=STYLE_CACHE_GENERIC["color"], ls="-",
                marker="o", label="Prefix cache", markersize=5,
            )
            ax.plot(
                sps_vals, y_n, color=STYLE_NO_CACHE["color"], ls=STYLE_NO_CACHE["ls"],
                marker="s", label=STYLE_NO_CACHE["label"], markersize=5,
            )
        ax.set_title(f"art = {art} s", fontsize=fontsize)
        ax.set_xlabel("Sessions/s (sps)", fontsize=fontsize - 1)
        ax.set_axisbelow(True)
        ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)
        if col == 0:
            ax.set_ylabel("Median per-session P95 TTFT (ms)", fontsize=fontsize)

    axs[ncols // 2].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3 if resolved == "live" else 2,
        fontsize=8, frameon=False,
    )
    fig.suptitle(
        "SWEBench: three configurations vs load (inter-request time ∝ art)",
        fontsize=fontsize + 1, y=1.08,
    )
    fig.tight_layout()

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"swebench_art_breakdown.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_p95_ttft_cdf_three_policies(results_dir: str, output_dir: str, layout: LayoutMode) -> None:
    """Empirical CDF of per-session P95 TTFT (ms): all three scenarios (cf. paper TTFT distributions)."""
    resolved = layout if layout != "auto" else detect_layout(results_dir)
    if resolved != "live":
        print("Skipping three-policy TTFT CDF: requires live layout.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fontsize = 12
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)

    for fig_id, dataset in enumerate(DATASETS):
        ax = axs[fig_id]
        groups, _ = discover_groups(results_dir, dataset, "live")
        if not groups:
            ax.set_visible(False)
            continue
        vm = collect_pooled_session_p95s(groups, "marconi_path")
        vl = collect_pooled_session_p95s(groups, "lru_path")
        vn = collect_pooled_session_p95s(groups, "no_cache_path")
        for vals, st in ((vm, STYLE_MARCONI), (vl, STYLE_LRU), (vn, STYLE_NO_CACHE)):
            if vals:
                sx, sy = values_to_cdf(vals)
                ax.plot(sx, sy, color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
        title = f"({chr(97 + fig_id)}) {DATASET_TITLES[dataset]}"
        ax.set_xlabel(f"{title}\nP95 TTFT per session (ms)", fontsize=fontsize)
        if fig_id == 0:
            ax.set_ylabel("Fraction of sessions", fontsize=fontsize)
        ax.set_axisbelow(True)
        ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)

    axs[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=3, fontsize=8.5, frameon=False,
    )
    fig.suptitle(
        "CDF of per-session P95 TTFT — three serving configurations\n"
        "(pooled over all sps and, for SWEBench, all art=)",
        fontsize=11, y=1.09,
    )
    fig.tight_layout()

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"p95_ttft_cdf_three_policies.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_ttft_ratio_vs_nocache_cdf(results_dir: str, output_dir: str, layout: LayoutMode) -> None:
    """Supplementary: only caching policies, normalized by scenario (1) no prefix cache."""
    resolved = layout if layout != "auto" else detect_layout(results_dir)
    if resolved != "live":
        print("Skipping ratio-to-nocache CDF: requires live layout.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fontsize = 12
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)

    for fig_id, dataset in enumerate(DATASETS):
        ax = axs[fig_id]
        groups, _ = discover_groups(results_dir, dataset, "live")
        ratios_m: list[float] = []
        ratios_l: list[float] = []
        for g in groups:
            m_p95 = compute_session_p95_ttft(load_results(g["marconi_path"]))
            l_p95 = compute_session_p95_ttft(load_results(g["lru_path"]))
            n_p95 = compute_session_p95_ttft(load_results(g["no_cache_path"]))
            ratios_m.extend(compute_ttft_ratio(m_p95, n_p95))
            ratios_l.extend(compute_ttft_ratio(l_p95, n_p95))

        if ratios_m:
            sx, sy = values_to_cdf(ratios_m)
            ax.plot(
                sx, sy, color=STYLE_MARCONI["color"], ls=STYLE_MARCONI["ls"], lw=2.0,
                label="Marconi ÷ (no prefix cache)",
            )
        if ratios_l:
            sx, sy = values_to_cdf(ratios_l)
            ax.plot(
                sx, sy, color=STYLE_LRU["color"], ls=STYLE_LRU["ls"], lw=2.0,
                label="LRU ÷ (no prefix cache)",
            )

        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.85)
        title = f"({chr(97 + fig_id)}) {DATASET_TITLES[dataset]}"
        ax.set_xlabel(title, fontsize=fontsize)
        if fig_id == 0:
            ax.set_ylabel("CDF", fontsize=fontsize)
        ax.set_axisbelow(True)
        ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)

    axs[1].legend(
        loc="upper center", ncols=2, fontsize=9,
        bbox_to_anchor=(0.5, 1.3), frameon=False,
    )
    fig.suptitle(
        "Normalized P95 TTFT (caching policies only)\n"
        "Each curve: TTFT with caching ÷ TTFT with prefix caching disabled (same session)",
        fontsize=10.5, y=1.12,
    )
    fig.tight_layout()

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"ttft_ratio_to_nocache_baseline.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_marconi_vs_lru_ratio_cdf(results_dir: str, output_dir: str, layout: LayoutMode) -> None:
    """CDF of TTFT_marconi / TTFT_lru per session (both use prefix cache; isolates eviction policy)."""
    resolved = layout if layout != "auto" else detect_layout(results_dir)
    if resolved != "live":
        print("Skipping Marconi vs LRU ratio CDF: requires live layout.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fontsize = 12
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for fig_id, dataset in enumerate(DATASETS):
        ax = axs[fig_id]
        groups, _ = discover_groups(results_dir, dataset, "live")
        ratios: list[float] = []
        for g in groups:
            m_p95 = compute_session_p95_ttft(load_results(g["marconi_path"]))
            l_p95 = compute_session_p95_ttft(load_results(g["lru_path"]))
            ratios.extend(compute_ttft_ratio(m_p95, l_p95))
        if ratios:
            sx, sy = values_to_cdf(ratios)
            ax.plot(sx, sy, color=STYLE_MARCONI["color"], ls="-", lw=2.0, label="Marconi ÷ LRU")
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.85)
        title = f"({chr(97 + fig_id)}) {DATASET_TITLES[dataset]}"
        ax.set_xlabel(title, fontsize=fontsize)
        if fig_id == 0:
            ax.set_ylabel("CDF", fontsize=fontsize)
        ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)

    fig.suptitle(
        "Marconi vs LRU (both with prefix cache on)\n"
        "Ratio = P95 TTFT Marconi / P95 TTFT LRU per session; <1 means Marconi faster",
        fontsize=10, y=1.08,
    )
    fig.tight_layout()

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"marconi_vs_lru_ttft_ratio_cdf.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_cache_hit_summary(results_dir: str, output_dir: str, layout: LayoutMode) -> None:
    """Mean cache_hit_pct by policy and dataset (bars)."""
    resolved = layout if layout != "auto" else detect_layout(results_dir)
    os.makedirs(output_dir, exist_ok=True)
    fontsize = 12
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DATASETS))
    width = 0.28

    if resolved == "live":
        means_m = []
        means_l = []
        means_n = []
        for dataset in DATASETS:
            groups, _ = discover_groups(results_dir, dataset, "live")
            m_hits: list[float] = []
            l_hits: list[float] = []
            n_hits: list[float] = []
            for g in groups:
                hm = mean_cache_hit_pct(load_results(g["marconi_path"]))
                hl = mean_cache_hit_pct(load_results(g["lru_path"]))
                hn = mean_cache_hit_pct(load_results(g["no_cache_path"]))
                if hm is not None:
                    m_hits.append(hm)
                if hl is not None:
                    l_hits.append(hl)
                if hn is not None:
                    n_hits.append(hn)
            means_m.append(np.mean(m_hits) if m_hits else 0.0)
            means_l.append(np.mean(l_hits) if l_hits else 0.0)
            means_n.append(np.mean(n_hits) if n_hits else 0.0)
        ax.bar(x - width, means_m, width, label="Marconi", color=STYLE_MARCONI["color"])
        ax.bar(x, means_l, width, label="LRU", color=STYLE_LRU["color"])
        ax.bar(x + width, means_n, width, label="No prefix cache", color=STYLE_NO_CACHE["color"])
    else:
        means_c = []
        means_n = []
        for dataset in DATASETS:
            groups, _ = discover_groups(results_dir, dataset, "legacy")
            c_hits: list[float] = []
            n_hits: list[float] = []
            for g in groups:
                hc = mean_cache_hit_pct(load_results(g["cache_path"]))
                hn = mean_cache_hit_pct(load_results(g["no_cache_path"]))
                if hc is not None:
                    c_hits.append(hc)
                if hn is not None:
                    n_hits.append(hn)
            means_c.append(np.mean(c_hits) if c_hits else 0.0)
            means_n.append(np.mean(n_hits) if n_hits else 0.0)
        ax.bar(x - width / 2, means_c, width, label="Prefix cache", color=STYLE_CACHE_GENERIC["color"])
        ax.bar(x + width / 2, means_n, width, label="No prefix cache", color=STYLE_NO_CACHE["color"])

    ax.set_ylabel("Mean cache hit % (avg over files, then datasets)", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_TITLES[d] for d in DATASETS])
    ax.legend(fontsize=10, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(color="lightgrey", linestyle="dashed", axis="y", linewidth=0.8)
    ax.set_title("Cache hit rate summary", fontsize=fontsize + 1)

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"cache_hit_summary.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_cache_hit_by_turn_sample(
    results_dir: str, output_dir: str, layout: LayoutMode,
    dataset: str = "lmsys", sps: float = 1.0,
) -> None:
    """Example: mean cache hit % vs turn_id for one workload file (Marconi vs LRU)."""
    resolved = layout if layout != "auto" else detect_layout(results_dir)
    if resolved != "live":
        return
    groups, _ = discover_groups(results_dir, dataset, "live")
    target = None
    for g in groups:
        if g["art"] is not None:
            continue
        if g["sps"] is not None and float(g["sps"]) == sps:
            target = g
            break
    if target is None and groups:
        target = groups[0]
    if target is None:
        print("Skipping cache hit by turn: no matching file.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fontsize = 12
    fig, ax = plt.subplots(figsize=(6, 3.5))
    m_by = mean_cache_hit_by_turn(load_results(target["marconi_path"]))
    l_by = mean_cache_hit_by_turn(load_results(target["lru_path"]))
    turns = sorted(set(m_by) | set(l_by))
    ax.plot(
        turns, [m_by.get(t, np.nan) for t in turns],
        "o-", label=STYLE_MARCONI["label"], color=STYLE_MARCONI["color"], markersize=4,
    )
    ax.plot(
        turns, [l_by.get(t, np.nan) for t in turns],
        "s--", label=STYLE_LRU["label"], color=STYLE_LRU["color"], markersize=4,
    )
    ax.set_xlabel("turn_id", fontsize=fontsize)
    ax.set_ylabel("Mean cache hit %", fontsize=fontsize)
    ax.set_title(
        f"{DATASET_TITLES.get(dataset, dataset)} sps={sps} "
        f"({Path(target['marconi_path']).name})",
        fontsize=11,
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(color="lightgrey", linestyle="dashed", axis="both", linewidth=0.8)

    for fmt, dpi in [("pdf", 500), ("png", 200)]:
        out_path = os.path.join(output_dir, f"cache_hit_by_turn_{dataset}_sps={sps}.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def plot_ttft_cdf(results_dir: str, output_dir: str) -> None:
    """Deprecated name: use plot_ttft_vs_sps with layout auto."""
    plot_ttft_vs_sps(results_dir, output_dir, "auto")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse TTFT from trace replay results")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--output-dir", default="results/figures", help="Where to save plots")
    parser.add_argument(
        "--layout",
        choices=["auto", "legacy", "live"],
        default="auto",
        help="auto: use marconi/ if present else cache+no_cache",
    )
    parser.add_argument(
        "--skip-cdf",
        action="store_true",
        help="Skip supplementary ratio CDFs (still draws three-policy absolute TTFT CDF)",
    )
    parser.add_argument(
        "--skip-cache-plots",
        action="store_true",
        help="Skip cache hit summary and by-turn sample plot",
    )
    args = parser.parse_args()

    plot_ttft_vs_sps(args.results_dir, args.output_dir, args.layout)
    plot_swebench_art_curves(args.results_dir, args.output_dir, args.layout)
    plot_p95_ttft_cdf_three_policies(args.results_dir, args.output_dir, args.layout)
    if not args.skip_cdf:
        plot_ttft_ratio_vs_nocache_cdf(args.results_dir, args.output_dir, args.layout)
        plot_marconi_vs_lru_ratio_cdf(args.results_dir, args.output_dir, args.layout)
    if not args.skip_cache_plots:
        plot_cache_hit_summary(args.results_dir, args.output_dir, args.layout)
        plot_cache_hit_by_turn_sample(args.results_dir, args.output_dir, args.layout)


if __name__ == "__main__":
    main()
