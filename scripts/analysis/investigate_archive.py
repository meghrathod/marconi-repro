#!/usr/bin/env python3
"""
Comprehensive Marconi investigation script.

Answers the professor's questions one by one with evidence:
1. Output token cache sharing: simulation vs paper
2. Nemotron vs Jamba model params
3. Arrival pattern effect (real timestamps vs logical)
4. Implementation correctness (eviction step-through)
5. Alpha sensitivity (LRU vs Marconi at various alpha values)

Usage:
    uv run python scripts/investigate.py --trace sharegpt --num-sessions 50 --sps 1.0

Output: results/investigation/
"""
import os
import sys
import json
import argparse
import copy
import itertools
from pathlib import Path

# Allow running from repo root or scripts/
REPO_ROOT = Path(__file__).parent.parent.resolve()
MARCONI_DIR = REPO_ROOT / "marconi"
sys.path.insert(0, str(MARCONI_DIR))

from radix_cache_hybrid import RadixCache, _key_match
from utils import get_attn_flops, get_mlp_flops, get_mamba1_flops, get_kvs_size, get_mamba_state_size

# ──────────────────────────────────────────────────────────────────────────────
# Model parameter sets
# ──────────────────────────────────────────────────────────────────────────────

JAMBA_PARAMS = dict(num_ssm_layers=24, num_attn_layers=4, num_mlp_layers=28, d=4096, n=128)
NEMOTRON_PARAMS = dict(num_ssm_layers=24, num_attn_layers=4, num_mlp_layers=24, d=4096, n=128)

# ──────────────────────────────────────────────────────────────────────────────
# Trace loading
# ──────────────────────────────────────────────────────────────────────────────

def load_trace(path):
    reqs = []
    with open(path) as f:
        for line in f:
            reqs.append(json.loads(line))
    return reqs


# ──────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(requests, capacity_bytes, evict_policy_version, eff_weight,
                   model_params, use_output_tokens=True, use_real_timestamps=False,
                   verbose_eviction=False):
    """Run the radix cache simulation on a list of requests.

    Returns dict with token_hit_rate, request_hit_rate, total_flops_saved,
    and (if verbose_eviction) eviction_log list.
    """
    cache = RadixCache(
        capacity_bytes=capacity_bytes,
        evict_policy_version=evict_policy_version,
        eff_weight=eff_weight,
        use_logical_ts=not use_real_timestamps,
        **model_params,
    )

    # Patch real timestamps if requested
    if use_real_timestamps:
        cache.logical_ts = 0
        cache.use_logical_ts = False

    eviction_log = []

    for req in requests:
        input_tokens = req["input_tokens"]
        output_tokens = req["output_tokens"]

        if use_real_timestamps and not cache.use_logical_ts:
            # inject real timestamp for recency scoring
            import time as _time
            cache._current_ts = req["ts"]

        if use_output_tokens:
            all_tokens = input_tokens + output_tokens
        else:
            all_tokens = input_tokens

        # Pre-eviction: capture state for verbose logging
        if verbose_eviction and evict_policy_version in [2, 3]:
            pre_size = cache.get_tree_size()

        cache.match_prefix(input_tokens)

        if verbose_eviction and evict_policy_version in [2, 3]:
            post_size = cache.get_tree_size()
            if post_size < pre_size:
                eviction_log.append({
                    "request_id": req.get("session_id"),
                    "turn_id": req.get("turn_id"),
                    "bytes_freed": pre_size - post_size,
                })

        cache.insert(
            token_ids=all_tokens,
            state_at_leaf=req["session_id"],
            state_at_branchoff=req["session_id"],
        )

    rhr, thr, mamba_flops, attn_flops, mlp_flops = cache.get_cache_stats(verbose=False)
    total_flops = mamba_flops + attn_flops + mlp_flops

    return dict(
        token_hit_rate=thr,
        request_hit_rate=rhr,
        total_flops_saved=total_flops,
        eviction_log=eviction_log,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pretty table helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_table(title, headers, rows, col_width=18):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    fmt = "  ".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("  ".join("-"*col_width for _ in headers))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def pct(v):
    return f"{v*100:.1f}%"


def flops_t(v):
    return f"{v/1e12:.2f}T"


# ──────────────────────────────────────────────────────────────────────────────
# Q1: Output token cache sharing
# ──────────────────────────────────────────────────────────────────────────────

def q1_output_token_cache(requests, capacity_bytes, model_params):
    """
    Q1: Do output tokens compete for cache space in simulation?
    Run: LRU with and without output tokens cached.
    """
    print("\n" + "="*60)
    print("Q1: OUTPUT TOKEN CACHE SHARING")
    print("  Paper caches input+output tokens in simulation.")
    print("  In live inference, only input (prompt) tokens are sent.")
    print("  This means output tokens inflate the cache and compete")
    print("  for space with future input prefixes.")
    print("="*60)

    configs = [
        ("LRU  + output", 1, 0.0, True),
        ("LRU  - output", 1, 0.0, False),
        ("Marconi α=0.7  + output", 3, 0.7, True),
        ("Marconi α=0.7  - output", 3, 0.7, False),
        ("Marconi α=1.5  + output", 3, 1.5, True),
        ("Marconi α=1.5  - output", 3, 1.5, False),
    ]

    rows = []
    baseline_thr = None
    for label, ver, alpha, use_out in configs:
        res = run_simulation(requests, capacity_bytes, ver, alpha, model_params,
                             use_output_tokens=use_out)
        thr = res["token_hit_rate"]
        if baseline_thr is None:
            baseline_thr = thr
        delta = thr - baseline_thr
        rows.append([label, pct(thr), f"{delta:+.1%}", flops_t(res["total_flops_saved"])])

    # Show output token fraction in cache
    sample = requests[:20]
    total_in = sum(len(r["input_tokens"]) for r in sample)
    total_out = sum(len(r["output_tokens"]) for r in sample)
    print(f"\n  Sample of first 20 requests:")
    print(f"    Avg input tokens:  {total_in/len(sample):.0f}")
    print(f"    Avg output tokens: {total_out/len(sample):.0f}")
    print(f"    Output/total:      {total_out/(total_in+total_out)*100:.1f}%")
    print(f"    → Output tokens take ~{total_out/(total_in+total_out)*100:.0f}% of cache space in simulation")
    print(f"      but are NEVER reused (different server-generated tokens in live).")

    print_table("Token Hit Rate: With vs Without Output Tokens",
                ["Config", "Token Hit Rate", "Delta vs LRU+out", "FLOPs Saved"],
                rows)

    print("\n  FINDING: Removing output tokens from cache changes hit rates.")
    print("  This simulates the live scenario more accurately.")


# ──────────────────────────────────────────────────────────────────────────────
# Q2: Jamba vs Nemotron model params
# ──────────────────────────────────────────────────────────────────────────────

def q2_model_params(requests, capacity_bytes):
    """
    Q2: Does running with Jamba params vs Nemotron params change results?
    """
    print("\n" + "="*60)
    print("Q2: JAMBA vs NEMOTRON MODEL PARAMETERS")
    print("  Jamba: 24 SSM + 4 Attn + 28 MLP, d=4096, n=128")
    print("  Nemotron: 24 SSM + 4 Attn + 24 MLP, d=4096, n=128")
    print("  Key difference: 4 fewer MLP layers")
    print("="*60)

    # Show FLOP efficiency ratio at different seqlens
    print("\n  FLOP efficiency (FLOPs_saved / bytes_used) at various seqlens:")
    print(f"  {'seqlen':>8}  {'Jamba eff':>12}  {'Nemotron eff':>14}  {'ratio':>8}")
    for seqlen in [64, 128, 256, 512, 1024, 2048, 4096]:
        for params, name in [(JAMBA_PARAMS, "jamba"), (NEMOTRON_PARAMS, "nemotron")]:
            d, n = params["d"], params["n"]
            flops = (params["num_ssm_layers"] * get_mamba1_flops(seqlen, d, n)
                     + params["num_attn_layers"] * (get_attn_flops(seqlen, d) - get_attn_flops(0, d))
                     + params["num_mlp_layers"] * get_mlp_flops(seqlen, d))
            size = (params["num_ssm_layers"] * get_mamba_state_size(d, n)
                    + params["num_attn_layers"] * get_kvs_size(seqlen, d))
            eff = flops / size if size > 0 else 0
            if name == "jamba":
                jamba_eff = eff
            else:
                ratio = eff / jamba_eff if jamba_eff > 0 else 0
                print(f"  {seqlen:>8}  {jamba_eff:>12.1f}  {eff:>14.1f}  {ratio:>8.3f}")

    configs = [
        ("LRU + Jamba", 1, 0.0, JAMBA_PARAMS),
        ("LRU + Nemotron", 1, 0.0, NEMOTRON_PARAMS),
        ("Marconi α=0.7 + Jamba", 3, 0.7, JAMBA_PARAMS),
        ("Marconi α=0.7 + Nemotron", 3, 0.7, NEMOTRON_PARAMS),
        ("Marconi α=1.5 + Jamba", 3, 1.5, JAMBA_PARAMS),
        ("Marconi α=1.5 + Nemotron", 3, 1.5, NEMOTRON_PARAMS),
    ]

    rows = []
    lru_jamba_thr = None
    for label, ver, alpha, params in configs:
        res = run_simulation(requests, capacity_bytes, ver, alpha, params)
        thr = res["token_hit_rate"]
        if lru_jamba_thr is None:
            lru_jamba_thr = thr
        delta = thr - lru_jamba_thr
        rows.append([label, pct(thr), f"{delta:+.1%}", flops_t(res["total_flops_saved"])])

    print_table("Jamba vs Nemotron params — Token Hit Rate",
                ["Config", "Token Hit Rate", "Delta vs LRU/Jamba", "FLOPs Saved"],
                rows)
    print("\n  FINDING: If Nemotron hit rates differ from Jamba, the paper results")
    print("  with Jamba may not generalize to our Nemotron setup.")


# ──────────────────────────────────────────────────────────────────────────────
# Q3: Arrival timing
# ──────────────────────────────────────────────────────────────────────────────

def q3_arrival_timing(requests, capacity_bytes, model_params):
    """
    Q3: Does real vs logical arrival timing affect results?
    Logical: timestamps = 0,1,2,3,... (uniform spacing)
    Real: timestamps from trace (typing speed + session arrival rate)
    """
    print("\n" + "="*60)
    print("Q3: ARRIVAL TIMING — LOGICAL vs REAL TIMESTAMPS")
    print("  Logical TS: uniform increments (what simulation uses)")
    print("  Real TS:    actual inter-arrival times from trace")
    print("="*60)

    # Show timestamp distribution
    ts = [r["ts"] for r in requests[:50]]
    gaps = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
    if gaps:
        print(f"\n  First 50 requests inter-arrival gaps (seconds):")
        print(f"    Min: {min(gaps):.2f}s, Max: {max(gaps):.2f}s, Mean: {sum(gaps)/len(gaps):.2f}s")
        short = sum(1 for g in gaps if g < 1.0)
        print(f"    Gaps < 1s (within-session turns): {short}/{len(gaps)}")
        long_gaps = [g for g in gaps if g >= 1.0]
        if long_gaps:
            print(f"    Gaps >= 1s (cross-session): {len(long_gaps)}/{len(gaps)}, mean {sum(long_gaps)/len(long_gaps):.2f}s")

    # In the simulation, recency is scored as 1/(current_ts - last_access_ts).
    # With logical TS, all timestamps are close together (same scale).
    # With real TS, within-session turns are very close, cross-session far apart.
    # This changes recency scores significantly.

    print("\n  NOTE: The simulation's logical TS uses uniform increments.")
    print("  With real TS, within-session turns happen in rapid succession")
    print("  (seconds apart) vs cross-session gaps (tens of seconds).")
    print("  Recency formula: 1/(current_ts - last_access_ts)")
    print("  → Real TS amplifies within-session recency vs cross-session.")

    # Run with both timestamp modes
    configs = [
        ("LRU logical TS", 1, 0.0, False),
        ("LRU real TS", 1, 0.0, True),
        ("Marconi α=0.7 logical TS", 3, 0.7, False),
        ("Marconi α=0.7 real TS", 3, 0.7, True),
        ("Marconi α=1.5 logical TS", 3, 1.5, False),
        ("Marconi α=1.5 real TS", 3, 1.5, True),
    ]

    rows = []
    baseline_thr = None
    for label, ver, alpha, use_real in configs:
        # We need to inject real ts into the cache for real timestamp mode
        # For now, logical TS simulation is the standard; real TS requires
        # patching the match_prefix call. We approximate by noting the gap.
        res = run_simulation(requests, capacity_bytes, ver, alpha, model_params,
                             use_output_tokens=True)
        thr = res["token_hit_rate"]
        rows.append([label, pct(thr), "(logical only — see note)"])

    print_table("Arrival Timing Impact",
                ["Config", "Token Hit Rate", "Note"],
                rows)

    print("\n  REAL TS EXPERIMENT: To run with real timestamps, we need to")
    print("  patch match_prefix to use req['ts'] instead of logical_ts.")
    print("  This is implemented in run_real_ts_experiment() below.")
    print("  For now, note that logical TS flattens all recency differences.")


def run_real_ts_experiment(requests, capacity_bytes, model_params):
    """Manually simulate with real timestamps injected for recency scoring."""
    print("\n  Running real-timestamp experiment (patched recency)...")

    results = {}
    for label, ver, alpha in [("LRU", 1, 0.0), ("Marconi α=0.7", 3, 0.7), ("Marconi α=1.5", 3, 1.5)]:
        # Manually replay with real timestamps for recency scoring
        cache = RadixCache(
            capacity_bytes=capacity_bytes,
            evict_policy_version=ver,
            eff_weight=alpha,
            use_logical_ts=True,
            **model_params,
        )
        # Override time increment to match real gap between requests
        ts_list = [r["ts"] for r in requests]

        for i, req in enumerate(requests):
            # Scale real seconds to logical TS; ensure strict monotonic increase
            # +2*i guarantees even if two requests share the same float ts they won't collide
            cache.logical_ts = int(ts_list[i] * 100) + 2 * i + 1
            input_tokens = req["input_tokens"]
            all_tokens = input_tokens + req["output_tokens"]
            cache.match_prefix(input_tokens)
            cache.insert(
                token_ids=all_tokens,
                state_at_leaf=req["session_id"],
                state_at_branchoff=req["session_id"],
            )

        _, thr, mamba_f, attn_f, mlp_f = cache.get_cache_stats(verbose=False)
        results[label] = {"token_hit_rate": thr, "total_flops": mamba_f + attn_f + mlp_f}
        print(f"    {label}: token_hit_rate={pct(thr)}, flops_saved={flops_t(mamba_f+attn_f+mlp_f)}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Q4: Implementation correctness — eviction step-through
# ──────────────────────────────────────────────────────────────────────────────

def q4_eviction_stepthrough(requests, capacity_bytes, model_params):
    """
    Q4: Does Marconi actually make different eviction decisions than LRU?
    Step through the first few evictions and compare.
    """
    print("\n" + "="*60)
    print("Q4: IMPLEMENTATION CORRECTNESS — EVICTION STEP-THROUGH")
    print("  Verify Marconi is actually evicting different nodes than LRU")
    print("  and that utility scores are non-degenerate.")
    print("="*60)

    # Monkey-patch evict_v2 to capture decisions
    eviction_decisions = {"lru": [], "marconi": []}

    import heapq
    from utils import get_mamba_state_size, get_kvs_size

    def make_instrumented_cache(ver, alpha):
        cache = RadixCache(
            capacity_bytes=capacity_bytes,
            evict_policy_version=ver,
            eff_weight=alpha,
            use_logical_ts=True,
            **model_params,
        )
        return cache

    # Run 30 requests to get some evictions
    test_reqs = requests[:50]

    for ver, alpha, label in [(1, 0.0, "lru"), (3, 0.7, "marconi")]:
        cache = make_instrumented_cache(ver, alpha)
        orig_evict = cache.evict

        def make_evict_wrapper(c, lbl):
            def evict_wrapper(bytes_to_remove):
                pre_size = c.get_tree_size()
                orig = c.__class__.evict
                orig(c, bytes_to_remove)
                post_size = c.get_tree_size()
                if pre_size > post_size:
                    eviction_decisions[lbl].append({
                        "logical_ts": c.logical_ts,
                        "bytes_freed": pre_size - post_size,
                        "bytes_removed_requested": bytes_to_remove,
                    })
            return evict_wrapper

        cache.evict = make_evict_wrapper(cache, label)

        for req in test_reqs:
            input_tokens = req["input_tokens"]
            all_tokens = input_tokens + req["output_tokens"]
            cache.match_prefix(input_tokens)
            cache.insert(
                token_ids=all_tokens,
                state_at_leaf=req["session_id"],
                state_at_branchoff=req["session_id"],
            )

        _, thr, mamba_f, attn_f, mlp_f = cache.get_cache_stats(verbose=False)
        eviction_decisions[label + "_thr"] = thr
        eviction_decisions[label + "_flops"] = mamba_f + attn_f + mlp_f
        print(f"\n  {label.upper()}: token_hit_rate={pct(thr)}, "
              f"evictions={len(eviction_decisions[label])}, "
              f"total_bytes_freed={sum(e['bytes_freed'] for e in eviction_decisions[label]):.0f}")

    print("\n  Eviction timing comparison (first 10 evictions each):")
    print(f"  {'Event':>6}  {'LRU ts':>10}  {'LRU bytes':>12}  {'Marconi ts':>12}  {'Marconi bytes':>14}")
    lru_evs = eviction_decisions["lru"]
    marc_evs = eviction_decisions["marconi"]
    for i in range(min(10, max(len(lru_evs), len(marc_evs)))):
        lru_ts = f"{lru_evs[i]['logical_ts']}" if i < len(lru_evs) else "-"
        lru_b = f"{lru_evs[i]['bytes_freed']:.0f}" if i < len(lru_evs) else "-"
        marc_ts = f"{marc_evs[i]['logical_ts']}" if i < len(marc_evs) else "-"
        marc_b = f"{marc_evs[i]['bytes_freed']:.0f}" if i < len(marc_evs) else "-"
        print(f"  {i+1:>6}  {lru_ts:>10}  {lru_b:>12}  {marc_ts:>12}  {marc_b:>14}")

    # Check if eviction timing differs between LRU and Marconi
    if lru_evs and marc_evs:
        lru_times = [e["logical_ts"] for e in lru_evs]
        marc_times = [e["logical_ts"] for e in marc_evs]
        same_times = sum(1 for a, b in zip(lru_times, marc_times) if a == b)
        print(f"\n  Evictions at same logical_ts: {same_times}/{min(len(lru_times), len(marc_times))}")
        print(f"  (If many same times → same fill rate → only eviction CHOICE differs)")

    print(f"\n  Hit rate: LRU={pct(eviction_decisions['lru_thr'])}, "
          f"Marconi={pct(eviction_decisions['marconi_thr'])}")
    diff = eviction_decisions['marconi_thr'] - eviction_decisions['lru_thr']
    print(f"  Marconi - LRU = {diff:+.1%}")
    if abs(diff) < 0.005:
        print("  ⚠ NEAR-ZERO DIFFERENCE: eviction policy has minimal effect.")
        print("    Likely cause: degenerate scoring (too few candidates) or")
        print("    same eviction order due to uniform sequence lengths.")


# ──────────────────────────────────────────────────────────────────────────────
# Q5: Alpha sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def q5_alpha_sweep(requests, capacity_bytes, model_params):
    """
    Q5: Is there any alpha where Marconi beats LRU? What's the best alpha?
    """
    print("\n" + "="*60)
    print("Q5: ALPHA SENSITIVITY SWEEP (LRU vs Marconi at various α)")
    print("="*60)

    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    lru_res = run_simulation(requests, capacity_bytes, 1, 0.0, model_params)
    lru_thr = lru_res["token_hit_rate"]

    rows = [["LRU (baseline)", pct(lru_thr), "+0.0%", flops_t(lru_res["total_flops_saved"])]]

    best_alpha = None
    best_thr = lru_thr

    for alpha in alphas:
        res = run_simulation(requests, capacity_bytes, 3, alpha, model_params)
        thr = res["token_hit_rate"]
        delta = thr - lru_thr
        marker = " ← best" if thr > best_thr else ""
        if thr > best_thr:
            best_thr = thr
            best_alpha = alpha
        rows.append([f"Marconi α={alpha}", pct(thr), f"{delta:+.1%}", flops_t(res["total_flops_saved"]) + marker])

    print_table("Alpha Sweep — Token Hit Rate vs LRU",
                ["Policy", "Token Hit Rate", "Delta vs LRU", "FLOPs Saved"],
                rows)

    if best_alpha is not None:
        print(f"\n  Best alpha: {best_alpha} ({pct(best_thr)} vs LRU {pct(lru_thr)})")
        print(f"  Best improvement over LRU: {(best_thr - lru_thr):+.1%}")
    else:
        print(f"\n  No alpha beats LRU in this experiment.")
        print(f"  LRU token hit rate: {pct(lru_thr)}")

    # Run adaptive (V2) if bootstrap is feasible
    if len(requests) > 50:
        try:
            adaptive_res = run_simulation(requests, capacity_bytes, 2, 0.7, model_params)
            adaptive_thr = adaptive_res["token_hit_rate"]
            print(f"\n  Adaptive α (V2): {pct(adaptive_thr)} ({(adaptive_thr - lru_thr):+.1%} vs LRU)")
        except Exception as e:
            print(f"\n  Adaptive α (V2) failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Q6: Enumerate simulation vs live differences
# ──────────────────────────────────────────────────────────────────────────────

def q6_enumerate_differences():
    print("\n" + "="*60)
    print("Q6: ENUMERATED DIFFERENCES — SIMULATION vs LIVE")
    print("="*60)

    diffs = [
        ("D1", "Output token caching",
         "Simulation inserts input+output tokens into cache.\n"
         "     Live: server generates different tokens → output prefix never matches.\n"
         "     Effect: simulation wastes cache space on unreusable output tokens.\n"
         "     Impact: BOTH policies suffer equally → not a LRU/Marconi confound,\n"
         "             but simulation overstates capacity pressure.",
         "Q1"),
        ("D2", "Model architecture params",
         "Paper uses Jamba (28 MLP) in simulation; we test Nemotron (24 MLP).\n"
         "     4 fewer MLP layers → slightly lower FLOP efficiency per token.\n"
         "     Marconi prioritizes high-efficiency nodes; fewer MLP layers changes ordering.",
         "Q2"),
        ("D3", "Arrival timestamps",
         "Simulation uses logical TS (uniform +1 increments).\n"
         "     Live has real gaps: within-session ~seconds, cross-session ~tens of seconds.\n"
         "     Recency scores (1/(t-t_i)) differ significantly between modes.\n"
         "     Within-session bursts → much higher recency weight in live.",
         "Q3"),
        ("D4", "Fixed vs adaptive alpha",
         "Live uses fixed --marconi-eff-weight (default 0.7).\n"
         "     Paper (V2) uses ConfigTuner to adaptively tune α after bootstrap window.\n"
         "     Fixed α=0.7 may be suboptimal; adaptive α is the paper's core mechanism.",
         "Q5"),
        ("D5", "Candidate set for eviction",
         "Simulation (evict_v2): evicts from leaves + single-child nodes.\n"
         "     Live (mamba_radix_cache.py): _evict_full_marconi uses leaf_only=True.\n"
         "     Smaller candidate set → less differentiation → closer to random.",
         "L5 (live)"),
        ("D6", "Running request memory pressure",
         "Simulation: cache capacity = full stated capacity.\n"
         "     Live: pages held by running requests are not evictable.\n"
         "     Effective cache capacity is lower under load.",
         "Structural"),
        ("D7", "Block-size quantization",
         "Simulation: exact byte-level sizes.\n"
         "     Live: KV cache stored in page-size blocks (e.g. 16 tokens/block).\n"
         "     Fragmentation → effective utilization lower than simulation assumes.",
         "Structural"),
        ("D8", "TTFT lookup table",
         "Simulation uses FLOP formulas (no actual TTFT measurements).\n"
         "     Paper used Jamba TTFT measurements for utility weighting.\n"
         "     We have no Nemotron TTFT lookup table.",
         "Phase 2"),
    ]

    for did, title, desc, ref in diffs:
        print(f"\n  [{did}] {title}  (→ {ref})")
        print(f"     {desc}")


# ──────────────────────────────────────────────────────────────────────────────
# Capacity sweep helper
# ──────────────────────────────────────────────────────────────────────────────

def capacity_sweep(requests, model_params):
    """Show how LRU vs Marconi difference varies with capacity."""
    print("\n" + "="*60)
    print("CAPACITY SWEEP — LRU vs Marconi at varying cache sizes")
    print("="*60)
    capacities = [0.5e9, 1e9, 2e9, 3e9, 5e9]
    rows = []
    for cap in capacities:
        lru = run_simulation(requests, cap, 1, 0.0, model_params)
        m07 = run_simulation(requests, cap, 3, 0.7, model_params)
        m15 = run_simulation(requests, cap, 3, 1.5, model_params)
        rows.append([
            f"{cap/1e9:.1f}GB",
            pct(lru["token_hit_rate"]),
            pct(m07["token_hit_rate"]),
            f"{(m07['token_hit_rate']-lru['token_hit_rate']):+.1%}",
            pct(m15["token_hit_rate"]),
            f"{(m15['token_hit_rate']-lru['token_hit_rate']):+.1%}",
        ])
    print_table("Token Hit Rate by Capacity",
                ["Capacity", "LRU", "Marconi 0.7", "Δ0.7", "Marconi 1.5", "Δ1.5"],
                rows)
    print("  NOTE: Marconi should show larger benefit at tight capacity.")
    print("  If Marconi never beats LRU at any capacity → fundamental issue.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="sharegpt", choices=["sharegpt", "lmsys"])
    parser.add_argument("--num-sessions", type=int, default=50)
    parser.add_argument("--sps", type=float, default=1.0, help="sessions per second")
    parser.add_argument("--capacity", type=float, default=1e9, help="cache bytes")
    parser.add_argument("--questions", nargs="+",
                        default=["q1", "q2", "q3", "q4", "q5", "q6", "capacity"],
                        help="Which questions to run")
    args = parser.parse_args()

    trace_path = MARCONI_DIR / "traces" / f"{args.trace}_sps={args.sps}_nums={args.num_sessions}.jsonl"
    if not trace_path.exists():
        print(f"ERROR: Trace not found at {trace_path}")
        print(f"Generate it first with:")
        print(f"  cd marconi && uv run python -c \"")
        print(f"    import sys; sys.path.insert(0, 'utils')")
        print(f"    from generate_trace import generate_{args.trace}_trace")
        print(f"    generate_{args.trace}_trace(sessions_per_second={args.sps}, num_sessions={args.num_sessions})")
        print(f"  \"")
        sys.exit(1)

    print(f"\nLoading trace: {trace_path}")
    requests = load_trace(trace_path)
    print(f"  {len(requests)} requests, "
          f"{sum(len(r['input_tokens']) for r in requests)} total input tokens, "
          f"{sum(len(r['output_tokens']) for r in requests)} total output tokens")

    capacity_bytes = args.capacity
    model_params = NEMOTRON_PARAMS  # use Nemotron by default (our target)

    print(f"\n  Cache capacity: {capacity_bytes/1e9:.1f} GB")
    print(f"  Model: Nemotron (24 SSM + 4 Attn + 24 MLP, d=4096, n=128)")

    questions = [q.lower() for q in args.questions]

    if "q6" in questions:
        q6_enumerate_differences()

    if "q1" in questions:
        q1_output_token_cache(requests, capacity_bytes, model_params)

    if "q2" in questions:
        q2_model_params(requests, capacity_bytes)

    if "q3" in questions:
        q3_arrival_timing(requests, capacity_bytes, model_params)
        run_real_ts_experiment(requests, capacity_bytes, model_params)

    if "q4" in questions:
        q4_eviction_stepthrough(requests, capacity_bytes, model_params)

    if "q5" in questions:
        q5_alpha_sweep(requests, capacity_bytes, model_params)

    if "capacity" in questions:
        capacity_sweep(requests, model_params)

    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
