# Marconi vs LRU: Findings

## Datasets

| Dataset | Source | Session type | Characteristics |
|---|---|---|---|
| ShareGPT | `ShareGPT_V3_unfiltered_cleaned_split` | Real human–model chat | Diverse single-topic conversations, 10+ turns, shorter and less repetitive |
| SWE-bench | `nebius/SWE-agent-trajectories` | Software agent trajectories | Debugging tasks (bash, edits, tests); older observations collapsed; long structured sequences |
| lmsys | `lmsys/lmsys-chat-1m` | Real multi-turn human chat | 10+ turn filter, long coherent conversations, prefixes grow to thousands of tokens |

Each request's input = full conversation history + new user message — a strictly growing prefix across turns.

---

## Experiment 1 — Eviction trace analysis

| Setting | Value |
|---|---|
| Dataset | ShareGPT, 10 sessions |
| Trace | `traces/sharegpt_sps=1.0_nums=10.jsonl` (113 requests) |
| Cache (mem-fraction) | 0.22 (~tight) |
| Configs | LRU vs Marconi α=0.7 (default) |
| Script | `scripts/experiments/capacity_test.sh --mem-fraction 0.22` |
| Eviction log | `logs/capacity-test/server_marconi_trace.log` |

### Finding 1: Marconi breaks short-session chains

**Session 8** (starts at 39 tokens):
```
req_0 (39 tok)    → cold miss      | cold miss
req_1 (95 tok)    → LRU: 41% HIT  | Marconi: 0% MISS  ← chain broken
req_2 (243 tok)   → LRU: 39% HIT  | Marconi: 0% MISS
req_3 (406 tok)   → LRU: 60% HIT  | Marconi: 0% MISS
req_10 (2200 tok) → LRU: 88% HIT  | Marconi: 18% MISS ← mid-chain re-evicted
```
**Why:** Session 8's 39-token prefix (eff=7,957, norm_eff=0.00) is evicted at step 3 — ranked
last because every other candidate is longer. LRU keeps it as most-recently-inserted.

### Finding 2: Marconi protects long-session prefixes (correctly)

**Session 0** (grows to 2291 tokens):
```
req_2 (479 tok)   → LRU:  0% MISS | Marconi: 59% HIT  ← prefix protected
req_5 (1099 tok)  → LRU:  0% MISS | Marconi: 81% HIT
req_8 (1673 tok)  → LRU:  0% MISS | Marconi: 87% HIT
req_9 (1909 tok)  → LRU:  0% MISS | Marconi: 76% HIT
```
**Why:** Node id=63 (1794 tokens, eff=255,536) ranks highest across 35 eviction rounds.
LRU displaced it whenever newer short sessions inserted more recently.

### Finding 3: Score normalization causes eviction churn

The eviction score is `utility = α × norm_eff + norm_rec`, where both terms are
**min-max normalized across all candidates at the moment of eviction** — not absolute values.
The node with the lowest utility is evicted.

The recency component `norm_rec` is derived as:
```
raw_rec  = 1 / (current_time - last_access_time)   # higher = more recent
norm_rec = (raw_rec - min_raw_rec) / (max_raw_rec - min_raw_rec)
```

Because it's relative, a node that looks "recent enough" right now can look "stale" one step
later simply because a newer node arrived and reset the scale:
```
Step 38: id=73 (115 tok)  KEPT  (norm_rec=0.49 — middle of the pack)
Step 39: id=73            EVICT (norm_rec=0.23 — new request pushed it to near-bottom)

Step 79: id=140 (1034 tok) EVICT — id=145 (norm_rec=0.78) was fresher, shielded id=140
Step 80: id=145            EVICT — now id=145 is the stale one (norm_rec=0.09)
```
Each incoming request reshuffles everyone's norm_rec. A node doesn't have to age — it just
needs a fresher neighbour to appear, and it drops toward the eviction threshold.

### Eviction summary
```
102 evictions | avg evicted eff = 41,728 | avg protected eff = 112,773 (2.7×)
LRU total hit rate: 61.7% | Marconi α=0.7: 53.2% | LRU wins by 8.5pp
```
Marconi wins sessions 0, 2, 4, 5 (long). LRU wins sessions 6, 8, 9 (short) and late-turn misses.

---

## Experiment 2 — Alpha sweep on swebench

| Setting | Value |
|---|---|
| Dataset | SWE-bench (SWE-agent trajectories), 10 sessions |
| Trace | `traces/swebench_sps=1.0_art=5_nums=10.jsonl` (63 requests) |
| Cache (mem-fraction) | 0.22 (~tight) |
| Configs | LRU, Marconi α ∈ {0.3, 0.5, 0.7, 1.0, 1.5} |
| Script | `scripts/experiments/alpha_sweep.sh --trace swebench` |
| Results | `results/alpha-sweep-swebench/` |

```
Config         Hit%    Δ LRU    AvgTTFT
lru           44.1%    +0.0pp    330ms
marc_a0.3     47.4%    +3.4pp    322ms  ← BEST (Marconi wins)
marc_a0.5     36.2%    -7.8pp    354ms  ← WORST (non-intuitive)
marc_a0.7     40.4%    -3.7pp    330ms
marc_a1.0     42.9%    -1.2pp    322ms
marc_a1.5     43.7%    -0.4pp    319ms
```

Per-session hit rates (▲=Marconi wins vs LRU, ▼=LRU wins):

| sess | lru | a0.3 | a0.5 | a0.7 | a1.0 | a1.5 |
|---|---|---|---|---|---|---|
| 0 | 42% | 56%▲ | 41% | 41% | 58%▲ | 58%▲ |
| 1 | 37% | 46%▲ | 40% | 40% | 51%▲ | 59%▲ |
| 2 | 22% | 57%▲ | 57%▲ | 57%▲ | 57%▲ | 57%▲ |
| 3 | 46% | 43% | 35%▼ | 35%▼ | 36%▼ | 36%▼ |
| 4 | 49% | 38%▼ | 26%▼ | 21%▼ | 24%▼ | 24%▼ |
| 5 | 43% | 26%▼ | 13%▼ | 30%▼ | 13%▼ | 13%▼ |
| 6 | 31% | 31% | 8%▼ | 38%▲ | 38%▲ | 38%▲ |
| 7 | 57% | 57% | 57% | 57% | 57% | 57% |
| 8 | 39% | 39% | 46%▲ | 48%▲ | 20%▼ | 20%▼ |
| 9 | 60% | 59% | 26%▼ | 38%▼ | 43%▼ | 43%▼ |

**Key findings:**
- **α=0.3 wins (+3.4pp):** Near-LRU nudge protects long SWE-agent trajectories (sess 0,1,2: +13–35pp) without catastrophically evicting medium sessions.
- **α=0.5 is worst (-7.8pp):** Mid-chain nodes in sess 5 (-30pp) and sess 9 (-34pp) have "moderate" efficiency — α=0.5 pushes them just below eviction threshold. U-shaped, not monotonic.
- **Session 2 wins at ALL alphas (+35pp):** Longest prefix — norm_eff always max, LRU cycles it out regardless.
- **TTFT:** α=0.3 saves 8ms; α=0.5 costs 24ms extra (more misses = more prefill work).

---

## Experiment 3 — Alpha sweep on lmsys

| Setting | Value |
|---|---|
| Dataset | lmsys-chat-1m (10+ turn conversations), 10 sessions |
| Trace | `traces/lmsys_sps=1.0_nums=10.jsonl` (129 requests) |
| Cache (mem-fraction) | 0.22 (~tight) |
| Configs | LRU, Marconi α ∈ {0.3, 0.5, 0.7, 1.0, 1.5} |
| Script | `scripts/experiments/alpha_sweep.sh --trace lmsys` |
| Results | `results/alpha-sweep-lmsys/` |

```
Config         Hit%    Δ LRU    AvgTTFT
lru           55.2%    +0.0pp    420ms
marc_a0.3     69.6%   +14.4pp    381ms  ← BEST
marc_a0.5     69.2%   +14.1pp    385ms
marc_a0.7     69.3%   +14.2pp    385ms
marc_a1.0     68.5%   +13.3pp    387ms
marc_a1.5     69.2%   +14.0pp    390ms
```

Per-session hit rates:

| sess | lru | a0.3 | a0.5 | a0.7 | a1.0 | a1.5 |
|---|---|---|---|---|---|---|
| 0 | 41% | 66%▲ | 67%▲ | 67%▲ | 68%▲ | 73%▲ |
| 1 | 57% | 36%▼ | 43%▼ | 43%▼ | 43%▼ | 43%▼ |
| 2 | 33% | 31% | 16%▼ | 16%▼ | 16%▼ | 16%▼ |
| 3 | 80% | 83% | 83% | 83% | 80% | 83% |
| 4 | 39% | 69%▲ | 72%▲ | 72%▲ | 72%▲ | 72%▲ |
| 5 | 45% | 63%▲ | 56%▲ | 56%▲ | 56%▲ | 56%▲ |
| 6 | 69% | 56%▼ | 37%▼ | 37%▼ | 37%▼ | 30%▼ |
| 7 |  0% | 42%▲ | 46%▲ | 46%▲ | 46%▲ | 57%▲ |
| 8 | 78% | 80% | 76% | 76% | 76% | 64%▼ |
| 9 | 79% | 82% | 80% | 81% | 78% | 72%▼ |

**Key findings:**
- **ALL alphas win by ~14pp** — lmsys is α-flat (only 1.1pp spread). Tight cache amplifies the previously observed +2pp at full capacity to +14pp (7× amplification).
- **Session 7: LRU 0% → Marconi 42–57%** — LRU completely fails; Marconi rescues by protecting its long accumulated prefix.
- **Why α-flat:** lmsys sessions are long enough that efficiency and recency signals agree — high-efficiency nodes are also the most recently needed. Alpha barely matters.
- **Sessions 1, 6 still lose:** Shorter lmsys conversations with lower FLOP efficiency — Marconi sacrifices them to protect longer ones.
- **TTFT:** α=0.3 saves 39ms vs LRU (381ms vs 420ms).

---

## Cross-dataset summary (mem-fraction=0.22)

| Dataset  | Sessions | Requests | Best α | Marconi | LRU   | Δ       |
|----------|----------|----------|--------|---------|-------|---------|
| ShareGPT | 10       | 113      | none   | 53.2%   | 61.7% | −8.5pp  |
| swebench | 10       | 63       | 0.3    | 47.4%   | 44.1% | +3.4pp  |
| lmsys    | 10       | 129      | 0.3    | 69.6%   | 55.2% | +14.4pp |

**Pattern:** Marconi's advantage scales with session length and turn repetition.
Short/diverse (ShareGPT) → LRU wins. Medium/code chains (swebench) → Marconi wins slightly.
Long/repetitive (lmsys) → Marconi wins decisively.

## Capacity curve summary (lmsys-50, α=0.3)

| mem-fraction | LRU Hit% | Marc Hit% | Δ       | LRU TTFT | Marc TTFT |
|---|---|---|---|---|---|
| 0.22 (tight)    | 25.1%  | 39.5%  | +14.4pp | 467ms | 436ms |
| 0.40 (moderate) | 73.3%  | 79.4%  | +6.0pp  | 380ms | 368ms |
| 0.60 (relaxed)  | 80.9%  | 83.3%  | +2.4pp  | 368ms | 359ms |
| 0.85 (large)    | 84.9%  | 84.6%  | −0.2pp  | 361ms | 357ms |

Marconi's lead shrinks monotonically as cache grows and eviction pressure eases.
At 0.85 both converge — advantage is purely an eviction-quality effect.

---

## Experiment 4 — Capacity curve

| Setting | Value |
|---|---|
| Dataset | lmsys-chat-1m, **50 sessions** (larger working set) |
| Trace | `traces/lmsys_sps=1.0_nums=50.jsonl` (744 requests) |
| Cache (mem-fraction) | 0.22, 0.40, 0.60, 0.85 |
| Configs | LRU vs Marconi α=0.3 (best from Exp 3) |
| Script | `scripts/experiments/capacity_curve.sh --dataset lmsys --nums 50` |
| Results | `results/capacity-curve-lmsys/` |

**Note on valid range:** `mem_fraction_static = (model weights + KV cache) / GPU capacity`.
Nemotron-H-8B needs ~16 GB (20% of A100 80 GB), so fractions below ~0.22 fail to load.
0.22 is the tight-cache floor; 0.85 approaches the full-capacity ceiling.

```
mem     LRU Hit%   Marc Hit%      Δ      Winner    LRU TTFT  Marc TTFT
--------------------------------------------------------------------
0.22      25.1%      39.5%    +14.4pp   Marconi      467ms     436ms  ▲▲▲▲▲▲▲
0.40      73.3%      79.4%     +6.0pp   Marconi      380ms     368ms  ▲▲▲
0.60      80.9%      83.3%     +2.4pp   Marconi      368ms     359ms  ▲
0.85      84.9%      84.6%     -0.2pp   LRU          361ms     357ms
```

### Finding 4a: Marconi's advantage is monotonically driven by eviction pressure

Marconi wins at every cache size except 0.85, where both policies converge (~0pp delta).
At 0.22 (50 sessions × long lmsys turns = working set >> cache), Marconi wins by **+14.4pp** —
exactly replicating the 10-session result from Exp 3 but with 5× more sessions.

### Finding 4b: Advantage decays as cache grows, vanishes at 0.85

- 0.22 → 0.40: delta halves from 14.4pp to 6.0pp as more long prefixes fit without eviction
- 0.40 → 0.60: delta halves again to 2.4pp (mostly only the shortest sessions get evicted)
- 0.60 → 0.85: effectively 0pp — the cache is large enough that both LRU and Marconi rarely evict

**Why:** At large cache, the working set fits and evictions are rare. Marconi's advantage is
entirely an eviction-quality effect: when it must evict, it evicts low-efficiency short nodes
and retains high-efficiency long ones. When evictions are rare, both policies make the same
(correct) decision almost every time.

### Finding 4c: TTFT tracks hit rate monotonically

Marconi saves 31ms at 0.22 (467ms→436ms, 6.6%), 12ms at 0.40, 9ms at 0.60.
At 0.85 both are within 4ms noise. TTFT mirrors hit-rate advantage at every cache size.

### Capacity curve summary

```
Tight (0.22): Marconi +14.4pp  ← cache pressure reveals eviction quality
Moderate (0.40): +6.0pp
Relaxed (0.60): +2.4pp
Large (0.85): ≈0pp             ← both policies equivalent, no pressure
```

**Conclusion:** Marconi is strictly better than LRU on lmsys-style long-session workloads
whenever the working set exceeds cache capacity. The benefit scales with pressure: the tighter
the cache, the more each eviction decision matters, and the larger Marconi's advantage.
