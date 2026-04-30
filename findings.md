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

---

## Experiment 5 — Live server evaluation (live-minimal-32K-v2)

| Setting | Value |
|---|---|
| Dataset | lmsys, sharegpt, swebench (100 sessions each) |
| Server | SGLang with Nemotron-H-8B-Reasoning-128K (4×A100) |
| Max Mamba slots | 318 (binding constraint) |
| Configs | LRU, Marconi α=0.3, Marconi α=1.0 |
| Results | `results/live-minimal-32K-v2/` |
| Logs | `logs/live-minimal-32K-v2/server_{lru,marconi_a0.3}.log` |

### Performance summary

```
Dataset                                  LRU hit%   Marc-a0.3   Marc-a1.0   Δ(a0.3−lru)
------------------------------------------------------------------------------------------
lmsys sps=0.25 (slow arrival)             67.8%      69.1%       68.2%        +1.3pp
lmsys sps=1                               59.1%      36.5%       35.5%       −22.6pp ▼
lmsys sps=5 (fast arrival)                54.9%      41.3%       39.5%       −13.5pp ▼
sharegpt sps=0.25                         63.6%      61.8%       59.5%        −1.8pp
sharegpt sps=1                            59.3%      31.5%       29.8%       −27.8pp ▼
sharegpt sps=5                            57.4%      33.3%       26.2%       −24.0pp ▼
swebench sps=1                            40.8%      23.4%       21.8%       −17.4pp ▼
swebench sps=5 art=5                      34.3%      19.9%       18.5%       −14.4pp ▼
swebench sps=5 art=7.5                    32.7%      18.1%       17.2%       −14.6pp ▼
```

LRU wins in 8 of 9 configurations. Marconi's only positive result is the slow-arrival lmsys
setting (+1.3pp), where request interarrival gaps are long enough that Mamba slot pressure
is rarely the binding constraint.

**Eviction activity (TP0 totals):**

| Policy | Eviction events | Mamba nodes evicted | `evict_full` fires |
|---|---|---|---|
| LRU | 23,798 | 41,475 | 0 |
| Marconi α=0.3 | 11,998 | 21,077 | 0 |

`evict_full` (the KV-token eviction path) never fires in either run. The KV token pool
(17.26 M tokens, 65.84 GB) is never full; the Mamba slot pool (318 slots) is the sole
binding constraint throughout.

---

### Finding 5a: SGLang's LRU tombstones branching internal nodes — Marconi cannot

This is the primary cause of LRU's dominance in the live server runs.

**The paper's LRU baseline (`evict_v1` in `marconi/radix_cache_hybrid.py`)** was
leaf-only: it collected leaves with `_collect_leaves()`, evicted them, and promoted
a parent to the leaf pool only after all its children were removed. A branching node
(2+ children) could never be evicted directly — it had to shed children first.

**SGLang's LRU** (`_evict_mamba_lru` in `mamba_radix_cache.py`) walks the
`mamba_lru_list` from the oldest end and operates differently:

```python
if len(x.children) > 0:
    # Internal node: free Mamba slot, tombstone — KV tokens preserved
    self.req_to_token_pool.mamba_pool.free(x.mamba_value)
    self._tombstone_internal_node(x)   # mamba_value = None; node stays in tree
else:
    # Leaf: evict both Mamba slot and KV tokens
    self._evict_leaf_node(x, True)
```

**Branching internal nodes (2+ children) are tombstoned in SGLang's LRU**: the Mamba
slot is freed and the node's `mamba_value` is cleared, but the node itself and its KV
tokens remain in the radix tree. Future requests can still get KV hits through that node.
This is a "free harvest" — recovering a scarce Mamba slot at zero KV cost.

**Marconi's filter** (`_collect_unlocked_candidates` in `mamba_radix_cache.py:915`)
reads `if len(x.children) <= 1: candidates.append(x)`, which is correct per paper §4.3
("nodes with multiple children represent the common prefixes shared by multiple requests
and should not be evicted"). Marconi cannot access branching internal nodes at all.

**In the LRU log, 45,600 of 95,192 total Mamba evictions (48%) are branching internal
nodes (is_leaf=False).** For each one, a full Mamba slot is recovered while the KV
prefix survives. Marconi must instead evict a leaf or 1-child internal node, destroying
both its Mamba slot and its KV tokens.

**Concrete examples from the LRU log:**

```
ts=3545  | evict lru num: 2 | n_cands=296
  [0] id=7 toks=6/path=7 age=3526.0 is_leaf=False
  → 6-token segment freed (Mamba slot reclaimed); 7 KV tokens stay

ts=3804  | evict lru num: 3 | n_cands=298
  [0] id=72 toks=247/path=9130 age=3340.0 is_leaf=False
  → 247-token segment freed; 9,130 KV tokens stay in tree

ts=784196| evict lru num: 1 | n_cands=314
  [0] id=55334 toks=8192/path=17634 age=8416.0 is_leaf=False
  → 8,192-token segment freed; 17,634 KV tokens stay in tree
```

The id=55334 event is especially striking: LRU recovers a Mamba slot from a
17,634-token prefix node without losing a single cached KV token. That prefix remains
fully searchable in the radix tree — only the Mamba state (needed for recurrent
state replay) is gone.

**The paper's comparison was:** leaf-only LRU vs Marconi (leaf + 1-child tombstone).
Marconi's ability to tombstone 1-child internal nodes was a novel advantage over that baseline.

**Our SGLang comparison is:** (leaf + 1-child + branching) LRU vs Marconi (leaf + 1-child).
LRU now has an additional capability Marconi cannot match. The paper's claimed improvement
is over a weaker LRU than what SGLang actually implements.

---

### Finding 5b: FLOP efficiency scoring collapses when seqlen_child ≈ 1

From the first Marconi eviction event in the live run (TP0, ts=4306, n_cands=287):

```
[0] id=9   toks=1/path=145  eff=  747.5 (n=0.00)  rec_n=0.00  util=0.000
[1] id=12  toks=1/path=14   eff=  762.5 (n=0.00)  rec_n=0.00  util=0.000
[2] id=188 toks=1/path=15   eff=  762.4 (n=0.00)  rec_n=0.00  util=0.000
[3] id=223 toks=1/path=14   eff=  762.5 (n=0.00)  rec_n=0.00  util=0.000
[4] id=155 toks=2/path=19   eff= 1523.8 (n=0.00)  rec_n=0.00  util=0.000
```

All five candidates have `n=0.00`, meaning they sit at the bottom of the efficiency
distribution among all 287 candidates. Their `toks` values (1–2) are the segment lengths
stored in that radix-tree node; `path` is the full prefix length from root.

The FLOP efficiency formula for Mamba layers in `marconi_utils.py` uses `seqlen_child`
(not `seqlen_total`) for SSM savings:
```python
elif layer_type == "linear_attention":
    flops += get_linear_attn_flops(seqlen_child, ...)  # seqlen_child = toks
```

When `seqlen_child = 1`, Mamba-layer savings are near-zero for all such nodes. The
discriminating term becomes the attention layers' quadratic savings
`(seqlen_total² − seqlen_parent²)`, which depends on path depth — correctly ranking
deeper nodes higher. The scoring is working as designed: these 1-token segments at
path=7–19 are legitimately lowest-efficiency.

The visible collapse (`n=0.00`) is the min-max normalization for the five candidates
shown, all of which are the eviction targets (lowest-ranked). This is not a scoring
failure — it is the algorithm correctly identifying which nodes to evict.

---

### Finding 5c: Architecture comparison — Nemotron vs Jamba-1.5-Large vs paper's 7B model

Side-by-side comparison (via `AutoConfig` from HuggingFace):

```
Model                          Layers   SSM   Attn   MLP    SSM:Attn   MoE     Params
---------------------------------------------------------------------------------------
Paper primary model (§5)         56     24     4      28      6:1       No       7B
Nemotron-H-8B-Reasoning-128K     52     24     4      24      6:1       No       8B
Jamba-Large-1.5 (accessible)     72     63     9       0      7:1   16 exp/2    52B
Jamba-1.5-Mini                  N/A    N/A   N/A     N/A      —        —        12B  ← gated
```

Key observations:

**Nemotron-H-8B is architecturally near-identical to the paper's primary 7B model**: both
have exactly 4 Attention layers and 24 SSM (Mamba) layers (6:1 ratio). Nemotron has 24
dedicated MLP-only blocks vs the paper's 28 — a 4-layer difference on a 52-layer model.
The user is correct: there is no fundamental architecture mismatch between our test model
and the paper's evaluation model.

**Jamba-Large-1.5 is architecturally distinct from both**: 63 SSM + 9 Attention (7:1
ratio), no standalone MLP layers (the MoE FFN is embedded within each Mamba/Attention
layer via `expert_layer_period=2`), and 2.6× more SSM layers than either. It is a 52B
model operating in a different regime. The claim that "Nemotron only has 4 extra SSM
layers" relative to Jamba doesn't match the data — Jamba-Large-1.5 has 39 more SSM
layers than Nemotron. The architectural similarity is actually between Nemotron and the
paper's 7B model, not between Nemotron and Jamba.

**Jamba-1.5-Mini** (the paper's cited secondary model) is not accessible with the
available credentials (403 on the gated repo), so a direct comparison is not possible.

The primary finding (Finding 5a) is architecture-independent: it is a property of the
SGLang LRU implementation traversing `mamba_lru_list` without the `len(x.children) <= 1`
guard. This would disadvantage Marconi equally on any hybrid model.

---

### Root cause summary

| Factor | Impact |
|---|---|
| SGLang LRU tombstones branching-internal nodes; Marconi filter excludes them | **Primary** — 48% of LRU Mamba evictions are "free" KV-preserving tombstones |
| Tight Mamba slot pool (318 slots) with zero KV pool pressure | Amplifies the tombstone advantage: every Mamba slot recovered without KV loss is worth more |
| Paper's LRU baseline was leaf-only; SGLang's LRU is not | The comparison is against a weaker LRU than what ships in production |
| FLOP efficiency scoring degenerates for 1-token segments | Secondary — scoring still ranks candidates correctly within the accessible pool |
