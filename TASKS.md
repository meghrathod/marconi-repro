# Marconi Reproduction — Task Tracker

Tracks progress against the [approved investigation plan](~/.claude/plans/foamy-purring-graham.md).
Update this file as tasks complete.

---

## Phase 1A — Simulation Only (MacBook, no GPU)

### Setup
- [x] `uv sync` — install deps (sglang-kernel now Mac-conditional)
- [ ] `uv run huggingface-cli login` — authenticate for gated lmsys dataset *(blocked on A100: just `export HF_TOKEN=...`)*
- [ ] Generate small trace: `uv run python scripts/gen_small_traces.py --num-sessions 20 --sps 1`

### S1 — Fix simulation bugs
- [x] **B1 fixed** — `marconi/radix_cache_hybrid.py:526`: MLP FLOP formula (`get_attn_flops` → `get_mlp_flops`)
- [x] **B2 fixed** — `marconi/radix_cache_hybrid.py:555`: bytes_evicted variable (`node.value` → `node_to_evict.value`)
- [ ] Run `quick_sim_compare.py` — compare buggy vs fixed token hit rates, quantify impact

### S2 — Alpha sensitivity in simulation
- [ ] Run `quick_sim_compare.py` — compare V1(LRU) vs V3(α=0.0/0.7/1.0) vs V2(adaptive)
- [ ] Answer: does any fixed α match adaptive? Does α=0.7 (live default) hurt Marconi?

### S3 — Nemotron vs Jamba params in simulation
- [x] Fetched actual Nemotron-H-8B-Base-8K architecture from HuggingFace config:
  - Pattern `M-M-M-M*-M-M-M-M-M*-…` → **24 Mamba + 4 Attn + 24 MLP**, d=4096, n=128
  - Same d/n as Jamba; 4 fewer MLP layers
- [ ] Run `quick_sim_compare.py` — Jamba params vs Nemotron params, LRU vs Marconi
- [ ] Answer: does architecture change the LRU vs Marconi ordering?

### S4 — Output token caching ablation *(deferred, after S1–S3)*
- [ ] Add flag to simulation to skip output token insertion
- [ ] Run with/without output caching; compare ordering

---

## Phase 1B — Live Inference (GPU required)

### Setup
- [x] `generate_trace.py` patched to use Nemotron tokenizer, CPU-safe (no `return_tensors="pt"`)
- [x] `scripts/gen_small_traces.py` — thin wrapper over patched `generate_trace.py`
- [x] `scripts/quick_sim_compare.py` — S1/S2/S3 comparison script (table output)
- [ ] Start SGLang server with Nemotron on GPU

### L1 — Step-through eviction verification *(do first)*
- [ ] Add logging to `_evict_full_marconi` and `_evict_full_lru` in `mamba_radix_cache.py`
- [ ] Run 5–10 request trace with both policies
- [ ] Confirm: are different nodes evicted? Is utility scoring non-trivial?
- [ ] Check: does min-max normalization collapse to degenerate scores with small candidate sets?

### L2 — Alpha sweep on live system
- [x] `run_live_experiments.sh --phase1` — swebench sps=5, α=[0.5, 1.0, 1.5, 2.0] (already done)
- [ ] Extend to lmsys + sharegpt; add α=0.0, 0.1
- [ ] **Latency**: run `bench_serving` (streaming) for each α → TTFT P50/P99, throughput
- [ ] **Cache hit rate**: run `trace_replayer` for each α → `cached_tokens` per request
- [ ] Answer: is there any α where Marconi beats LRU live?

### L3 — Output token mismatch verification
- [ ] Log per-turn `cached_tokens` vs `prompt_tokens` for one session
- [ ] Confirm: does `cached_tokens` plateau at turn-0 length (not grow with conv depth)?
- [ ] If confirmed: investigate fix (replay server's actual generated tokens as next-turn context)

### L4 — Real inter-arrival timing experiment
- [ ] Pass `--speed-factor 1.0` to `trace_replayer` (real timestamps from trace)
- [ ] Run lmsys sps=1, compare LRU vs Marconi hit rate/TTFT vs AFAP results

### L5 — Candidate set check
- [ ] Confirm: does `_evict_full_marconi` use `leaf_only=True` (only leaves, not intermediate)?
- [ ] Compare with simulation which evicts leaves + single-child nodes for both full and mamba
- [ ] If mismatch: try `leaf_only=False`, measure impact

---

## Phase 2 — H100 (coming soon)

- [ ] Full trace sweep: all sps values, all capacity sizes
- [ ] Build Nemotron TTFT lookup table from real H100 TTFT measurements
- [ ] Port ConfigTuner adaptive alpha to live SGLang (online α tuning after bootstrap window)
- [ ] Re-run full LRU vs Marconi sweep with adaptive alpha
- [ ] Reproduce paper Figures 3, 5, 6 with Nemotron instead of Jamba
- [ ] Multi-GPU (tensor parallel) experiments

---

## Key Files

| File | Role |
|---|---|
| `marconi/radix_cache_hybrid.py` | Simulation — bugs fixed at :526 and :555 |
| `marconi/utils/generate_trace.py` | Patched — Nemotron tokenizer, CPU-safe |
| `scripts/gen_small_traces.py` | Generate small traces (wraps generate_trace.py) |
| `scripts/quick_sim_compare.py` | S1/S2/S3 comparison runner |
| `sglang/…/mamba_radix_cache.py` | Live eviction — add logging for L1 |
| `sglang/python/sglang/bench_serving.py` | Better TTFT/latency for live experiments |
| `src/trace_replayer.py` | Cache hit rate collection for live experiments |
| `src/run_live_experiments.sh` | Alpha sweep (L2), timing (L4) |

## Working Hypothesis (most likely first)

1. **Fixed α=0.7 is suboptimal** — adaptive tuning is the paper's core; without it Marconi may behave worse than LRU
2. **Simulation bugs inflate paper results** — MLP formula inflates efficiency for short seqs; fixing may close sim vs live gap
3. **Output token prefix mismatch** — live cache hits lower than sim because server generates different tokens than trace; affects both policies equally
4. **Memory pressure from running requests** — live cache gets less space than simulation assumes
5. **Degenerate utility scoring** — with few eviction candidates, min-max normalization may make Marconi behave randomly
