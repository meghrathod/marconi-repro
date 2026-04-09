# Marconi Reproduction — Running Context for A100

**Problem**: Every live test (results/live-limited/) shows LRU beating Marconi. Goal: find why.
**Model under test**: `nvidia/Nemotron-H-8B-Base-8K`
**Task checklist**: `TASKS.md` (repo root)

---

## What We've Studied

### Paper (arxiv 2411.19379) https://arxiv.org/pdf/2411.19379
Marconi = FLOP-aware eviction for hybrid SSM+Attention LLMs.
- `utility = α × normalize(FLOP_efficiency) + normalize(recency)` — evict min-utility node
- `FLOP_efficiency = FLOPs_saved / bytes_used`
- Attention is O(L²) → long prefixes exponentially more valuable
- Mamba SSM state: fixed-size, O(L) compute savings
- Alpha tuned **online** by ConfigTuner after bootstrap window (V2 = adaptive)

### Simulation code (`marconi/`) provided by paper authors, few updates by us:
- `radix_cache_hybrid.py` — core radix tree, V1=LRU, V2=adaptive Marconi, V3=fixed-α oracle
- `policy_exploration.py` — sweeps (dataset × capacity × sps × policy)
- `config_tuner.py` — grid-searches α on past window after bootstrap
- `utils.py` — FLOP formulas (attn O(L²), MLP O(L), Mamba O(L))
- Hardcoded Jamba 1.5 Mini params: `num_ssm_layers=24, num_attn_layers=4, num_mlp_layers=28, d=4096, n=128`
- Simulation inserts `input_tokens + output_tokens` per request — output tokens compete for cache space

### SGLang PR (`sglang/python/sglang/srt/mem_cache/`) - https://github.com/sgl-project/sglang/pull/20045
- `mamba_radix_cache.py` — `MambaRadixCache`, two separate LRU lists (full KV + mamba SSM)
- `marconi_utils.py` — FLOP formulas for Nemotron's pattern-string arch (correct — no MLP bug)
- `cache_init_params.py` — config object
- Eviction: `evict_mamba()` + `evict_full()` dispatch to `_evict_*_marconi/lru/seglen`
- Live uses **fixed** `--marconi-eff-weight` (no ConfigTuner, no adaptive alpha)

### Nemotron-H-8B-Base-8K Architecture (fetched from HF AutoConfig)
```
hybrid_override_pattern = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
52 total layers:  M=24 Mamba,  *=4 Attention,  -=24 MLP,  E=0 MoE
hidden_size=4096, mamba_num_heads=128, mamba_head_dim=64, ssm_state_size=128
num_attention_heads=32, num_key_value_heads=8, head_dim=128
```
**Key insight**: Same d=4096, n=128, 24 SSM, 4 attn as Jamba — only difference is 24 vs 28 MLP layers.

### Trace Replayer (`src/trace_replayer.py`)
- Non-streaming `/v1/completions`, `cached_tokens` from `usage.prompt_tokens_details`
- TTFT from Prometheus histogram deltas (indirect — `bench_serving` is better for latency)
- Default `speed_factor=0` (AFAP); `--speed-factor 1.0` for real timestamps
- **Output token mismatch**: trace `input_tokens` for turn N+1 include original dataset's output tokens; server generated different Nemotron tokens → cache hits only on shared input prefix, not response portion. Affects both policies equally.

### bench_serving (`sglang/python/sglang/bench_serving.py`)
- Streaming-based true TTFT, P50/P90/P99 latency, throughput
- Does NOT report cache hit rate
- Use for latency; use `trace_replayer` for cache hit rate

---

## What's Done ✅

**Bugs fixed in `marconi/radix_cache_hybrid.py`**:
- `:526` — MLP FLOP formula: `get_attn_flops(seqlen_parent)` → `get_mlp_flops(seqlen_parent)` *(was copy-paste from attn line; for lmsys/sharegpt depths inflated MLP savings; for SWEBench depths >8192 tokens went negative)*
- `:555` — bytes_evicted: `len(node.value)` → `len(node_to_evict.value)` *(used last loop variable instead of evicted node — miscounted freed bytes)*

**`marconi/utils/generate_trace.py` patched**:
- `device = "cpu"` (was `"cuda:0"`) - we can update it back to cuda as we are running on A100
- `TOKENIZER_MODEL` default → `nvidia/Nemotron-H-8B-Base-8K` (was Llama-2)
- All `tokenizer(text, return_tensors="pt").input_ids[0].tolist()` → `tokenizer.encode(text)` (no PyTorch required)


**`pyproject.toml` fixed**:
- `sglang-kernel` now conditional on Linux (`sys_platform == 'linux'`)
- Added: `transformers, datasets, tqdm, numpy, pandas, pytz, aiohttp, prometheus-client, matplotlib, scipy`

---

## What Remains

### Phase 1A — Simulation (no GPU)

**Blocked by HF auth** — `lmsys/lmsys-chat-1m` is gated:
```bash
uv run huggingface-cli login   # or export HF_TOKEN=...
uv run python scripts/gen_small_traces.py --num-sessions 20 --sps 1
uv run python scripts/quick_sim_compare.py   # runs S1 + S2 + S3
```
`quick_sim_compare.py` outputs a table covering:
- **S1**: buggy vs fixed token hit rates for LRU + Marconi
- **S2**: V1(LRU) vs V3(α=0.0/0.7/1.0) vs V2(adaptive) — quantifies adaptive alpha benefit
- **S3**: Jamba params vs Nemotron params — quantifies model mismatch effect

**S4** (deferred): ablate output token caching — add flag to simulation to not insert output tokens, compare ordering.

### Phase 1B — Live (GPU required)

**L1 — Step-through eviction** *(do first)*:
Add per-eviction logging to `sglang/…/mamba_radix_cache.py:_evict_full_marconi` and `_evict_full_lru`. Run 10-request trace side-by-side. Check if utility scores are degenerate (min-max normalization with few candidates collapses scores → random eviction).

**L2 — Alpha sweep**:
`src/run_live_experiments.sh --phase1` already done for swebench. Extend to lmsys/sharegpt; add α=0.0, 0.1. Use `bench_serving` for TTFT + `trace_replayer` for cache hit rate.

**L3 — Output token mismatch**:
Log per-turn `cached_tokens` vs `prompt_tokens` for one session. If `cached_tokens` plateaus at turn-0 prompt length → D2 confirmed. Fix: replay server's actual generated tokens as next-turn context.

**L4** — Real inter-arrival timing: pass `--speed-factor 1.0` to replayer.

**L5** — Candidate set: confirm `_evict_full_marconi` uses `leaf_only=True` (only evicts leaves for full KV); simulation evicts leaves+single-child. Try `leaf_only=False`.

### Phase 2 — H100

- Build Nemotron TTFT lookup table from real measurements
- Port ConfigTuner adaptive alpha to live SGLang
- Full sweep: all sps × capacity × dataset
- Reproduce paper Figures 3/5/6 with Nemotron

---

## Hypothesis Ranking

1. **Fixed α=0.7 suboptimal** — adaptive tuning is paper's core mechanism; S2/L2 will confirm
2. **Simulation bugs inflate paper results** — S1 will quantify (both bugs hurt Marconi's simulation accuracy)
3. **Output token mismatch** — explains sim vs live gap; not a LRU/Marconi confound
4. **Running-request memory pressure** — live cache gets less space than sim assumes
5. **Degenerate utility scoring** — L1 will check if scoring is actually differentiating candidates