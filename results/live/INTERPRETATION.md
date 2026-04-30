# Trace replay: three scenarios and figure guide

This directory compares **three** serving configurations (see `marconi/`, `lru/`, `no-cache/`):

| # | Configuration | Meaning |
|---|----------------|--------|
| 1 | **No prefix cache** | Prefix caching disabled at the server — upper bound on prefill work per request. |
| 2 | **LRU prefix cache** | Prefix caching on, **LRU eviction** (standard SGLang-style baseline for hybrid radix cache). |
| 3 | **Marconi** | Prefix caching on, **Marconi (FLOP-aware) eviction** — [PR #20045](https://github.com/sgl-project/sglang/pull/20045) integration. |

**Why some plots use “÷ no prefix cache”:**  
The paper and serving literature often report *speedup vs no caching*. Ratios like “LRU ÷ (no prefix cache)” compare scenarios **2 or 3** to **1** only; they are **not** a two-way world without LRU. A separate figure reports **Marconi ÷ LRU** with both caches on, i.e. eviction policy only.

**Alignment with [Marconi (arXiv:2411.19379)](https://arxiv.org/abs/2411.19379):**

- **`p95_ttft_cdf_three_policies`** — Empirical CDF of per-session P95 TTFT (ms) for all three configurations (same spirit as TTFT distribution comparisons in the paper, e.g. discussion around Fig. 10(b)).
- **`ttft_vs_sps_three_way`** — TTFT vs session arrival rate (cf. Fig. 13(a) style: load on x-axis).
- **`swebench_art_breakdown`** — Same three lines per panel; columns vary inter-request delay (`art`), analogous to Fig. 13(b) (effect of spacing within a session).
- **`ttft_ratio_to_nocache_baseline`** — Supplementary: caching policies normalized by scenario (1).
- **`marconi_vs_lru_ttft_ratio_cdf`** — Supplementary: Marconi vs LRU only (both prefix caching on).

**Why results may not match paper headlines:** PR #20045 stresses **eviction**; the paper’s largest gains combine **admission + eviction** and use workloads with strong shared prefixes. Chat traces and tokenizer/serving mismatch can shrink gaps between Marconi and LRU. See also [PR #17898](https://github.com/sgl-project/sglang/pull/17898) for a broader SGLang implementation with admission/tuning.

---

## Debugging: `cached_tokens` always 0

**Cause (fixed in `src/run_live_experiments.sh`):** SGLang’s OpenAI-compatible usage block only includes `prompt_tokens_details.cached_tokens` when the server is started with **`--enable-cache-report`**. The default is `False` (see `ServerArgs.enable_cache_report` in `sglang/.../server_args.py` and `UsageProcessor` in `entrypoints/openai/usage_processor.py`). Without it, the replayer’s `prompt_details.get("cached_tokens", 0)` is always 0 even on cache hits.

**Verify:** In `logs/live/server_*.log`, the printed `server_args=` line must contain `enable_cache_report=True`.

**Note:** OpenAI-style usage often **omits** `prompt_tokens_details` when `cached_tokens == 0` (no cache hit on that request). You will still see zeros on cold prefixes; look for non-zero values on later `turn_id` rows after fixing `--enable-cache-report`.

### Host runs: one listener on the port

If **`docker compose up sglang-server`** (or any other process) is bound to **30000** on the host, `run_live_experiments.sh` will talk to that server—not the subprocess it thinks it started. Stop Docker (or change `PORT` in the script) before a host-native run. The script now runs **`fuser` + `pkill`** and **`setsid`/`kill` by process group** so a previous SGLang tree does not survive between no-cache / LRU / Marconi.

### Limited trace run (before the full 30-file matrix)

Use `./src/run_live_experiments.sh --limited` → three traces (LMSys, ShareGPT, SWEBench @ sps=1) under `results/live-limited/`. Full matrix: run the same script with **no** `--limited` → `results/live/`.

### TTFT was zero in JSONL (fixed in trace_replayer)

SGLang exposes TTFT as a Prometheus **histogram** with **one `_sum` / `_count` line per label set**. The replayer used to keep only the last line, so cumulative sum/count were wrong and per-request TTFT deltas were often zero. The parser now **sums** all `time_to_first_token_*_sum` and `*_count` lines. A short retry loop was added after each request before scraping `/metrics` so the histogram can update.

---

## Debugging: TTFT CDFs for Marconi / LRU / no-cache almost identical

That can be **real** (little differentiation) or a measurement artifact:

1. **Mostly cache misses** — Traces are sorted globally by timestamp. Many sessions contribute turn 0 before any turn 1, so parent prefixes are often evicted before reuse; Marconi vs LRU only diverges under **contention** when different entries compete. With few hits, all three policies pay full prefill → similar TTFT. **Check** non-zero `cached_tokens` on later turns after fixing `--enable-cache-report`.

2. **Tokenizer vs served model** — `marconi/utils/generate_trace.py` defaults to `TOKENIZER_MODEL=meta-llama/Llama-2-7b-hf` while live runs use `nvidia/Nemotron-H-8B-Base-8K`. Cumulative `input_token` *IDs* are consistent for radix matching, but they are not Nemotron’s real tokenization of the text. Regenerate traces with `TOKENIZER_MODEL` matching the served model for realistic hits and TTFT.

3. **Nemotron vs Qwen in upstream PRs** — [PR #17898](https://github.com/sgl-project/sglang/pull/17898) benchmarks use `Qwen3-Next-80B` with `--mamba-scheduler-strategy extra_buffer` and a **synthetic shared-prefix** dataset (`bench_serving`, `generated-shared-prefix`). In this SGLang tree, **Nemotron-H** sets `support_mamba_cache_extra_buffer=False` (`server_args.py`), so **`extra_buffer` is not available** for that architecture; you cannot copy Qwen’s scheduler flags verbatim.

**Where PR bench numbers come from:** They run `python -m sglang.bench_serving` (see the PR description), which drives many concurrent requests and reads server-side throughput/latency and cache-related stats (e.g. token hit rate) from the running engine—not from the same JSONL path as `trace_replayer.py`. To align with that style locally, use the same `bench_serving` CLI after your server flags match the PR as closely as your model allows.
