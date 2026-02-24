# Marconi Integration: From Simulation to Live Inference

This document outlines the **immediate next steps** for reproducing [Marconi](https://arxiv.org/abs/2411.19379) results in a live [SGLang](https://github.com/sgl-project/sglang) inference system, starting from a working sglang server Docker image.

> [!NOTE]
> The Marconi trace-driven simulation (CPU-only, conda env) has already been verified and is producing correct graphs. This document focuses on the **live GPU inference** path.

---

## Table of Contents

1. [Paper Analysis](#1-paper-analysis)
2. [Model Selection](#2-model-selection)
3. [Phase 0: SGLang Smoke Test](#3-phase-0-sglang-smoke-test)
4. [Phase 1: Live Inference Baseline](#4-phase-1-live-inference-baseline)
5. [Phase 2: Implement & Compare Marconi](#5-phase-2-implement--compare-marconi)
6. [Key References](#6-key-references)

---

## 1. Paper Analysis

### 1.1 What Experiments Does the Paper Actually Run?

The paper uses **two distinct evaluation modes**. Understanding this is critical for reproducing results correctly.

| Evaluation | Mode | Model | Hardware | Figures |
|---|---|---|---|---|
| Token hit rate, FLOP savings | **Trace-driven simulation** (CPU-only) | NVIDIA Mamba2-Hybrid 7B config | Any CPU machine | Figs 7, 8, 10–13 |
| TTFT measurements | **Live GPU inference** | Jamba-1.5-Mini (12B/52B) | 4× A100-40GB (vLLM) | Fig 9 |

> [!IMPORTANT]
> The **main results** (token hit rate improvements of 4.5–34.4× over vLLM+) come entirely from the trace-driven simulator. The simulator uses pre-tokenized inputs (tokenized with `meta-llama/Llama-2-7b-hf`) and scores nodes using FLOP/memory formulas—**no actual model weights are loaded**.

### 1.2 The Simulation Model Config

The trace-driven simulator ([`policy_exploration.py`](../marconi/policy_exploration.py) lines 196–202) uses:

```python
# NVIDIA's Attention-Mamba2 Hybrid 7B model (https://arxiv.org/pdf/2406.07887)
num_ssm_layers = 24
num_attn_layers = 4
num_mlp_layers = 28
d = 4096
n = 128
```

This config is from the [NVIDIA Mamba-2 paper](https://arxiv.org/abs/2406.07887). The model available on HuggingFace is [`nvidia/mamba2-hybrid-8b-3t-128k`](https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-128k), but it uses custom Megatron-LM code and is not directly loadable in SGLang.

### 1.3 Baselines in the Paper

| System | Admission | Eviction | Description |
|---|---|---|---|
| **vLLM+** | Fine-grained (every 32 tokens) | LRU | Extended vLLM with hybrid support |
| **SGLang+** | Judicious (Marconi's admission) | LRU | SGLang with Marconi's admission but LRU eviction |
| **Marconi** | Judicious | FLOP-aware (α-weighted) | Full Marconi system |

### 1.4 Key Metrics

- **Token hit rate** (%): ratio of tokens that skipped prefill / total input tokens
- **TTFT** (ms): Time to first token at P5, P50, P95 percentiles
- **FLOP saved**: total compute savings (proxy for latency)

### 1.5 Marconi's Core Mechanisms

**Judicious Admission**: Only cache SSM states when reuse is likely:
- Purely-input prefixes (system prompts): speculative insertion at branch points
- Input-and-output prefixes (conversations): cache only the last decoded token's SSM state

**FLOP-Aware Eviction**: Score nodes by `S(n) = recency(n) + α · flop_efficiency(n)`, where FLOP efficiency = compute_saved / memory_footprint. α=0 degenerates to LRU.

**Online α Tuning**: Bootstrap window → replay past requests with α ∈ {0, 0.1, …, 2.0} → pick best α.

See [`radix_cache_hybrid.py`](../marconi/radix_cache_hybrid.py) and [`config_tuner.py`](../marconi/config_tuner.py) for the simulation implementations.

### 1.6 FLOP Formulas (from [`utils.py`](../marconi/utils.py))

| Layer | FLOPs (forward only) |
|---|---|
| **Attention** | `8·L·D² + 4·L²·D` |
| **MLP** | `16·L·D²` |
| **Mamba1/2** | `12·L·D² + 16·L·D·N + 10·L·D` |

Memory per node: `num_ssm_layers × mamba_state_size(d,n) + num_attn_layers × kv_size(L,d)`

---

## 2. Model Selection

### 2.1 Recommended Model: Nemotron-H 8B

**Model**: [`nvidia/Nemotron-H-8B-Base-4096`](https://huggingface.co/nvidia/Nemotron-H-8B-Base-4096)

| Config | Paper (Mamba2-Hybrid 7B) | Nemotron-H 8B | Match? |
|---|---|---|---|
| Attention layers | 4 | 4 | ✅ |
| SSM (Mamba-2) layers | 24 | 24 | ✅ |
| MLP layers | 28 | 24 | ⚠️ close |
| d_model | 4096 | 4096 | ✅ |
| SSM state dim (n) | 128 | 128 | ✅ |
| Total layers | 56 | 52 | ⚠️ |

### 2.2 Rationale

1. **Architecture match**: 4 attn + 24 SSM layers with d=4096, n=128 are identical. Only MLP count differs (24 vs 28).
2. **SGLang native support**: `NemotronHForCausalLM` is supported in SGLang with `MambaRadixCache` via [PR #11214](https://github.com/sgl-project/sglang/pull/11214).
3. **Size**: ~8B parameters, fits on 1–2 A100-40GB GPUs.
4. **MLP difference impact**: MLP layers only affect the `flop_efficiency` denominator/numerator in eviction scoring. The relative ordering of Marconi vs LRU eviction decisions is largely preserved since the attn/SSM ratio (the main driver of Marconi's advantage) is identical.

### 2.3 Alternatives Considered

| Model | Why not? |
|---|---|
| `nvidia/mamba2-hybrid-8b-3t-128k` | Uses Megatron-LM custom code, not HF-native, likely incompatible with SGLang |
| `ai21labs/Jamba-1.5-Mini` | SGLang Jamba support still maturing (open feature request); MoE architecture adds complexity |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | Too large (80B), requires 4+ GPUs, very different architecture from paper's 7B hybrid |

### 2.4 Simulation Config Adjustment

When comparing live results with the trace-driven simulator, update the simulator's config to match Nemotron-H:

```python
# In policy_exploration.py, change:
num_mlp_layers = 24  # was 28, now matches Nemotron-H 8B
```

This ensures apples-to-apples comparison between simulated and live results.

---

## 3. Phase 0: SGLang Smoke Test

**Goal**: Verify the sglang Docker image correctly loads and serves a hybrid model.

### 3.1 Start the Server

```bash
# Option A: via docker-compose (recommended)
MODEL_NAME=nvidia/Nemotron-H-8B-Base-4096 TENSOR_PARALLEL=1 docker compose up sglang-server

# Option B: in container directly
python3 -m sglang.launch_server \
    --model nvidia/Nemotron-H-8B-Base-4096 \
    --host 0.0.0.0 --port 30000 \
    --tp 1
```

> [!TIP]
> Use `--tp 1` for single-GPU. If the model doesn't fit, try `--tp 2`. The Nemotron-H 8B in FP16 needs ~16GB VRAM for weights alone.

### 3.2 Verify Health

```bash
# Health check
curl http://localhost:30000/health

# Model info
curl http://localhost:30000/v1/models
```

### 3.3 Test Inference

```bash
# Simple completion
curl -s http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-H-8B-Base-4096",
    "prompt": "The capital of France is",
    "max_tokens": 20,
    "temperature": 0
  }' | python3 -m json.tool

# Chat completion
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-H-8B-Base-4096",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50,
    "temperature": 0
  }' | python3 -m json.tool
```

### 3.4 Verify Hybrid Architecture Detection

Check the server logs for:
- Mamba/SSM layer detection
- `MambaRadixCache` initialization (if radix cache is enabled)
- No errors about unsupported model type

```bash
# Check logs for hybrid-specific messages
docker compose logs sglang-server 2>&1 | grep -i "mamba\|hybrid\|radix\|ssm"
```

### 3.5 Smoke Test Checklist

- [ ] Server starts without errors
- [ ] `/health` returns 200
- [ ] `/v1/models` lists the model
- [ ] Completions return coherent text
- [ ] Server logs show hybrid architecture recognized
- [ ] `MambaRadixCache` is initialized (when radix cache enabled)

---

## 4. Phase 1: Live Inference Baseline

**Goal**: Establish baseline performance with SGLang's default caching (LRU) using Marconi's trace workloads.

### 4.1 Trace Format Adaptation

Marconi's pre-tokenized traces are JSONL with fields:
```json
{
  "session_id": "...",
  "turn_id": 0,
  "ts": 1234567890.0,
  "num_input_tokens": 512,
  "num_output_tokens": 128,
  "input_tokens": [1, 2, 3, ...],
  "output_tokens": [4, 5, 6, ...]
}
```

> [!WARNING]
> The traces are tokenized with `meta-llama/Llama-2-7b-hf` tokenizer, but we're serving Nemotron-H which uses a different tokenizer. For hit-rate comparison, we need to either:
> 1. **Re-tokenize** the traces with Nemotron-H's tokenizer (cleanest approach), or
> 2. **Use text-level replay** — decode traces back to text, then send as text prompts to SGLang (which re-tokenizes)
>
> Option 2 is simpler and more realistic but may lose some token-level alignment with simulation results. For initial experiments, Option 2 is sufficient.

### 4.2 Trace Replayer Script

Build a custom trace replayer that:
1. Reads Marconi JSONL traces
2. Decodes token IDs back to text using `meta-llama/Llama-2-7b-hf` tokenizer
3. Sends requests to SGLang via HTTP API
4. Respects inter-request timing from traces
5. Records per-request TTFT and cache hit metrics from SGLang

### 4.3 Baseline Experiments

```bash
# Run 1: No radix cache (vanilla inference)
python3 -m sglang.launch_server \
    --model nvidia/Nemotron-H-8B-Base-4096 \
    --tp 1 --disable-radix-cache

# Run 2: With radix cache (LRU eviction — this is "SGLang+" baseline)
python3 -m sglang.launch_server \
    --model nvidia/Nemotron-H-8B-Base-4096 \
    --tp 1

# Run 3: Quick bench_serving sanity check (generated workload)
python3 -m sglang.bench_serving \
    --backend sglang --base-url http://localhost:30000 \
    --dataset-name generated-shared-prefix \
    --num-prompts 100
```

### 4.4 Metrics to Collect

| Metric | Source | Notes |
|---|---|---|
| Token hit rate | SGLang metrics endpoint or logs | May need to enable verbose cache logging |
| TTFT (P50, P95, P99) | Trace replayer timestamps | `time_to_first_token = first_token_time - request_sent_time` |
| Input throughput (tok/s) | bench_serving output | Baseline comparison |
| Output throughput (tok/s) | bench_serving output | Baseline comparison |

---

## 5. Phase 2: Implement & Compare Marconi

**Goal**: Add FLOP-aware eviction to SGLang and compare against baselines.

### 5.1 Existing Work: SGLang PR #17898

There is a **WIP implementation** of Marconi in SGLang: [PR #17898](https://github.com/sgl-project/sglang/pull/17898) by [@qimcis](https://github.com/qimcis) (opened Jan 28, 2026).

#### What PR #17898 adds:

| Component | Files | Description |
|---|---|---|
| **Server args** | `server_args.py` | `--enable-marconi`, `--marconi-eff-weight`, `--marconi-bootstrap-window-size`, `--marconi-bootstrap-multiplier`, `--marconi-tuning-interval` |
| **Admission control** | `marconi_admission_cache.py` | Judicious admission (branch point + last token caching) |
| **FLOP-aware eviction** | Modified `mamba_radix_cache.py` | Utility scoring = recency + α·efficiency, candidate collection |
| **Config tuning** | `marconi_tuning_cache.py` | Online α tuning with process pool |
| **Utilities** | `marconi_utils.py` | FLOP/memory calculation functions (ported from paper's `utils.py`) |
| **Config** | `marconi_config.py` | Centralized Marconi configuration |
| **Request tracking** | `data.py` (Req class) | New fields: `kv_cache_protected_len`, `kv_cache_inserted_start/end`, `marconi_cache_len` |

#### Status and gaps:

- The PR is **WIP** (30 commits, "fix" commit messages suggest active iteration)
- No tests or benchmarks included yet
- Not merged; may have integration issues with latest SGLang main
- Based on SGLang's `MambaRadixCache` — correct target for our work

### 5.2 Strategy: Build on PR #17898

1. **Fork or cherry-pick** the PR's changes onto the SGLang version in our Docker image (v0.5.6.post2)
2. **Test** with Nemotron-H 8B and verify the server starts with `--enable-marconi`
3. **Validate** eviction behavior with simple workloads before full trace replay
4. **Benchmark** with Marconi traces and compare against Phase 1 baselines

### 5.3 Running with Marconi Enabled

```bash
# With Marconi eviction (via docker-compose)
MODEL_NAME=nvidia/Nemotron-H-8B-Base-4096 \
TENSOR_PARALLEL=1 \
SGLANG_EXTRA_ARGS="--enable-marconi --marconi-eff-weight 0.5" \
docker compose up sglang-server

# Or directly
python3 -m sglang.launch_server \
    --model nvidia/Nemotron-H-8B-Base-4096 \
    --tp 1 \
    --enable-marconi \
    --marconi-eff-weight 0.5
```

### 5.4 Expected Results Comparison

Based on the trace-driven simulation and the paper:

| Metric | vLLM+ → Marconi | SGLang+ → Marconi |
|---|---|---|
| Token hit rate | 4.5–34.4× improvement | 19–220% improvement |
| P95 TTFT | Up to 71.1% reduction | Up to 24.7% reduction |
| Best on | SWEBench (long, diverse sequences) | SWEBench |

> [!NOTE]
> Live inference results will likely show **smaller gains** than the simulation because:
> 1. We're using a different model (Nemotron-H 8B vs the paper's abstract 7B config)
> 2. The simulation assumes perfect compute-bound prefill (FLOP ≈ time), but real GPUs have memory bandwidth effects
> 3. Without judicious admission, the gap narrows
>
> The directional improvement (Marconi > LRU) should still hold clearly.

---

## 6. Key References

| Resource | Link |
|---|---|
| Marconi paper | [arXiv:2411.19379](https://arxiv.org/abs/2411.19379) |
| Marconi repo (simulation) | [ruipeterpan/marconi](https://github.com/ruipeterpan/marconi) |
| Marconi in SGLang (WIP PR) | [PR #17898](https://github.com/sgl-project/sglang/pull/17898) |
| SGLang MambaRadixCache | [PR #11214](https://github.com/sgl-project/sglang/pull/11214) |
| Nemotron-H 8B (HuggingFace) | [nvidia/Nemotron-H-8B-Base-4096](https://huggingface.co/nvidia/Nemotron-H-8B-Base-4096) |
| NVIDIA Mamba-2 Hybrid paper | [arXiv:2406.07887](https://arxiv.org/abs/2406.07887) |
| Original Mamba-2 Hybrid model | [nvidia/mamba2-hybrid-8b-3t-128k](https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-128k) |
| Artifact evaluation | [marconi/artifact_evaluation.md](../marconi/artifact_evaluation.md) |
| SGLang overview | [01-sglang-overview.md](01-sglang-overview.md) |
| Mamba implementation | [02-mamba-sglang-implementation.md](02-mamba-sglang-implementation.md) |
| Setup notebook | [setup-vm.ipynb](../setup-vm.ipynb) |
