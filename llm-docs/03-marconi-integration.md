# Marconi Integration: Strategy, Plan, and Replication

This document combines the **Marconi strategy** (admission + eviction), **integration plan** for SGLang, **test plan**, and **replication strategy** for bringing [Marconi](https://arxiv.org/abs/2411.19379) from trace-driven simulation to live inference.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Marconi Strategy](#2-marconi-strategy)
3. [Integration Plan](#3-integration-plan)
4. [Test Plan](#4-test-plan)
5. [Replication Strategy](#5-replication-strategy)
6. [Quick Start](#6-quick-start)

---

## 1. Project Overview

### 1.1 Goal

Replicate [Marconi's theoretical results](https://arxiv.org/abs/2411.19379) in a **live inference system** by integrating Marconi's FLOP-aware eviction (and optionally judicious admission) into [SGLang](https://github.com/sgl-project/sglang)'s [Mamba radix cache](02-mamba-sglang-implementation.md) (PR #11214).

### 1.2 Repository Structure

| Path | Description |
|------|-------------|
| [marconi/](marconi/) | Submodule: trace-driven simulation (no GPU) from [ruipeterpan/marconi](https://github.com/ruipeterpan/marconi) |
| [llm-docs/](llm-docs/) | Documentation (this folder) |
| [setup-vm.ipynb](setup-vm.ipynb) | Chameleon GPU instance provisioning |
| [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) | Container setup |

### 1.3 Marconi vs Live Inference

| Aspect | Marconi (simulation) | SGLang PR #11214 (live) |
|--------|----------------------|--------------------------|
| **Scope** | Trace-driven, CPU-only | GPU inference |
| **Admission** | Judicious (branch points + last token) | All tokens |
| **Eviction** | FLOP-aware (α tuning) | LRU only |
| **Data** | `HybridStates` (abstract) | KV indices + Mamba indices |

---

## 2. Marconi Strategy

### 2.1 Theoretical Motivation

**Problem**: In hybrid models, SSM states are fixed-size and updated in-place. Fine-grained checkpointing (e.g., every 32 tokens) creates many **sparsely-hit** entries that thrash the cache. See [SGLang overview §3.3](01-sglang-overview.md#33-hybrid-specific-challenge-prefix-caching).

**Marconi's insight**: Not all tokens are equal. Evict based on **FLOP efficiency** (compute saved per byte) and **recency**, not just recency.

### 2.2 Judicious Admission

Marconi caches SSM states only when reuse likelihood is high:

1. **Purely-input prefixes** (system prompts, few-shot): Use **speculative insertion** before prefetch—if inserting the input creates a new branch, checkpoint the prefix's SSM state.
2. **Input-and-output prefixes** (conversation history): Cache only the **last decoded token**'s SSM state (conversations typically resume from there).

Implementation: [marconi/radix_cache_hybrid.py](marconi/radix_cache_hybrid.py) (insert logic, `state_at_branchoff`).

### 2.3 FLOP-Aware Eviction (evict_v2)

**Candidates**: Nodes with ≤1 child (leaf + single-child intermediate).

**Utility score**:
$$S(n) = \text{recency}(n) + \alpha \cdot \text{flop\_efficiency}(n)$$

- **recency**: Normalized `1 / (current_ts - last_access_ts)` (higher = more recent).
- **flop_efficiency**: Normalized `total_flops_saved / total_memory`.
- **α (eff_weight)**: Balances recency vs efficiency. α=0 → LRU; α=1 → efficiency-only.

**Evict** the node with the **lowest** score.

Implementation: [marconi/radix_cache_hybrid.py](marconi/radix_cache_hybrid.py) `evict_v2` (lines ~503–561).

### 2.4 FLOP Formulas

From [marconi/utils.py](marconi/utils.py):

| Layer | Formula | Reference |
|-------|---------|-----------|
| **Attention** | `8*L*D² + 4*L²*D` | [Chinchilla](https://arxiv.org/abs/2203.15556) |
| **MLP** | `16*L*D²` | Chinchilla |
| **Mamba1** | `12*L*D² + 16*L*D*N + 10*L*D` | [Mamba](https://github.com/state-spaces/mamba) |

Where: L = sequence length, D = model dim, N = SSM state dim.

**Savings per node** (relative to parent):
- `flops_savings_mamba = num_ssm_layers * get_mamba1_flops(seqlen_child, d, n)`
- `flops_savings_attn = num_attn_layers * (get_attn_flops(seqlen_total, d) - get_attn_flops(seqlen_parent, d))`
- `flops_savings_mlp = num_mlp_layers * (get_mlp_flops(seqlen_total, d) - get_mlp_flops(seqlen_parent, d))`

**Total memory** per node: `num_ssm_layers * get_mamba_state_size(d, n) + num_attn_layers * get_kvs_size(seqlen_total, d)`.

### 2.5 Online α Tuning (ConfigTuner)

Marconi tunes α retrospectively:

1. **Bootstrap**: After first eviction, set `bootstrap_window_size = bootstrap_multiplier * num_reqs_before_eviction`.
2. **Replay**: When window is full, replay past requests with α ∈ {0, 0.1, …, 2.0} in parallel.
3. **Select**: Pick α that maximizes FLOPs saved (or token hit rate).

Implementation: [marconi/config_tuner.py](marconi/config_tuner.py).

---

## 3. Integration Plan

### 3.1 Add Marconi Eviction to MambaRadixCache

**Target**: SGLang [PR #11214](https://github.com/sgl-project/sglang/pull/11214) branch (`support_mamba_radix_cache`).

**Steps**:

1. **Server args**: Add `--marconi-eff-weight` (α), `--marconi-evict-policy` (1=LRU, 2=Marconi).
2. **Model config**: Pass `num_ssm_layers`, `num_attn_layers`, `num_mlp_layers`, `d`, `n` from model config to `MambaRadixCache`.
3. **FLOP helpers**: Port [marconi/utils.py](marconi/utils.py) `get_attn_flops`, `get_mamba1_flops`, `get_mlp_flops`, `get_kvs_size`, `get_mamba_state_size` (or equivalent using SGLang's model config).
4. **evict_mamba / evict**: Replace LRU selection with Marconi scoring:
   - Collect candidates (leaf + single-child nodes).
   - Compute FLOP efficiency and recency; normalize; compute utility = recency + α * efficiency.
   - Evict node with min utility.
5. **Optional**: Implement judicious admission (speculative insertion, branch-point checkpointing)—larger change.

### 3.2 File Mapping

| Marconi (simulation) | SGLang (live) |
|----------------------|---------------|
| [radix_cache_hybrid.py](marconi/radix_cache_hybrid.py) `evict_v2` | `mamba_radix_cache.py` `evict_mamba`, `evict` |
| [utils.py](marconi/utils.py) FLOP/memory | New `marconi_utils.py` or inline in cache |
| [config_tuner.py](marconi/config_tuner.py) | Optional: async tuning thread |

### 3.3 Challenges

- **Admission**: Judicious admission requires speculative insertion before prefill; SGLang's flow may need scheduler changes.
- **Timestamp**: Use logical timestamps or `time.monotonic()`; ensure consistency across TP ranks.
- **Lock refs**: Marconi's simulation does not have lock refs; SGLang's `full_lock_ref`/`mamba_lock_ref` must be respected when collecting eviction candidates.

---

## 4. Test Plan

### 4.1 Unit Test (Marconi Strategy)

**Location**: [tests/test_marconi_eviction.py](../tests/test_marconi_eviction.py) in this repo.

**Run** (requires marconi conda env):
```bash
conda activate marconi
pytest tests/test_marconi_eviction.py -v
```

**Cases**:

1. **match_prefix + cow_mamba**: Insert sequences with shared prefixes; verify `match_prefix` returns correct prefix length and Mamba copy-on-write.
2. **Eviction order (LRU vs Marconi)**: With fixed capacity, insert sequences that create contention. Assert Marconi evicts different nodes than LRU when α > 0.
3. **FLOP efficiency**: Insert one long sequence and several short ones; under eviction, verify Marconi prefers keeping the long sequence (higher FLOP efficiency).

### 4.2 Benchmark Test (bench_serving)

**Dataset**: `generated-shared-prefix` (shared system prompts across groups).

**Commands**:
```bash
# Baseline: no radix cache
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4 --disable-radix-cache

# With radix cache (LRU)
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4

# With Marconi (when implemented)
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4 --marconi-evict-policy 2 --marconi-eff-weight 0.5
```

**Metrics**: Token hit rate, TTFT (P50, P95, P99), input/output throughput.

### 4.3 Trace Replay (Bridge to Paper)

**Goal**: Replay [Marconi traces](https://github.com/ruipeterpan/marconi) (LMSys, ShareGPT, SWEBench) against live SGLang.

**Steps**:

1. **Trace format**: Marconi traces are JSONL with `input_tokens`, `output_tokens`, `session_id`. Build an adapter to SGLang's bench format or a custom client.
2. **Replay**: Send requests in trace order with appropriate inter-request delays.
3. **Compare**: Token hit rate, TTFT vs [Marconi simulation results](marconi/artifact_evaluation.md).

---

## 5. Replication Strategy

### 5.1 Paper Results ([arXiv:2411.19379](https://arxiv.org/abs/2411.19379))

- **Token hit rate**: 4.5–34.4× over vLLM+ (LMSys, ShareGPT, SWEBench).
- **P95 TTFT**: Up to 71.1% (617 ms) reduction vs vanilla; up to 36.1–71.1% vs vLLM+.
- **FLOP-aware vs LRU**: 19–220% token hit rate improvement over SGLang+ (LRU).

### 5.2 Replication Phases

| Phase | Action |
|-------|--------|
| **1. Environment** | GPU instance ([setup-vm.ipynb](setup-vm.ipynb)), SGLang with PR #11214 |
| **2. Model** | Hybrid GDN model (e.g., Qwen3-Next-80B-A3B, Jamba-1.5-Mini) |
| **3. Traces** | Download [Marconi traces](https://github.com/ruipeterpan/marconi#datasets) or generate via [marconi/utils/generate_trace.py](marconi/utils/generate_trace.py) |
| **4. Baseline** | Run with `--disable-radix-cache` and with radix cache (LRU) |
| **5. Marconi** | Implement FLOP-aware eviction; run same traces |
| **6. Compare** | Token hit rate, TTFT percentiles, FLOP saved |

### 5.3 Expected Gaps

- **Admission**: Without judicious admission, gains may be smaller than paper.
- **Model config**: Paper uses 7B Hybrid (4 attn, 24 SSM, 28 MLP, d=4096, n=128); SGLang PR tested with Qwen3-Next-80B—configs differ; adjust FLOP formulas accordingly.

---

## 6. Quick Start

### 6.1 Run Marconi Simulation (No GPU)

```bash
# Setup
conda env create -f marconi/environment.yml
conda activate marconi

# Download traces (see marconi/artifact_evaluation.md)
cd marconi && gdown --fuzzy 'https://drive.google.com/file/d/1D8f68sBWJHyCfJZdEYCBK2M0iHmSDE6M/view?usp=sharing'
tar -xzvf traces.tar.gz && mkdir -p logs results figures/eval

# Run experiments
bash run_all_experiments.sh
```

See [marconi/artifact_evaluation.md](marconi/artifact_evaluation.md) for full instructions.

### 6.2 Run SGLang with Hybrid Model (GPU)

```bash
# Install SGLang (see https://docs.sglang.io/get_started/install.html)
pip install sglang

# With PR #11214 (checkout support_mamba_radix_cache branch)
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

### 6.3 Key Links

| Resource | Link |
|----------|------|
| Marconi paper | [arXiv:2411.19379](https://arxiv.org/abs/2411.19379) |
| Marconi repo | [ruipeterpan/marconi](https://github.com/ruipeterpan/marconi) |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| PR #11214 | [Mamba radix cache](https://github.com/sgl-project/sglang/pull/11214) |
| SGLang overview | [01-sglang-overview.md](01-sglang-overview.md) |
| Mamba implementation | [02-mamba-sglang-implementation.md](02-mamba-sglang-implementation.md) |
