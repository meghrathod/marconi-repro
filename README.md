# Marconi: KV Cache Management for Hybrid LLMs

This repository aims to replicate [Marconi's theoretical results](https://arxiv.org/abs/2411.19379) (MLSys 2025) in a **live inference system** by integrating Marconi's FLOP-aware eviction into [SGLang](https://github.com/sgl-project/sglang)'s Mamba radix cache.

---

## Overview

Marconi is a prefix caching system for **Hybrid LLMs** (Mamba + Transformer, e.g., Jamba). It introduces:

- **Judicious admission**: Cache SSM states only at branch points and last decoded token
- **FLOP-aware eviction**: Score = recency + α × flop_efficiency; evict lowest score

The [marconi/](marconi/) submodule contains the **trace-driven simulation** (no GPU). Live inference integration targets [SGLang PR #11214](https://github.com/sgl-project/sglang/pull/11214) (Mamba radix cache).

---

## Documentation

| Doc | Description |
|-----|-------------|
| [llm-docs/03-marconi-integration.md](llm-docs/03-marconi-integration.md) | **Full guide**: Marconi strategy, integration plan, test plan, replication, quick start |
| [llm-docs/01-sglang-overview.md](llm-docs/01-sglang-overview.md) | SGLang basics and advanced (RadixAttention, HiCache, hybrid models) |
| [llm-docs/02-mamba-sglang-implementation.md](llm-docs/02-mamba-sglang-implementation.md) | Mamba radix cache implementation in SGLang |
| [llm-docs/README.md](llm-docs/README.md) | Index of all docs |

---

## Quick Start

### Run Marconi simulation (no GPU)

```bash
conda env create -f marconi/environment.yml
conda activate marconi
cd marconi && bash run_all_experiments.sh
```

See [marconi/artifact_evaluation.md](marconi/artifact_evaluation.md) for trace download and full instructions.

### Run SGLang with hybrid model (GPU)

```bash
pip install sglang
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

---

## Tests

Run Marconi eviction tests (requires `conda activate marconi`):
```bash
pytest tests/test_marconi_eviction.py -v
```

---

## Links

- [Marconi paper](https://arxiv.org/abs/2411.19379)
- [Marconi repo](https://github.com/ruipeterpan/marconi)
- [SGLang](https://github.com/sgl-project/sglang)
- [SGLang PR #11214 (Mamba radix cache)](https://github.com/sgl-project/sglang/pull/11214)
