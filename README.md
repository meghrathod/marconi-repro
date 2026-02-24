# Marconi: KV Cache Management for Hybrid LLMs

This repository aims to replicate [Marconi's theoretical results](https://arxiv.org/abs/2411.19379) (MLSys 2025) in a **live inference system** by integrating Marconi's FLOP-aware eviction into [SGLang](https://github.com/sgl-project/sglang)'s Mamba radix cache.

---

## Overview

Marconi is a prefix caching system for **Hybrid LLMs** (Mamba + Transformer, e.g., Nemotron-H). It introduces:

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

### 1. Marconi trace-driven simulation (CPU only, no GPU needed)

```bash
conda env create -f marconi/environment.yml
conda activate marconi
cd marconi && bash run_all_experiments.sh
```

See [marconi/artifact_evaluation.md](marconi/artifact_evaluation.md) for trace download and full instructions.

### 2. SGLang inference server (bare-metal GPU node)

Install SGLang with FlashInfer support (CUDA 12.4 + PyTorch 2.5):

```bash
pip install "sglang==0.5.6.post2" \
    --find-links "https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python"
pip install -r requirements.txt
```

Start the server (we use [nvidia/Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K) — a Mamba hybrid model):

```bash
export HF_TOKEN=<your_token>

python3 -m sglang.launch_server \
    --model nvidia/Nemotron-H-8B-Base-8K \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 4
```

Verify it's running:

```bash
curl http://localhost:30000/get_model_info
```

> **Tip:** Run the server in a `tmux` session so you can detach and run benchmarks in a second terminal.

### 3. Run the trace replayer

```bash
# In a second terminal (or tmux pane)
python3 src/trace_replayer.py --url http://localhost:30000
```

---

## Tests

```bash
# Marconi eviction tests (conda env, no GPU)
pytest tests/test_marconi_eviction.py -v

# Trace replayer tests (SGLang server must be running)
pytest tests/test_trace_replayer.py -v
```

---

## Links

- [Marconi paper](https://arxiv.org/abs/2411.19379)
- [Marconi repo](https://github.com/ruipeterpan/marconi)
- [SGLang](https://github.com/sgl-project/sglang)
- [SGLang PR #11214 (Mamba radix cache)](https://github.com/sgl-project/sglang/pull/11214)
- [Nemotron-H-8B-Base-8K on HuggingFace](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K)
