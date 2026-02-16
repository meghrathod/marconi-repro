# SGLang: LLM Serving from Basics to Advanced

This document provides a comprehensive overview of [SGLang](https://github.com/sgl-project/sglang), a high-performance serving framework for large language models and multimodal models. It covers fundamentals, prefix caching with RadixAttention, and advanced features including HiCache and hybrid model support.

---

## Table of Contents

1. [What is SGLang?](#1-what-is-sglang)
2. [Prefix Caching and RadixAttention](#2-prefix-caching-and-radixattention)
3. [Advanced Features: HiCache and Hybrid Models](#3-advanced-features-hicache-and-hybrid-models)
4. [Further Reading](#4-further-reading)

---

## 1. What is SGLang?

[SGLang](https://github.com/sgl-project/sglang) is a high-performance serving framework designed to deliver **low-latency** and **high-throughput** inference across setups ranging from a single GPU to large distributed clusters. Key characteristics include:

| Feature | Description |
|---------|-------------|
| **RadixAttention** | Core optimization for automatic [KV cache](#2-prefix-caching-and-radixattention) reuse across generation calls via a radix tree |
| **Zero-overhead CPU scheduler** | Efficient request scheduling without GPU stalls |
| **Prefill-decode disaggregation** | Separate prefill and decode stages for better resource utilization |
| **Speculative decoding** | Accelerate generation with draft models |
| **Continuous batching** | Dynamically batch requests for higher throughput |
| **Paged attention** | Memory-efficient KV cache management (inspired by [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)) |
| **Parallelism** | Tensor, pipeline, expert, and data parallelism |
| **Quantization** | FP4/FP8/INT4/AWQ/GPTQ support |

SGLang supports a wide range of models (Llama, Qwen, DeepSeek, GLM, Mistral, etc.), [OpenAI-compatible APIs](https://docs.sglang.io/basic_usage/openai_api.html), and runs on [NVIDIA GPUs](https://docs.sglang.io/get_started/install.html), [AMD GPUs](https://docs.sglang.io/platforms/amd_gpu.html), and [other platforms](https://docs.sglang.io/).

**Installation**: See the [official install guide](https://docs.sglang.io/get_started/install.html) for pip, Docker, Kubernetes, and cloud deployment options.

---

## 2. Prefix Caching and RadixAttention

### 2.1 The Problem

When multiple prompts share the same prefix—such as:
- **System prompts** in chatbots ([ChatGPT](https://openai.com/index/chatgpt/), [Claude](https://www.anthropic.com/news/claude-3-family))
- **Few-shot examples** in in-context learning ([Brown et al., 2020](https://arxiv.org/abs/2005.14165))
- **Conversation history** in multi-turn chat
- **Self-consistency** reasoning with multiple samples ([Wang et al., 2022](https://arxiv.org/abs/2203.11171))

—recomputing identical KV cache for each request wastes **memory** and **compute**. The prefill phase (converting input tokens to KV cache) becomes a bottleneck, especially for long contexts.

### 2.2 RadixAttention: The Solution

[RadixAttention](https://docs.sglang.io/) is SGLang's core optimization. Instead of discarding KV cache after each request, it retains it in a **radix tree** data structure:

- **Radix tree**: A space-efficient prefix tree where edges can be labeled with sequences of tokens of varying lengths
- **Path root-to-leaf** = prefix of a request
- **Shared prefixes** reuse the same nodes, avoiding redundant storage and computation
- **Automatic handling** of diverse reuse patterns: few-shot learning, self-consistency, multi-turn chat, tree-of-thought search

For a detailed comparison of SGLang's token-level radix tree vs vLLM's block-level hashing, see [Prefix Caching: SGLang vs vLLM](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1).

### 2.3 Eviction Policy

For standard Transformer models, SGLang uses **LRU (Least Recently Used)** eviction combined with cache-aware scheduling to maximize cache hit rates. When the cache is full, the least recently accessed entries are evicted first.

**Note**: For hybrid models (Mamba + Transformer), eviction is more complex—see [Mamba Implementation in SGLang](02-mamba-sglang-implementation.md) and [Marconi Integration](03-marconi-integration.md).

---

## 3. Advanced Features: HiCache and Hybrid Models

### 3.1 HiCache: Hierarchical KV Caching

[HiCache](https://docs.sglang.io/advanced_features/hicache_design.html) extends RadixAttention with a three-tier cache hierarchy inspired by CPU cache design:

| Tier | Storage | Role |
|------|----------|------|
| **L1** | GPU memory | Fastest access, hottest data |
| **L2** | Host (CPU) memory | Larger capacity, moderate latency |
| **L3** | Distributed storage (Mooncake, 3FS, NIXL, etc.) | Largest capacity, shared across instances |

This significantly expands KV cache capacity while maintaining performance for multi-turn QA and long-context workloads. See the [HiCache design doc](https://docs.sglang.io/advanced_features/hicache_design.html) for prefetch strategies, write-back policies, and L3 backend integration.

### 3.2 Hybrid Models: Mamba + Transformer

**Hybrid models** (e.g., [Jamba](https://arxiv.org/abs/2403.19887), [Mamba](https://arxiv.org/abs/2312.00752), [Qwen3-Next](https://huggingface.co/Qwen)) mix two layer types:

| Layer Type | Architecture | Complexity | State |
|------------|---------------|-------------|-------|
| **Attention** | Transformer | $O(L^2)$ | KV cache grows with sequence length |
| **SSM (Mamba)** | Recurrent / State Space | $O(L)$ | Fixed-size recurrent state, in-place updates |

**Why hybrid?** SSM layers offer linear complexity and constant memory per layer, while Attention layers provide superior recall and in-context learning. Hybrid models balance efficiency and capability.

### 3.3 Hybrid-Specific Challenge: Prefix Caching

A critical difference affects prefix caching:

- **Attention KV cache**: Can be **sliced**—if you have KVs for tokens 1..100, you can reuse KVs for tokens 1..80 by truncation
- **SSM state**: Updated **in-place**—the state at token 100 cannot be "rolled back" to represent token 80

**Implication**: SSM states require **exact-match** cache hits. You cannot reuse a cached state for a shorter prefix. This complicates:
1. **Admission**: Which SSM states to cache (fine-grained checkpointing creates many sparsely-hit entries)
2. **Eviction**: How to value SSM vs KV entries (different memory/compute tradeoffs)

The [Marconi paper](https://arxiv.org/abs/2411.19379) addresses these challenges with judicious admission and FLOP-aware eviction. See [Mamba Implementation in SGLang](02-mamba-sglang-implementation.md) for how SGLang's PR #11214 implements radix cache for hybrid models, and [Marconi Integration](03-marconi-integration.md) for integrating Marconi's strategy.

---

## 4. Further Reading

| Topic | Link |
|-------|------|
| SGLang documentation | [docs.sglang.io](https://docs.sglang.io/) |
| SGLang GitHub | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| HiCache design | [docs.sglang.io/advanced_features/hicache_design.html](https://docs.sglang.io/advanced_features/hicache_design.html) |
| vLLM PagedAttention | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| Mamba paper | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| Marconi paper | [arXiv:2411.19379](https://arxiv.org/abs/2411.19379) |
| Next: Mamba in SGLang | [02-mamba-sglang-implementation.md](02-mamba-sglang-implementation.md) |
| Next: Marconi integration | [03-marconi-integration.md](03-marconi-integration.md) |
