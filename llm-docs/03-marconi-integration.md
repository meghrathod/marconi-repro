# Marconi Integration in SGLang: A Comprehensive Guide

This document explains the integration of the **Marconi FLOP-aware Cache Eviction Policy** into SGLang for Hybrid Large Language Models (LLMs), drawing from the official PR by the `abdelfattah-lab` and the Marconi research paper (arXiv:2411.19379). To fully grasp the integration, we first break down foundational concepts including Transformers, SSMs, and SGLang's memory management.

## 1. Foundation: Transformers and Attention
### What is Attention?
The Transformer architecture revolutionized sequence modeling by replacing recurrence with **Self-Attention**. Self-Attention allows a model to weigh the importance of all other tokens in a sequence when processing a specific token, giving it deep contextual understanding.

**Example:** Consider the sentence: *"The bank of the river was muddy, so the bank closed early."* 
- The first "bank" refers to a riverbank.
- The second "bank" refers to a financial institution. 
Self-Attention allows the model to look at surrounding words like "river" and "muddy" for the first "bank", and "closed early" for the second "bank", thus understanding the completely different contexts of the exact same word.

### QKV (Query, Key, Value)
At the heart of self-attention are three vectors generated for each token:
- **Query (Q):** What the current token is "looking for". (e.g., "bank" asking, "What context am I in?")
- **Key (K):** What the token "has" (its label or index). (e.g., "river" advertising, "I am a body of water.")
- **Value (V):** The actual content or meaning the token contributes. (e.g., the conceptual meaning of a river environment).

**How it works step-by-step:**
1. The Query for "bank" is compared (via dot product) against the Keys of all other words: "The", "river", "was", etc.
2. High similarity between "bank" (Q) and "river" (K) produces a high **attention score**.
3. These scores are converted to probabilities (using softmax).
4. The probabilities are multiplied by the Value (V) of each word. The final representation for "bank" becomes heavily weighted by the Value of "river", helping the model understand it means "riverbank".

**Crucially**, Transformers suffer from quadratic computational complexity ($O(N^2)$) relative to sequence length $N$ because every token must compute its QKV dot products against every other token in the sequence.

### What are Attention Heads and KV Heads?
In modern Transformers, a single token doesn't just do *one* QKV lookup. Instead, it splits its embedding into multiple parallel "heads." This allows the model to look at different parts of the sentence simultaneously (e.g., one head looks at grammar, one head looks at sentiment, one head tracks pronouns).

- **Attention Heads (Queries):** Numerically, a "Head" is simply a learned linear projection (a matrix multiplication) that maps a token's high-dimensional embedding into a smaller, specialized vector space (the `head_dim`). 
  - **Example:** If $h$ (hidden size) is 4096 and the model has 32 Attention Heads, each token is multiplied by 32 different, independent weight matrices. This produces 32 separate Query vectors, each of dimension 128 ($4096 / 32$). The model learns through training that one subspace might mathematically attend to nearest-neighbors, another to sentence subjects, etc. No human programs a "grammar head"; they are just 32 mathematically independent $h \times d$ projection matrices operating in parallel.
- **KV Heads (Keys & Values):** Similarly, these are the independent projection matrices that generate the Key and Value vectors for the vocabulary. The 32 Query vectors will compute dot products against the Key vectors to generate attention scores.

#### Evolution of Heads (MHA vs. MQA vs. GQA)
Historically, the number of KV Heads was the same as the number of Attention Heads. But as models grew, storing that massive KV cache became a memory bottleneck.
1. **Multi-Head Attention (MHA):** 32 Attention Heads $\rightarrow$ 32 KV Heads. (High quality, but huge memory).
2. **Multi-Query Attention (MQA):** 32 Attention Heads $\rightarrow$ 1 single shared KV Head. (Extremely memory efficient, but slightly lower quality).
3. **Grouped-Query Attention (GQA):** The modern compromise. 32 Attention Heads $\rightarrow$ 8 KV Heads. Every 4 Attention Queries share 1 KV Memory. (Great balance of memory efficiency and quality).

The Marconi FLOP equations explicitly use $k$ (KV heads) and $a$ (Attention heads) because models like Qwen3.5 heavily rely on GQA to save memory!

## 2. Beyond Transformers: State Space Models (SSMs) and Mamba
### What are SSMs?
State Space Models (SSMs) are mathematical models used to describe dynamic systems. In deep learning, they map an input sequence to a latent state representation, which then predicts the output. 

### Mamba and Continuous vs. Discrete States
**Mamba** is a specialized, selective SSM (S6) that makes the state transition parameters input-dependent. Unlike Transformers, SSMs process tokens with **linear complexity** ($O(N)$), summarizing the entire history into a fixed-size hidden state (similar to an RNN). This means they don't need to look back at every single previous token; they just update their current "state" with the new token.

**Example:** Imagine reading a massive book. 
- A **Transformer** has a perfect photographic memory but has to reread the entire book from page 1 every time it reads a new word to understand the context. This gets incredibly slow ($O(N^2)$).
- A **Mamba (SSM)** model acts like a reader taking brief, focused notes. When it reads a new chapter, it doesn't reread the whole book; it just looks at its notebook (the state) and updates it with the new plot points. This is much faster and scales linearly ($O(N)$).

### The Anatomy of the Mamba State
In Mamba, the hidden state isn't just a single vector, but a combination of two representations:
- **Convolutional State:** A short-term, local memory. E.g., If the sequence is "The quick brown fox", the convolutional state for "fox" might remember "quick brown" to understand immediate context. It uses a 1D convolution window.
- **Temporal State (SSM State):** The long-term memory that compresses the entire sequence history through the state space equations. E.g., The notebook keeping track of the overarching story plot from chapter 1 to chapter 50.

## 3. Hybrid LLMs: The Best of Both Worlds
So, why use both? Models like `Qwen3.5-27B` or `Jamba` are **Hybrid Architecture Models**. They interleave Attention layers with SSM layers to achieve a balance between the high recall/reasoning capabilities of Transformers and the massive context-window efficiency of SSMs.

### How Layer Interleaving Works
Imagine a 64-layer architecture block. Instead of doing 64 Attention layers (incredibly slow for 100k+ tokens), a hybrid model might follow a pattern like:
`[SSM, SSM, SSM, Attention, SSM, SSM, SSM, Attention, ...]`

- The **SSM layers** act as the highly efficient "readers" that compress long stretches of text into the hidden state in linear time.
- The **Attention layers** periodically "sync up", pulling out specific, high-fidelity facts from the sequence history that the SSMs might have over-compressed.

This means for a single token pass-through, the system must manage *both* the KV Cache needed by the scattered Attention layers and the Continuous Hidden State needed by the SSM layers.

## 4. SGLang Basics: RadixAttention and Memory Pools
SGLang is a high-performance serving framework for LLMs. Two of its primary concepts are RadixAttention and Memory Pools.

### RadixAttention
To avoid redundant computation during inference (e.g., when sharing a system prompt or multi-turn chat), SGLang caches the states of past tokens. **RadixAttention** stores these states in a **radix tree** (a compressed prefix tree). 
- If a new request shares a prefix (like the first 1,000 words of a system prompt) with an old request, SGLang traverses the Radix Tree, finds the node corresponding to those 1,000 words, and skips computing them entirely.

### Memory Pools in SGLang (Hybrid Setup)
Because hybrid models have two fundamentally different types of memory, SGLang must pre-allocate separate GPU memory blocks to prevent fragmentation and Out-Of-Memory (OOM) crashes:
1. **TokenToKVPool (KV Cache Pool):** Stores the explicit $K$ and $V$ tensors for the dense Attention layers. This grows proportionally with the sequence length.
2. **MambaPool (SSM Pool):** Stores the convolutional and temporal states for Mamba/linear-attention layers. Unlike KV Cache, the SSM state size is *fixed* regardless of sequence length—a 10-token prefix and a 10,000-token prefix take up the exact same amount of memory in the SSM State pool.

## 5. The Marconi Paper: FLOP-Aware Prefix Caching
Because **Hybrid LLMs** use both Attention layers and SSM layers, caching them is notoriously tricky. Traditional caching uses pure "Least Recently Used" (LRU) policies—if a sequence hasn't been accessed recently, evict it.

**The Problem:** SSM states are updated *in-place*. 
- If you evict the KV cache of a 10,000-token Transformer prompt, you just have to recompute the KV cache.
- If you evict the SSM state of a 10,000-token Hybrid prompt, you have to run 10,000 tokens through *every single SSM and Attention layer* from scratch to perfectly recreate that fixed-size state matrix. **Recomputing a long sequence's SSM state is astronomically expensive.**

### The Solution (Marconi): FLOP-Aware Eviction
Instead of just looking at recency (LRU), Marconi uses a **FLOP-aware eviction policy**. It calculates a **FLOP efficiency score ($\mathcal{E}$)**: the number of floating-point operations (FLOPs) you *save* by keeping a cache entry, divided by the memory bytes it physically occupies on the GPU.

### The Marconi FLOP Equations Explained
Let's go "wild" into the math. The goal is to calculate $\mathcal{E}(p) = \frac{\Delta \text{FLOPs}(p)}{\text{Memory}(p)}$ where $p$ is the prefix length.

#### 1. Calculating FLOP Savings ($\Delta \text{FLOPs}$)
When we hit a cached prefix of length $L_p$ for a total sequence of length $L$, we skip recomputing those $L_p$ tokens. But the savings differ by layer type:

**A. Attention Layer FLOP Savings:**
The computational cost of Attention is quadratic: it involves projecting the vectors (linear) and then computing the attention scores ($QA^T$, which is quadratic).
Total FLOPs for Attention on length $L$:
$$ \text{FLOPs}_{attn}(L) = 4 L h (h + k \cdot d) + 4 L^2 a \cdot d $$
*(Where $h$=hidden size, $k$=KV heads, $a$=attention heads, $d$=head dim)*

The *savings* from caching a prefix of length $L_p$ when processing a sequence of length $L$ is the cost of processing the full sequence minus the cost of processing just the *uncached* overlapping part ($L - L_p$):
$$ \Delta \text{FLOPs}_{attn} = \text{FLOPs}_{attn}(L) - \text{FLOPs}_{attn}(L - L_p) $$
Notice that because of the $L^2$ term, caching a 100-token prefix for a 10,000-token prompt saves *way more* FLOPs than caching a 100-token prefix for a 200-token prompt!

**B. SSM (Mamba) Layer FLOP Savings:**
SSMs process tokens linearly. The FLOPs are dominated by state transitions and projections:
$$ \text{FLOPs}_{ssm}(L_p) = 2 L_p v \cdot s + 8 L_p h \cdot v $$
*(Where $v$=intermediate size, $s$=state size)*. 
Because it's linear, the FLOPs saved are directly proportional to the prefix length $L_p$.

**C. Feed-Forward Network (MLP/MoE) FLOP Savings:**
Hybrid models often use Mixture of Experts (MoE) or dense MLPs. Like SSMs, these are linear with respect to sequence length:
$$ \text{FLOPs}_{mlp}(L_p) = 6 L_p h \cdot i $$
*(Where $i$=intermediate size)*.

**Total FLOPs Saved:**
We sum these up across all layers in the specific model's architecture (e.g., $N_{attn}$ attention layers, $N_{ssm}$ Mamba layers):
$$ \Delta \text{FLOPs}_{total} = \sum (\Delta \text{FLOPs}_{attn}) + \sum (\text{FLOPs}_{ssm}) + \sum (\text{FLOPs}_{mlp}) $$

#### 2. Calculating Memory Cost ($\text{Memory}(p)$)
The memory footprint of caching prefix $p$ includes both the KV cache and the SSM state.

- **KV Cache Size:** Grows linearly with prefix length $L_p$.
  $$ \text{Mem}_{KV} = 2 \times 2 \times L_p \times k \times d \times N_{attn} $$
  *(Multiplying by 2 for both K and V, by 2 for FP16 bytes)*.
- **SSM State Size:** A massive, *fixed* block of memory, regardless of $L_p$.
  $$ \text{Mem}_{SSM} = \text{constant state size depending on Mamba config} $$

**Total Memory:**
$$ \text{Memory}(p) = \text{Mem}_{KV}(L_p) + \text{Mem}_{SSM} $$

#### 3. The Final Efficiency Score
$$ \mathcal{E}(p) = \frac{\Delta \text{FLOPs}_{total}(p)}{\text{Memory}(p)} $$

**The profound realization of Marconi:** Because $\text{Mem}_{SSM}$ is a massive fixed constant, short prefixes have terrible efficiency (very little FLOP savings relative to the huge fixed memory cost of storing an SSM state). But as $L_p$ grows large, the $L^2$ FLOP savings from the Attention layers absolutely explodes, dwarfing the memory cost. Therefore, **Marconi ruthlessly evicts short prefixes and aggressively protects long prefixes.**

## 6. SGLang Integration: The PR and RFC_MARCONI
The `abdelfattah-lab` pull request implements Marconi's logic directly into SGLang. 

### Core Changes in the Code
1. **Marconi Efficiency Math (`marconi_utils.py`):**
   New mathematical utilities compute the FLOPs saved by full-attention layers, linear-attention (Mamba), and MoE (Mixture of Experts) layers. It divides this by the exact memory footprint of the SSM state + KV state to get the `flop_efficiency` score.

2. **Dual-State LRU and Tombstoning (`mamba_radix_cache.py` / `RFC_MARCONI.md`):**
   SGLang enforces an invariant: **KV cache cannot be evicted without evicting the SSM state, but the SSM state can be evicted independently**.
   Under the new policy, `evict_mamba_marconi` evaluates all unlocked eviction candidates using the equation:
   `Utility = (α * normalized_efficiency) + normalized_recency`
   If an internal node in the Radix Tree is chosen for eviction, its SSM state is freed (to save memory), but its KV cache is kept—a process called **Tombstoning**.

3. **Node Scoring (`TreeNode`):**
   Nodes in the Radix Tree are augmented with `num_cached_tokens` and `_flop_efficiency`. This efficiency is lazily invalidated upon cache access and recalculated when memory pressure triggers eviction.

By integrating these features via the `--eviction-policy marconi` argument, SGLang can serve hybrid models (like `Qwen3.5-27B`) with vastly improved Time-To-First-Token (TTFT) and cache hit rates compared to standard LRU.

---

## 7. Running the Implementation

### 7.1 Running Marconi Unit Tests
You can verify the mathematical correctness of the Marconi FLOP efficiency implementation by running the SGLang test suite. From the root of the project, using the `uv` environment with the `PYTHONPATH` correctly mapped to the `sglang` submodule:

```bash
# Run the 28 Marconi cache utility tests successfully
PYTHONPATH=$(pwd)/sglang/python uv run --project . python3 sglang/test/registered/radix_cache/test_marconi_utils.py
```

### 7.2 Launching the Server with Marconi Caching
To start the SGLang server and actually utilize the Marconi caching policy for benchmarking, you need to use the newly implemented arguments: `--eviction-policy marconi` and `--marconi-eff-weight`.

```bash
# Launch SGLang with a Hybrid Model (e.g. Qwen3.5-27B) using Marconi Eviction from source
PYTHONPATH=$(pwd)/sglang/python uv run --project . python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-27B \
    --radix-eviction-policy marconi \
    --marconi-eff-weight 0.7 \
    --port 30000
```
- **`--radix-eviction-policy marconi`**: Shifts the Radix Cache from pure LRU to the tombstoning FLOP-aware math discussed in Section 5.
- **`--marconi-eff-weight 0.7`**: Tunes the $\alpha$ parameter (how much priority is given to FLOP savings vs Recency). The paper recommends balancing this based on total KV cache size.

---

## 8. Adapting Nemotron Models to Marconi
While the current Marconi integration in SGLang was built targeting `Qwen3.5-27B` (and similar Qwen hybrid models), adapting it to work with **Nemotron** (e.g., Nemotron-H) is entirely feasible but requires bridging configuration mismatches. 

The Marconi mathematics relies on specific config attributes to calculate FLOP savings in `sglang/srt/mem_cache/marconi_utils.py`. The `NemotronHConfig` has the correct architectural components (Attention, Mamba, MLP, MoE), but names them differently or infers them through a pattern string.

### Required Adaptations
To adapt Marconi to work with Nemotron, the `compute_flops_saved` function in `marconi_utils.py` needs a translation layer (or `NemotronHConfig` needs aliases) for the following:

1. **Layer Types Resolution:**
   - **Marconi Expects:** `config.layers_block_type` (a list of strings like `"attention"`, `"linear_attention"`).
   - **Nemotron Reality:** Uses a `hybrid_override_pattern` string (e.g., `"M-M-M-M*-M..."`) where `M`=Mamba, `*`=Attention, `-`=MLP, `E`=MoE.

2. **Mamba (Linear Attention) Parameters:**
   - **Marconi Expects:** `config.linear_num_value_heads`, `config.linear_value_head_dim`, `config.linear_key_head_dim`.
   - **Nemotron Reality:** Uses `mamba_num_heads`, `mamba_head_dim`, and `ssm_state_size`.

3. **MoE Parameters (If Applicable):**
   - **Marconi Expects:** `config.moe_intermediate_size` and `config.shared_expert_intermediate_size`.
   - **Nemotron Reality:** Uses identical names, but `is_moe` check in Marconi checks for `num_experts > 1`, while Nemotron uses `n_routed_experts`.

**Implementation Path:**
To enable Nemotron support, modify `compute_flops_saved` to detect `model_type == "nemotron_h"` and dynamically map the Nemotron attributes into the expected variables, and parse the `hybrid_override_pattern` string into the sequential `layers_block_type` list. Once mapped, the exact same FLOP efficiency calculations and tombstoning logic will work perfectly with Nemotron.

