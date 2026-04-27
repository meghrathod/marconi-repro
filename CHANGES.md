# Changes to Marconi Submodule vs Original Authors

All changes are relative to the upstream `init upload for ae` commit (`6889729`).

---

## Functional Changes

Changes that affect simulation behavior, correctness, or what data the plots use.

### Bug Fixes in `radix_cache_hybrid.py`

- **MLP FLOPs savings typo** (`line 526`): `flops_savings_mlp` was calculated using `get_attn_flops(seqlen_parent)` instead of `get_mlp_flops(seqlen_parent)`. This caused MLP eviction scores to be computed with the wrong function, corrupting the efficiency score used to rank eviction candidates when `eff_weight > 0`.

- **Wrong node variable in bytes_evicted** (`line 555`): `get_kvs_size(len(node.value))` referenced the loop variable `node` instead of `node_to_evict`. This caused bytes-evicted accounting to use whichever node happened to be bound last, producing incorrect cache capacity tracking.

### Plotting: Replace Hardcoded Data with Real Experiment Results

- **`microbenchmark_arrivalrate.py` (Figs 13a/13b)**: Completely rewritten. Original used a hardcoded dict of hit-rate values with no documented source (similar to Palmer Penguins placeholder data). Now parses `logs/swebench.txt` at 5 GB cache size to extract V1/V2 hit rates across all `sps` and `art` combinations.

- **`microbenchmark_contention.py` (Fig 11)**: Completely rewritten. Original used a hardcoded dict with cache sizes `(60, 80, 100, 120, 140)` GB and undocumented values. Now reads directly from result pickle files across all available cache sizes `[1, 5, 10, 20, 40, 60, 80, 100]` GB for the `sps=5, art=10` config, using `v1_max_hit_rate` / `v2_max_hit_rate` keys.

- **`fine_grained_analysis.py` (Fig 10)**: Config changed from `(swebench, 80 GB, sps=10, art=7.5)` to `(swebench, 5 GB, sps=0.25, art=10)`. The original config produced a flat V1=V2 result because the bootstrap window never fired at 80 GB (first eviction at request ~343, bootstrap window = 1715 > trace length 748). The new config evicts at request ~61, bootstrap fires at 305, producing meaningful V2 > V1 differentiation.

### Plotting: Crash Fix in `fine_grained_analysis.py`

- **Empty bin crash**: `statistics.mean([...])` over an empty list raises `StatisticsError`. Added `if bin_data else 0` guard for the binned hit-rate-diff bar chart.

### Plotting: Visual/Scale Fix in `sglang_comparison.py` (Fig 8)

- **x-axis clipped to ±20%**: Original had no `set_xlim`, making the plot dominated by outliers. Now clipped to `[-20, 20]` with explicit ticks to highlight the main improvement region.
- **Curly quote syntax error**: Unicode `'`/`'` (U+2018/U+2019) in string literals caused `SyntaxError: invalid character`. Replaced with ASCII straight quotes.
- **Removed `set_label_coords`**: Prevented xlabel from rendering correctly on the horizontal boxplot.

### Figure Output Naming

- All plotting scripts updated to save with `fig7_`, `fig8_`, `fig10_`, `fig11_`, `fig12a_`, `fig12b_`, `fig13a_`, `fig13b_` prefixes, matching paper figure numbers.

### Trace Generation (`utils/generate_trace.py`)

- **Tokenizer customization**: Added `--tokenizer` CLI argument (default: `meta-llama/Llama-2-7b-hf`). Allows generating traces for other models without code changes.
- **Max token length**: Reduced from 32768 → 8192 to avoid memory-intensive sequences that cause OOM in trace generation.
- **SWEBench dataset source**: Switched to a working HuggingFace dataset after the original source became unavailable.
- **Dataset loading**: Optimized to avoid loading the full dataset into memory before filtering.
- **`if __name__ == "__main__":` guard**: Moved trace generation code into a guard block so the file is safely importable.

---

## Cosmetic Changes

Changes that affect style, paths, or readability but not behavior.

### All Plotting Scripts: Dynamic Path Resolution

Every plotting script under `plotting/` had hardcoded relative paths like `../figures/eval/` and `../logs/`. All replaced with `os.path.dirname(os.path.abspath(__file__))` based resolution so scripts run correctly from any working directory:

- `cache_usage_breakdown.py`
- `context_window_over_time.py`
- `fine_grained_analysis.py`
- `microbenchmark_arrivalrate.py`
- `microbenchmark_contention.py`
- `microbenchmark_dstate.py`
- `microbenchmark_layer_composition.py`
- `sglang_comparison.py`
- `state_size_comparison.py`
- `token_hit_rate.py`
- `ttft.py`

### `ttft.py`: Hardcoded Log Filename List Removed

Original had a commented-out list of old internal log filenames (e.g., `1029_lmsys_initw0.0_wind=1000.txt`). Replaced with a loop over `["lmsys", "sharegpt", "swebench"]` mapping to the standard log paths.

### `run_all_experiments.sh`: Datasets Line Uncommented

The `datasets=("sharegpt" "lmsys" "swebench")` line was active; a commented-out partial version was cleaned up. (Minor — both ran all three datasets.)
