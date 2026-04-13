#!/usr/bin/env bash
# test_capacity_hypothesis.sh
#
# HYPOTHESIS: Constraining cache to tight capacity makes Marconi beat LRU.
# At default A100 capacity the cache is ~130x too large → LRU wins.
#
# Usage:
#   bash scripts/test_capacity_hypothesis.sh [--mem-fraction 0.22] [--full-capacity]
set -e

MODEL="nvidia/Nemotron-H-8B-Base-8K"
PORT=30000
SERVER_URL="http://127.0.0.1:${PORT}"
UV_RUN=(uv run --project .)
TRACE_DIR="traces"
TRACE_NAMES="sharegpt_sps=1.0_nums=10.jsonl"

# Parse args
MEM_FRACTION="0.22"
LABEL_SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mem-fraction)   MEM_FRACTION="$2"; shift 2;;
        --mem-fraction=*) MEM_FRACTION="${1#*=}"; shift;;
        --full-capacity)  MEM_FRACTION="0.9"; LABEL_SUFFIX="-full"; shift;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

OUTPUT_DIR="results/capacity-test${LABEL_SUFFIX}"
LOG_DIR="logs/capacity-test${LABEL_SUFFIX}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================================"
echo "  HYPOTHESIS: tight cache → Marconi beats LRU"
echo "  mem-fraction-static : ${MEM_FRACTION}"
echo "  Trace               : ${TRACE_NAMES}"
echo "  Output              : ${OUTPUT_DIR}/"
echo "========================================================"

# ── copy from run_live_experiments.sh ────────────────────────────────────────

free_port_and_sglang() {
    local p=$1
    echo "Releasing port ${p}..."
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -n tcp "$p" 2>/dev/null || true
    fi
    pkill -9 -f "python3 -m sglang.launch_server.*--model-path ${MODEL}.*--port ${p}" 2>/dev/null || true
    pkill -9 -f "python.*sglang.launch_server.*--port ${p}" 2>/dev/null || true
    sleep 2
    local n=0
    while nc -z localhost "$p" 2>/dev/null; do
        echo "  port ${p} still busy; retry ($((++n))/15)..."
        fuser -k -n tcp "$p" 2>/dev/null || true
        sleep 2
        [[ $n -ge 15 ]] && { echo "ERROR: port stuck"; exit 1; }
    done
    echo "  port ${p} free."
}

start_server() {
    local config=$1 extra_args=$2
    free_port_and_sglang "${PORT}"
    echo "Starting server: ${config}"
    setsid env PYTHONPATH="$(pwd)/sglang/python" "${UV_RUN[@]}" python3 -m sglang.launch_server \
        --model-path "${MODEL}" --port "${PORT}" \
        --disable-custom-all-reduce --enable-metrics --enable-cache-report \
        --mem-fraction-static "${MEM_FRACTION}" \
        ${extra_args} > "${LOG_DIR}/server_${config}.log" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"
    echo "Waiting for port ${PORT}..."
    local n=0
    while ! nc -z localhost "${PORT}"; do
        sleep 2; n=$((n+1))
        if [[ $n -gt 120 ]]; then
            echo "ERROR: timeout (see ${LOG_DIR}/server_${config}.log)"
            kill -9 -- -"${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
    done
    sleep 5
    echo "Server ready (${config})."
}

stop_server() {
    local config="${1:-unknown}"
    echo "Stopping server (${config})..."
    kill -9 -- -"${SERVER_PID}" 2>/dev/null || kill -9 "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    free_port_and_sglang "${PORT}"
}

run_replayer_for_config() {
    local config=$1
    "${UV_RUN[@]}" python src/trace_replayer.py \
        --trace-dir "${TRACE_DIR}" \
        --trace-names "${TRACE_NAMES}" \
        --output-dir "${OUTPUT_DIR}/${config}" \
        --server-url "${SERVER_URL}" \
        --model "${MODEL}" \
        > "${LOG_DIR}/replayer_${config}.log" 2>&1
}

run_experiment() {
    local config=$1 extra_args=$2
    echo "==========================================================="
    echo "Experiment: ${config}"
    echo "==========================================================="
    start_server "${config}" "${extra_args}"
    echo "Replaying trace..."
    run_replayer_for_config "${config}"
    stop_server "${config}"
}

# ── Run LRU then Marconi ──────────────────────────────────────────────────────

run_experiment "lru"     "--radix-eviction-policy lru"
run_experiment "marconi" "--radix-eviction-policy marconi --marconi-eff-weight 0.7"

# ── Summarise results ─────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  RESULTS  (mem-fraction-static=${MEM_FRACTION})"
echo "========================================================"

export OUTPUT_DIR MEM_FRACTION
python3 - <<'PYEOF'
import json, glob, os

base = os.environ["OUTPUT_DIR"]
mem = os.environ["MEM_FRACTION"]

results = {}
for policy in ["lru", "marconi"]:
    files = glob.glob(f"{base}/{policy}/*.jsonl")
    if not files:
        print(f"  {policy.upper()}: no output file found")
        continue
    reqs = [json.loads(l) for f in files for l in open(f)]
    total_prompt  = sum(r.get("prompt_tokens", 0) for r in reqs)
    total_cached  = sum(r.get("cached_tokens", 0) for r in reqs)
    errors        = sum(1 for r in reqs if r.get("error"))
    hit_pct       = total_cached / total_prompt * 100 if total_prompt > 0 else 0
    ttfts         = [r["ttft_ms"] for r in reqs if r.get("ttft_ms", 0) > 0]
    avg_ttft      = sum(ttfts) / len(ttfts) if ttfts else 0
    results[policy] = hit_pct

    # per-turn breakdown
    by_turn = {}
    for r in reqs:
        t = r.get("turn_id", 0)
        by_turn.setdefault(t, {"prompt": 0, "cached": 0, "n": 0})
        by_turn[t]["prompt"]  += r.get("prompt_tokens", 0)
        by_turn[t]["cached"]  += r.get("cached_tokens", 0)
        by_turn[t]["n"]       += 1

    print(f"\n  {policy.upper()}: token_hit={hit_pct:.1f}%  avg_ttft={avg_ttft:.0f}ms  errors={errors}  n={len(reqs)}")
    print(f"  {'Turn':>5}  {'Hit%':>6}  {'Cached':>8}  {'Prompt':>8}  {'N':>4}")
    for t in sorted(by_turn):
        d = by_turn[t]
        p = d["cached"] / d["prompt"] * 100 if d["prompt"] > 0 else 0
        print(f"  {t:>5}  {p:>5.0f}%  {d['cached']:>8}  {d['prompt']:>8}  {d['n']:>4}")

print()
if "lru" in results and "marconi" in results:
    delta = results["marconi"] - results["lru"]
    winner = "Marconi" if delta > 0 else "LRU"
    print(f"  WINNER: {winner}  (Marconi - LRU = {delta:+.1f}pp)  [mem-fraction={mem}]")
    if float(mem) < 0.5:
        if delta > 2:
            print("  HYPOTHESIS CONFIRMED: tight cache → Marconi wins")
        elif delta > -2:
            print("  MARGINAL: try --mem-fraction 0.21")
        else:
            print("  LRU still wins — cache may still be too large; try lower mem-fraction")
PYEOF

echo ""
echo "Logs   : ${LOG_DIR}/"
echo "Results: ${OUTPUT_DIR}/"
