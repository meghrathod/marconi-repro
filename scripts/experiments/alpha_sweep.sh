#!/usr/bin/env bash
# alpha_sweep.sh
# Run LRU + Marconi at α=0.3,0.5,0.7,1.0,1.5 on swebench-10 trace.
# Generates the swebench trace if missing, then sweeps alphas sequentially.
#
# Usage:
#   bash scripts/alpha_sweep.sh [--mem-fraction 0.22] [--trace swebench]
set -e

MODEL="nvidia/Nemotron-H-8B-Reasoning-128K"
PORT=30000
SERVER_URL="http://127.0.0.1:${PORT}"
UV_RUN=(uv run --project .)
MEM_FRACTION="0.22"
DATASET="swebench"
SPS="1.0"
ART="5"
NUMS="10"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mem-fraction)   MEM_FRACTION="$2"; shift 2 ;;
        --mem-fraction=*) MEM_FRACTION="${1#*=}"; shift ;;
        --trace)          DATASET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$DATASET" == "swebench" ]]; then
    TRACE_NAME="swebench_sps=${SPS}_art=${ART}_nums=${NUMS}.jsonl"
elif [[ "$DATASET" == "lmsys" ]]; then
    TRACE_NAME="lmsys_sps=${SPS}_nums=${NUMS}.jsonl"
else
    TRACE_NAME="sharegpt_sps=${SPS}_nums=${NUMS}.jsonl"
fi

OUTPUT_DIR="results/alpha-sweep-${DATASET}"
LOG_DIR="logs/alpha-sweep-${DATASET}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "========================================================"
echo "  ALPHA SWEEP — ${DATASET}"
echo "  trace         : ${TRACE_NAME}"
echo "  mem-fraction  : ${MEM_FRACTION}"
echo "  output        : ${OUTPUT_DIR}/"
echo "========================================================"

# ── Generate swebench trace if needed ─────────────────────────────────────────

TRACE_PATH="traces/${TRACE_NAME}"
if [[ ! -f "$TRACE_PATH" ]]; then
    echo ""
    if [[ "$DATASET" == "swebench" ]]; then
        echo "Generating swebench trace (downloads from HuggingFace, ~2 min)..."
        "${UV_RUN[@]}" python3 scripts/generate/swebench.py
    elif [[ "$DATASET" == "lmsys" ]]; then
        echo "Generating lmsys trace (downloads from HuggingFace, ~2 min)..."
        "${UV_RUN[@]}" python3 scripts/generate/lmsys.py
    fi
fi

if [[ ! -f "$TRACE_PATH" ]]; then
    # Try symlink from marconi/traces
    SRC="marconi/traces/${TRACE_NAME}"
    if [[ -f "$SRC" ]]; then
        cp "$SRC" "$TRACE_PATH"
        echo "Copied trace from $SRC"
    else
        echo "ERROR: trace not found at $TRACE_PATH"; exit 1
    fi
fi

# Count requests
N_REQS=$(wc -l < "$TRACE_PATH")
echo "Trace: ${N_REQS} requests"
echo ""

# ── Server management ──────────────────────────────────────────────────────────

free_port() {
    local p=$1
    fuser -k -n tcp "$p" 2>/dev/null || true
    pkill -9 -f "sglang.launch_server.*--port ${p}" 2>/dev/null || true
    sleep 2
    local n=0
    while nc -z localhost "$p" 2>/dev/null; do
        fuser -k -n tcp "$p" 2>/dev/null || true
        sleep 2
        [[ $((++n)) -ge 15 ]] && { echo "ERROR: port ${p} stuck"; exit 1; }
    done
}

start_server() {
    local tag=$1 extra=$2
    free_port "${PORT}"
    echo "[${tag}] Starting server..."
    setsid env PYTHONPATH="$(pwd)/sglang/python" "${UV_RUN[@]}" python3 -m sglang.launch_server \
        --model-path "${MODEL}" --port "${PORT}" \
        --disable-custom-all-reduce --enable-metrics --enable-cache-report \
        --mem-fraction-static "${MEM_FRACTION}" \
        ${extra} > "${LOG_DIR}/server_${tag}.log" 2>&1 &
    SERVER_PID=$!
    echo "[${tag}] PID=${SERVER_PID}, waiting for port..."
    local n=0
    while ! nc -z localhost "${PORT}"; do
        sleep 2; n=$((n+1))
        [[ $n -gt 120 ]] && { echo "ERROR: timeout"; kill -9 -- -"${SERVER_PID}" 2>/dev/null; exit 1; }
    done
    sleep 5
    echo "[${tag}] Server ready."
}

stop_server() {
    local tag=$1
    echo "[${tag}] Stopping..."
    kill -9 -- -"${SERVER_PID}" 2>/dev/null || kill -9 "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    free_port "${PORT}"
}

run_replayer() {
    local tag=$1
    mkdir -p "${OUTPUT_DIR}/${tag}"
    "${UV_RUN[@]}" python src/trace_replayer.py \
        --trace-dir "traces" \
        --trace-names "${TRACE_NAME}" \
        --output-dir "${OUTPUT_DIR}/${tag}" \
        --server-url "${SERVER_URL}" \
        --model "${MODEL}" \
        > "${LOG_DIR}/replayer_${tag}.log" 2>&1
    echo "[${tag}] Replayer done."
}

run_one() {
    local tag=$1 extra=$2
    echo ""
    echo "─────────────────────────────────────"
    echo " Config: ${tag}"
    echo "─────────────────────────────────────"
    start_server "${tag}" "${extra}"
    run_replayer "${tag}"
    stop_server "${tag}"
}

# ── Run all configs ────────────────────────────────────────────────────────────

run_one "lru"        "--radix-eviction-policy lru"
run_one "marc_a0.3"  "--radix-eviction-policy marconi --marconi-eff-weight 0.3"
run_one "marc_a0.5"  "--radix-eviction-policy marconi --marconi-eff-weight 0.5"
run_one "marc_a0.7"  "--radix-eviction-policy marconi --marconi-eff-weight 0.7"
run_one "marc_a1.0"  "--radix-eviction-policy marconi --marconi-eff-weight 1.0"
run_one "marc_a1.5"  "--radix-eviction-policy marconi --marconi-eff-weight 1.5"

# ── Summarise ─────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  RESULTS  (mem-fraction=${MEM_FRACTION}, dataset=${DATASET})"
echo "========================================================"

export OUTPUT_DIR TRACE_NAME MEM_FRACTION DATASET
python3 - <<'PYEOF'
import json, glob, os, statistics

base      = os.environ["OUTPUT_DIR"]
trace     = os.environ["TRACE_NAME"]
mem       = os.environ["MEM_FRACTION"]
dataset   = os.environ["DATASET"]

configs = ["lru", "marc_a0.3", "marc_a0.5", "marc_a0.7", "marc_a1.0", "marc_a1.5"]
results = {}

print(f"\n{'Config':>12}  {'Hit%':>7}  {'AvgTTFT':>9}  {'Errors':>7}  {'N':>6}")
print("  " + "-" * 52)

for cfg in configs:
    files = glob.glob(f"{base}/{cfg}/*.jsonl")
    if not files:
        print(f"  {cfg:>12}  -- no output --")
        continue
    reqs = [json.loads(l) for f in files for l in open(f)]
    total_p = sum(r.get("prompt_tokens", 0) for r in reqs)
    total_c = sum(r.get("cached_tokens", 0) for r in reqs)
    errors  = sum(1 for r in reqs if r.get("error"))
    ttfts   = [r["ttft_ms"] for r in reqs if r.get("ttft_ms", 0) > 0]
    hit_pct = total_c / total_p * 100 if total_p else 0
    avg_ttft = statistics.mean(ttfts) if ttfts else 0
    results[cfg] = {"hit": hit_pct, "ttft": avg_ttft, "n": len(reqs)}
    print(f"  {cfg:>12}  {hit_pct:>6.1f}%  {avg_ttft:>8.0f}ms  {errors:>7}  {len(reqs):>6}")

if "lru" in results:
    lru_hit = results["lru"]["hit"]
    print(f"\n  Δ vs LRU:")
    for cfg in configs:
        if cfg == "lru" or cfg not in results:
            continue
        delta = results[cfg]["hit"] - lru_hit
        winner = "Marconi" if delta > 0 else "LRU   "
        bar = "+" * int(abs(delta)) if delta > 0 else "-" * int(abs(delta))
        print(f"    {cfg:>12}: {delta:>+6.1f}pp  [{winner}] {bar}")

print(f"\n  mem-fraction={mem}  dataset={dataset}")
PYEOF

echo ""
echo "Logs   : ${LOG_DIR}/"
echo "Results: ${OUTPUT_DIR}/"
