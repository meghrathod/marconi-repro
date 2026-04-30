#!/usr/bin/env bash
# experiments/capacity_curve.sh
#
# PURPOSE: Show how Marconi's advantage changes with cache size.
# Runs LRU vs marc_a0.3 (best alpha) at 4 memory fractions on a given dataset.
# Use a larger session count so the working set exceeds even the large cache.
#
# Usage:
#   bash scripts/experiments/capacity_curve.sh                        # lmsys-50, default fractions
#   bash scripts/experiments/capacity_curve.sh --dataset lmsys --nums 50
#   bash scripts/experiments/capacity_curve.sh --dataset swebench --nums 20
set -e

MODEL="nvidia/Nemotron-H-8B-Reasoning-128K"
PORT=30000
SERVER_URL="http://127.0.0.1:${PORT}"
UV_RUN=(uv run --project .)
DATASET="lmsys"
NUMS=50
SPS="1.0"
ART="5"
MEM_FRACTIONS="0.22 0.40 0.60 0.85"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --nums)       NUMS="$2";    shift 2 ;;
        --fractions)  MEM_FRACTIONS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$DATASET" == "lmsys" ]]; then
    TRACE_NAME="lmsys_sps=${SPS}_nums=${NUMS}.jsonl"
elif [[ "$DATASET" == "swebench" ]]; then
    TRACE_NAME="swebench_sps=${SPS}_art=${ART}_nums=${NUMS}.jsonl"
else
    TRACE_NAME="sharegpt_sps=${SPS}_nums=${NUMS}.jsonl"
fi

OUTPUT_BASE="results/capacity-curve-${DATASET}"
LOG_BASE="logs/capacity-curve-${DATASET}"
mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

echo "========================================================"
echo "  CAPACITY CURVE — ${DATASET}-${NUMS}"
echo "  trace         : ${TRACE_NAME}"
echo "  mem-fractions : ${MEM_FRACTIONS}"
echo "  configs       : lru  marc_a0.3"
echo "  output        : ${OUTPUT_BASE}/"
echo "========================================================"

# ── Generate trace if needed ───────────────────────────────────────────────────
TRACE_PATH="traces/${TRACE_NAME}"
if [[ ! -f "$TRACE_PATH" ]]; then
    echo ""
    if [[ "$DATASET" == "lmsys" ]]; then
        echo "Generating lmsys-${NUMS} trace..."
        HF_TOKEN=$(grep '^HF_TOKEN=' .env 2>/dev/null | cut -d= -f2)
        [[ -n "$HF_TOKEN" ]] && export HF_TOKEN
        "${UV_RUN[@]}" python3 scripts/generate/lmsys.py --nums "$NUMS" --sps "$SPS"
    elif [[ "$DATASET" == "swebench" ]]; then
        echo "Generating swebench-${NUMS} trace..."
        "${UV_RUN[@]}" python3 scripts/generate/swebench.py --nums "$NUMS"
    fi
    # Fallback: copy from marconi/traces/
    [[ ! -f "$TRACE_PATH" && -f "marconi/traces/${TRACE_NAME}" ]] && cp "marconi/traces/${TRACE_NAME}" "$TRACE_PATH"
fi
[[ ! -f "$TRACE_PATH" ]] && { echo "ERROR: trace not found at $TRACE_PATH"; exit 1; }
echo "Trace: $(wc -l < "$TRACE_PATH") requests"

# ── Server helpers ─────────────────────────────────────────────────────────────
free_port() {
    local p=$1
    fuser -k -n tcp "$p" 2>/dev/null || true
    pkill -9 -f "sglang.launch_server.*--port ${p}" 2>/dev/null || true
    sleep 2
    local n=0
    while nc -z localhost "$p" 2>/dev/null; do
        fuser -k -n tcp "$p" 2>/dev/null || true; sleep 2
        [[ $((++n)) -ge 15 ]] && { echo "ERROR: port $p stuck"; exit 1; }
    done
}

start_server() {
    local tag=$1 mem=$2 extra=$3
    free_port "${PORT}"
    echo "[${tag}] Starting (mem=${mem})..."
    setsid env PYTHONPATH="$(pwd)/sglang/python" "${UV_RUN[@]}" python3 -m sglang.launch_server \
        --model-path "${MODEL}" --port "${PORT}" \
        --disable-custom-all-reduce --enable-metrics --enable-cache-report \
        --mem-fraction-static "${mem}" \
        ${extra} > "${LOG_BASE}/server_${tag}.log" 2>&1 &
    SERVER_PID=$!
    local n=0
    while ! nc -z localhost "${PORT}"; do
        sleep 2; n=$((n+1))
        [[ $n -gt 120 ]] && { echo "ERROR: timeout"; kill -9 -- -"${SERVER_PID}" 2>/dev/null; exit 1; }
    done
    sleep 5
    echo "[${tag}] Server ready."
}

stop_server() {
    kill -9 -- -"${SERVER_PID}" 2>/dev/null || kill -9 "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    free_port "${PORT}"
}

run_replayer() {
    local tag=$1
    mkdir -p "${OUTPUT_BASE}/${tag}"
    "${UV_RUN[@]}" python src/trace_replayer.py \
        --trace-dir "traces" \
        --trace-names "${TRACE_NAME}" \
        --output-dir "${OUTPUT_BASE}/${tag}" \
        --server-url "${SERVER_URL}" \
        --model "${MODEL}" \
        > "${LOG_BASE}/replayer_${tag}.log" 2>&1
    echo "[${tag}] Replayer done."
}

# ── Run each (mem_fraction, policy) pair ──────────────────────────────────────
for MEM in $MEM_FRACTIONS; do
    MEM_LABEL="${MEM//./_}"
    for CONFIG in lru marc_a0.3; do
        TAG="${CONFIG}_m${MEM_LABEL}"
        echo ""
        echo "─────────────────────────────────────"
        echo " ${TAG}"
        echo "─────────────────────────────────────"
        if [[ "$CONFIG" == "lru" ]]; then
            EXTRA="--radix-eviction-policy lru"
        else
            EXTRA="--radix-eviction-policy marconi --marconi-eff-weight 0.3"
        fi
        start_server "$TAG" "$MEM" "$EXTRA"
        run_replayer "$TAG"
        stop_server "$TAG"
    done
done

# ── Summary table ──────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  CAPACITY CURVE RESULTS — ${DATASET}-${NUMS}"
echo "========================================================"

export OUTPUT_BASE MEM_FRACTIONS TRACE_NAME DATASET NUMS
python3 - <<'PYEOF'
import json, glob, os, statistics

base      = os.environ["OUTPUT_BASE"]
mems      = os.environ["MEM_FRACTIONS"].split()
dataset   = os.environ["DATASET"]
nums      = os.environ["NUMS"]

print(f"\n{'mem':>8}  {'LRU Hit%':>10}  {'Marc Hit%':>10}  {'Δ':>8}  {'Winner':>8}  {'LRU TTFT':>10}  {'Marc TTFT':>10}")
print("  " + "-" * 72)

for mem in mems:
    mem_label = mem.replace('.', '_')
    row = {}
    for cfg in ["lru", "marc_a0.3"]:
        tag = f"{cfg}_m{mem_label}"
        files = glob.glob(f"{base}/{tag}/*.jsonl")
        if not files:
            row[cfg] = None
            continue
        reqs = [json.loads(l) for f in files for l in open(f)]
        tp = sum(r.get("prompt_tokens", 0) for r in reqs)
        tc = sum(r.get("cached_tokens", 0) for r in reqs)
        ttfts = [r["ttft_ms"] for r in reqs if r.get("ttft_ms", 0) > 0]
        row[cfg] = {
            "hit": tc / tp * 100 if tp else 0,
            "ttft": statistics.mean(ttfts) if ttfts else 0,
        }
    lru = row.get("lru")
    marc = row.get("marc_a0.3")
    if lru and marc:
        delta = marc["hit"] - lru["hit"]
        winner = "Marconi" if delta > 0 else "LRU    "
        bar = ("▲" * max(0, int(abs(delta)/2))) if delta > 0 else ("▼" * max(0, int(abs(delta)/2)))
        print(f"  {mem:>6}  {lru['hit']:>9.1f}%  {marc['hit']:>9.1f}%  {delta:>+7.1f}pp  {winner:>8}  "
              f"{lru['ttft']:>8.0f}ms  {marc['ttft']:>8.0f}ms  {bar}")
    else:
        missing = "lru" if not lru else "marc_a0.3"
        print(f"  {mem:>6}  -- missing {missing} --")

print(f"\n  dataset={dataset}-{nums}")
PYEOF

echo ""
echo "Logs   : ${LOG_BASE}/"
echo "Results: ${OUTPUT_BASE}/"
