#!/usr/bin/env bash
# scripts/experiments/run_live.sh
#
# Runs server configs (no-cache, LRU, Marconi) and replays traces via trace_replayer.
# Default mode is --v2 (fixed-alpha sweep).
#
# Modes:
#   ./scripts/experiments/run_live.sh [--v2]
#       Fixed-alpha sweep: lru + marconi α=0.3 + marconi α=1.0 on 9 minimal traces.
#       no-cache results are copied from results/live-minimal-32K/.
#       Default output: results/live-minimal-32K-v2/
#
#   ./src/run_live_experiments.sh --limited
#       Representative subset (one trace per dataset) for quick analysis.
#       Default output: results/live-limited-32K/
#
#   ./src/run_live_experiments.sh --full
#       Full matrix: every traces/*.jsonl → results/live-32K/
#
#   ./src/run_live_experiments.sh --trace-names 'a.jsonl,b.jsonl' [--output-dir DIR]
#       Custom trace list; compatible with all modes.
#
# Server tuning (env vars with defaults):
#   TP=4                       Tensor-parallel degree
#   MAX_MAMBA_CACHE_SIZE=318   Mamba cache size
#   EXTRA_SERVER_ARGS=""       Any additional sglang flags
#
# Logs: logs/$(basename OUTPUT_DIR)/

set -e

MODEL="nvidia/Nemotron-H-8B-Reasoning-128K"
PORT=30000
SERVER_URL="http://127.0.0.1:${PORT}"
TRACE_DIR="marconi/traces/traces_nemotron_32K"
UV_RUN=(/home/cc/.local/bin/uv run --project .)

# Server tuning — override via env
TP=${TP:-4}
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-318}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-""}

LIMITED_TRACE_NAMES="lmsys_sps=1_nums=100.jsonl,sharegpt_sps=1_nums=100.jsonl,swebench_sps=1_art=5_nums=100.jsonl"
V2_TRACE_NAMES="lmsys_sps=0.25_nums=100.jsonl,lmsys_sps=1_nums=100.jsonl,lmsys_sps=5_nums=100.jsonl,sharegpt_sps=0.25_nums=100.jsonl,sharegpt_sps=1_nums=100.jsonl,sharegpt_sps=5_nums=100.jsonl,swebench_sps=1_art=5_nums=100.jsonl,swebench_sps=5_art=5_nums=100.jsonl,swebench_sps=5_art=7.5_nums=100.jsonl"
V2_TRACE_DIR="traces"

MODE="v2"
OUTPUT_DIR=""
TRACE_NAMES_ARG=""

usage() {
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --v2)           MODE="v2";      shift ;;
        --limited)      MODE="limited"; shift ;;
        --full)         MODE="full";    shift ;;
        --trace-names)  TRACE_NAMES_ARG="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";       shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ "$MODE" == "v2" ]]; then
    TRACE_DIR="$V2_TRACE_DIR"
    [[ -n "$TRACE_NAMES_ARG" ]] || TRACE_NAMES_ARG="$V2_TRACE_NAMES"
    [[ -n "$OUTPUT_DIR" ]]      || OUTPUT_DIR="results/live-minimal-32K-v2"
elif [[ "$MODE" == "limited" ]]; then
    [[ -n "$TRACE_NAMES_ARG" ]] || TRACE_NAMES_ARG="$LIMITED_TRACE_NAMES"
    [[ -n "$OUTPUT_DIR" ]]      || OUTPUT_DIR="results/live-limited-32K"
else
    [[ -n "$OUTPUT_DIR" ]]      || OUTPUT_DIR="results/live-32K"
fi

LOG_DIR="logs/$(basename "${OUTPUT_DIR}")"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ "$MODE" == "v2" ]]; then
    echo "=========================================="
    echo "  Live experiments  mode=v2 (fixed-alpha sweep)"
    echo "  Results         ${OUTPUT_DIR}/{no-cache(copied),lru,marconi_a0.3,marconi_a1.0}/"
    echo "  Traces          ${TRACE_NAMES_ARG}"
else
    echo "=========================================="
    echo "  Live experiments  mode=${MODE}"
    echo "  Results         ${OUTPUT_DIR}/{no-cache,lru,marconi}/"
    [[ -n "$TRACE_NAMES_ARG" ]] && echo "  Trace filter    ${TRACE_NAMES_ARG}"
fi
echo "  TP=${TP}  MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE}"
echo "  Logs            ${LOG_DIR}/"
echo "=========================================="

# Free PORT and GPU children: launch_server forks schedulers/workers; kill -9 $! often
# leaves sglang::scheduler (and port 30000) alive, so the next "start" still talks to the
# first config.
free_port_and_sglang() {
    local p=$1
    echo "Releasing port ${p} and cleaning SGLang processes..."
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -n tcp "$p" 2>/dev/null || true
    fi
    pkill -9 -f "python3 -m sglang.launch_server.*--model-path ${MODEL}.*--port ${p}" 2>/dev/null || true
    pkill -9 -f "python.*sglang.launch_server.*--port ${p}" 2>/dev/null || true
    sleep 2
    local n=0
    while nc -z localhost "$p" 2>/dev/null; do
        echo "  port ${p} still busy; retry cleanup ($((++n))/15)..."
        fuser -k -n tcp "$p" 2>/dev/null || true
        sleep 2
        if [[ $n -ge 15 ]]; then
            echo "ERROR: port ${p} still in use. Run: nvidia-smi  # then kill leftover PIDs"
            exit 1
        fi
    done
    echo "  port ${p} is free."
}

start_server() {
    local config=$1 extra_args=$2
    free_port_and_sglang "${PORT}"
    echo "Starting server: ${config}"
    local launch_cmd=(env PYTHONPATH="$(pwd)/sglang/python" "${UV_RUN[@]}" python3 -m sglang.launch_server
        --model-path "${MODEL}" --port "${PORT}"
        --tp "${TP}" --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}"
        --disable-custom-all-reduce --enable-metrics --enable-cache-report
        $extra_args $EXTRA_SERVER_ARGS)
    # setsid: new session + process group so kill -- -PGID tears down uv + python + workers
    if command -v setsid >/dev/null 2>&1; then
        setsid "${launch_cmd[@]}" > "${LOG_DIR}/server_${config}.log" 2>&1 &
    else
        "${launch_cmd[@]}" > "${LOG_DIR}/server_${config}.log" 2>&1 &
    fi
    SERVER_PID=$!
    echo "  Server PGID/PID: ${SERVER_PID}"
    echo "Waiting for port ${PORT}..."
    local n=0
    while ! nc -z localhost "$PORT"; do
        sleep 2
        n=$((n + 1))
        if [[ $n -gt 120 ]]; then
            echo "ERROR: server did not open port ${PORT} (see ${LOG_DIR}/server_${config}.log)"
            kill -9 -- -"${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
    done
    sleep 5
    echo "Server ready (${config})."
}

stop_server() {
    local config="${1:-unknown}"
    echo "Stopping server (PID ${SERVER_PID}, config was ${config})..."
    if command -v setsid >/dev/null 2>&1; then
        kill -9 -- -"${SERVER_PID}" 2>/dev/null || kill -9 "${SERVER_PID}" 2>/dev/null || true
    else
        kill -9 "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
    free_port_and_sglang "${PORT}"
}

run_replayer_for_config() {
    local config="$1"
    local -a cmd
    cmd=("${UV_RUN[@]}" python src/trace_replayer.py
        --trace-dir "${TRACE_DIR}"
        --output-dir "${OUTPUT_DIR}/${config}"
        --server-url "${SERVER_URL}"
        --model "${MODEL}")
    [[ -n "$TRACE_NAMES_ARG" ]] && cmd+=(--trace-names "$TRACE_NAMES_ARG")
    "${cmd[@]}"
}

run_experiment() {
    local config=$1 extra_args=$2
    echo "==========================================================="
    echo "Experiment: ${config}"
    echo "==========================================================="
    start_server "$config" "$extra_args"
    echo "Trace replay..."
    run_replayer_for_config "$config" > "${LOG_DIR}/replayer_${config}.log" 2>&1
    stop_server "$config"
}

if [[ "$MODE" == "v2" ]]; then
    mkdir -p "${OUTPUT_DIR}/no-cache"
    cp results/live-minimal-32K/no-cache/*.jsonl "${OUTPUT_DIR}/no-cache/" 2>/dev/null || true
    echo "Copied no-cache results from results/live-minimal-32K/no-cache/"
    run_experiment "lru"          "--radix-eviction-policy lru"
    run_experiment "marconi_a0.3" "--radix-eviction-policy marconi --marconi-eff-weight 0.3"
    run_experiment "marconi_a1.0" "--radix-eviction-policy marconi --marconi-eff-weight 1.0"
else
    run_experiment "no-cache" "--disable-radix-cache"
    run_experiment "lru"      "--radix-eviction-policy lru"
    run_experiment "marconi"  "--radix-eviction-policy marconi --marconi-eff-weight 0.7"
fi

echo "Done. Results: ${OUTPUT_DIR}/  Logs: ${LOG_DIR}/"
