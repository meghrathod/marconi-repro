#!/usr/bin/env bash
# src/run_live_experiments.sh
#
# Runs three server configs (no-cache, LRU, Marconi) and replays traces via trace_replayer.
#
# Modes:
#   ./src/run_live_experiments.sh
#       Full matrix: every traces/*.jsonl → results/live/{no-cache,lru,marconi}/
#
#   ./src/run_live_experiments.sh --limited
#       Representative subset (one trace per dataset, sps=1) for analysis before the full run.
#       Default output: results/live-limited/{no-cache,lru,marconi}/
#
#   ./src/run_live_experiments.sh --phase1
#       Alpha calibration: SWEBench sps=5 art=5 with no-cache, lru, and
#       marconi at eff_weight=[0.5, 1.0, 1.5, 2.0].
#       Default output: results/live-phase1/{no-cache,lru,marconi_a0.5,...}/
#
#   ./src/run_live_experiments.sh --minimal
#       Key subset for graph alignment: lmsys/sharegpt sps=0.25,1,5 + swebench sps=1,5 art=5,7.5
#       Default output: results/live-minimal-32K/{no-cache,lru,marconi}/
#
#   ./src/run_live_experiments.sh --trace-names 'a.jsonl,b.jsonl' [--output-dir DIR]
#       Custom list (any mode); does not imply --limited unless you only pass a few files.
#
# Back-compat:  --subset verify  ==  --limited
#
# Logs: logs/$(basename OUTPUT_DIR)/
# Replayer uses the same uv env as the server (see UV_RUN).

set -e

MODEL="nvidia/Nemotron-H-8B-Reasoning-128K"
PORT=30000
SERVER_URL="http://127.0.0.1:${PORT}"
TRACE_DIR="marconi/traces/traces_nemotron_32K"
UV_RUN=(uv run --project .)

# Default traces for --limited: one file per dataset (multi-turn + cache-friendly load)
LIMITED_TRACE_NAMES="lmsys_sps=1_nums=100.jsonl,sharegpt_sps=1_nums=100.jsonl,swebench_sps=1_art=5_nums=100.jsonl"

# Default traces for --minimal: low/mid/high pressure per dataset, covers key graph axes
MINIMAL_TRACE_NAMES="lmsys_sps=0.25_nums=100.jsonl,lmsys_sps=1_nums=100.jsonl,lmsys_sps=5_nums=100.jsonl,sharegpt_sps=0.25_nums=100.jsonl,sharegpt_sps=1_nums=100.jsonl,sharegpt_sps=5_nums=100.jsonl,swebench_sps=1_art=5_nums=100.jsonl,swebench_sps=5_art=5_nums=100.jsonl,swebench_sps=5_art=7.5_nums=100.jsonl"
MINIMAL_TRACE_DIR="traces_nemotron_32K_minimal"

MODE="full"
OUTPUT_DIR=""
TRACE_NAMES_ARG=""

# Alpha values to sweep in --phase1 mode
PHASE1_TRACE="swebench_sps=5_art=5_nums=100.jsonl"
PHASE1_ALPHAS=(0.5 1.0 1.5 2.0)

usage() {
    sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
    echo ""
    echo "Flags:"
    echo "  --minimal              9 key traces (low/mid/high sps × 3 datasets) → results/live-minimal-32K
  --limited              Subset traces + default output results/live-limited"
    echo "  --phase1               Alpha calibration on SWEBench sps=5 (see header)"
    echo "  --subset verify        Same as --limited (deprecated alias)"
    echo "  --trace-names LIST     Comma-separated basenames under ${TRACE_DIR}/"
    echo "  --output-dir DIR       Override result root (default: live or live-limited)"
    echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase1)
            MODE="phase1"
            shift
            ;;
        --minimal)
            MODE="minimal"
            shift
            ;;
        --limited)
            MODE="limited"
            shift
            ;;
        --subset)
            if [[ "${2:-}" != "verify" ]]; then
                echo "Error: use --subset verify (or prefer --limited)"
                exit 1
            fi
            MODE="limited"
            shift 2
            ;;
        --trace-names)
            TRACE_NAMES_ARG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$MODE" == "phase1" ]]; then
    [[ -n "$TRACE_NAMES_ARG" ]] || TRACE_NAMES_ARG="$PHASE1_TRACE"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="results/live-phase1-32K"
elif [[ "$MODE" == "minimal" ]]; then
    TRACE_DIR="$MINIMAL_TRACE_DIR"
    [[ -n "$TRACE_NAMES_ARG" ]] || TRACE_NAMES_ARG="$MINIMAL_TRACE_NAMES"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="results/live-minimal-32K"
elif [[ "$MODE" == "limited" ]]; then
    [[ -n "$TRACE_NAMES_ARG" ]] || TRACE_NAMES_ARG="$LIMITED_TRACE_NAMES"
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="results/live-limited-32K"
else
    [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="results/live-32K"
fi

LOG_DIR="logs/$(basename "${OUTPUT_DIR}")"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ "$MODE" == "phase1" ]]; then
    MARCONI_SUBDIRS=$(printf "marconi_a%s " "${PHASE1_ALPHAS[@]}")
    echo "=========================================="
    echo "  Live experiments  mode=${MODE} (alpha calibration)"
    echo "  Results         ${OUTPUT_DIR}/{no-cache,lru,${MARCONI_SUBDIRS}}"
    echo "  Alphas          ${PHASE1_ALPHAS[*]}"
    echo "  Trace           ${TRACE_NAMES_ARG}"
    echo "  Logs            ${LOG_DIR}/"
    echo "=========================================="
else
    echo "=========================================="
    echo "  Live experiments  mode=${MODE}"
    echo "  Results         ${OUTPUT_DIR}/{no-cache,lru,marconi}/"
    echo "  Logs            ${LOG_DIR}/"
    [[ -n "$TRACE_NAMES_ARG" ]] && echo "  Trace filter    ${TRACE_NAMES_ARG}"
    echo "=========================================="
fi

# Free PORT and GPU children: launch_server forks schedulers/workers; kill -9 $! often
# leaves sglang::scheduler (and port 30000) alive, so the next "start" still talks to the
# first config. Also stop any Docker/other proxy publishing the same host port.
free_port_and_sglang() {
    local p=$1
    echo "Releasing port ${p} and cleaning SGLang processes..."
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -n tcp "$p" 2>/dev/null || true
    fi
    # Match our launch line (model + port); avoids killing unrelated sglang if unique enough
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
            echo "       If using Docker: docker compose stop sglang-server"
            exit 1
        fi
    done
    echo "  port ${p} is free."
}

start_server() {
    local config=$1
    local extra_args=$2
    free_port_and_sglang "${PORT}"
    echo "Starting server: ${config}"
    # setsid: new session + process group so kill -- -PGID tears down uv + python + workers
    if command -v setsid >/dev/null 2>&1; then
        setsid env PYTHONPATH="$(pwd)/sglang/python" "${UV_RUN[@]}" python3 -m sglang.launch_server \
            --model-path "${MODEL}" --port "${PORT}" \
            --disable-custom-all-reduce --enable-metrics --enable-cache-report \
            $extra_args > "${LOG_DIR}/server_${config}.log" 2>&1 &
    else
        env PYTHONPATH="$(pwd)/sglang/python" "${UV_RUN[@]}" python3 -m sglang.launch_server \
            --model-path "${MODEL}" --port "${PORT}" \
            --disable-custom-all-reduce --enable-metrics --enable-cache-report \
            $extra_args > "${LOG_DIR}/server_${config}.log" 2>&1 &
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
    if [[ -n "$TRACE_NAMES_ARG" ]]; then
        cmd+=(--trace-names "$TRACE_NAMES_ARG")
    fi
    "${cmd[@]}"
}

run_experiment() {
    local config=$1
    local extra_args=$2
    echo "==========================================================="
    echo "Experiment: ${config}"
    echo "==========================================================="
    start_server "$config" "$extra_args"
    echo "Trace replay..."
    run_replayer_for_config "$config" > "${LOG_DIR}/replayer_${config}.log" 2>&1
    stop_server "$config"
}

if [[ "$MODE" == "phase1" ]]; then
    run_experiment "no-cache" "--disable-radix-cache"
    run_experiment "lru"      "--radix-eviction-policy lru"
    for alpha in "${PHASE1_ALPHAS[@]}"; do
        run_experiment "marconi_a${alpha}" \
            "--radix-eviction-policy marconi --marconi-eff-weight ${alpha}"
    done
else
    run_experiment "lru"      "--radix-eviction-policy lru"
    run_experiment "no-cache" "--disable-radix-cache"
    run_experiment "marconi"  "--radix-eviction-policy marconi --marconi-eff-weight 0.7"
fi

echo "Done. Results: ${OUTPUT_DIR}/  Logs: ${LOG_DIR}/"
