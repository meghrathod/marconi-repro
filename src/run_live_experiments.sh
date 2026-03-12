#!/bin/bash
# marconi/run_live_experiments.sh
# Automates the execution of live inference experiments across no-cache, LRU, and Marconi.

set -e

MODEL="nvidia/Nemotron-H-8B-Base-8K"
PORT=30000
SERVER_URL="http://127.0.0.1:${PORT}"
TRACE_DIR="traces"
OUTPUT_DIR="results/live"
LOG_DIR="logs/live"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Common server launch prefix
LAUNCH_PREFIX="env PYTHONPATH=$(pwd)/sglang/python uv run --project . python3 -m sglang.launch_server --model-path ${MODEL} --port ${PORT} --disable-custom-all-reduce --enable-metrics"

start_server() {
    local config=$1
    local extra_args=$2
    
    echo "Starting Server for configuration: ${config}"
    
    $LAUNCH_PREFIX $extra_args > "${LOG_DIR}/server_${config}.log" 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port ${PORT}..."
    while ! nc -z localhost $PORT; do   
      sleep 2
    done
    sleep 5 # extra wait for model loading initialization buffer
    echo "Server is fully ready."
}

stop_server() {
    echo "Stopping server (PID: ${SERVER_PID})"
    kill -9 $SERVER_PID || true
    wait $SERVER_PID || true
    echo "Server stopped."
    sleep 2
}

run_experiment() {
    local config=$1
    local extra_args=$2
    
    echo "==========================================================="
    echo "Running Experiment: ${config}"
    echo "==========================================================="
    
    start_server "$config" "$extra_args"
    
    # Run the trace replayer across the directory
    echo "Starting trace replay..."
    python3 src/trace_replayer.py \
        --trace-dir ${TRACE_DIR} \
        --output-dir "${OUTPUT_DIR}/${config}" \
        --server-url ${SERVER_URL} \
        --model ${MODEL} \
        > "${LOG_DIR}/replayer_${config}.log" 2>&1
        
    stop_server
}

# 1. Baseline: No Cache
run_experiment "no-cache" "--disable-radix-cache"

# 2. Baseline: LRU Cache
run_experiment "lru" "--radix-eviction-policy lru"

# 3. Marconi Cache
run_experiment "marconi" "--radix-eviction-policy marconi --marconi-eff-weight 0.7"

echo "All experiments completed. Check ${OUTPUT_DIR}/ and ${LOG_DIR}/ for results."
