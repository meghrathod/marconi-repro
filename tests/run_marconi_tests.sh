#!/bin/bash
# Run Marconi eviction tests.
# Requires marconi conda env: conda env create -f marconi/environment.yml && conda activate marconi

set -e
cd "$(dirname "$0")/.."

# Use marconi's Python if conda env is active
if command -v pytest &>/dev/null; then
    pytest tests/test_marconi_eviction.py -v
else
    echo "Activate marconi env first: conda activate marconi"
    echo "Or install pytest: pip install pytest"
    exit 1
fi
