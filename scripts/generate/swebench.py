#!/usr/bin/env python3
"""Generate swebench trace.

Usage:
    uv run python scripts/generate/swebench.py              # 10 sessions (default)
    uv run python scripts/generate/swebench.py --nums 50
"""
import sys, os, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "marconi", "utils"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

from generate_trace import process_swebench_trace

ap = argparse.ArgumentParser()
ap.add_argument("--nums", type=int, default=10)
ap.add_argument("--sps", type=float, default=1.0)
ap.add_argument("--art", type=int, default=5)
args = ap.parse_args()

SPS, ART, NUMS = args.sps, args.art, args.nums
print(f"Generating swebench trace ({NUMS} sessions, sps={SPS}, art={ART})...")
reqs = process_swebench_trace(
    sessions_per_second=SPS,
    avg_response_time=ART,
    num_sessions=NUMS,
)

src = f"marconi/traces/swebench_sps={SPS}_art={ART}_nums={NUMS}.jsonl"
dst = f"traces/swebench_sps={SPS}_art={ART}_nums={NUMS}.jsonl"
if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy(src, dst)
    print(f"Copied {src} → {dst}")
elif os.path.exists(dst):
    print(f"Already at {dst}")
else:
    print(f"WARNING: source not found at {src}")

print(f"Done. {len(reqs)} requests total.")
