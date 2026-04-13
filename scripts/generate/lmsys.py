#!/usr/bin/env python3
"""Generate lmsys trace.

Usage:
    uv run python scripts/generate/lmsys.py              # 10 sessions (default)
    uv run python scripts/generate/lmsys.py --nums 50    # 50 sessions
    uv run python scripts/generate/lmsys.py --nums 50 --sps 1.0
"""
import sys, os, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "marconi", "utils"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

from generate_trace import generate_lmsys_trace

ap = argparse.ArgumentParser()
ap.add_argument("--nums", type=int, default=10)
ap.add_argument("--sps", type=float, default=1.0)
args = ap.parse_args()

SPS, NUMS = args.sps, args.nums
print(f"Generating lmsys trace ({NUMS} sessions, sps={SPS})...")
reqs, _ = generate_lmsys_trace(sessions_per_second=SPS, num_sessions=NUMS)

src = f"marconi/traces/lmsys_sps={SPS}_nums={NUMS}.jsonl"
dst = f"traces/lmsys_sps={SPS}_nums={NUMS}.jsonl"
if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy(src, dst)
    print(f"Copied {src} → {dst}")
elif os.path.exists(dst):
    print(f"Already at {dst}")

print(f"Done. {len(reqs)} requests total.")
