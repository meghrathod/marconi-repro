"""
Trace Replayer — replay Marconi JSONL traces against a live SGLang server.

Reads pre-tokenized traces (produced by marconi/utils/generate_trace.py),
and sends them to an SGLang /v1/completions endpoint with streaming to
measure TTFT and throughput.

Two prompt modes:
  - token-ids mode (default): sends input_tokens directly as token ID array.
    Requires the trace to be tokenized with the target model's tokenizer
    (use generate_trace.py with the appropriate tokenizer).
  - text mode (--text-mode): decodes input_tokens back to text using the
    meta-llama/Llama-2-7b-hf tokenizer and sends as a string prompt.
    Useful for debugging or when traces use a different tokenizer.

Usage:
  python src/trace_replayer.py \
      --trace traces/lmsys_sps=1_nums=100.jsonl \
      --server-url http://localhost:30000 \
      --model nvidia/Nemotron-H-8B-Base-8K \
      --output results/baseline_lru.jsonl \
      --speed-factor 0 \
      --max-requests 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TraceRequest:
    """A single request from a Marconi JSONL trace."""

    session_id: int
    turn_id: int
    ts: float
    num_input_tokens: int
    num_output_tokens: int
    input_tokens: list[int]
    output_tokens: list[int]


@dataclass
class ReplayResult:
    """Metrics collected for one replayed request."""

    session_id: int
    turn_id: int
    ts: float
    num_input_tokens: int
    num_output_tokens: int
    prompt_len: int = 0  # chars if text mode, token count if token-ids mode
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    generated_tokens: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_trace(trace_path: str | Path) -> list[TraceRequest]:
    """Load a Marconi JSONL trace file.

    Returns requests in the order they appear (already sorted by ``ts``
    by generate_trace.py).
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    requests: list[TraceRequest] = []
    with open(trace_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            requests.append(
                TraceRequest(
                    session_id=obj["session_id"],
                    turn_id=obj["turn_id"],
                    ts=obj["ts"],
                    num_input_tokens=obj["num_input_tokens"],
                    num_output_tokens=obj["num_output_tokens"],
                    input_tokens=obj["input_tokens"],
                    output_tokens=obj.get("output_tokens", []),
                )
            )
    return requests


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def decode_tokens(token_ids: list[int], tokenizer=None) -> str:
    """Decode token IDs to text using the Llama-2-7b-hf tokenizer.

    Only used in ``--text-mode``.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_fast=True
        )
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_prompt(
    req: TraceRequest,
    text_mode: bool,
    tokenizer=None,
) -> str | list[int]:
    """Return the prompt for a request.

    In token-ids mode (default), returns the raw ``input_tokens`` list.
    In text mode, decodes token IDs back to a string.
    """
    if text_mode:
        return decode_tokens(req.input_tokens, tokenizer)
    return req.input_tokens


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


def compute_sleep_durations(
    requests: list[TraceRequest],
    speed_factor: float,
) -> list[float]:
    """Compute per-request sleep durations based on timestamp deltas.

    Args:
        requests: Trace requests sorted by ts.
        speed_factor: Multiplier for inter-request delay.
            0 = no delay (as-fast-as-possible).
            1.0 = real-time replay.

    Returns:
        List of sleep durations in seconds (same length as requests).
    """
    if not requests or speed_factor <= 0:
        return [0.0] * len(requests)

    durations = [0.0]  # first request has no delay
    for i in range(1, len(requests)):
        delta = requests[i].ts - requests[i - 1].ts
        durations.append(max(0.0, delta * speed_factor))
    return durations


# ---------------------------------------------------------------------------
# HTTP request sender (streaming)
# ---------------------------------------------------------------------------


async def send_completion_request(
    session: aiohttp.ClientSession,
    server_url: str,
    model: str,
    prompt: str | list[int],
    max_tokens: int,
) -> ReplayResult:
    """Send a streaming /v1/completions request and measure TTFT.

    ``prompt`` can be a string (text mode) or a list of ints (token-ids mode).
    The OpenAI /v1/completions spec accepts both.
    """
    url = f"{server_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    prompt_len = len(prompt)  # chars or token count
    result = ReplayResult(
        session_id=0,
        turn_id=0,
        ts=0.0,
        num_input_tokens=0,
        num_output_tokens=0,
        prompt_len=prompt_len,
    )

    t_start = time.perf_counter()
    first_token_time: float | None = None
    generated_tokens = 0

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:500]}"
                result.total_latency_ms = (time.perf_counter() - t_start) * 1000
                return result

            # Parse SSE stream
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text"):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        generated_tokens += 1
                except json.JSONDecodeError:
                    continue

    except Exception as exc:
        result.error = str(exc)
        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return result

    t_end = time.perf_counter()
    result.total_latency_ms = (t_end - t_start) * 1000
    if first_token_time is not None:
        result.ttft_ms = (first_token_time - t_start) * 1000
    result.generated_tokens = generated_tokens
    return result


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------


async def replay_trace(
    requests: list[TraceRequest],
    server_url: str,
    model: str,
    max_output_tokens: int,
    speed_factor: float,
    text_mode: bool,
    dry_run: bool,
) -> list[ReplayResult]:
    """Replay trace requests against SGLang, returning per-request metrics."""
    sleep_durations = compute_sleep_durations(requests, speed_factor)
    results: list[ReplayResult] = []

    # Lazy-load tokenizer only if text mode is needed
    tokenizer = None
    if text_mode:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_fast=True
        )

    if dry_run:
        for i, req in enumerate(tqdm(requests, desc="Dry run")):
            prompt = build_prompt(req, text_mode, tokenizer)
            max_tok = min(req.num_output_tokens, max_output_tokens)
            result = ReplayResult(
                session_id=req.session_id,
                turn_id=req.turn_id,
                ts=req.ts,
                num_input_tokens=req.num_input_tokens,
                num_output_tokens=req.num_output_tokens,
                prompt_len=len(prompt),
            )
            results.append(result)
            preview = str(prompt)[:80].replace("\n", "\\n")
            logger.info(
                "[dry-run] session=%d turn=%d input_tokens=%d max_out=%d prompt=%s...",
                req.session_id,
                req.turn_id,
                req.num_input_tokens,
                max_tok,
                preview,
            )
        return results

    connector = aiohttp.TCPConnector(limit=64)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, req in enumerate(tqdm(requests, desc="Replaying")):
            # Inter-request delay
            if sleep_durations[i] > 0:
                await asyncio.sleep(sleep_durations[i])

            prompt = build_prompt(req, text_mode, tokenizer)
            max_tok = min(req.num_output_tokens, max_output_tokens)

            result = await send_completion_request(
                session=session,
                server_url=server_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tok,
            )
            # Fill in trace metadata
            result.session_id = req.session_id
            result.turn_id = req.turn_id
            result.ts = req.ts
            result.num_input_tokens = req.num_input_tokens
            result.num_output_tokens = req.num_output_tokens

            results.append(result)

            if result.error:
                logger.warning(
                    "Request %d (session=%d turn=%d) failed: %s",
                    i,
                    req.session_id,
                    req.turn_id,
                    result.error,
                )

    return results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def write_results(results: list[ReplayResult], output_path: str | Path) -> None:
    """Write results as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            json.dump(asdict(r), f)
            f.write("\n")
    logger.info("Wrote %d results to %s", len(results), output_path)


def print_summary(results: list[ReplayResult]) -> None:
    """Print a summary of replay results to stdout."""
    if not results:
        print("No results to summarize.")
        return

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    print(f"\n{'='*60}")
    print("Trace Replay Summary")
    print(f"{'='*60}")
    print(f"Total requests:  {len(results)}")
    print(f"Successful:      {len(successful)}")
    print(f"Failed:          {len(failed)}")

    if successful:
        ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
        latencies = [r.total_latency_ms for r in successful]

        if ttfts:
            ttfts_sorted = sorted(ttfts)
            n = len(ttfts_sorted)
            print(f"\nTTFT (ms):")
            print(f"  P5:   {ttfts_sorted[max(0, int(n*0.05))]:.1f}")
            print(f"  P50:  {ttfts_sorted[int(n*0.50)]:.1f}")
            print(f"  P95:  {ttfts_sorted[min(n-1, int(n*0.95))]:.1f}")
            print(f"  P99:  {ttfts_sorted[min(n-1, int(n*0.99))]:.1f}")
            print(f"  Mean: {sum(ttfts)/len(ttfts):.1f}")

        lat_sorted = sorted(latencies)
        n = len(lat_sorted)
        print(f"\nTotal latency (ms):")
        print(f"  P50:  {lat_sorted[int(n*0.50)]:.1f}")
        print(f"  P95:  {lat_sorted[min(n-1, int(n*0.95))]:.1f}")
        print(f"  Mean: {sum(latencies)/len(latencies):.1f}")

        total_gen = sum(r.generated_tokens for r in successful)
        total_time = sum(r.total_latency_ms for r in successful) / 1000
        if total_time > 0:
            print(f"\nOutput throughput: {total_gen/total_time:.1f} tok/s (sequential)")

    if failed:
        print("\nFirst 3 errors:")
        for r in failed[:3]:
            print(f"  session={r.session_id} turn={r.turn_id}: {r.error}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Marconi JSONL traces against a live SGLang server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to Marconi JSONL trace file.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:30000",
        help="SGLang server base URL (default: http://localhost:30000).",
    )
    parser.add_argument(
        "--model",
        default="nvidia/Nemotron-H-8B-Base-8K",
        help="Model name for the completions API.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path for per-request results. "
        "Defaults to results/replay_<trace_stem>.jsonl.",
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Decode input_tokens to text via Llama-2 tokenizer before sending. "
        "Default sends token IDs directly (requires trace tokenized with "
        "the target model's tokenizer).",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=0.0,
        help="Replay speed multiplier. 0=AFAP, 1.0=real-time (default: 0).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Limit the number of requests to replay.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Cap on max_tokens per request (default: 256).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and log them without sending HTTP requests.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load trace
    logger.info("Loading trace: %s", args.trace)
    requests = load_trace(args.trace)
    logger.info("Loaded %d requests from trace.", len(requests))

    if args.max_requests is not None:
        requests = requests[: args.max_requests]
        logger.info("Truncated to %d requests.", len(requests))

    # Determine output path
    output_path = args.output
    if output_path is None:
        trace_stem = Path(args.trace).stem
        output_path = f"results/replay_{trace_stem}.jsonl"

    # Run replay
    results = asyncio.run(
        replay_trace(
            requests=requests,
            server_url=args.server_url,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            speed_factor=args.speed_factor,
            text_mode=args.text_mode,
            dry_run=args.dry_run,
        )
    )

    # Write output
    write_results(results, output_path)
    print_summary(results)


if __name__ == "__main__":
    main()
