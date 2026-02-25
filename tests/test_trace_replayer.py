"""
Tests for the trace replayer.

Covers trace loading, prompt construction, scheduling, and dry-run replay.
Runs locally without a GPU or SGLang server.

Run:
  python -m pytest tests/test_trace_replayer.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trace_replayer import (
    TraceRequest,
    load_trace,
    build_prompt,
    compute_sleep_durations,
    write_results,
    ReplayResult,
    ServerMetricsSnapshot,
    _parse_prometheus_text,
    parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRACE = [
    {
        "session_id": 0,
        "turn_id": 0,
        "ts": 0.0,
        "num_input_tokens": 5,
        "num_output_tokens": 3,
        "input_tokens": [1, 2, 3, 4, 5],
        "output_tokens": [10, 11, 12],
    },
    {
        "session_id": 0,
        "turn_id": 1,
        "ts": 2.5,
        "num_input_tokens": 10,
        "num_output_tokens": 4,
        "input_tokens": [1, 2, 3, 4, 5, 10, 11, 12, 6, 7],
        "output_tokens": [20, 21, 22, 23],
    },
    {
        "session_id": 1,
        "turn_id": 0,
        "ts": 1.0,
        "num_input_tokens": 4,
        "num_output_tokens": 2,
        "input_tokens": [1, 2, 8, 9],
        "output_tokens": [30, 31],
    },
]


@pytest.fixture
def trace_file(tmp_path: Path) -> Path:
    """Write a small synthetic trace to a temporary JSONL file."""
    p = tmp_path / "test_trace.jsonl"
    with open(p, "w") as f:
        for req in SAMPLE_TRACE:
            json.dump(req, f)
            f.write("\n")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_trace(trace_file: Path):
    """Loading a JSONL trace produces the right number of TraceRequest objects."""
    requests = load_trace(trace_file)
    assert len(requests) == 3
    assert requests[0].session_id == 0
    assert requests[0].turn_id == 0
    assert requests[0].input_tokens == [1, 2, 3, 4, 5]
    assert requests[0].num_output_tokens == 3


def test_load_trace_missing_file():
    """Loading a non-existent trace raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_trace("/nonexistent/trace.jsonl")


def test_load_trace_fields(trace_file: Path):
    """All required fields are parsed correctly."""
    requests = load_trace(trace_file)
    r = requests[1]
    assert r.session_id == 0
    assert r.turn_id == 1
    assert r.ts == 2.5
    assert r.num_input_tokens == 10
    assert r.num_output_tokens == 4
    assert r.output_tokens == [20, 21, 22, 23]


def test_build_prompt_token_ids_mode():
    """In default (token-ids) mode, build_prompt returns the raw token list."""
    req = TraceRequest(
        session_id=0,
        turn_id=0,
        ts=0.0,
        num_input_tokens=3,
        num_output_tokens=1,
        input_tokens=[100, 200, 300],
        output_tokens=[400],
    )
    prompt = build_prompt(req, text_mode=False)
    assert prompt == [100, 200, 300]
    assert isinstance(prompt, list)


def test_schedule_no_delay():
    """Speed factor 0 should produce zero delays."""
    reqs = [
        TraceRequest(0, 0, 0.0, 5, 3, [1], [2]),
        TraceRequest(0, 1, 5.0, 5, 3, [1], [2]),
        TraceRequest(1, 0, 10.0, 5, 3, [1], [2]),
    ]
    durations = compute_sleep_durations(reqs, speed_factor=0.0)
    assert all(d == 0.0 for d in durations)


def test_schedule_realtime():
    """Speed factor 1.0 should reproduce trace timestamp deltas."""
    reqs = [
        TraceRequest(0, 0, 0.0, 5, 3, [1], [2]),
        TraceRequest(0, 1, 2.0, 5, 3, [1], [2]),
        TraceRequest(1, 0, 5.0, 5, 3, [1], [2]),
    ]
    durations = compute_sleep_durations(reqs, speed_factor=1.0)
    assert durations[0] == 0.0
    assert abs(durations[1] - 2.0) < 1e-9
    assert abs(durations[2] - 3.0) < 1e-9


def test_schedule_fast_forward():
    """Speed factor 0.5 should halve the delays."""
    reqs = [
        TraceRequest(0, 0, 0.0, 5, 3, [1], [2]),
        TraceRequest(0, 1, 4.0, 5, 3, [1], [2]),
    ]
    durations = compute_sleep_durations(reqs, speed_factor=0.5)
    assert abs(durations[1] - 2.0) < 1e-9


def test_schedule_empty():
    """Empty request list produces empty durations."""
    assert compute_sleep_durations([], speed_factor=1.0) == []


def test_write_results(tmp_path: Path):
    """Results are written as valid JSONL."""
    results = [
        ReplayResult(
            session_id=0,
            turn_id=0,
            ts=0.0,
            num_input_tokens=5,
            num_output_tokens=3,
            prompt_len=5,
            ttft_ms=12.5,
            total_latency_ms=50.0,
            generated_tokens=3,
        ),
        ReplayResult(
            session_id=1,
            turn_id=0,
            ts=1.0,
            num_input_tokens=4,
            num_output_tokens=2,
            prompt_len=4,
            ttft_ms=8.0,
            total_latency_ms=30.0,
            generated_tokens=2,
            error=None,
        ),
    ]
    output = tmp_path / "results.jsonl"
    write_results(results, output)

    lines = output.read_text().strip().split("\n")
    assert len(lines) == 2

    obj = json.loads(lines[0])
    assert obj["session_id"] == 0
    assert obj["ttft_ms"] == 12.5
    assert obj["cached_tokens"] == 0
    assert obj["error"] is None


def test_dry_run_produces_results(trace_file: Path):
    """Dry run should return results without sending HTTP requests."""
    import asyncio

    from trace_replayer import replay_trace

    requests = load_trace(trace_file)
    results, server_metrics_delta = asyncio.run(
        replay_trace(
            requests=requests,
            server_url="http://localhost:99999",  # won't be contacted
            model="test-model",
            max_output_tokens=128,
            speed_factor=0.0,
            text_mode=False,
            dry_run=True,
        )
    )
    assert len(results) == 3
    assert all(r.error is None for r in results)
    assert results[0].session_id == 0
    assert results[0].num_input_tokens == 5
    assert server_metrics_delta == {}


def test_replay_result_cache_fields():
    """ReplayResult should include cache metrics and serialize them."""
    r = ReplayResult(
        session_id=0,
        turn_id=0,
        ts=0.0,
        num_input_tokens=100,
        num_output_tokens=20,
        prompt_len=100,
        ttft_ms=50.0,
        total_latency_ms=100.0,
        generated_tokens=20,
        cached_tokens=60,
        prompt_tokens=100,
        completion_tokens=20,
        cache_hit_pct=60.0,
    )
    from dataclasses import asdict

    d = asdict(r)
    assert d["cached_tokens"] == 60
    assert d["prompt_tokens"] == 100
    assert d["completion_tokens"] == 20
    assert d["cache_hit_pct"] == 60.0


def test_parse_prometheus_text():
    """Prometheus text format is parsed into a dict."""
    text = """
# HELP sglang_cache_hit_rate Cache hit rate
# TYPE sglang_cache_hit_rate gauge
sglang:cache_hit_rate 0.75
sglang:cache_hit_count 150
sglang:cache_query_count 200
# HELP other_metric Some other metric
other_metric{label="foo"} 42.5
"""
    parsed = _parse_prometheus_text(text)
    assert parsed["sglang:cache_hit_rate"] == 0.75
    assert parsed["sglang:cache_hit_count"] == 150.0
    assert parsed["sglang:cache_query_count"] == 200.0
    assert parsed["other_metric"] == 42.5


def test_directory_mode_args_mutually_exclusive():
    """--trace and --trace-dir are mutually exclusive."""
    with pytest.raises(SystemExit):
        parse_args(["--trace", "a.jsonl", "--trace-dir", "traces/"])


def test_directory_mode_requires_one():
    """At least one of --trace or --trace-dir is required."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_directory_discovery(tmp_path: Path):
    """Directory mode discovers all .jsonl files."""
    for name in ["a.jsonl", "b.jsonl", "c.txt"]:
        (tmp_path / name).write_text('{"session_id":0,"turn_id":0,"ts":0,'
                                     '"num_input_tokens":1,"num_output_tokens":1,'
                                     '"input_tokens":[1],"output_tokens":[2]}\n')
    found = sorted(tmp_path.glob("*.jsonl"))
    assert len(found) == 2
    assert found[0].name == "a.jsonl"
    assert found[1].name == "b.jsonl"


def test_no_stream_arg():
    """--no-stream flag is parsed correctly."""
    args = parse_args(["--trace", "test.jsonl", "--no-stream"])
    assert args.no_stream is True

    args = parse_args(["--trace", "test.jsonl"])
    assert args.no_stream is False
