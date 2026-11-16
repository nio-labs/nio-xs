#!/usr/bin/env python3
"""
benchmark_heads_5_to_10.py - Focused benchmark on Hydra Council Heads #005 through #010

Heads Tested:
  • Head #005: ClickHouse MergeTree Analytics & Columnar Storage
  • Head #006: Dynamic Programming & Formal SMT Invariants (Z3)
  • Head #007: Kubernetes CRD Operator & Kube-rs Controller
  • Head #008: Distributed Consensus & Raft State Machine
  • Head #009: Hardware Vector SIMD (AVX2 / AVX-512) Acceleration
  • Head #010: WebAssembly Component Model & WASI 0.2 Sandbox
"""

import os
import sys
import time
import json
import threading
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "nioxs"

CASES = [
    {
        "head_id": "005",
        "title": "Head #005: ClickHouse MergeTree Analytics",
        "prompt": "Design a ClickHouse ReplacingMergeTree table schema with versioning and monthly partitioning for immutable events.",
        "expected_head": "005",
        "expected_badge": "ClickHouse MergeTree",
        "max_tokens": 120,
    },
    {
        "head_id": "006",
        "title": "Head #006: Formal Logic & Dynamic Programming",
        "prompt": "Implement a cache-efficient lazy propagation segment tree in Rust with range addition and range min queries.",
        "expected_head": "006",
        "expected_badge": "Formal Logic",
        "max_tokens": 120,
    },
    {
        "head_id": "007",
        "title": "Head #007: Kubernetes CRD Operator (Kube-rs)",
        "prompt": "Write an asynchronous Kubernetes operator reconcile function in Rust using kube-rs with finalizer cleanup.",
        "expected_head": "007",
        "expected_badge": "Kubernetes CRD",
        "max_tokens": 120,
    },
    {
        "head_id": "008",
        "title": "Head #008: Distributed Consensus & Raft",
        "prompt": "Implement the core AppendEntries handler for a distributed Raft consensus node in Rust with term validation.",
        "expected_head": "008",
        "expected_badge": "Distributed Consensus",
        "max_tokens": 120,
    },
    {
        "head_id": "009",
        "title": "Head #009: Hardware Vector SIMD (AVX2)",
        "prompt": "Write an unrolled 8-way AVX2 FMA dot product kernel in Rust using core::arch::x86_64 intrinsics.",
        "expected_head": "009",
        "expected_badge": "Hardware Vector SIMD",
        "max_tokens": 120,
    },
    {
        "head_id": "010",
        "title": "Head #010: WASM Component Model & WASI 0.2",
        "prompt": "Configure a Wasmtime host engine in Rust enforcing 64MB memory limits, epoch interruption, and fuel consumption.",
        "expected_head": "010",
        "expected_badge": "WASM Component Model",
        "max_tokens": 120,
    },
]

def get_process_rss_mb(pid):
    try:
        status_path = Path(f"/proc/{pid}/status")
        if status_path.exists():
            for line in status_path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def run_case(case):
    prompt = case["prompt"]
    max_tokens = case["max_tokens"]
    
    cmd = [
        str(BINARY),
        "hydra",
        "--cpu",
        "-n", str(max_tokens),
        prompt
    ]
    
    env = os.environ.copy()
    env["NIOXS_DEBUG"] = "1"
    
    start_wall = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    
    stdout_chunks = []
    stderr_chunks = []
    
    def read_pipe(stream, chunks):
        for line in iter(stream.readline, ''):
            chunks.append(line)
        stream.close()

    t_out = threading.Thread(target=read_pipe, args=(proc.stdout, stdout_chunks))
    t_err = threading.Thread(target=read_pipe, args=(proc.stderr, stderr_chunks))
    t_out.start()
    t_err.start()

    peak_rss = 0.0
    while proc.poll() is None:
        rss = get_process_rss_mb(proc.pid)
        if rss > peak_rss:
            peak_rss = rss
        time.sleep(0.01)
        
    t_out.join()
    t_err.join()
    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    total_time = time.monotonic() - start_wall
    
    # Parse hot-swap time from stderr
    hotswap_ms = None
    for line in stderr.splitlines():
        if "Hot-swapped to Head" in line and "in" in line:
            try:
                parts = line.split("in ")
                ms_part = parts[1].split("ms")[0]
                hotswap_ms = float(ms_part)
            except Exception:
                pass

    # Parse route
    routed_ok = case["expected_badge"] in stdout or f"head = {case['head_id']}" in stdout
    words = len(stdout.split())
    words_per_sec = words / max(total_time, 0.001)

    # Extract think trace
    think_trace = ""
    if "<think>" in stdout and "</think>" in stdout:
        think_trace = stdout.split("<think>")[1].split("</think>")[0].strip()

    return {
        "head_id": case["head_id"],
        "title": case["title"],
        "routed_ok": routed_ok,
        "total_time_s": round(total_time, 2),
        "hotswap_ms": hotswap_ms,
        "peak_rss_mb": round(peak_rss, 1),
        "words": words,
        "words_per_sec": round(words_per_sec, 1),
        "think_trace": think_trace,
        "stdout_snippet": stdout.replace("\n\n", "\n")[:260].strip(),
    }

def main():
    print("=" * 86)
    print("      HYDRA SOVEREIGN COUNCIL: FOCUSED BENCHMARK (HEADS #005 TO #010)")
    print("=" * 86)
    print(f"Target Binary: {BINARY}")
    print(f"Test Suite:    6 Dedicated Heads | Invariants, Hot-Swap, RSS & Throughput\n")

    results = []
    for c in CASES:
        print(f"▶ Testing {c['title']} ...")
        t0 = time.time()
        res = run_case(c)
        status_badge = "✅ PASS" if res["routed_ok"] else "❌ MISMATCH"
        swap_str = f"{res['hotswap_ms']:.2f}ms" if res["hotswap_ms"] else "N/A"
        print(f"  • Route Status:   {status_badge}")
        print(f"  • Hot-Swap MMAP:  {swap_str} (Target: < 1000ms)")
        print(f"  • Total Time:     {res['total_time_s']}s ({res['words_per_sec']} w/s)")
        print(f"  • Peak RSS:       {res['peak_rss_mb']} MB (Anchor + Head resident)")
        if res["think_trace"]:
            print(f"  • Invariant:      {res['think_trace'].splitlines()[1] if len(res['think_trace'].splitlines()) > 1 else res['think_trace']}")
        print(f"  • Code Preview:\n    " + res['stdout_snippet'].split('```')[-1].replace('\n', '\n    ')[:160] + "...")
        print("-" * 86)
        results.append(res)

    print("\n" + "=" * 86)
    print("                      HEADS #005 TO #010 BENCHMARK SUMMARY")
    print("=" * 86)
    print(f"{'Head':<10} | {'Domain':<36} | {'HotSwap':<9} | {'RSS (MB)':<8} | {'Speed':<8} | {'Route':<6}")
    print("-" * 86)
    for r in results:
        swap_str = f"{r['hotswap_ms']:.2f}ms" if r["hotswap_ms"] else "N/A"
        route_str = "PASS" if r["routed_ok"] else "FAIL"
        domain = r['title'].split(': ')[1]
        print(f"Head #{r['head_id']:<4} | {domain:<36} | {swap_str:<9} | {r['peak_rss_mb']:<8} | {r['words_per_sec']:<4} w/s | {route_str:<6}")
    print("-" * 86)

    out_json = ROOT / "hydra_heads_5_to_10_benchmark.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed telemetry saved to: {out_json}")

if __name__ == "__main__":
    main()
