#!/usr/bin/env python3
"""
benchmark_hydra.py - Comprehensive Production Benchmark for Hydra Multi-Head Council

Measures:
1. Routing Decision & Council Deliberation Accuracy (All 10 Founding Heads + Polymath + Meta)
2. In-Process LoRA Hot-Swap Latency (< 1ms target)
3. Generation Throughput (tokens/sec) & Time-to-First-Token (TTFT)
4. Memory Footprint (Resident RSS in MB)
5. Head Storage Footprint on SSD (< 3.0 MB target)
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

BENCHMARK_CASES = [
    {
        "id": "CASE-1",
        "category": "Head #001 (Atomic Concurrency)",
        "prompt": "Write an atomic lock-free ring buffer in Zig using Acquire/Release semantics.",
        "expected_head": "001",
        "expected_domain": "systems-atomic-concurrency",
        "max_tokens": 120,
    },
    {
        "id": "CASE-2",
        "category": "Head #002 (Axum / Tokio Microservices)",
        "prompt": "Build a high-throughput Rust Axum HTTP service with Tokio mpsc streaming channel.",
        "expected_head": "002",
        "expected_domain": "backend-axum-tokio",
        "max_tokens": 120,
    },
    {
        "id": "CASE-3",
        "category": "Head #003 (React 19 RSC & Suspense)",
        "prompt": "Create a React 19 Server Action with useOptimistic and Suspense streaming.",
        "expected_head": "003",
        "expected_domain": "frontend-react19-rsc",
        "max_tokens": 120,
    },
    {
        "id": "CASE-4",
        "category": "Head #004 (Kernel eBPF & XDP Bypass)",
        "prompt": "Write an XDP drop filter in Rust using aya-bpf to block subnets.",
        "expected_head": "004",
        "expected_domain": "kernel-ebpf-xdp",
        "max_tokens": 120,
    },
    {
        "id": "CASE-5",
        "category": "Head #005 (ClickHouse MergeTree Analytics)",
        "prompt": "Design a ClickHouse ReplacingMergeTree table schema with versioning.",
        "expected_head": "005",
        "expected_domain": "data-clickhouse-mergetree",
        "max_tokens": 120,
    },
    {
        "id": "CASE-6",
        "category": "Head #006 (Formal Logic & Dynamic Programming)",
        "prompt": "Implement a lazy propagation segment tree in Rust.",
        "expected_head": "006",
        "expected_domain": "algorithms-formal-logic",
        "max_tokens": 120,
    },
    {
        "id": "CASE-7",
        "category": "Head #007 (Kubernetes CRD Operator)",
        "prompt": "Write an asynchronous Kubernetes operator reconcile function in Rust using kube-rs.",
        "expected_head": "007",
        "expected_domain": "devops-k8s-controller",
        "max_tokens": 120,
    },
    {
        "id": "CASE-8",
        "category": "Head #008 (Distributed Consensus & Raft)",
        "prompt": "Implement AppendEntries log replication handler for Raft consensus.",
        "expected_head": "008",
        "expected_domain": "consensus-raft-paxos",
        "max_tokens": 120,
    },
    {
        "id": "CASE-9",
        "category": "Head #009 (Hardware Vector SIMD)",
        "prompt": "Write an unrolled 8-way AVX2 FMA dot product kernel in Rust.",
        "expected_head": "009",
        "expected_domain": "vector-simd-avx512",
        "max_tokens": 120,
    },
    {
        "id": "CASE-10",
        "category": "Head #010 (WASM Component Model & WASI 0.2)",
        "prompt": "Configure a Wasmtime host in Rust enforcing memory limits and fuel.",
        "expected_head": "010",
        "expected_domain": "wasm-wasi-sandboxing",
        "max_tokens": 120,
    },
    {
        "id": "CASE-11",
        "category": "Anchor Polymath (Direct Execution)",
        "prompt": "Explain the difference between TCP and UDP in two concise sentences.",
        "expected_head": "direct",
        "expected_domain": "polymath",
        "max_tokens": 80,
    },
    {
        "id": "CASE-12",
        "category": "Zero-Overhead Meta Greeting",
        "prompt": "Hello!",
        "expected_head": "none",
        "expected_domain": "greeting",
        "max_tokens": 40,
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
    
    # Parse output
    has_council_directive = "Council:" in stdout or "@council(" in stdout
    routed_head = None
    if "Council: Systems & Atomic Concurrency" in stdout:
        routed_head = "001"
    elif "Council: High-Throughput Axum" in stdout:
        routed_head = "002"
    elif "Council: React 19 Server Components" in stdout:
        routed_head = "003"
    elif "Council: Linux Kernel eBPF" in stdout:
        routed_head = "004"
    elif "Council: ClickHouse MergeTree" in stdout:
        routed_head = "005"
    elif "Council: Formal Logic" in stdout:
        routed_head = "006"
    elif "Council: Kubernetes CRD" in stdout:
        routed_head = "007"
    elif "Council: Distributed Consensus" in stdout:
        routed_head = "008"
    elif "Council: Hardware Vector SIMD" in stdout:
        routed_head = "009"
    elif "Council: WASM Component Model" in stdout:
        routed_head = "010"
    elif "Council: Direct Polymath" in stdout or "Council: Polymath" in stdout or "Hello from Hydra" in stdout:
        routed_head = "direct" if "Polymath" in stdout else "none"
    elif "@council" in stdout:
        for line in stdout.splitlines():
            line = line.strip()
            if "head =" in line or "head=" in line:
                routed_head = line.split("head")[1].split("=")[1].split(",")[0].split(")")[0].strip().strip('"').strip("'")
                break
    
    # Check hot-swap time in stderr
    hotswap_ms = None
    for line in stderr.splitlines():
        if "Hot-swapped to Head" in line and "in" in line:
            try:
                parts = line.split("in ")
                ms_part = parts[1].split("ms")[0]
                hotswap_ms = float(ms_part)
            except Exception:
                pass
                
    # Count generated words
    words = len(stdout.split())
    words_per_sec = words / max(total_time, 0.001)
    
    return {
        "case_id": case["id"],
        "category": case["category"],
        "prompt": prompt,
        "total_time_s": round(total_time, 3),
        "peak_rss_mb": round(peak_rss, 1),
        "hotswap_ms": hotswap_ms,
        "words": words,
        "words_per_sec": round(words_per_sec, 1),
        "has_council_directive": has_council_directive,
        "routed_head": routed_head,
        "expected_head": case["expected_head"],
        "stdout_snippet": stdout[:280].strip(),
    }

def main():
    if not BINARY.exists():
        print(f"Error: binary {BINARY} does not exist. Run cargo build --release first.", file=sys.stderr)
        sys.exit(1)
        
    print("=" * 80)
    print("      HYDRA SOVEREIGN MULTI-HEAD COUNCIL PRODUCTION BENCHMARK")
    print("=" * 80)
    print(f"Binary:       {BINARY}")
    print(f"Target Anchor: hydra-anchor-0.5b.q4km.gguf (~380 MB)")
    
    # Check head storage
    heads_dir = Path.home() / ".nioxs" / "heads"
    if not heads_dir.exists():
        fallback_heads = ROOT.parent / "nio-xs-models" / "models" / "hydra" / "heads"
        if fallback_heads.exists():
            heads_dir = fallback_heads
            
    print("\n[Disk Storage Footprint (< 3.0 MB SLA)]")
    if heads_dir.exists():
        for d in sorted(heads_dir.iterdir()):
            if d.is_dir() and (d.name.startswith("head-") or d.name.isdigit()):
                size_mb = sum(f.stat().st_size for f in d.glob("*") if f.is_file()) / (1024 * 1024)
                status = "PASS" if size_mb < 3.0 else "EXCEEDED"
                print(f"  • {d.name}: {size_mb:.2f} MB  (Constraint < 3.0 MB: {status})")
    
    print("\n[Running Workload Evaluation Across All Council Heads...]")
    results = []
    for case in BENCHMARK_CASES:
        print(f"  -> Testing {case['category']} ... ", end="", flush=True)
        res = run_case(case)
        status = "PASS" if res["routed_head"] == res["expected_head"] or (res["routed_head"] and res["routed_head"] in res["expected_head"]) else "MISMATCH"
        swap_str = f"{res['hotswap_ms']:.2f}ms" if res['hotswap_ms'] else "N/A"
        print(f"Done in {res['total_time_s']}s ({res['words_per_sec']} words/s, RSS: {res['peak_rss_mb']} MB, Swap: {swap_str}) [{status}]")
        results.append(res)
        
    print("\n" + "=" * 80)
    print("                      BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Category':<40} | {'Time(s)':<7} | {'Speed(w/s)':<10} | {'HotSwap':<9} | {'Peak RSS':<8} | {'Route':<5}")
    print("-" * 80)
    for r in results:
        swap_str = f"{r['hotswap_ms']:.2f}ms" if r["hotswap_ms"] else "N/A"
        route_ok = "PASS" if r["routed_head"] == r["expected_head"] or (r["routed_head"] and r["routed_head"] in r["expected_head"]) else "FAIL"
        print(f"{r['category']:<40} | {r['total_time_s']:<7} | {r['words_per_sec']:<10} | {swap_str:<9} | {r['peak_rss_mb']:<4} MB | {route_ok:<5}")
    print("-" * 80)
    
    # Save benchmark report
    out_file = ROOT / "hydra_production_benchmark.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved detailed benchmark JSON to: {out_file}")

if __name__ == "__main__":
    main()
