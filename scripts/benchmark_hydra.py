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
    
    # Check hot-swap time and generation telemetry in stderr
    hotswap_ms = None
    tok_per_sec = None
    ttft_s = None
    tok_count = None
    gen_time_s = None
    for line in stderr.splitlines():
        if "Hot-swapped to Head" in line and "in" in line:
            try:
                parts = line.split("in ")
                ms_part = parts[1].split("ms")[0]
                hotswap_ms = float(ms_part)
            except Exception:
                pass
        if "Hydra Generation Complete:" in line:
            # Format: [DEBUG] Hydra Generation Complete: 104 tokens in 10.85s (9.58 tok/s, TTFT: 1.88s)
            try:
                parts = line.split("Hydra Generation Complete:")[1].strip()
                tok_count = int(parts.split("tokens")[0].strip())
                gen_time_s = float(parts.split("in ")[1].split("s ")[0].strip())
                tok_per_sec = float(parts.split("(")[1].split("tok/s")[0].strip())
                ttft_s = float(parts.split("TTFT:")[1].split("s")[0].strip())
            except Exception:
                pass
                
    # Count generated words
    words = len(stdout.split())
    words_per_sec = words / max(gen_time_s if gen_time_s else total_time, 0.001)
    
    return {
        "case_id": case["id"],
        "category": case["category"],
        "prompt": prompt,
        "total_time_s": round(total_time, 3),
        "peak_rss_mb": round(peak_rss, 1),
        "hotswap_ms": hotswap_ms,
        "tokens": tok_count,
        "tok_per_sec": tok_per_sec,
        "ttft_s": ttft_s,
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
    print(f"Target Anchor: hydra-anchor-180m.q4km.gguf (~242 MB, 10 layers)")
    
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
        tok_str = f"{res['tok_per_sec']} tok/s" if res['tok_per_sec'] else f"{res['words_per_sec']} w/s"
        ttft_str = f"TTFT: {res['ttft_s']}s" if res['ttft_s'] else ""
        print(f"Done in {res['total_time_s']}s ({tok_str}, {res['words_per_sec']} w/s, {ttft_str}, RSS: {res['peak_rss_mb']} MB, Swap: {swap_str}) [{status}]")
        results.append(res)
        
    print("\n" + "=" * 90)
    print("                      BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Category':<38} | {'Decode':<11} | {'Speed(w/s)':<10} | {'TTFT':<7} | {'HotSwap':<9} | {'Peak RSS':<8} | {'Route':<5}")
    print("-" * 90)
    for r in results:
        swap_str = f"{r['hotswap_ms']:.2f}ms" if r["hotswap_ms"] else "N/A"
        tok_str = f"{r['tok_per_sec']} tok/s" if r["tok_per_sec"] else "N/A"
        ttft_str = f"{r['ttft_s']}s" if r["ttft_s"] else "N/A"
        route_ok = "PASS" if r["routed_head"] == r["expected_head"] or (r["routed_head"] and r["routed_head"] in r["expected_head"]) else "FAIL"
        print(f"{r['category']:<38} | {tok_str:<11} | {r['words_per_sec']:<10} | {ttft_str:<7} | {swap_str:<9} | {r['peak_rss_mb']:<4} MB | {route_ok:<5}")
    print("-" * 90)
    
    # Save benchmark report
    out_file = ROOT / "hydra_production_benchmark.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved detailed benchmark JSON to: {out_file}")

if __name__ == "__main__":
    main()
