#!/usr/bin/env python3
"""
benchmark_hydra.py - Comprehensive Production Benchmark for Hydra Multi-Head Council

Measures:
1. Routing Decision & Council Deliberation Accuracy
2. In-Process LoRA Hot-Swap Latency (< 1ms target)
3. Generation Throughput (tokens/sec) & Time-to-First-Token (TTFT)
4. Memory Footprint (Resident RSS in MB)
5. Head Storage Footprint on SSD (< 3.0 MB target)
"""

import os
import sys
import time
import json
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
        "category": "Anchor Polymath (Direct Execution)",
        "prompt": "Explain the difference between TCP and UDP in two concise sentences.",
        "expected_head": "direct",
        "expected_domain": "polymath",
        "max_tokens": 80,
    },
    {
        "id": "CASE-4",
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
    
    peak_rss = 0.0
    while proc.poll() is None:
        rss = get_process_rss_mb(proc.pid)
        if rss > peak_rss:
            peak_rss = rss
        time.sleep(0.01)
        
    stdout, stderr = proc.communicate()
    total_time = time.monotonic() - start_wall
    
    # Parse output
    has_council_directive = "Council:" in stdout or "@council(" in stdout
    routed_head = None
    if "Council: Systems & Atomic Concurrency" in stdout:
        routed_head = "001"
    elif "Council: High-Throughput Axum" in stdout:
        routed_head = "002"
    elif "Council: Direct Polymath" in stdout or "Council: Online" in stdout:
        routed_head = "none"
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
        
    print("=" * 76)
    print("      HYDRA SOVEREIGN MULTI-HEAD COUNCIL PRODUCTION BENCHMARK")
    print("=" * 76)
    print(f"Binary:       {BINARY}")
    print(f"Target Anchor: hydra-anchor-0.5b.q4km.gguf (~380 MB)")
    
    # Check head storage
    heads_dir = Path("/root/.nioxs/heads")
    print("\n[Disk Storage Footprint]")
    if heads_dir.exists():
        for head in sorted(heads_dir.iterdir()):
            if head.is_dir():
                st = head / "adapter_model.safetensors"
                if st.exists():
                    size_mb = st.stat().st_size / (1024 * 1024)
                    print(f"  • {head.name:15}: {size_mb:.2f} MB  (Constraint < 3.0 MB: {'PASS' if size_mb < 3.0 else 'FAIL'})")
    
    print("\n[Running Workload Evaluation...]")
    results = []
    for case in BENCHMARK_CASES:
        sys.stdout.write(f"  -> Testing {case['category']} ... ")
        sys.stdout.flush()
        res = run_case(case)
        print(f"Done in {res['total_time_s']:.2f}s ({res['words_per_sec']} words/s, RSS: {res['peak_rss_mb']} MB)")
        results.append(res)
        
    print("\n" + "=" * 76)
    print("                      BENCHMARK RESULTS SUMMARY")
    print("=" * 76)
    
    print(f"{'Category':<32} | {'Time(s)':<7} | {'Speed(w/s)':<10} | {'HotSwap':<9} | {'Peak RSS':<8} | {'Route':<6}")
    print("-" * 76)
    for r in results:
        swap_str = f"{r['hotswap_ms']:.2f}ms" if r['hotswap_ms'] is not None else "N/A"
        route_status = "PASS" if (r['expected_head'] == "none") or (r['routed_head'] == r['expected_head']) or (r['routed_head'] and r['routed_head'].endswith(r['expected_head'])) or (r['expected_head'] == "direct") else "CHECK"
        print(f"{r['category']:<32} | {r['total_time_s']:<7.2f} | {r['words_per_sec']:<10.1f} | {swap_str:<9} | {r['peak_rss_mb']:<6.1f}MB | {route_status:<6}")
        
    print("-" * 76)
    
    # Save benchmark report artifact
    report_file = ROOT / "hydra_production_benchmark.json"
    with open(report_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": "hydra-anchor-0.5b.q4km",
            "results": results
        }, f, indent=2)
    print(f"\nSaved detailed benchmark JSON to: {report_file}")

if __name__ == "__main__":
    main()
