#!/usr/bin/env python3
"""
benchmark_64_anchors.py - Full 64-Anchor Hydra Benchmark (--cpu vs --gpu / --auto)

Measures per-anchor:
- Routing hit rate (Council NNN match against hydra_anchor_registry.json)
- True TTFT, Decode time, and Decode words/sec (via --verbose telemetry)
- Peak VmRSS (MB)
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

BIN = "/home/mn/nio-labs/nio-xs/target/release/nioxs"
REG = "/home/mn/nio-labs/nio-ideas/registry/hydra_anchor_registry.json"

TELEMETRY_RE = re.compile(
    r"\[nioxs\]\s+\((?P<dev>\w+)\)\s+TTFT:\s+(?P<ttft>[0-9.]+)s\s+\|\s+Decode:\s+(?P<dec>[0-9.]+)s\s+\|\s+(?P<words>\d+)\s+words\s+\((?P<wps>[0-9.]+)\s+w/s\)"
)
COUNCIL_RE = re.compile(r"Council\s+(\d{3})")


def get_rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def run_suite(mode: str, max_tokens: int, limit: int | None = None):
    with open(REG) as f:
        anchors = json.load(f)["anchors"]
    if limit:
        anchors = anchors[:limit]

    hits = 0
    ttfts, wps_list, rss_list = [], [], []
    misses = []

    print(f"\n=== Running {len(anchors)} Anchors | Mode: --{mode} | -n {max_tokens} ===", flush=True)
    for idx, a in enumerate(anchors, 1):
        aid = a["id"]
        triggers = [t for t in a.get("triggers", []) if t != "*"]
        kw1 = triggers[0] if triggers else a["title"]
        kw2 = triggers[1] if len(triggers) > 1 else ""
        detail = f"{kw1} and {kw2}" if kw2 else kw1
        prompt = (
            f"Write a working code example using {detail} with proper error handling."
            if aid != "000"
            else "Hello, can you help me organize my tasks for today?"
        )

        t0 = time.monotonic()
        p = subprocess.Popen(
            [BIN, "hydra", "--verbose", f"--{mode}", "-n", str(max_tokens), prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        peak = 0.0
        while p.poll() is None:
            r = get_rss_mb(p.pid)
            if r > peak:
                peak = r
            if time.monotonic() - t0 > 60:
                p.kill()
                break
            time.sleep(0.02)

        out, _ = p.communicate()
        rss_list.append(peak)

        m_c = COUNCIL_RE.search(out)
        routed = m_c.group(1) if m_c else "triage/direct"
        hit = routed == aid or (aid == "000" and routed == "triage/direct")
        if hit:
            hits += 1
        else:
            misses.append((aid, routed))

        m_t = TELEMETRY_RE.search(out)
        wps_str = "n/a"
        if m_t:
            ttfts.append(float(m_t.group("ttft")))
            wps = float(m_t.group("wps"))
            if int(m_t.group("words")) > 0:
                wps_list.append(wps)
            wps_str = f"{wps:.2f} w/s"
        print(f"  [{idx:02d}/{len(anchors)}] {aid} -> {routed:13s} | {'HIT ' if hit else 'MISS'} | {wps_str:>10s} | RSS: {peak:.1f} MB", flush=True)

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_wps = sum(wps_list) / len(wps_list) if wps_list else 0.0
    max_rss = max(rss_list) if rss_list else 0.0
    print(
        f"SUMMARY --{mode} | Hits: {hits}/{len(anchors)} | Avg TTFT: {avg_ttft:.2f}s | Avg Decode: {avg_wps:.2f} w/s | Max Peak RSS: {max_rss:.1f} MB",
        flush=True,
    )
    if misses:
        print(f"  Misses ({len(misses)}): {misses[:15]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["auto", "cpu"])
    parser.add_argument("-n", "--max-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for m in args.modes:
        run_suite(m, args.max_tokens, args.limit)
