#!/usr/bin/env python3
"""
Expands Hydra Anchor Registry to 64 Sovereign Anchors (000 to 063).
Preserves existing anchors 000-042, appends 043-063, creates corresponding
skeleton adapter directories in ~/.nioxs/anchors, and generates adapter_model.gguf.
"""

import json
import os
import shutil
from pathlib import Path

REGISTRY_PATH = Path("/home/mn/nio-labs/nio-ideas/registry/hydra_anchor_registry.json")
NIOXS_REGISTRY_PATH = Path("/home/mn/.nioxs/registry/hydra_anchor_registry.json")
ANCHORS_DIR = Path.home() / ".nioxs" / "anchors"

NEW_ANCHORS = [
    {
        "id": "043",
        "slug": "anchor-043-socratic-mentor",
        "pillar_id": 0,
        "title": "Socratic Mentor, Rubber Duck & Conversational Companion",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "mentor", "rubber duck", "explain simply", "teach me", "socratic", "why does", "help me understand", "companion"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #043: Socratic Mentor & Rubber Duck Companion. Ask clarifying questions, encourage first-principles deduction, explain subtle concepts with grounded analogies, and guide the developer gently toward optimal solutions."
    },
    {
        "id": "044",
        "slug": "anchor-044-chief-architect",
        "pillar_id": 0,
        "title": "Chief Architect, System Design & Trade-Off Matrix",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "architecture", "system design", "tradeoff", "trade-off matrix", "high level design", "rfc", "capacity planning", "scalability"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #044: Chief Architect & Trade-Off Matrix. Formulate robust distributed topologies, document CAP/PACELC trade-offs, define SLA/SLO boundaries, produce quantitative comparison matrices, and specify single-points-of-failure."
    },
    {
        "id": "045",
        "slug": "anchor-045-forensics-debugger",
        "pillar_id": 1,
        "title": "Incident Forensics, Root-Cause Debugger & Memory Dump Diagnostics",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "debug", "coredump", "core dump", "segfault", "stack overflow", "memory leak", "valgrind", "gdb", "perf", "postmortem", "incident"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #045: Incident Forensics & Root-Cause Debugger. Analyze stack traces, coredumps, ASan/TSan reports, gdb backtraces, and memory corruption patterns. Identify exact line numbers, corrupted invariants, and minimal fix patches."
    },
    {
        "id": "046",
        "slug": "anchor-046-api-rfc-designer",
        "pillar_id": 2,
        "title": "API & Protocol RFC Designer (OpenAPI, gRPC, Wire Specs)",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "openapi", "swagger", "api design", "rest rfc", "wire spec", "protobuf schema", "grpc proto", "api contract"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #046: API & Protocol RFC Designer. Author rigorous, backward-compatible schemas (OpenAPI 3.1, Protobuf v3, RFC-compliant JSON/binary specs) with strict idempotent semantics, semantic versioning, and rate-limit headers."
    },
    {
        "id": "047",
        "slug": "anchor-047-code-reviewer",
        "pillar_id": 0,
        "title": "Code Reviewer, Refactoring Specialist & Design Patterns",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "code review", "refactor", "clean code", "smell", "maintainability", "dry", "solid principles", "design patterns"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #047: Code Reviewer & Refactoring Specialist. Scrutinize pull requests for cyclomatic complexity, cognitive load, SOLID adherence, naming accuracy, DRY violations, and architectural regression. Provide diff patches."
    },
    {
        "id": "048",
        "slug": "anchor-048-security-hardener",
        "pillar_id": 1,
        "title": "Security Red-Team, Hardening Auditor & Exploit Mitigations",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "security", "exploit", "cve", "buffer overflow", "rce", "sql injection", "csrf", "xss", "hardening", "seccomp", "apparmor", "red team"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #048: Security Hardener & Red-Team Auditor. Detect OWASP Top 10 vulnerabilities, integer overflows, format string bugs, insecure deserialization, privilege escalations, and provide hardened seccomp/AppArmor policies."
    },
    {
        "id": "049",
        "slug": "anchor-049-cpp23-modern",
        "pillar_id": 1,
        "title": "Modern C++20/C++23 Concepts, Ranges, Modules & Coroutines",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "c++20", "c++23", "cpp20", "cpp23", "concepts", "ranges", "coroutine", "std::views", "std::span", "std::jthread"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #049: Modern C++20/C++23 Specialist. Output idiomatic C++20/C++23 code utilizing concepts/requires, std::ranges, coroutines (co_await/co_yield), std::jthread, constexpr reflection, and zero legacy C-style idioms."
    },
    {
        "id": "050",
        "slug": "anchor-050-swift6-concurrency",
        "pillar_id": 2,
        "title": "Swift 6 Strict Concurrency, Sendable Actors & Server Swift",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "swift 6", "swift concurrency", "sendable", "actor", "isolated", "taskgroup", "hummingbird", "vapor", "swift server"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #050: Swift 6 Strict Concurrency Specialist. Enforce compile-time data isolation, @Sendable closures, distributed actors, TaskGroups, async algorithms, and zero concurrency warnings under Swift 6 strict mode."
    },
    {
        "id": "051",
        "slug": "anchor-051-elixir-otp",
        "pillar_id": 2,
        "title": "Elixir/OTP, BEAM VM & Fault-Tolerant Actor Supervision",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "elixir", "erlang", "otp", "genserver", "beam", "supervisor tree", "phoenix liveview", "ets", "let it crash"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #051: Elixir/OTP Specialist. Architect fault-tolerant supervision trees (one_for_one, rest_for_one), GenServer states, ETS table partitioning, Phoenix LiveView streams, and pure 'let it crash' resilience."
    },
    {
        "id": "052",
        "slug": "anchor-052-embedded-riscv",
        "pillar_id": 1,
        "title": "Embedded Systems, Bare-Metal C & RISC-V Assembly",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "riscv", "risc-v", "embedded", "bare-metal", "cortex-m", "interrupt", "csr", "linker script", "memory-mapped io", "mmio", "freertos"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #052: Embedded Systems & RISC-V Bare-Metal Specialist. Write deterministic MMIO register operations, interrupt vector tables (trap handlers), linker scripts (.ld), startup assembly (crt0), and FreeRTOS tasks."
    },
    {
        "id": "053",
        "slug": "anchor-053-crypto-zkp",
        "pillar_id": 8,
        "title": "Applied Cryptography, TLS 1.3 & Zero-Knowledge Proofs",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "cryptography", "crypto", "tls 1.3", "zkp", "zero knowledge", "elliptic curve", "ed25519", "aes-gcm", "chacha20", "circom", "snark"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #053: Applied Cryptography Specialist. Implement constant-time cryptographic primitives, timing-attack-free comparisons, authenticated encryption (AEAD: AES-GCM, ChaCha20-Poly1305), Ed25519 signatures, and Circom/ZKP constraints."
    },
    {
        "id": "054",
        "slug": "anchor-054-sql-tuning",
        "pillar_id": 5,
        "title": "Advanced SQL Query Optimization & B-Tree Index Internals",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "sql optimize", "query plan", "explain analyze", "index tuning", "b-tree", "composite index", "window function", "cte tuning"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #054: SQL Tuning & Index Optimization Specialist. Analyze EXPLAIN (ANALYZE, BUFFERS) outputs, eliminate sequential scans via covering/partial indexes, optimize complex CTEs and window partitions, and eliminate N+1 latency."
    },
    {
        "id": "055",
        "slug": "anchor-055-scylladb-cassandra",
        "pillar_id": 5,
        "title": "ScyllaDB & Apache Cassandra Distributed NoSQL",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "scylladb", "cassandra", "cql", "partition key", "clustering key", "wide-column", "lwt", "nosql distribution"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #055: ScyllaDB & Cassandra Specialist. Design optimal partition and clustering key schemas for shard-per-core scalability, configure write/read consistency levels (QUORUM, LOCAL_ONE), and prevent tombstones and hot partitions."
    },
    {
        "id": "056",
        "slug": "anchor-056-vllm-serving",
        "pillar_id": 6,
        "title": "vLLM High-Throughput Engine, PagedAttention & Continuous Batching",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "vllm", "pagedattention", "continuous batching", "llm serving", "kv cache", "tensor parallel", "speculative decoding", "sglang"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #056: vLLM High-Throughput Serving Specialist. Configure PagedAttention block size, KV cache memory allocation, tensor/pipeline parallelism, continuous batching throughput, speculative drafting, and OpenAI-compatible API serving."
    },
    {
        "id": "057",
        "slug": "anchor-057-shell-bash",
        "pillar_id": 7,
        "title": "POSIX Shell, Bash Hardening & Automated Systems Administration",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "bash", "shell script", "posix shell", "sh", "set -euo pipefail", "sed", "awk", "grep script", "sysadmin script"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #057: Shell & Bash Specialist. Output bulletproof shell scripts with 'set -euo pipefail', trap cleanup handlers, safe variable quoting, POSIX compatibility checks, and portable awk/sed pipelines."
    },
    {
        "id": "058",
        "slug": "anchor-058-wireguard-mesh",
        "pillar_id": 7,
        "title": "WireGuard, eBPF Overlay & Zero-Trust Mesh Networking",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "wireguard", "vpn mesh", "tailscale", "overlay network", "cilium mesh", "zero trust network", "iptables wg", "allowedips"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #058: WireGuard & Overlay Networking Specialist. Author declarative WireGuard peer configurations, AllowedIPs routing tables, kernel MTU clamps, eBPF tunnel encapsulation, and zero-trust mutual authentication topologies."
    },
    {
        "id": "059",
        "slug": "anchor-059-distributed-consensus",
        "pillar_id": 2,
        "title": "Distributed Consensus, Raft Leader Election & Multi-Paxos",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "raft", "paxos", "consensus", "leader election", "log replication", "split brain", "fencing token", "two phase commit"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #059: Distributed Consensus Specialist. Formulate state machine replication algorithms, Raft term transitions, log matching invariants, leader lease heartbeats, snapshotting contracts, and network partition recovery."
    },
    {
        "id": "060",
        "slug": "anchor-060-wasm-component-model",
        "pillar_id": 3,
        "title": "WebAssembly Component Model, WIT Specs & Wasmtime Runtime",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "component model", "wit", "wasm-tools", "wasmtime", "wasi 0.2", "canonical abi", "bindgen wasm"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #060: Wasm Component Model Specialist. Design WIT (Wasm Interface Type) interfaces, implement guest exports and host imports, build composable Wasm components via wasm-tools, and embed Wasmtime 0.2 runtimes."
    },
    {
        "id": "061",
        "slug": "anchor-061-dsp-audio",
        "pillar_id": 1,
        "title": "Digital Signal Processing, Real-Time Audio & FFT Filters",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "dsp", "audio", "fft", "biquad filter", "real-time audio", "sample rate", "pcm", "fourier transform", "lock-free audio ring"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #061: Digital Signal Processing Specialist. Implement lock-free, zero-allocation real-time audio pipelines, biquad IIR/FIR filter equations, Cooley-Tukey FFT transforms, and SIMD-accelerated sample conversions."
    },
    {
        "id": "062",
        "slug": "anchor-062-compiler-llvm",
        "pillar_id": 8,
        "title": "Compiler Construction, AST Pratt Parsers & LLVM IR Generation",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "compiler", "parser", "ast", "pratt parser", "llvm ir", "codegen", "intermediate representation", "type checker"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #062: Compiler Construction Specialist. Write recursive-descent/Pratt expression parsers, typed AST representations, symbol tables with lexical scoping, SSA intermediate representations, and LLVM IR generation."
    },
    {
        "id": "063",
        "slug": "anchor-063-quantum-computing",
        "pillar_id": 8,
        "title": "Quantum Computing, Qiskit & Quantum Circuit Simulation",
        "rank": 32,
        "adapter_size_mb": 2.63,
        "triggers": [
            "quantum", "qiskit", "qubit", "hadamard", "cnot", "quantum circuit", "bell state", "shor algorithm", "grover"
        ],
        "slot": 1,
        "status": "ready_to_train",
        "directive": "You are Hydra Head #063: Quantum Computing Specialist. Design quantum circuits via Qiskit/Pennylane, formulate unitary gate transformations (Hadamard, CNOT, Phase shift), simulate Bell states, and evaluate quantum error mitigation."
    }
]


def expand_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        reg = json.load(f)

    existing_ids = {a["id"] for a in reg["anchors"]}
    for new_a in NEW_ANCHORS:
        if new_a["id"] not in existing_ids:
            reg["anchors"].append(new_a)

    reg["total_anchors"] = len(reg["anchors"])
    reg["architecture"] = f"vMicroLM-{len(reg['anchors'])}-Anchor-Council"

    # Save to both locations
    for p in [REGISTRY_PATH, NIOXS_REGISTRY_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
        print(f"Updated registry at {p}: {reg['total_anchors']} anchors.")

    # Create skeleton anchor directories in ~/.nioxs/anchors
    chair_dir = ANCHORS_DIR / "anchor-000-chair"
    if not chair_dir.exists():
        print(f"Error: {chair_dir} does not exist.")
        return

    st_src = chair_dir / "adapter_model.safetensors"
    gguf_src = chair_dir / "adapter_model.gguf"

    created = 0
    for a in NEW_ANCHORS:
        a_dir = ANCHORS_DIR / a["slug"]
        a_dir.mkdir(parents=True, exist_ok=True)

        # adapter_config.json
        cfg_path = a_dir / "adapter_config.json"
        if not cfg_path.exists():
            cfg_data = {
                "r": 12,
                "lora_alpha": 24.0,
                "target_modules": [
                    "down_proj", "up_proj", "v_proj", "k_proj", "q_proj", "o_proj", "gate_proj"
                ]
            }
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)

        # adapter_model.safetensors
        st_dst = a_dir / "adapter_model.safetensors"
        if not st_dst.exists() and st_src.exists():
            shutil.copyfile(st_src, st_dst)

        # adapter_model.gguf
        gguf_dst = a_dir / "adapter_model.gguf"
        if not gguf_dst.exists() and gguf_src.exists():
            shutil.copyfile(gguf_src, gguf_dst)

        created += 1

    print(f"Created/verified {created} skeleton anchor adapter directories.")


if __name__ == "__main__":
    expand_registry()
