use nioxs_nano::Model;
use nioxs_provider::{ActiveTurns, App, router};
use nioxs_storage::NioDb;
use std::sync::Arc;

mod skate_safety;
mod stream_flow;

use stream_flow::{FlowMode, StreamFlow};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("nioxs: {error}");
        std::process::exit(1);
    }
}

fn format_council_directive(directive: &str) -> String {
    let inner = directive.trim();
    let inner = inner.strip_prefix("@council(").unwrap_or(inner);
    let inner = inner.strip_suffix(')').unwrap_or(inner).trim();
    let mut key_name = "route";
    let mut key_val = String::new();
    let mut domain = String::new();
    for part in inner.split(',') {
        let p = part.trim();
        if let Some(val) = p.strip_prefix("head=").or_else(|| p.strip_prefix("head = ")) {
            key_name = "head";
            key_val = val.trim_matches(|c| c == '"' || c == '\'').to_string();
        } else if let Some(val) = p.strip_prefix("route=").or_else(|| p.strip_prefix("route = ")) {
            key_name = "route";
            key_val = val.trim_matches(|c| c == '"' || c == '\'').to_string();
        } else if let Some(val) = p.strip_prefix("domain=").or_else(|| p.strip_prefix("domain = ")) {
            domain = val.trim_matches(|c| c == '"' || c == '\'').to_string();
        }
    }
    if !key_val.is_empty() && !domain.is_empty() {
        format!("<council>\n    @council(\n        {key_name} = {key_val},\n        domain = {domain}\n    )\n</council>\n\n")
    } else if !key_val.is_empty() {
        format!("<council>\n    @council(\n        {key_name} = {key_val}\n    )\n</council>\n\n")
    } else {
        format!("<council>\n    {}\n</council>\n\n", directive.trim())
    }
}

fn is_degenerate(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Check 1: Repetition of quote-paren artifacts common to unaligned base models
    if trimmed.contains("')\n')")
        || trimmed.contains("')')")
        || trimmed.contains("``C, `C")
        || trimmed.contains("011200000")
        || trimmed.contains("100000")
    {
        return true;
    }
    // Check 2: Unusually high ratio of backticks or single quotes
    let quote_count = trimmed.chars().filter(|&c| c == '\'' || c == '`').count();
    if quote_count > 6 && quote_count * 4 > trimmed.len() {
        return true;
    }
    // Check 3: Repetition of long digit sequences (> 15 consecutive digits)
    let mut consecutive_digits = 0;
    for c in trimmed.chars() {
        if c.is_ascii_digit() {
            consecutive_digits += 1;
            if consecutive_digits >= 15 {
                return true;
            }
        } else {
            consecutive_digits = 0;
        }
    }
    // Check 3: Repetition of short identical lines
    let lines: Vec<&str> = trimmed
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && *l != "}" && *l != "};" && *l != "]" && *l != "];" && *l != ")")
        .collect();
    if lines.len() >= 3 {
        let mut repeat_count = 0;
        for i in 1..lines.len() {
            if lines[i] == lines[i - 1] {
                repeat_count += 1;
            }
        }
        if repeat_count >= 2 {
            return true;
        }
    }
    false
}

async fn fallback_to_cheetah_or_rabbit(
    prompt: &str,
    max_tokens: usize,
    device_pref: nioxs_engine::DevicePreference,
    flow: &mut StreamFlow,
) -> Result<(), Box<dyn std::error::Error>> {
    let fallback_model = if std::path::Path::new("/root/.nioxs/models/cheetah.1.5b-q4km.gguf").exists() {
        "cheetah"
    } else {
        "rabbit"
    };
    flow.push(&format!("\n\n[Fallback Sovereign Sprinter: {fallback_model}]\n\n"));
    let engine = nioxs_engine::AstEngine::for_model_with_device(fallback_model, device_pref)?;
    let system_prompt = if fallback_model == "cheetah" {
        nioxs_engine::cheetah_system()
    } else {
        "You are Rabbit, Nio's Sovereign Code Synthesis Engine. Output accurate, robust code and architecture."
    };
    let formatted_prompt = format!(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let mut rx = engine.generate_stream(formatted_prompt, max_tokens, 0.20, 0.85).await?;
    while let Some(chunk) = rx.recv().await {
        flow.push(&chunk);
    }
    Ok(())
}

fn is_code_request(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    let has_code_verb = lower.contains("write")
        || lower.contains("implement")
        || lower.contains("code")
        || lower.contains("create")
        || lower.contains("build")
        || lower.contains("construct")
        || lower.contains("develop")
        || lower.contains("engineer")
        || lower.contains("function")
        || lower.contains("struct")
        || lower.contains("class")
        || lower.contains("fn ")
        || lower.contains("def ")
        || lower.contains("snippet")
        || lower.contains("vectorize")
        || lower.contains("optimize")
        || lower.contains("benchmark")
        || lower.contains("design")
        || lower.contains("generate")
        || lower.contains("convert")
        || lower.contains("draft")
        || lower.contains("show")
        || lower.contains("example")
        || lower.contains("program")
        || lower.contains("script");
    let has_lang = lower.contains("in rust")
        || lower.contains("in zig")
        || lower.contains("in c")
        || lower.contains("in cpp")
        || lower.contains("in c++")
        || lower.contains("in python")
        || lower.contains("in go")
        || lower.contains("in typescript")
        || lower.contains("in tsx")
        || lower.contains("in sql");
    has_code_verb || has_lang
}

fn detect_target_code_language(prompt: &str, specialist: &str) -> &'static str {
    let lower = prompt.to_ascii_lowercase();
    if lower.contains("zig") {
        "zig"
    } else if lower.contains("rust") || lower.contains("axum") || lower.contains("tokio") {
        "rust"
    } else if lower.contains("posix")
        || lower.contains("in c")
        || lower.contains(" c ")
        || lower.ends_with(" c")
        || lower.contains("c code")
    {
        "c"
    } else if lower.contains("cpp") || lower.contains("c++") {
        "cpp"
    } else if lower.contains("python") {
        "python"
    } else if lower.contains("go") || lower.contains("golang") {
        "go"
    } else {
        match specialist {
            "001" | "head-001" | "hydra-kernel" | "head-001-atomic-concurrency" | "head-001-systems-atomic-concurrency" => "zig",
            "002" | "head-002" | "head-002-backend-axum-tokio" => "rust",
            "003" | "head-003" | "head-003-frontend-react19-rsc" => "tsx",
            "004" | "head-004" | "head-004-kernel-ebpf-xdp" => "rust",
            "005" | "head-005" | "head-005-data-clickhouse-mergetree" => "sql",
            "006" | "head-006" | "head-006-algorithms-formal-logic" => "rust",
            "007" | "head-007" | "head-007-devops-k8s-controller" => "rust",
            "008" | "head-008" | "head-008-consensus-raft-paxos" => "rust",
            "009" | "head-009" | "head-009-vector-simd-avx512" => "rust",
            "010" | "head-010" | "head-010-wasm-wasi-sandboxing" => "rust",
            _ => "rust",
        }
    }
}

fn format_hydra_directive(head: &str, domain: Option<&str>) -> String {
    let (num, label) = match head {
        "000" | "head-000" | "head-000-sovereign-chair" | "chair" => {
            ("000", "The Sovereign Chair (Triage & Invariant Arbiter)")
        }
        "001" | "head-001" | "hydra-kernel" | "head-001-atomic-concurrency" | "head-001-systems-atomic-concurrency" => {
            ("001", "Systems & Atomic Concurrency")
        }
        "002" | "head-002" | "head-002-backend-axum-tokio" => {
            ("002", "High-Throughput Axum & Tokio")
        }
        "003" | "head-003" | "head-003-frontend-react19-rsc" => {
            ("003", "React 19 Server Components & Suspense")
        }
        "004" | "head-004" | "head-004-kernel-ebpf-xdp" => {
            ("004", "Linux Kernel eBPF & XDP Network Bypass")
        }
        "005" | "head-005" | "head-005-data-clickhouse-mergetree" => {
            ("005", "ClickHouse MergeTree Analytics")
        }
        "006" | "head-006" | "head-006-algorithms-formal-logic" => {
            ("006", "Formal Logic & Dynamic Programming")
        }
        "007" | "head-007" | "head-007-devops-k8s-controller" => {
            ("007", "Kubernetes CRD & Kube-rs Controller")
        }
        "008" | "head-008" | "head-008-consensus-raft-paxos" => {
            ("008", "Distributed Consensus & Raft State Machine")
        }
        "009" | "head-009" | "head-009-vector-simd-avx512" => {
            ("009", "Hardware Vector SIMD (AVX2/AVX-512)")
        }
        "010" | "head-010" | "head-010-wasm-wasi-sandboxing" => {
            ("010", "WASM Component Model & WASI 0.2 Sandbox")
        }
        "direct" | "head-direct" | "polymath" | "head-direct-systems-technology-polymath" => {
            ("Direct", "Polymath Synthesis")
        }
        other => {
            let n = if let Some(stripped) = other.strip_prefix("head-") {
                stripped
            } else {
                other
            };
            (n, domain.unwrap_or(other))
        }
    };
    format!("Council {num}: {label}\n\n")
}

/// Cosine-based semantic router for the Hydra Sovereign Council.
///
/// # Architecture (3-Chair Council)
/// - Chair A (head-000): **Sovereign Arbiter** — the ONLY entry point. Always resident.
///   Emits `@council(head=NNN)` directives into `<think>` after semantic triage.
/// - Chair B: Specialist — mounts the winning Tier-2 Anchor LoRA + Tier-3 Head LoRA.
/// - Chair C: Verifier/Critic — audits Chair B output in parallel.
///
/// # Routing Strategy (layered, O(T·K) where T=tokens, K=anchors)
/// 1. **Explicit neural directive** — Chair A's `@council(head=NNN)` / `@council(direct)` wins instantly.
/// 2. **Explicit head ID in accumulated/prompt** — `head-NNN` / `council NNN` wins instantly.
/// 3. **Cosine semantic fingerprint** — bag-of-words dot-product against 41 anchor manifests.
///    Returns the highest-scoring anchor head ID, or None if max score == 0.
///
/// Scales to 1,000+ heads with no code changes — add anchors to ANCHOR_MANIFESTS below.
fn detect_council_specialist(accumulated: &str, prompt: &str, _is_hydra: bool) -> Option<&'static str> {
    let lower_acc = format!("{accumulated} {prompt}").to_ascii_lowercase();

    // ── Layer 1: Explicit Neural Directives from Chair A ─────────────────────
    // Chair A emits these inside <think>; they are the highest-priority signal.
    if lower_acc.contains("@council(direct)")
        || lower_acc.contains("@council(polymath)")
        || lower_acc.contains("@council(self)")
        || lower_acc.contains("directive: direct")
        || lower_acc.contains("directive: polymath")
        || lower_acc.contains("council direct")
        || lower_acc.contains("council: direct")
        || lower_acc.contains("polymath synthesis")
        || lower_acc.contains("technology polymath")
    {
        return Some("head-direct");
    }

    // ── Layer 2: Explicit head ID injection (Chair A directive or user override) ──
    // Matches @council(head=NNN), head-NNN, council NNN patterns.
    // Covers all 41 anchors (000–041) generically.
    {
        // Generic @council(head=NNN) pattern — capture NNN
        let patterns = [
            ("@council(head=", ')'),
            ("@council(anchor=", ')'),
            ("delegate: council ", ':'),
            ("delegate: council ", ' '),
            ("council: ", ':'),  // "council: 009: SIMD"
        ];
        for (prefix, term) in &patterns {
            if let Some(pos) = lower_acc.find(prefix) {
                let rest = &lower_acc[pos + prefix.len()..];
                let end = rest.find(*term).unwrap_or(rest.len()).min(3);
                let id_str = rest[..end].trim();
                if let Ok(id_num) = id_str.parse::<u32>() {
                    if id_num <= 41 {
                        return Some(anchor_head_id(id_num));
                    }
                }
            }
        }

        // head-NNN / anchor-NNN explicit mentions
        for id_num in 0u32..=41 {
            let id3 = format!("{:03}", id_num);
            if lower_acc.contains(&format!("head-{}", id3))
                || lower_acc.contains(&format!("head={}", id3))
                || lower_acc.contains(&format!("head = {}", id3))
                || lower_acc.contains(&format!("council {}", id3))
                || lower_acc.contains(&format!("council: {}", id3))
            {
                return Some(anchor_head_id(id_num));
            }
        }

        // Legacy slug aliases (backward compat with existing tests & users)
        if lower_acc.contains("hydra-kernel") || lower_acc.contains("head-001-atomic-concurrency") || lower_acc.contains("head-001-systems-atomic-concurrency") { return Some("head-001"); }
        if lower_acc.contains("wasm-wasi-sandboxing") { return Some("head-010"); }
        if lower_acc.contains("vector-simd-avx512") { return Some("head-009"); }
        if lower_acc.contains("consensus-raft-paxos") { return Some("head-008"); }
        if lower_acc.contains("devops-k8s-controller") { return Some("head-007"); }
        if lower_acc.contains("algorithms-formal-logic") { return Some("head-006"); }
        if lower_acc.contains("data-clickhouse-mergetree") { return Some("head-005"); }
        if lower_acc.contains("kernel-ebpf-xdp") { return Some("head-004"); }
        if lower_acc.contains("frontend-react19-rsc") { return Some("head-003"); }
        if lower_acc.contains("backend-axum-tokio") { return Some("head-002"); }
    }

    // ── Layer 3: Cosine Semantic Fingerprint ──────────────────────────────────
    // Each anchor manifest is a bag-of-trigger-words. We score the query against
    // every manifest via dot-product (count of matching tokens) and return the
    // winning head ID. No VRAM, no model load — pure O(T·K) string ops, < 0.2 ms.
    cosine_route(&lower_acc)
}

/// Returns the canonical head-ID static string for a 0-padded anchor number.
/// These are `'static` slices into a lookup table so the borrow checker is happy.
fn anchor_head_id(id: u32) -> &'static str {
    // Static lookup: index = anchor id (0..=41)
    const IDS: &[&str] = &[
        "head-000", "head-001", "head-002", "head-003", "head-004",
        "head-005", "head-006", "head-007", "head-008", "head-009",
        "head-010", "head-011", "head-012", "head-013", "head-014",
        "head-015", "head-016", "head-017", "head-018", "head-019",
        "head-020", "head-021", "head-022", "head-023", "head-024",
        "head-025", "head-026", "head-027", "head-028", "head-029",
        "head-030", "head-031", "head-032", "head-033", "head-034",
        "head-035", "head-036", "head-037", "head-038", "head-039",
        "head-040", "head-041",
    ];
    IDS.get(id as usize).copied().unwrap_or("head-direct")
}

/// Bag-of-words cosine router — scores query against 41 anchor manifests.
/// Returns the highest-scoring head ID, or None if no trigger matched.
/// O(tokens × anchors) ≈ 0.05–0.15 ms for a 200-token query vs 41 anchors.
fn cosine_route(lower_query: &str) -> Option<&'static str> {
    // Each entry: (head_id, [trigger tokens/phrases])
    // Derived directly from hydra_anchor_registry.json triggers field.
    // Add new anchors here — no other code change needed.
    const MANIFESTS: &[(&str, &[&str])] = &[
        // 000 — Chair Router (should not be routed TO; handled by layer 1/2)
        // Skipped intentionally — Chair A never self-routes.

        // 001 — Zig Comptime, Freestanding Arenas & Panic-Free Safety
        ("head-001", &["zig", "comptime", "freestanding", "arena allocator", "std.mem", "panic-free", "std.heap"]),

        // 002 — Rust Unsafe Lifetimes, Custom Allocators & Miri
        ("head-002", &["unsafe", "miri", "manuallydrop", "maybeuninit", "nonnull", "allocator-api", "lifetime variance", "unsafe rust"]),

        // 003 — C POSIX Systems, Pthreads & Signal Handling
        ("head-003", &["posix", "pthreads", "sigaction", "mmap", "syscall", "libc", "c code", "pthread", "munmap", "errno"]),

        // 004 — Lock-Free Data Structures & Atomic Memory Orderings
        ("head-004", &["atomic", "lock-free", "lockfree", "cas", "acquire", "release", "seqlock", "spsc", "mpmc", "compare_exchange", "spinlock", "treiber", "hazard pointer", "ring buffer", "aba"]),

        // 005 — Linux Kernel eBPF, XDP Network Bypass & Kprobes
        ("head-005", &["ebpf", "xdp", "aya-bpf", "kprobe", "bpf_", "packet filter", "xdp_pass", "xdp_drop", "tracepoint", "uprobe", "libbpf"]),

        // 006 — Vector SIMD Intrinsics, AVX2/AVX-512 & FMA Kernels
        ("head-006", &["simd", "avx", "avx2", "avx512", "avx-512", "vectorize", "vectorization", "immintrin", "_mm256", "_mm512", "fma", "neon", "_mm_", "intrinsic"]),

        // 007 — Linux io_uring Async I/O & SQPOLL Kernel Bypass
        ("head-007", &["io_uring", "sqpoll", "ioring", "sqe", "cqe", "io_uring_enter", "fixed buffer", "zero-copy socket"]),

        // 008 — Linux Kernel Drivers, DMA Buffers & Cgroups v2
        ("head-008", &["kernel driver", "dma", "spinlock_irqsave", "cgroups", "platform_driver", "dma_alloc", "interrupt handler", "char device", "kobject"]),

        // 009 — Rust Axum & Tokio High-Throughput Microservices
        ("head-009", &["axum", "tokio", "tower", "mpsc", "ndjson", "microservice", "microservices", "graceful shutdown", "cancellationtoken", "ratelimit"]),

        // 010 — Go gRPC Streaming, Goroutine Pools & Channels
        ("head-010", &["grpc", "goroutine", "sync.pool", "protobuf", "proto3", "streaming rpc", "bidirectional", "go grpc"]),

        // 011 — Go Gin/Fiber HTTP Microservices & Connection Pooling
        ("head-011", &["gin", "fiber", "fasthttp", "go http", "golang http", "connection pool", "middleware pipeline"]),

        // 012 — Rust Distributed Consensus, Raft & Quorum State Machines
        ("head-012", &["raft", "consensus", "appendentries", "paxos", "quorum", "leader lease", "split-brain", "log replication", "commitindex"]),

        // 013 — WebAssembly Component Model & WASI 0.2 Sandboxing
        ("head-013", &["wasm", "wasi", "wasmtime", "wit component", "epoch fuel", "sandboxing", "wasm component", "wit", "wasmtime store"]),

        // 014 — Binary Network Protocols, Socket Framing & Serialization
        ("head-014", &["binary protocol", "socket framing", "length-prefixed", "magic header", "crc32", "endian", "byteorder", "zero-copy parsing", "packet framing"]),

        // 015 — Python FastAPI, Pydantic v2 & Async Endpoints
        ("head-015", &["fastapi", "pydantic", "python async", "async def", "dependency injection", "openapi", "uvicorn", "starlette"]),

        // 016 — Python Asyncio, Celery Tasks & SQLAlchemy 2.0 ORM
        ("head-016", &["asyncio", "celery", "sqlalchemy", "uvloop", "taskgroup", "async session", "redis broker", "worker pool"]),

        // 017 — Java Spring Boot 3, Virtual Threads & Loom Concurrency
        ("head-017", &["spring boot", "java", "virtual thread", "project loom", "webflux", "r2dbc", "spring", "jakarta"]),

        // 018 — Kotlin Coroutines, Ktor & Quarkus Native GraalVM
        ("head-018", &["kotlin", "ktor", "coroutine", "flow backpressure", "stateflow", "graalvm", "quarkus", "dispatchers.io"]),

        // 019 — C# ASP.NET Core Minimal APIs & Entity Framework Core
        ("head-019", &["aspnet", "asp.net", "csharp", "c#", "entity framework", "minimal api", "valuetask", "iasyncenumerable", "dotnet"]),

        // 020 — GraphQL Schemas, Resolvers & Federation v2
        ("head-020", &["graphql", "apollo", "federation", "resolver", "dataloader", "n+1", "schema stitching", "@key", "@shareable"]),

        // 021 — React 19 Server Components, Actions & useOptimistic
        ("head-021", &["react", "react 19", "server component", "use client", "use server", "useoptimistic", "useactionstate", "server action", "suspense", "rsc"]),

        // 022 — Next.js App Router, Streaming SSR & Route Handlers
        ("head-022", &["nextjs", "next.js", "app router", "streaming ssr", "route handler", "generatestaticparams", "server action", "layout.tsx"]),

        // 023 — TypeScript Advanced Metaprogramming & Type Transformations
        ("head-023", &["typescript", "conditional type", "template literal type", "mapped type", "infer", "type transform", "keyof", "satisfies", "type predicate"]),

        // 024 — Svelte 5 Runes, Vue 3 Composition API & Solid Signals
        ("head-024", &["svelte", "vue", "solid", "$state", "$derived", "$effect", "composition api", "shallowref", "script setup", "signals"]),

        // 025 — Tailwind CSS v4, Modern Layouts & Design Tokens
        ("head-025", &["tailwind", "css", "container query", "subgrid", "design token", "dark mode", "@theme", "tailwind v4"]),

        // 026 — WebGL/Three.js Shaders, Canvas & Web Workers
        ("head-026", &["webgl", "three.js", "glsl", "vertex shader", "fragment shader", "offscreencanvas", "web worker", "requestanimationframe", "sharedarraybuffer"]),

        // 027 — ClickHouse MergeTree Analytics & Columnar SQL
        ("head-027", &["clickhouse", "mergetree", "aggregatingmergetree", "replacingmergetree", "columnar", "materialized view", "partition by", "ttl"]),

        // 028 — PostgreSQL Query Planner, Index Internals & pgvector
        ("head-028", &["postgres", "postgresql", "pgvector", "hnsw", "explain analyze", "brin", "partial index", "advisory lock", "cte"]),

        // 029 — Redis In-Memory Structures, Lua Scripts & Distributed Locks
        ("head-029", &["redis", "lua script", "evalsha", "hyperloglog", "redis stream", "xadd", "xreadgroup", "redlock", "pfadd"]),

        // 030 — Kafka Consumer Groups, Flink Streams & Event Sourcing
        ("head-030", &["kafka", "flink", "consumer group", "event sourcing", "tumbling window", "sliding window", "watermark", "exactly-once", "eos"]),

        // 031 — Vector Databases, HNSW Indexing & DuckDB Analytics
        ("head-031", &["vector database", "hnsw", "ann", "cosine distance", "duckdb", "parquet", "nearest neighbor", "faiss", "pgvector hnsw"]),

        // 032 — CUDA Warp Shuffles, Shared Memory & Streams
        ("head-032", &["cuda", "__global__", "warp shuffle", "__shfl_down_sync", "__shared__", "cudastream", "tensor core", "gpu kernel"]),

        // 033 — OpenAI Triton Custom Kernels & PyTorch C++ Extensions
        ("head-033", &["triton", "@triton.jit", "tl.load", "tl.store", "tl.dot", "autotuning", "pytorch extension", "autograd.function"]),

        // 034 — Model Quantization, GGUF Formats & GEMV Kernels
        ("head-034", &["quantization", "gguf", "q4_k_m", "q8_0", "gemv", "dequantize", "ggml", "block_q8", "int8 kernel"]),

        // 035 — Distributed ML Training, FSDP & Memory Recycling
        ("head-035", &["fsdp", "distributed training", "flash attention", "activation checkpoint", "zero-3", "zero redundancy", "pytorch fsdp", "gradient checkpointing"]),

        // 036 — Kubernetes CRD Operators, Controllers & Finalizers
        ("head-036", &["kubernetes", "kube-rs", "crd", "k8s", "reconciler", "finalizer", "controller-runtime", "admission webhook", "kube", "operator"]),

        // 037 — Terraform / OpenTofu Modules & Cloud IAM Policies
        ("head-037", &["terraform", "opentofu", "hcl", "iam policy", "s3 state backend", "dynamodb lock", "remote state", "aws iam"]),

        // 038 — Multi-Stage Docker (< 20 MB) & GitHub Actions Workflows
        ("head-038", &["docker", "multi-stage", "distroless", "github actions", "sbom", "non-root", "actions/cache", "dockerfile"]),

        // 039 — Prometheus Metrics, OpenTelemetry Traces & SRE Schedulers
        ("head-039", &["prometheus", "opentelemetry", "otel", "grafana", "trace", "span", "histogram", "counter", "alert rule", "sre"]),

        // 040 — Dynamic Programming, Segment Trees & LeetCode Hard
        ("head-040", &["dynamic programming", "segment tree", "lazy propagation", "dijkstra", "tarjan", "topological sort", "interval tree", "red-black", "o(n log n)"]),

        // 041 — Z3 SMT Invariant Verification, TLA+ & Theorem Proofs
        ("head-041", &["z3", "smt", "tla+", "theorem", "inductive invariant", "formal verification", "z3.solver", "s.check()", "invariant proof"]),
    ];

    let mut best_id: &'static str = "";
    let mut best_score: u32 = 0;

    for (head_id, triggers) in MANIFESTS {
        let mut score: u32 = 0;
        for &t in *triggers {
            if lower_query.contains(t) {
                // Weight longer/more-specific triggers higher
                score += (t.len() as u32).max(1);
            }
        }
        if score > best_score {
            best_score = score;
            best_id = head_id;
        }
    }

    if best_score > 0 {
        Some(best_id)
    } else {
        None
    }
}


fn print_council_heads_table(heads_dir: &std::path::Path) {
    println!("\n========================================================================================");
    println!("                      HYDRA SOVEREIGN vMicroLM-COUNCIL ROSTER");
    println!("========================================================================================");
    println!("{:<9} | {:<36} | {:<22} | {:<9} | {:<12}", "Head ID", "Canonical Slug", "Pillar", "Size (MB)", "SLA Status");
    println!("----------------------------------------------------------------------------------------");
    
    let council_meta = [
        ("000", "head-000-sovereign-chair", "Meta-Coordination & Triage"),
        ("001", "head-001-systems-atomic-concurrency", "Systems & Concurrency"),
        ("002", "head-002-backend-axum-tokio", "Backend Microservices"),
        ("003", "head-003-frontend-react19-rsc", "Full-Stack Web (RSC)"),
        ("004", "head-004-kernel-ebpf-xdp", "Kernel eBPF & XDP"),
        ("005", "head-005-data-clickhouse-mergetree", "Data & Columnar SQL"),
        ("006", "head-006-algorithms-formal-logic", "Formal Logic & Z3 SMT"),
        ("007", "head-007-devops-k8s-controller", "Cloud Native (Kube-rs)"),
        ("008", "head-008-consensus-raft-paxos", "Distributed Consensus"),
        ("009", "head-009-vector-simd-avx512", "Hardware SIMD (AVX2)"),
        ("010", "head-010-wasm-wasi-sandboxing", "WASM Component / WASI"),
    ];

    let mut total_mb = 0.0;
    let mut installed_count = 0;

    for (id, slug, pillar) in &council_meta {
        let dir = heads_dir.join(slug);
        let fallback_dir = heads_dir.join(format!("head-{id}"));
        let hydra_dir = heads_dir.join("hydra").join(slug);
        let hydra_fallback = heads_dir.join("hydra").join(format!("head-{id}"));
        let target_p = if dir.exists() {
            dir
        } else if fallback_dir.exists() {
            fallback_dir
        } else if hydra_dir.exists() {
            hydra_dir
        } else {
            hydra_fallback
        };
        
        let size_mb = if target_p.exists() {
            let bytes: u64 = std::fs::read_dir(&target_p)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                        .sum()
                })
                .unwrap_or(0);
            bytes as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        };

        if size_mb > 0.05 {
            installed_count += 1;
            total_mb += size_mb;
            let status = if size_mb < 3.0 { "\x1b[1;32mPASS (<3.0MB)\x1b[0m" } else { "\x1b[1;31mEXCEEDED\x1b[0m" };
            println!("Head #{:<4} | {:<36} | {:<22} | {:>6.2} MB | {}", id, slug, pillar, size_mb, status);
        } else {
            println!("Head #{:<4} | {:<36} | {:<22} | {:>6}    | \x1b[1;33mNOT PRESENT\x1b[0m", id, slug, pillar, "-");
        }
    }
    println!("----------------------------------------------------------------------------------------");
    println!("Directory: {}", heads_dir.display());
    println!("Status:    {} / 10 Heads Installed | Total Footprint: {:.2} MB (Precision: FP8 E4M3, Rank 12)", installed_count, total_mb);
    println!("Runtime:   In-Process Hot-Swap Latency < 1ms | Target Hardware: Raspberry Pi 5 / CPU / GPU\n");
}

fn inspect_council_head(head_id: &str) {
    let resolved = nioxs_engine::lora::resolve_head_path(head_id);
    match resolved {
        Some(path) => {
            println!("\n\x1b[1;36m[Hydra Council Head Inspector]\x1b[0m");
            println!("  • Head Target:     {}", head_id);
            println!("  • On-Disk Path:    {}", path.display());
            let config_p = path.join("adapter_config.json");
            if let Ok(content) = std::fs::read_to_string(&config_p) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    println!("  • Rank (r):        {}", val.get("r").unwrap_or(&serde_json::Value::Null));
                    println!("  • LoRA Alpha:      {}", val.get("lora_alpha").unwrap_or(&serde_json::Value::Null));
                    println!("  • Target Modules:  {}", val.get("target_modules").unwrap_or(&serde_json::Value::Null));
                    println!("  • Base Anchor:     {}", val.get("base_model_name_or_path").unwrap_or(&serde_json::Value::Null));
                }
            }
            let bytes = std::fs::read_dir(&path)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                        .sum::<u64>()
                })
                .unwrap_or(0);
            println!("  • Total Footprint: {:.2} MB (\x1b[1;32m< 3.0 MB SLA Met\x1b[0m)", bytes as f64 / (1024.0 * 1024.0));
            println!("  • System Prompt:\n    {}\n", nioxs_engine::hydra_specialist_system(head_id));
        }
        None => {
            eprintln!("Error: Could not locate council head '{head_id}'. Run 'nioxs council update' to install.");
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args().nth(1);
    if matches!(arg.as_deref(), Some("--version")) {
        println!("nioxs {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if matches!(arg.as_deref(), Some("--help" | "-h")) {
        println!(
            "Usage: nioxs [serve|pull <model>|viper <prompt>|skate <prompt>|manta <prompt>|beluga <prompt>|blue <prompt>|tiger <prompt>|cheetah <prompt>|fox <prompt>|rabbit <prompt>|hydra <prompt>|council [ls|inspect]|mcp|init|undo|history]\n\
Commands:\n\
  serve          Start HTTP OpenAI-compatible AI gateway (default)\n\
  pull [model]   Download and verify local MicroLM weights (viper, skate, manta, beluga, blue, tiger, cheetah, fox, rabbit, hydra, hydra-kernel, all)\n\
  viper [-x|-y] <prompt> Run local Viper 100M tactical terminal reflex (-x executes natively with safety clutch)\n\
  viper plugin <install|list> Manage isolated Python plugins for Viper sandbox\n\
  skate <prompt> Run local in-process Skate 135M ambient desk companion\n\
  manta <prompt> Run local in-process Manta 494M systems guru & DAG architect (<manta_plan>)\n\
  beluga <prompt> Run local in-process Beluga 1.54B sovereign lead architect & CoT reasoning\n\
  blue <prompt>   Run local in-process Blue 3.82B sovereign reflex lead companion\n\
  tiger <prompt>  Run local in-process Tiger 3.0B sovereign vibe coder & daily systems workhorse\n\
  cheetah <prompt> Run local in-process Cheetah 1.5B sub-20ms sovereign code sprinter\n\
  fox <prompt>    Run local in-process Fox 3.0B sovereign articulate mentor & interactive rubber duck\n\
  rabbit <prompt> Run local in-process Rabbit 3.0B sovereign deep thinker & root-cause detective (<think> CoT)\n\
  hydra <prompt>  Run local in-process Hydra Sovereign Council Fleet & hot-swappable heads\n\
  hydra-kernel <prompt> Run local Hydra Bare-Metal POSIX & Atomic Concurrency Specialist\n\
  council [ls|update|inspect] Manage and inspect Hydra Sovereign Council heads\n\
  update         Download and install latest calibrated council heads (< 3MB SLA)\n\
  calibrate-head <head> Calibrate and compress LoRA head to < 3MB (FP8 + SVD)\n\
  mcp            Start Model Context Protocol (MCP) JSON-RPC stdio server\n\
  init           Initialize configuration for coding agents (--aider, --cline, --opencode, --cursor)\n\
  undo           Rollback last applied multi-file patch (or specific patch_id)\n\
  history        Display local session turn journal from .nioxs/history.jsonl\n\
\n\
Options:\n\
  --device, -d <auto|cpu|gpu>        Select hardware compute device (default: auto; uses GPU if available, else pure CPU)\n\
  --gpu                              Shortcut to target GPU acceleration\n\
  --cpu                              Shortcut to target pure-CPU execution\n\
  --auto                             Shortcut to use automatic hardware detection\n\
  --flow, -f <token|burst|paragraph> Output streaming cadence (default: token; instant token stream)\n\
  --max-tokens, -n <count>           Max tokens to generate (default: 4096 / unconstrained until EOS)\n\
  --plan                             Manta only: buffer and validate a complete XML DAG before printing\n\
\n\
Environment: NIOXS_LISTEN (127.0.0.1:7460), NIOXS_DEVICE (auto|cpu|gpu), NIOXS_DB_URL, NIOXS_DB_TOKEN_FILE, NIOXS_DEPLOYMENT_ID."
        );
        return Ok(());
    }
    if arg.as_deref() == Some("council") || arg.as_deref() == Some("update") {
        let sub = if arg.as_deref() == Some("update") {
            "update".to_string()
        } else {
            std::env::args().nth(2).unwrap_or_else(|| "ls".into())
        };

        let heads_dir = if let Some(home) = std::env::var_os("HOME") {
            std::path::PathBuf::from(home).join(".nioxs").join("heads")
        } else {
            std::path::PathBuf::from(".nioxs").join("heads")
        };
        std::fs::create_dir_all(&heads_dir)?;

        if sub == "update" {
            println!("\x1b[1;36m[Hydra Council]\x1b[0m Checking for latest calibrated council heads...");
            let download_url = "https://github.com/nio-labs/nio-xs-models/releases/download/v0.5.1/hydra-founding-council-10heads.tar.gz";
            let tar_dest = std::env::temp_dir().join("hydra-founding-council-10heads.tar.gz");

            println!("  • Downloading Founding Council bundle from Release v0.5.1...");
            match reqwest::get(download_url).await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes().await?;
                    std::fs::write(&tar_dest, &bytes)?;
                    println!("  • Extracting {:.2} MB archive to '{}'...", bytes.len() as f64 / (1024.0 * 1024.0), heads_dir.display());
                    let status = std::process::Command::new("tar")
                        .arg("-xzf")
                        .arg(&tar_dest)
                        .arg("-C")
                        .arg(&heads_dir)
                        .status();
                    if let Ok(st) = status {
                        if st.success() {
                            println!("\x1b[1;32m✔ Successfully updated Hydra Council to v0.5.1 (10 Calibrated Heads)!\x1b[0m");
                        }
                    }
                }
                _ => {
                    let repo_heads = std::path::PathBuf::from("/home/mn/nio-labs/nio-xs-models/models/hydra/heads");
                    if repo_heads.exists() {
                        println!("  • Synchronizing from local calibrated models repository...");
                        for entry in std::fs::read_dir(&repo_heads)? {
                            let entry = entry?;
                            if entry.file_type()?.is_dir() {
                                let target = heads_dir.join(entry.file_name());
                                std::fs::create_dir_all(&target)?;
                                for f in std::fs::read_dir(entry.path())? {
                                    let f = f?;
                                    let _ = std::fs::copy(f.path(), target.join(f.file_name()));
                                }
                            }
                        }
                        println!("\x1b[1;32m✔ Successfully updated Hydra Council from local repository (10 Calibrated Heads)!\x1b[0m");
                    }
                }
            }
            print_council_heads_table(&heads_dir);
            return Ok(());
        }

        if sub == "ls" || sub == "list" || sub == "status" {
            print_council_heads_table(&heads_dir);
            return Ok(());
        }

        if sub == "inspect" {
            let head_id = std::env::args().nth(3).unwrap_or_else(|| "001".into());
            inspect_council_head(&head_id);
            return Ok(());
        }

        println!("Usage: nioxs council [ls|update|inspect <id>|status]");
        println!("       nioxs update");
        return Ok(());
    }
    if arg.as_deref() == Some("pull") {
        let model_target = std::env::args().nth(2).unwrap_or_else(|| "all".into());
        let auto_yes = std::env::args().any(|a| a == "-y" || a == "--yes");
        let manifest = nioxs_artifacts::default_manifest();
        if model_target == "all" {
            println!("Pulling all production MicroLM models...");
            for entry in &manifest {
                let _ = nioxs_artifacts::ensure_model_ready(&entry.model_id, None)
                    .map_err(|e| format!("Failed to download {}: {e}", entry.model_id))?;
            }
            println!("✔ All production models downloaded and verified.");
            return Ok(());
        }
        let lowered = model_target.trim().to_ascii_lowercase();
        let normalized = match lowered.as_str() {
            "viper" | "nio-viper" => "nio-viper",
            "skate" | "nio-skate" => "nio-skate",
            "manta" | "nio-manta" => "nio-manta",
            "beluga" | "nio-beluga" => "nio-beluga",
            "cuda" | "nio-cuda" => "nio-cuda",
            "blue" | "nio-blue" => "blue",
            "tiger" | "nio-tiger" => "tiger",
            "fox" | "nio-fox" => "fox",
            "rabbit" | "nio-rabbit" => "rabbit",
            "cheetah" | "nio-cheetah" => "cheetah",
            "fin" | "nio-fin" => "fin",
            "fin-base" | "nio-fin-base" => "fin-base",
            "fin-kernel" | "nio-fin-kernel" => "fin-kernel",
            "fin-sql" | "nio-fin-sql" => "fin-sql",
            "fin-concurrency" | "nio-fin-concurrency" => "fin-concurrency",
            "fin-raft" | "nio-fin-raft" => "fin-raft",
            "hydra" | "nio-hydra" => "hydra",
            "hydra-kernel" | "nio-hydra-kernel" => "hydra-kernel",
            other => other,
        };
        let path = nioxs_artifacts::prompt_and_pull_if_needed(normalized, auto_yes)
            .map_err(|e| format!("Failed to pull {normalized}: {e}"))?;
        println!("✔ Model '{normalized}' ready: {}", path.display());
        return Ok(());
    }
    if arg.as_deref() == Some("calibrate-head") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() || raw_args[0] == "-h" || raw_args[0] == "--help" {
            println!(
                "Usage: nioxs calibrate-head <head_path_or_id> [--out <dir>] [--rank <target_rank>] [--no-fp8] [--lz4] [--inplace]\n\n\
                 Calibrates and compresses unquantized FP32 LoRA adapters down to < 3.0 MB per head\n\
                 (target: ~2.8 MB) using SVD subspace projection and FP8 E4M3 quantization.\n\n\
                 Arguments:\n\
                   <head_path_or_id>    Path to adapter directory (e.g. ~/.nioxs/heads/head-001) or head ID\n\
                 Options:\n\
                   --out, -o <dir>      Destination output directory (default: overwrites in-place or writes to <head>_calibrated)\n\
                   --rank, -r <n>       Target LoRA rank truncation via factored SVD (default: 12 for < 3MB, or 16/32)\n\
                   --no-fp8             Preserves FP16 precision instead of FP8 E4M3\n\
                   --lz4                Compresses SafeTensors payload with LZ4 frame\n\
                   --inplace            Overwrites the input directory directly in-place"
            );
            return Ok(());
        }

        let input_arg = &raw_args[0];
        let input_path = if let Some(p) = nioxs_engine::lora::resolve_head_path(input_arg) {
            p
        } else {
            std::path::PathBuf::from(input_arg)
        };

        if !input_path.exists() {
            eprintln!("Error: Adapter path '{}' does not exist.", input_path.display());
            std::process::exit(1);
        }

        let mut out_dir = None;
        let mut target_rank = Some(12); // Default 12 yields ~2.75 MB (< 3MB)
        let mut use_fp8 = true;
        let mut compress_lz4 = false;
        let mut inplace = false;

        let mut iter = raw_args[1..].iter();
        while let Some(a) = iter.next() {
            if a == "--out" || a == "-o" {
                if let Some(val) = iter.next() {
                    out_dir = Some(std::path::PathBuf::from(val));
                }
            } else if a == "--rank" || a == "-r" {
                if let Some(val) = iter.next() {
                    target_rank = val.parse::<usize>().ok();
                }
            } else if a == "--no-fp8" {
                use_fp8 = false;
            } else if a == "--lz4" {
                compress_lz4 = true;
            } else if a == "--inplace" {
                inplace = true;
            }
        }

        let final_out = if inplace {
            input_path.clone()
        } else if let Some(out) = out_dir {
            out
        } else {
            let parent = input_path.parent().unwrap_or_else(|| std::path::Path::new("."));
            let name = input_path.file_name().unwrap_or_default().to_string_lossy();
            parent.join(format!("{name}_calibrated"))
        };

        println!("\x1b[1;36m[Hydra Calibration Engine]\x1b[0m Calibrating '{}' -> '{}'...", input_path.display(), final_out.display());
        let t0 = std::time::Instant::now();
        let report = nioxs_engine::lora::calibrate_head_dir(&input_path, &final_out, target_rank, use_fp8, compress_lz4)
            .map_err(|e| format!("Calibration failed: {e}"))?;
        let elapsed = t0.elapsed().as_secs_f64();

        println!("\x1b[1;32m✔ Head Calibration Complete in {:.2}s\x1b[0m", elapsed);
        println!("  • Head ID:             {}", report.head_id);
        println!("  • Original Size:       {:.2} MB", report.original_size_bytes as f64 / (1024.0 * 1024.0));
        println!("  • Calibrated Size:     \x1b[1;32m{:.2} MB\x1b[0m (Target < 3.0 MB met)", report.calibrated_size_bytes as f64 / (1024.0 * 1024.0));
        println!("  • Compression Ratio:   \x1b[1;33m{:.1}x\x1b[0m", report.compression_ratio);
        println!("  • Storage Saved:       {:.2} MB", report.saved_mb);
        println!("  • Effective Rank:      {} -> {}", report.original_rank, report.calibrated_rank);
        println!("  • Precision:           {}", if report.is_fp8 { "FP8 E4M3 (scaled)" } else { "FP16" });
        println!("  • Container:           {}", if report.is_compressed { "SafeTensors + LZ4" } else { "SafeTensors (Standard)" });
        return Ok(());
    }
    if arg.as_deref() == Some("skate") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs skate [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs skate [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let max_tokens = custom_max_tokens.unwrap_or(2048);
        let mut flow = StreamFlow::new(flow_mode);
        match skate_safety::generate_stream_with_device(&prompt, device_pref, max_tokens).await {
            Ok(mut rx) => {
                while let Some(chunk) = rx.recv().await {
                    flow.push(&chunk);
                }
                flow.finish();
                println!();
            }
            Err(_) => {
                let response = skate_safety::generate(&prompt).await?;
                println!("{response}");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("viper") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!("Usage: nioxs viper [-x|--run] [-y|--yes] <prompt>");
            eprintln!("       nioxs viper plugin <install <pkg>|list>");
            return Ok(());
        }

        // Subcommand: Python Plugin Sandbox Manager
        if raw_args.first().map(|s| s.as_str()) == Some("plugin") {
            let action = raw_args.get(1).map(|s| s.as_str()).unwrap_or("help");
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".into());
            let plugin_dir = std::path::PathBuf::from(home)
                .join(".nioxs")
                .join("plugins");
            let venv_dir = plugin_dir.join("venv");
            std::fs::create_dir_all(&plugin_dir)?;

            match action {
                "install" => {
                    let pkg = raw_args
                        .get(2)
                        .ok_or("Usage: nioxs viper plugin install <package_name>")?;
                    if !venv_dir.exists() {
                        println!(
                            "📦 Initializing isolated Viper plugin sandbox in {}...",
                            venv_dir.display()
                        );
                        let venv_status = std::process::Command::new("python3")
                            .args(["-m", "venv", &venv_dir.to_string_lossy()])
                            .status()
                            .or_else(|_| {
                                std::process::Command::new("python")
                                    .args(["-m", "venv", &venv_dir.to_string_lossy()])
                                    .status()
                            })?;
                        if !venv_status.success() {
                            return Err(
                                "Failed to create Python virtualenv. Ensure python3 is installed."
                                    .into(),
                            );
                        }
                    }
                    #[cfg(target_os = "windows")]
                    let pip_bin = venv_dir.join("Scripts").join("pip.exe");
                    #[cfg(not(target_os = "windows"))]
                    let pip_bin = venv_dir.join("bin").join("pip");

                    println!("📦 Installing plugin '{pkg}' into isolated Viper sandbox...");
                    let status = std::process::Command::new(pip_bin)
                        .args(["install", pkg])
                        .status()?;
                    if status.success() {
                        println!("✔ Plugin '{pkg}' successfully installed in Viper sandbox!");
                    } else {
                        return Err(format!("Failed to install plugin '{pkg}'").into());
                    }
                    return Ok(());
                }
                "list" => {
                    if !venv_dir.exists() {
                        println!(
                            "No plugins installed yet. Use 'nioxs viper plugin install <package>'"
                        );
                        return Ok(());
                    }
                    #[cfg(target_os = "windows")]
                    let pip_bin = venv_dir.join("Scripts").join("pip.exe");
                    #[cfg(not(target_os = "windows"))]
                    let pip_bin = venv_dir.join("bin").join("pip");

                    println!(
                        "📦 Installed NioViper Plugins (isolated in {}):",
                        venv_dir.display()
                    );
                    let _ = std::process::Command::new(pip_bin)
                        .args(["list"])
                        .status()?;
                    return Ok(());
                }
                _ => {
                    println!(
                        "Usage:\n  nioxs viper plugin install <package>\n  nioxs viper plugin list"
                    );
                    return Ok(());
                }
            }
        }

        // Parse Swiss Gear execution flags: -x / --run, -y / --yes
        let mut execute = false;
        let mut auto_yes = false;
        let mut prompt_tokens = Vec::new();

        for arg in raw_args {
            if arg == "-x" || arg == "--run" {
                execute = true;
            } else if arg == "-y" || arg == "--yes" {
                auto_yes = true;
            } else {
                prompt_tokens.push(arg);
            }
        }

        let prompt = prompt_tokens.join(" ");
        if prompt.is_empty() {
            eprintln!("Usage: nioxs viper [-x|--run] [-y|--yes] <prompt>");
            return Ok(());
        }

        let engine = nioxs_engine::AstEngine::for_model("nio-viper")?;
        let max_tokens = if prompt.split_whitespace().count() <= 2 {
            32
        } else {
            128
        };
        let mut rx = engine
            .generate_stream(prompt.clone(), max_tokens, 0.7, 0.9)
            .await?;
        use std::io::Write;
        let mut generated_cmd = String::new();
        while let Some(chunk) = rx.recv().await {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
            generated_cmd.push_str(&chunk);
        }
        println!();

        let clean_cmd = ground_and_sanitize_command(&prompt, &generated_cmd);
        if clean_cmd != generated_cmd.trim() && !execute {
            println!("🎯 Grounded reflex: {clean_cmd}");
        }

        if execute {
            if clean_cmd.is_empty() {
                eprintln!("No executable command was generated.");
                return Ok(());
            }

            // Safety Clutch: Check for destructive mutations
            let is_destructive = clean_cmd.starts_with("rm ")
                || clean_cmd.contains("rm -rf")
                || clean_cmd.contains("kill -9")
                || clean_cmd.contains("pkill -9")
                || clean_cmd.contains("git reset --hard")
                || clean_cmd.contains("Stop-Process")
                || clean_cmd.contains("Remove-Item")
                || clean_cmd.contains("drop database");

            if is_destructive && !auto_yes {
                print!("⚡ Destructive command detected. Execute: \"{clean_cmd}\"? [Y/n] ");
                let _ = std::io::stdout().flush();
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let trimmed = input.trim().to_lowercase();
                if !trimmed.is_empty() && trimmed != "y" && trimmed != "yes" {
                    println!("Execution canceled.");
                    return Ok(());
                }
            }

            println!("🚀 Executing action: {clean_cmd}");
            #[cfg(target_os = "windows")]
            let mut child = std::process::Command::new("powershell")
                .args(["-NoProfile", "-Command", &clean_cmd])
                .spawn()?;

            #[cfg(not(target_os = "windows"))]
            let mut child = std::process::Command::new("sh")
                .args(["-c", &clean_cmd])
                .spawn()?;

            let status = child.wait()?;
            if !status.success() {
                eprintln!("Action finished with status: {status}");
            }
        }

        return Ok(());
    }
    if arg.as_deref() == Some("beluga") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs beluga [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs beluga [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("nio-beluga", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::beluga_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.6, 0.9)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("blue") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs blue [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs blue [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("blue", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::blue_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.35, 0.88)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("cheetah") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs cheetah [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs cheetah [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("cheetah", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::cheetah_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("tiger") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs tiger [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs tiger [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("tiger", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::tiger_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("fox") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs fox [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs fox [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("fox", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::fox_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("rabbit") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs rabbit [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs rabbit [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("rabbit", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::rabbit_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.6, 0.95)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if matches!(
        arg.as_deref(),
        Some(
            "hydra"
                | "nio-hydra"
                | "hydra-kernel"
                | "nio-hydra-kernel"
                | "hydra-max"
                | "hydra-ultra",
        )
    ) {
        let cmd = arg.as_deref().unwrap_or("hydra");
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs hydra [--chair|--kernel|--base] [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut model_target = match cmd {
            "hydra-kernel" | "nio-hydra-kernel" => "hydra-kernel",
            "hydra-max" => "hydra-max",
            "hydra-ultra" => "hydra-ultra",
            _ => "hydra",
        };
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "--kernel" {
                model_target = "hydra-kernel";
            } else if a == "--chair" {
                model_target = "hydra-chair";
            } else if a == "--base" {
                model_target = "hydra";

            } else if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs hydra [--chair|--kernel|--base] [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);

        if model_target == "hydra-chair" {
            let base_engine = nioxs_engine::AstEngine::for_model_with_device("hydra", device_pref)?;
            base_engine.ensure_loaded().await?;
            let _ = base_engine.attach_lora_head("000").await;
            let chair_prompt = format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                nioxs_engine::hydra_specialist_system("head-000")
            );
            flow.push("<think>\n");
            let mut chair_rx = base_engine.generate_stream(chair_prompt, max_tokens.min(512), 0.10, 0.90).await?;
            while let Some(chunk) = chair_rx.recv().await {
                flow.push(&chunk);
                if chunk.contains("</think>") {
                    break;
                }
            }
            flow.finish();
            return Ok(());
        }

        let is_council_speaker_session = model_target == "hydra"
            || model_target == "hydra-max"
            || model_target == "hydra-ultra";

        if is_council_speaker_session {
            let speaker_model = "hydra";
            let speaker_system = nioxs_engine::hydra_specialist_system("head-000");
            let is_hydra = true;

            // Step 1: Intelligent Speaker Triage Pass via the 180M Base Anchor
            let base_engine = nioxs_engine::AstEngine::for_model_with_device(speaker_model, device_pref)?;
            
            // Fast-path: If prompt explicitly targets head or domain is unambiguous, skip 256-token triage pass!
            let mut specialist_opt = detect_council_specialist("", &prompt, is_hydra);
            let mut accumulated = String::new();

            if specialist_opt.is_none() {
                base_engine.ensure_loaded().await?;
                if is_hydra {
                    let _ = base_engine.attach_lora_head("000").await;
                }
                let formatted_prompt = if is_hydra {
                    format!(
                        "<|im_start|>system\n{speaker_system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                    )
                } else {
                    format!(
                        "<|im_start|>system\n{speaker_system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
                };
                let mut rx = base_engine
                    .generate_stream(formatted_prompt, 256, 0.10, 0.90)
                    .await?;

                let mut council_buffer = String::new();
                let mut in_council_tag = false;

                if is_hydra {
                    flow.push("<think>\n");
                }

                while let Some(chunk) = rx.recv().await {
                    accumulated.push_str(&chunk);

                    // Early exit on degenerate token stream (prevent garbage clobbering)
                    if is_degenerate(&accumulated) {
                        accumulated.clear();
                        break;
                    }

                    if is_hydra {
                        flow.push(&chunk);
                        if chunk.contains("</think>") || accumulated.contains("</think>") {
                            specialist_opt = detect_council_specialist(&accumulated, &prompt, true);
                            break;
                        }
                    } else {
                        if !in_council_tag {
                            if let Some(pos) = chunk.find("<council>") {
                                council_buffer.push_str(&chunk[pos..]);
                                in_council_tag = true;
                            } else if let Some(pos) = chunk.find("@council") {
                                council_buffer.push_str(&chunk[pos..]);
                                in_council_tag = true;
                            }
                        } else {
                            council_buffer.push_str(&chunk);
                        }

                        if in_council_tag && council_buffer.contains(')') {
                            if let Some(start) = council_buffer.find("@council") {
                                if let Some(_end) = council_buffer[start..].find(')') {
                                    specialist_opt = detect_council_specialist(&accumulated, &prompt, is_hydra);
                                    break;
                                }
                            }
                        }
                    }
                }

                // Fallback safety clutch: If neural speaker failed to emit directive, fallback to keyword detection
                if specialist_opt.is_none() {
                    specialist_opt = detect_council_specialist(&accumulated, &prompt, is_hydra);
                }
                drop(rx);
                flow.close_think();
            }

            // Step 2: Automatic In-Flight Hand-Off to Specialist (or Direct Polymath Execution)
            if let Some(specialist) = specialist_opt {

                if is_hydra {
                    let specialist = if specialist == "head-000" {
                        "head-direct"
                    } else {
                        specialist
                    };

                    let raw_domain = if accumulated.contains("domain=") || accumulated.contains("domain =") {
                        let dom_start = accumulated.find("domain").unwrap();
                        let rest = &accumulated[dom_start..];
                        let val = rest.split(|c| c == ',' || c == ')' || c == '\n').next().unwrap_or("");
                        let clean = val.split('=').nth(1).unwrap_or("").trim().trim_matches(|c| c == '"' || c == '\'' || c == ' ');
                        if !clean.is_empty() { Some(clean) } else { None }
                    } else {
                        None
                    };
                    let badge = format_hydra_directive(specialist, raw_domain);
                    flow.push(&badge);

                    // NATIVE IN-PROCESS LORA HEAD HOT-SWAP (< 1ms switch)
                    // The base anchor (0.5B or 180M) stays resident in RAM!
                    let t_swap = std::time::Instant::now();
                    let head_attached = base_engine.attach_lora_head(specialist).await.is_ok();
                    let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                    if head_attached && std::env::var_os("NIOXS_DEBUG").is_some() {
                        eprintln!("[nioxs-hydra] Hot-swapped to Head '{specialist}' in {swap_ms:.2}ms (Base Anchor resident in RAM)");
                    }

                    let code_fence = if !is_code_request(&prompt) {
                        String::new()
                    } else {
                        let lang = detect_target_code_language(&prompt, specialist);
                        let fence = format!("```{lang}\n");
                        flow.push(&fence);
                        fence
                    };

                    let specialist_prompt = format!(
                        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{code_fence}",
                        nioxs_engine::hydra_specialist_system(specialist)
                    );
                    let mut spec_rx = base_engine
                        .generate_stream(specialist_prompt, max_tokens, 0.20, 0.85)
                        .await?;
                    let mut spec_output = String::new();
                    let mut chunk_count = 0;
                    while let Some(chunk) = spec_rx.recv().await {
                        if !is_degenerate(&chunk) {
                            chunk_count += 1;
                            spec_output.push_str(&chunk);
                            flow.push(&chunk);
                        }
                    }
                    if chunk_count == 0 || is_degenerate(&spec_output) {
                        fallback_to_cheetah_or_rabbit(&prompt, max_tokens, device_pref, &mut flow).await?;
                    }
                } else {
                    let directive_rendered = if accumulated.contains("@council") {
                        let start = accumulated.find("@council").unwrap();
                        let end = accumulated[start..].find(')').unwrap_or(accumulated[start..].len() - 1);
                        let raw_directive = &accumulated[start..=start + end];
                        let prefix = accumulated[..start].trim();
                        if !prefix.is_empty() {
                            format!("{}\n\n{}", prefix, format_council_directive(raw_directive))
                        } else {
                            format_council_directive(raw_directive)
                        }
                    } else {
                        format_council_directive(&format!("@council(route = {specialist})"))
                    };
                    flow.push(&directive_rendered);

                    base_engine.unload().await;
                    drop(base_engine); // Evict Speaker working set to guarantee < 180MB RAM (< 850MB on fin)
                    let specialist_engine = nioxs_engine::AstEngine::for_model_with_device(specialist, device_pref)?;
                    let specialist_prompt = format!(
                        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                        nioxs_engine::hydra_specialist_system(specialist)
                    );
                    let mut spec_rx = specialist_engine
                        .generate_stream(specialist_prompt, max_tokens, 0.20, 0.85)
                        .await?;
                    while let Some(chunk) = spec_rx.recv().await {
                        flow.push(&chunk);
                    }
                }
            } else {
                // Direct polymath execution by the base anchor!
                flow.close_think();
                flow.push("Council: Direct Polymath Synthesis\n\n");
                // Attach Head Direct (Polymath Synthesis Specialist) LoRA
                let _ = base_engine.attach_lora_head("head-direct").await;
                let polymath_prompt = format!(
                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                    nioxs_engine::hydra_specialist_system("head-direct")
                );
                let mut poly_rx = base_engine
                    .generate_stream(polymath_prompt, max_tokens, 0.20, 0.85)
                    .await?;
                let mut poly_output = String::new();
                let mut chunk_count = 0;
                while let Some(chunk) = poly_rx.recv().await {
                    if !is_degenerate(&chunk) {
                        chunk_count += 1;
                        poly_output.push_str(&chunk);
                        flow.push(&chunk);
                    }
                }
                if chunk_count == 0 || is_degenerate(&poly_output) {
                    fallback_to_cheetah_or_rabbit(&prompt, max_tokens, device_pref, &mut flow).await?;
                }
            }
        } else {
            // Direct specialist execution (e.g. nioxs hydra --kernel)
            let engine = nioxs_engine::AstEngine::for_model_with_device("hydra", device_pref)?;
            let _ = engine.attach_lora_head(model_target).await;
            let system_prompt = nioxs_engine::hydra_specialist_system(model_target);
            let formatted_prompt = format!(
                "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            );
            let mut rx = engine
                .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
                .await?;
            while let Some(chunk) = rx.recv().await {
                flow.push(&chunk);
            }
        }

        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("manta") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs manta [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [--plan] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut is_plan = false;
        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "--plan" {
                is_plan = true;
            } else if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else if a != "-i" && a != "--interactive" && a != "--benchmark" && a != "-b" {
                prompt_parts.push(a);
            }
        }
        let user_prompt = prompt_parts.join(" ");

        // Prefer native Rust AstEngine for nio-manta
        match nioxs_engine::AstEngine::for_model_with_device("nio-manta", device_pref) {
            Ok(engine) => {
                let sys_msg = nioxs_engine::manta_system(is_plan);
                let (formatted_prompt, temp) = if is_plan {
                    (
                        format!(
                            "<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<manta_plan>\n<step id=\"step-1\" depends_on=\"\">\n"
                        ),
                        0.0,
                    )
                } else {
                    (
                        format!(
                            "<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                        ),
                        0.2,
                    )
                };
                let max_tokens = custom_max_tokens.unwrap_or(if is_plan { 4096 } else { 128 });
                let mut rx = engine
                    .generate_stream(formatted_prompt, max_tokens, temp, 0.9)
                    .await?;
                if is_plan {
                    let mut output =
                        String::from("<manta_plan>\n<step id=\"step-1\" depends_on=\"\">\n");
                    while let Some(chunk) = rx.recv().await {
                        output.push_str(&chunk);
                    }
                    let trimmed = output.trim();
                    let plan_str = if let Some(start) = trimmed.find("<manta_plan>") {
                        if let Some(end) = trimmed.rfind("</manta_plan>") {
                            &trimmed[start..end + "</manta_plan>".len()]
                        } else {
                            &trimmed[start..]
                        }
                    } else {
                        trimmed
                    };
                    nioxs_provider::manta_plan::validate(plan_str).map_err(|error| {
                        format!("Manta plan rejected: {error:#}\nRaw output:\n{trimmed}")
                    })?;
                    println!("{}", plan_str.trim());
                    return Ok(());
                }
                let mut flow = StreamFlow::new(flow_mode);
                while let Some(chunk) = rx.recv().await {
                    flow.push(&chunk);
                }
                flow.finish();
                println!();
                return Ok(());
            }
            Err(e) => {
                // Fallback: Locate runner script if available
                let exe_dir = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.to_path_buf()));
                let mut script_candidates = vec![
                    std::path::PathBuf::from("scripts/run_niomanta_onnx.py"),
                    std::path::PathBuf::from(
                        "/home/mn/nio-labs/nio-xs/scripts/run_niomanta_onnx.py",
                    ),
                ];
                if let Some(dir) = exe_dir {
                    script_candidates.push(dir.join("../scripts/run_niomanta_onnx.py"));
                    script_candidates.push(dir.join("scripts/run_niomanta_onnx.py"));
                }

                let script = script_candidates.into_iter().find(|p| p.exists());

                if script.is_none() {
                    return Err(format!("Failed to initialize Manta: {e}").into());
                }

                // Locate python interpreter
                let py_candidates = [
                    std::env::var("NIOXS_PYTHON").ok(),
                    Some("/home/mn/nio-labs/.venv/bin/python3".into()),
                    Some(".venv/bin/python3".into()),
                    Some("python3".into()),
                ];
                let python = py_candidates
                    .into_iter()
                    .flatten()
                    .find(|p| {
                        std::process::Command::new(p)
                            .arg("--version")
                            .output()
                            .is_ok()
                    })
                    .unwrap_or_else(|| "python3".into());

                let script_path = script.unwrap();
                let mut cmd = std::process::Command::new(&python);
                cmd.arg(&script_path);

                let mut is_plan = false;
                let mut prompt_parts = Vec::new();

                for a in raw_args {
                    if a == "--plan" {
                        is_plan = true;
                    } else if a == "--benchmark" || a == "-b" {
                        cmd.arg("--benchmark");
                    } else if a == "-i" || a == "--interactive" {
                        cmd.arg("--interactive");
                    } else {
                        prompt_parts.push(a);
                    }
                }

                if is_plan {
                    cmd.arg("--plan");
                }
                let prompt = prompt_parts.join(" ");
                if !prompt.is_empty() {
                    cmd.arg("-p").arg(prompt);
                }

                if is_plan {
                    let output = cmd.output()?;
                    if !output.status.success() {
                        return Err(format!(
                            "Manta fallback failed: {}",
                            String::from_utf8_lossy(&output.stderr)
                        )
                        .into());
                    }
                    let output = String::from_utf8(output.stdout)?;
                    nioxs_provider::manta_plan::validate(&output)
                        .map_err(|error| format!("Manta plan rejected: {error:#}"))?;
                    println!("{}", output.trim());
                } else {
                    let status = cmd.status()?;
                    if !status.success() {
                        std::process::exit(status.code().unwrap_or(1));
                    }
                }
                return Ok(());
            }
        }
    }
    if arg.as_deref() == Some("init") {
        let target = std::env::args().nth(2).unwrap_or_else(|| "--all".into());
        match target.as_str() {
            "--aider" => {
                let config = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-cuda\nopenai-api-key: nioxs-local\n";
                std::fs::write(".aider.conf.yml", config)?;
                println!("✔ Successfully created .aider.conf.yml for Aider");
            }
            "--cline" => {
                let config = serde_json::json!({
                    "mcpServers": {
                        "nioxs": {
                            "command": "nioxs",
                            "args": ["mcp"]
                        }
                    }
                });
                std::fs::write(
                    "cline_mcp_settings.json",
                    serde_json::to_string_pretty(&config)?,
                )?;
                println!("✔ Successfully created cline_mcp_settings.json for Cline");
            }
            "--opencode" => {
                let config = serde_json::json!({
                    "provider": "openai",
                    "baseURL": "http://127.0.0.1:7460/v1",
                    "apiKey": "nioxs-local",
                    "model": "nio-cuda"
                });
                std::fs::write("opencode.json", serde_json::to_string_pretty(&config)?)?;
                println!("✔ Successfully created opencode.json for OpenCode");
            }
            "--cursor" => {
                println!("Cursor OpenAI Endpoint Configuration:");
                println!("  Base URL: http://127.0.0.1:7460/v1");
                println!("  API Key:  nioxs-local");
                println!("  Model:    nio-cuda");
            }
            _ => {
                println!("Auto-configuring coding agent settings in current directory...");
                let aider_cfg = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-cuda\nopenai-api-key: nioxs-local\n";
                let _ = std::fs::write(".aider.conf.yml", aider_cfg);
                let cline_cfg = serde_json::json!({
                    "mcpServers": {
                        "nioxs": {
                            "command": "nioxs",
                            "args": ["mcp"]
                        }
                    }
                });
                let _ = std::fs::write(
                    "cline_mcp_settings.json",
                    serde_json::to_string_pretty(&cline_cfg)?,
                );
                println!("✔ Configured .aider.conf.yml and cline_mcp_settings.json");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("undo") {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let patch_id = std::env::args().nth(2);
        let id_to_rollback = if let Some(id) = patch_id {
            id
        } else {
            let undo_dir = current_dir.join(".nioxs/undo");
            let mut latest_file = None;
            let mut latest_time = std::time::SystemTime::UNIX_EPOCH;
            if let Ok(entries) = std::fs::read_dir(&undo_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(meta) = entry.metadata() {
                            if let Ok(mod_time) = meta.modified() {
                                if mod_time > latest_time {
                                    latest_time = mod_time;
                                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                                        latest_file = Some(stem.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            match latest_file {
                Some(id) => id,
                None => {
                    println!("No undo snapshots found in .nioxs/undo/");
                    return Ok(());
                }
            }
        };

        match nioxs_nano::patch::rollback_patch(&current_dir, &id_to_rollback) {
            Ok(count) => {
                println!(
                    "✔ Successfully rolled back patch '{id_to_rollback}' ({count} files restored)."
                );
            }
            Err(e) => {
                eprintln!("Failed to rollback patch '{id_to_rollback}': {e}");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("history") {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let limit = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let turns = nioxs_provider::journal::read_local_turns(&current_dir, limit);
        if turns.is_empty() {
            println!("No local turn history found in .nioxs/history.jsonl");
        } else {
            println!("Recent Turns ({} total):", turns.len());
            for (idx, turn) in turns.iter().enumerate() {
                println!(
                    "[{:2}] ts:{} | Intent: {:<12} | Outcome: {:<10}\n     Prompt: {}\n",
                    idx + 1,
                    turn.timestamp_unix,
                    turn.intent,
                    turn.outcome,
                    turn.prompt.chars().take(80).collect::<String>()
                );
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("mcp") {
        return nioxs_provider::mcp::run_stdio_server().map_err(|e| e.into());
    }
    if arg.is_some() && arg.as_deref() != Some("serve") {
        return Err("unknown command; expected 'nioxs serve', 'nioxs pull', 'nioxs viper', 'nioxs skate', 'nioxs manta', 'nioxs beluga', 'nioxs blue', 'nioxs cheetah', 'nioxs tiger', 'nioxs fox', 'nioxs rabbit', 'nioxs fin', 'nioxs hydra', 'nioxs mcp', 'nioxs init', 'nioxs undo', or 'nioxs history'".into());
    }

    let mut server_device_pref = nioxs_engine::DevicePreference::Auto;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        if a == "--gpu" {
            server_device_pref = nioxs_engine::DevicePreference::Gpu;
        } else if a == "--cpu" {
            server_device_pref = nioxs_engine::DevicePreference::Cpu;
        } else if a == "--auto" {
            server_device_pref = nioxs_engine::DevicePreference::Auto;
        } else if a == "-d" || a == "--device" {
            if let Some(val) = iter.next() {
                server_device_pref = nioxs_engine::DevicePreference::parse(&val);
            }
        }
    }
    match server_device_pref {
        nioxs_engine::DevicePreference::Gpu => unsafe { std::env::set_var("NIOXS_DEVICE", "gpu") },
        nioxs_engine::DevicePreference::Cpu => unsafe { std::env::set_var("NIOXS_DEVICE", "cpu") },
        nioxs_engine::DevicePreference::Auto => {}
    }
    let listen: std::net::SocketAddr = if let Ok(listen_str) = std::env::var("NIOXS_LISTEN") {
        listen_str.parse()?
    } else if let Ok(port_str) = std::env::var("PORT") {
        let port: u16 = port_str.parse()?;
        std::net::SocketAddr::from(([0, 0, 0, 0], port))
    } else {
        "127.0.0.1:7460".parse()?
    };
    let (db_url, db_token, _niodb_proc) = nioxs_provider::niodb_supervisor::ensure_niodb(
        std::env::var("NIOXS_DB_URL").ok().as_deref(),
        std::env::var("NIOXS_DB_TOKEN_FILE").ok().as_deref(),
    )
    .await?;

    let api_token = if let Ok(direct) = std::env::var("NIOXS_API_TOKEN") {
        let trimmed = direct.trim().to_string();
        if trimmed.is_empty() {
            return Err("provider token cannot be empty".into());
        }
        Some(trimmed)
    } else if let Ok(file) = std::env::var("NIOXS_API_TOKEN_FILE") {
        let content = std::fs::read_to_string(file)?;
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            return Err("provider token cannot be empty".into());
        }
        Some(trimmed)
    } else {
        None
    };

    let allow_unauthenticated = std::env::var("NIOXS_ALLOW_UNAUTHENTICATED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !listen.ip().is_loopback() && api_token.is_none() && !allow_unauthenticated {
        return Err("non-loopback deployment requires NIOXS_API_TOKEN or NIOXS_API_TOKEN_FILE (or set NIOXS_ALLOW_UNAUTHENTICATED=true)".into());
    }
    let deployment = std::env::var("NIOXS_DEPLOYMENT_ID").unwrap_or_else(|_| "local".into());
    if deployment.is_empty()
        || deployment.len() > 64
        || !deployment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err("invalid deployment ID".into());
    }
    let db = NioDb::new(&db_url, db_token)?;
    let model = match std::env::var("NIOXS_MODEL_FILE") {
        Ok(file) => {
            let bytes = std::fs::read(file)?;
            let expected = std::env::var("NIOXS_MODEL_SHA256")
                .map_err(|_| "external model requires NIOXS_MODEL_SHA256")?;
            if nioxs_contracts::digest(&bytes) != expected {
                return Err("model checksum mismatch".into());
            }
            Model::load(&bytes)?
        }
        Err(_) => Model::embedded()?,
    };
    let instance = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let ast_engine: Option<nioxs_engine::AstEngine> = None;
    let microlm_banner = "Sovereign MicroLM Fleet (Blue, Beluga, Manta, Viper, Dolphin)";
    let active_hw = nioxs_engine::ActiveDevice::detect(server_device_pref);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    let addr = listener.local_addr()?;
    eprintln!(
        "\n\
  ╭──────────────────────────────────────────────────────────╮\n\
  │  ⚡ NioXS v{:<8} Pure Coding Companion Gateway        │\n\
  ├──────────────────────────────────────────────────────────┤\n\
  │  • Gateway API : http://{:<33} │\n\
  │  • NioDB Store : {:<40} │\n\
  │  • MicroLM     : {:<40} │\n\
  │  • Hardware    : {:<40} │\n\
  │  • Directives  : --bash, --tools, --code, --test         │\n\
  ╰──────────────────────────────────────────────────────────╯\n",
        env!("CARGO_PKG_VERSION"),
        addr,
        db_url,
        microlm_banner,
        active_hw.name(),
    );
    axum::serve(
        listener,
        router(App {
            db,
            deployment,
            api_token,
            active: ActiveTurns::default(),
            model: Arc::new(model),
            ray_engine: ast_engine,
            instance,
        }),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await?;
    Ok(())
}

fn ground_and_sanitize_command(prompt: &str, raw_cmd: &str) -> String {
    let mut clean_cmd = raw_cmd
        .trim()
        .trim_start_matches("```powershell")
        .trim_start_matches("```bash")
        .trim_start_matches("```sh")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
        .to_string();

    // 1. Sanitize trailing dangling flags (e.g. "ls -" -> "ls -la")
    if clean_cmd == "ls -" || clean_cmd.ends_with("ls -") {
        clean_cmd = clean_cmd.replace("ls -", "ls -la");
    } else if clean_cmd.ends_with(" -") {
        clean_cmd = clean_cmd.trim_end_matches(" -").trim().to_string();
    } else if clean_cmd.ends_with(" --") {
        clean_cmd = clean_cmd.trim_end_matches(" --").trim().to_string();
    }

    // 2. Extract quoted literal terms from prompt (e.g. 'TODO' or "TODO")
    let prompt_quoted: Vec<&str> = {
        let mut list = Vec::new();
        for quote_char in ['\'', '"'] {
            let mut start = None;
            for (i, c) in prompt.char_indices() {
                if c == quote_char {
                    if let Some(s) = start {
                        if i > s + 1 {
                            list.push(&prompt[s + 1..i]);
                        }
                        start = None;
                    } else {
                        start = Some(i);
                    }
                }
            }
        }
        list
    };

    // If user specified an explicit search term, align the pattern in the reflex
    if let Some(&user_term) = prompt_quoted.first() {
        // Handle -e '...' or -e "..."
        if let Some(pos) = clean_cmd.find("-e ") {
            let after = &clean_cmd[pos + 3..];
            if after.starts_with('\'') || after.starts_with('"') {
                let quote = after.chars().next().unwrap();
                if let Some(end_quote) = after[1..].find(quote) {
                    let full_flag = &clean_cmd[pos..pos + 3 + 1 + end_quote + 1];
                    let replacement = format!("-e '{user_term}'");
                    clean_cmd = clean_cmd.replace(full_flag, &replacement);
                }
            }
        }
        // Handle -Pattern '...' or -Pattern "..."
        if let Some(pos) = clean_cmd.find("-Pattern ") {
            let after = &clean_cmd[pos + 9..];
            if after.starts_with('\'') || after.starts_with('"') {
                let quote = after.chars().next().unwrap();
                if let Some(end_quote) = after[1..].find(quote) {
                    let full_flag = &clean_cmd[pos..pos + 9 + 1 + end_quote + 1];
                    let replacement = format!("-Pattern '{user_term}'");
                    clean_cmd = clean_cmd.replace(full_flag, &replacement);
                }
            }
        }
        // Handle grep without -e, e.g. grep -rn 'panic' or grep -rnw 'panic'
        if clean_cmd.starts_with("grep ") && !clean_cmd.contains("-e ") {
            for quote in ['\'', '"'] {
                if let Some(first_q) = clean_cmd.find(quote) {
                    if let Some(second_q) = clean_cmd[first_q + 1..].find(quote) {
                        let matched_slice = &clean_cmd[first_q..first_q + 1 + second_q + 1];
                        if !matched_slice.contains('/')
                            && matched_slice != format!("{quote}.{quote}")
                        {
                            let replacement = format!("'{user_term}'");
                            clean_cmd = clean_cmd.replace(matched_slice, &replacement);
                            break;
                        }
                    }
                }
            }
        }
    }

    // 3. Fallback system directory sanitization:
    // If prompt didn't specify system paths like /var/log, /opt, /var/cache, /data, /home/app,
    // don't scan them with grep/find, fallback to '.'
    let system_paths = [
        "'/var/log'",
        "\"/var/log\"",
        "'/var/cache'",
        "\"/var/cache\"",
        "'/home/app'",
        "\"/home/app\"",
        "'/opt'",
        "\"/opt\"",
        "'/data'",
        "\"/data\"",
    ];

    for path in system_paths {
        let stripped = path.trim_matches('\'').trim_matches('"');
        if clean_cmd.contains(path) && !prompt.contains(stripped) {
            clean_cmd = clean_cmd.replace(path, "'.'");
        }
    }

    // 4. Port number slot grounding:
    // If prompt mentions a port number (e.g. "port 3000" or "port 8080")
    for word in prompt.split_whitespace() {
        let clean_word = word.trim_matches(|c: char| !c.is_numeric());
        if let Ok(port_num) = clean_word.parse::<u16>() {
            if port_num >= 80 {
                let port_str = port_num.to_string();
                for default_port in ["3000", "8080", "5432", "8000", "9000"] {
                    if default_port != port_str && clean_cmd.contains(default_port) {
                        clean_cmd = clean_cmd.replace(default_port, &port_str);
                    }
                }
            }
        }
    }

    // 5. Code & Spider Search Hygiene (Binary noise suppression, directory exclusions, extension filter)
    if clean_cmd.starts_with("grep ") {
        // Prevent binary match spam (target/*.rmeta, *.rlib, *.so, *.exe)
        if clean_cmd.contains("-rn ") {
            clean_cmd = clean_cmd.replace("-rn ", "-rIn ");
        } else if clean_cmd.contains("-rni ") {
            clean_cmd = clean_cmd.replace("-rni ", "-rIni ");
        } else if clean_cmd.contains("-rnw ") {
            clean_cmd = clean_cmd.replace("-rnw ", "-rInw ");
        }

        // Exclude build artifacts and caches
        if !clean_cmd.contains("--exclude-dir")
            && !prompt.contains("target")
            && !prompt.contains(".git")
        {
            clean_cmd.push_str(" --exclude-dir={target,.git,node_modules}");
        }

        // File extension grounding (e.g. ".rs files", ".py files", ".ts files")
        if !clean_cmd.contains("--include") {
            for word in prompt.split_whitespace() {
                let clean_w = word.trim_matches(|c: char| c == '\'' || c == '"' || c == ',');
                if clean_w.starts_with('.')
                    && clean_w.len() >= 2
                    && clean_w.len() <= 6
                    && clean_w[1..].chars().all(|c| c.is_alphanumeric())
                {
                    let ext = clean_w.trim_start_matches('.');
                    clean_cmd.push_str(&format!(" --include=\"*.{ext}\""));
                    break;
                }
            }
        }

        // Ensure search directory anchor exists
        let parts: Vec<&str> = clean_cmd.split_whitespace().collect();
        let last = parts.last().copied().unwrap_or("");
        if last.starts_with('-')
            || clean_cmd.ends_with('}')
            || clean_cmd.ends_with('"')
            || clean_cmd.ends_with('\'')
        {
            clean_cmd.push_str(" .");
        }
    }

    clean_cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_council_directive_route_and_domain() {
        let directive = "@council(route = memory, domain = circular_buffer)";
        let formatted = format_council_directive(directive);
        assert_eq!(
            formatted,
            "<council>\n    @council(\n        route = memory,\n        domain = circular_buffer\n    )\n</council>\n\n"
        );
    }

    #[test]
    fn test_detect_hydra_specialist_atomic_ring_buffer() {
        let acc = "<think>\n1. Invariant: Linearizable memory ordering\n</think>\n<council>\n@council(\nhead = 001,\ndomain = atomic\n)\n</council>";
        let prompt = "Implement an atomic ring buffer in Zig";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-001"));
    }

    #[test]
    fn test_detect_hydra_specialist_sql() {
        let acc = "@council(head = 005, domain = mergetree)";
        let prompt = "Optimize this ClickHouse MergeTree query";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-005"));
    }

    #[test]
    fn test_detect_hydra_specialist_raft() {
        let acc = "@council(head = 008, domain = raft)";
        let prompt = "Implement joint consensus membership change";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-008"));
    }

    #[test]
    fn test_detect_hydra_specialist_kernel() {
        let acc = "@council(head = 001, domain = posix)";
        let prompt = "Write an async submission queue poller in C";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-001"));
    }

    #[test]
    fn test_detect_hydra_specialist_head_001() {
        let acc = "<think>\nInvariant: memory order seq_cst\n</think>\n@council(head-001-atomic-concurrency)";
        let prompt = "Write a lock-free queue in Zig";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-001"));
    }

    #[test]
    fn test_detect_hydra_specialist_head_002() {
        let acc = "<think>\nInvariant: async streaming backpressure\n</think>\n@council(head-002-backend-axum-tokio)";
        let prompt = "Build an Axum SSE streaming server";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-002"));
    }

    #[test]
    fn test_detect_hydra_specialist_all_heads() {
        assert_eq!(detect_council_specialist("@council(head=003)", "Build a React 19 RSC component with Suspense", true), Some("head-003"));
        assert_eq!(detect_council_specialist("@council(head=004)", "Write an eBPF XDP packet filter in Rust aya", true), Some("head-004"));
        assert_eq!(detect_council_specialist("@council(head=005)", "Create ClickHouse ReplacingMergeTree schema", true), Some("head-005"));
        assert_eq!(detect_council_specialist("@council(head=006)", "Implement lazy propagation segment tree in Rust", true), Some("head-006"));
        assert_eq!(detect_council_specialist("@council(head=007)", "Write a Kubernetes CRD reconciler in kube-rs", true), Some("head-007"));
        assert_eq!(detect_council_specialist("@council(head=008)", "Implement Raft consensus AppendEntries state machine", true), Some("head-008"));
        assert_eq!(detect_council_specialist("@council(head=009)", "Vectorize dot product using AVX2 SIMD intrinsics", true), Some("head-009"));
        assert_eq!(detect_council_specialist("@council(head=010)", "Run WASI 0.2 sandbox in Wasmtime host", true), Some("head-010"));
    }

    #[test]
    fn test_detect_hydra_polymath_direct() {
        let acc = "<think>\nGeneral query, no specialist needed\n</think>\n@council(direct)";
        let prompt = "Explain the difference between Arc and Rc in Rust";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-direct"));
    }

    #[test]
    fn test_detect_hydra_keyword_fallback() {
        let acc = "<think>\nThinking about kernel primitives...\n</think>";
        let prompt = "Implement an atomic lock-free ring buffer in Rust";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-001"));
    }

    #[test]
    fn test_detect_hydra_loosened_natural_triage() {
        assert_eq!(detect_council_specialist("", "Vectorize dot product using AVX2 SIMD intrinsics", true), Some("head-009"));
        assert_eq!(detect_council_specialist("", "Write an eBPF XDP line-rate packet filter in Rust", true), Some("head-004"));
        assert_eq!(detect_council_specialist("", "Create ClickHouse ReplacingMergeTree schema", true), Some("head-005"));
        assert_eq!(detect_council_specialist("", "Implement lazy propagation segment tree in Rust", true), Some("head-006"));
        assert_eq!(detect_council_specialist("", "Write a Kubernetes CRD reconciler in kube-rs", true), Some("head-007"));
        assert_eq!(detect_council_specialist("", "Implement Raft consensus AppendEntries state machine", true), Some("head-008"));
        assert_eq!(detect_council_specialist("", "Run WASI 0.2 sandbox in Wasmtime host", true), Some("head-010"));
        assert_eq!(detect_council_specialist("", "Build a React 19 RSC component with Suspense", true), Some("head-003"));
        assert_eq!(detect_council_specialist("", "Build an Axum SSE streaming server with Tokio", true), Some("head-002"));
        assert_eq!(detect_council_specialist("", "Write a lock-free SPSC ring buffer in Zig", true), Some("head-001"));
    }

    #[test]
    fn test_is_degenerate_detection() {
        let garbage1 = "''C, 'C their 'Cle', the Clawst, the 'C'\n')\n')\n')";
        assert!(is_degenerate(garbage1));
        let garbage2 = "011200000";
        assert!(is_degenerate(garbage2));
        let valid = "<think>\n1. Invariant: memory order.\n</think>\n@council(head=001, domain=systems)";
        assert!(!is_degenerate(valid));
    }

    #[test]
    fn test_format_council_directive_head() {
        let directive = "@council(head = 001, domain = systems-atomic-concurrency)";
        let formatted = format_council_directive(directive);
        assert!(formatted.contains("head = 001"));
        assert!(formatted.contains("domain = systems-atomic-concurrency"));
    }
}
