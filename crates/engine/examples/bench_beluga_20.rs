use anyhow::{anyhow, Result};
use nioxs_engine::ast_engine::AstMicroEngine;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Clone)]
struct BenchmarkResult {
    prompt_id: usize,
    category: String,
    prompt: String,
    prompt_tokens: usize,
    prefill_time_ms: f64,
    prefill_tps: f64,
    gen_tokens: usize,
    gen_time_s: f64,
    tokens_per_sec: f64,
    word_count: usize,
    words_per_sec: f64,
    vm_rss_mb: f64,
    vm_hwm_mb: f64,
    has_think_tags: bool,
    response: String,
}

#[derive(Debug, Serialize)]
struct Beluga20Summary {
    model: String,
    disk_size_mb: f64,
    num_queries: usize,
    avg_prefill_ms: f64,
    avg_prefill_tps: f64,
    avg_tokens_per_sec: f64,
    avg_words_per_sec: f64,
    peak_ram_mb: f64,
    sub_250mb_sla_passed: bool,
    results: Vec<BenchmarkResult>,
}

fn memory_mb() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get_val = |key: &str| -> f64 {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    (get_val("VmRSS:"), get_val("VmHWM:"))
}

fn get_file_size_mb(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf"));
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    if !model_path.exists() {
        return Err(anyhow!("Beluga model missing at {:?}", model_path));
    }
    if !tokenizer_path.exists() {
        return Err(anyhow!("Beluga tokenizer missing at {:?}", tokenizer_path));
    }

    println!("\n================================================================================");
    println!("  🐋 BENCHMARKING: NIOBELUGA (1.54B) - 20 COMPREHENSIVE PRODUCTION PROMPTS");
    println!("  Model Path:     {}", model_path.display());
    println!("  Disk Size:      {:.2} MB", get_file_size_mb(&model_path));
    println!("  Memory SLA:     < 250 MB Peak RAM");
    println!("================================================================================");

    let (start_rss, start_hwm) = memory_mb();
    println!("Process Start RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB", start_rss, start_hwm);

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let t0 = Instant::now();
    let mut engine = AstMicroEngine::new(&model_path)?;
    let load_time = t0.elapsed().as_secs_f64();
    let (load_rss, load_hwm) = memory_mb();
    println!("Engine loaded in {:.2}s | Post-load RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB\n", load_time, load_rss, load_hwm);

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let system_prompt = "You are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions. Defer raw low-level code generation to Cuda.";

    let test_cases = [
        // Systems & Architecture
        ("Arch: PubSub", "Design a distributed pub-sub message broker in Rust"),
        ("Arch: RateLimiter", "Design a sliding window rate limiter for a high-throughput API gateway"),
        ("Arch: KeyValue", "Architect an in-memory key-value cache with LRU eviction and persistence"),
        ("Arch: DistributedLock", "Design a distributed locking mechanism using consensus and leases"),
        ("Arch: MetricsPipeline", "Design a real-time streaming metrics ingestion pipeline"),
        // Explanations & Systems Concepts
        ("Explain: Mutex", "Explain the trade-offs between Arc<Mutex<T>> and RwLock<T> in async Tokio"),
        ("Explain: ZeroCopy", "What is zero-copy networking and how does splice/sendfile achieve it?"),
        ("Explain: Epoll", "How does Linux epoll work under the hood compared to select and poll?"),
        ("Explain: TailCall", "Explain tail call optimization and why it matters in compilers"),
        ("Explain: MVCC", "How does Multi-Version Concurrency Control (MVCC) work in relational databases?"),
        // Coding & Algorithms
        ("Code: LRU", "Write an LRU cache implementation in Rust with O(1) get and put"),
        ("Code: Trie", "Implement a concurrent Trie prefix tree for autocomplete"),
        ("Code: RingBuffer", "Implement a lock-free single-producer single-consumer ring buffer in Rust"),
        ("Code: Merkle", "Explain how Merkle trees verify data integrity in distributed systems"),
        ("Code: Backoff", "Implement an exponential backoff with jitter retry strategy in Rust"),
        // Debugging & Reflex
        ("Debug: MemoryLeak", "How do you detect memory leaks and page fragmentation in Linux user-space?"),
        ("Debug: Deadlock", "What causes deadlock in concurrent thread pools and how to prevent it?"),
        ("Reflex: GitConflict", "How do you resolve a 3-way git merge conflict cleanly in CLI?"),
        ("Reflex: DockerPID1", "Why is running a process as PID 1 in Docker containers risky?"),
        ("Reflex: TCPHalfOpen", "What is a TCP half-open connection and how does SO_KEEPALIVE mitigate it?"),
    ];

    let mut results = Vec::new();
    let mut total_prefill_ms = 0.0;
    let mut total_prefill_tps = 0.0;
    let mut total_tps = 0.0;
    let mut total_wps = 0.0;
    let mut peak_ram_observed: f64 = load_hwm;

    for (idx, &(category, prompt)) in test_cases.iter().enumerate() {
        println!("--------------------------------------------------------------------------------");
        println!("[{:02}/{:02}] [{}] \"{}\"", idx + 1, test_cases.len(), category, prompt);

        engine.reset_cache();
        engine.set_sampling(0.65, 0.90, 1.15, 64);

        let formatted = format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let encoded = tokenizer.encode(formatted.as_str(), false).map_err(|e| anyhow!("{e}"))?;
        let prompt_tokens = encoded.get_ids().to_vec();

        // Prefill
        let t_prefill = Instant::now();
        let _ = engine.prefill(&prompt_tokens)?;
        let prefill_duration = t_prefill.elapsed();
        let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
        let prefill_tps = prompt_tokens.len() as f64 / prefill_duration.as_secs_f64();

        // Decode loop
        let t_gen = Instant::now();
        let mut gen_tokens = Vec::new();
        let max_gen_tokens = 64;

        for _ in 0..max_gen_tokens {
            let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
            if next_tok == eos_id || next_tok == 0 {
                break;
            }
            gen_tokens.push(next_tok);
            let _ = engine.forward_stream(&[next_tok], prompt_tokens.len() + gen_tokens.len() - 1)?;
        }
        let gen_duration = t_gen.elapsed();
        let gen_time_s = gen_duration.as_secs_f64();
        let tps = if gen_time_s > 0.0 { gen_tokens.len() as f64 / gen_time_s } else { 0.0 };

        let decoded_response = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
        let words = decoded_response.split_whitespace().count();
        let wps = if gen_time_s > 0.0 { words as f64 / gen_time_s } else { 0.0 };

        let (cur_rss, cur_hwm) = memory_mb();
        if cur_hwm > peak_ram_observed {
            peak_ram_observed = cur_hwm;
        }

        let has_think = decoded_response.contains("<think>") || decoded_response.contains("think");

        println!("  Prefill: {:.1} ms ({:.1} t/s) | Gen: {} tok in {:.2}s -> {:.2} tok/s | {:.2} w/s", prefill_ms, prefill_tps, gen_tokens.len(), gen_time_s, tps, wps);
        println!("  RAM: VmRSS = {:.1} MB | VmHWM = {:.1} MB (Peak: {:.1} MB) | Think tags: {}", cur_rss, cur_hwm, peak_ram_observed, if has_think { "YES" } else { "NO" });
        println!("  Response preview:\n  {}", decoded_response.replace('\n', " ").trim());

        total_prefill_ms += prefill_ms;
        total_prefill_tps += prefill_tps;
        total_tps += tps;
        total_wps += wps;

        results.push(BenchmarkResult {
            prompt_id: idx + 1,
            category: category.to_string(),
            prompt: prompt.to_string(),
            prompt_tokens: prompt_tokens.len(),
            prefill_time_ms: prefill_ms,
            prefill_tps,
            gen_tokens: gen_tokens.len(),
            gen_time_s,
            tokens_per_sec: tps,
            word_count: words,
            words_per_sec: wps,
            vm_rss_mb: cur_rss,
            vm_hwm_mb: cur_hwm,
            has_think_tags: has_think,
            response: decoded_response,
        });
    }

    let n = test_cases.len() as f64;
    let summary = Beluga20Summary {
        model: "NioBeluga (1.54B Sovereign Cognitive Companion)".to_string(),
        disk_size_mb: get_file_size_mb(&model_path),
        num_queries: test_cases.len(),
        avg_prefill_ms: total_prefill_ms / n,
        avg_prefill_tps: total_prefill_tps / n,
        avg_tokens_per_sec: total_tps / n,
        avg_words_per_sec: total_wps / n,
        peak_ram_mb: peak_ram_observed,
        sub_250mb_sla_passed: peak_ram_observed < 250.0,
        results,
    };

    println!("\n================================================================================");
    println!("  🐋 NIOBELUGA 20-PROMPT BENCHMARK SUMMARY");
    println!("  Peak Resident RAM (VmHWM): {:.2} MB", summary.peak_ram_mb);
    println!("  Sub-250MB SLA Compliance:  {}", if summary.sub_250mb_sla_passed { "✅ PASSED (< 250 MB)" } else { "❌ FAILED" });
    println!("  Avg Generation Speed:      {:.2} tok/s  ({:.2} words/sec)", summary.avg_tokens_per_sec, summary.avg_words_per_sec);
    println!("  Avg Prefill Speed:         {:.2} tok/s  ({:.1} ms)", summary.avg_prefill_tps, summary.avg_prefill_ms);
    println!("================================================================================");

    let output_path = PathBuf::from("/home/mn/nio-labs/beluga_20_benchmark_report.json");
    std::fs::write(&output_path, serde_json::to_string_pretty(&summary)?)?;
    println!("✅ Full 20-Prompt Report saved to: {}", output_path.display());

    Ok(())
}
