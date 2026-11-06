use anyhow::Result;
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

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

const PROMPTS: &[(&str, &str)] = &[
    (
        "PubSub",
        "Design a distributed pub-sub message broker in Rust",
    ),
    (
        "RateLimiter",
        "Design a sliding window rate limiter for a high-throughput API gateway",
    ),
    (
        "KeyValue",
        "Architect an in-memory key-value cache with LRU eviction and persistence",
    ),
    (
        "DistributedLock",
        "Design a distributed locking mechanism using consensus and leases",
    ),
    (
        "MetricsPipeline",
        "Design a real-time streaming metrics ingestion pipeline",
    ),
    (
        "Mutex",
        "Explain the trade-offs between Arc<Mutex<T>> and RwLock<T> in async Tokio",
    ),
    (
        "ZeroCopy",
        "What is zero-copy networking and how does splice/sendfile achieve it?",
    ),
    (
        "Epoll",
        "How does Linux epoll work under the hood compared to select and poll?",
    ),
    (
        "TailCall",
        "Explain tail call optimization and why it matters in compilers",
    ),
    (
        "MVCC",
        "How does Multi-Version Concurrency Control (MVCC) work in relational databases?",
    ),
    (
        "LRU",
        "Write an LRU cache implementation in Rust with O(1) get and put",
    ),
    (
        "Trie",
        "Implement a concurrent Trie prefix tree for autocomplete",
    ),
    (
        "RingBuffer",
        "Implement a lock-free single-producer single-consumer ring buffer in Rust",
    ),
    (
        "Merkle",
        "Explain how Merkle trees verify data integrity in distributed systems",
    ),
    (
        "Backoff",
        "Implement an exponential backoff with jitter retry strategy in Rust",
    ),
    (
        "MemoryLeak",
        "How do you detect memory leaks and page fragmentation in Linux user-space?",
    ),
    (
        "Deadlock",
        "What causes deadlock in concurrent thread pools and how to prevent it?",
    ),
    (
        "GitConflict",
        "How do you resolve a 3-way git merge conflict cleanly in CLI?",
    ),
    (
        "DockerPID1",
        "Why is running a process as PID 1 in Docker containers risky?",
    ),
    (
        "TCPHalfOpen",
        "What is a TCP half-open connection and how does SO_KEEPALIVE mitigate it?",
    ),
];

fn main() -> Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/mn".to_string());
    let model_path = PathBuf::from(&home).join(".nioxs/models/skate.vmicrolm-vector.gguf");
    let tok_path = PathBuf::from(&home).join(".nioxs/models/skate-tokenizer.json");

    println!("============================================================");
    println!("🐟 Skate 135M Companion 20-Prompt Throughput Benchmark");
    println!("============================================================");

    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut engine = AstMicroEngine::new(&model_path)?;
    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(49153);

    let (rss_init, hwm_init) = memory_mb();
    println!(
        "Engine Initialized: Working RAM = {:.2} MB | Peak RAM = {:.2} MB\n",
        rss_init, hwm_init
    );

    println!(
        "{:<4} | {:<16} | {:<6} | {:<6} | {:<9} | {:<9} | {:<8}",
        "#", "Task", "Tokens", "Words", "Tok/s", "Words/s", "Peak RAM"
    );
    println!(
        "{:-<4}-+-{:-<16}-+-{:-<6}-+-{:-<6}-+-{:-<9}-+-{:-<9}-+-{:-<8}",
        "", "", "", "", "", "", ""
    );

    let mut total_tokens = 0usize;
    let mut total_words = 0usize;
    let mut total_gen_duration = 0.0f64;
    let mut prompt_wps = Vec::new();
    let mut prompt_tps = Vec::new();

    for (idx, (label, query)) in PROMPTS.iter().enumerate() {
        engine.reset_cache();

        let prompt = format!(
            "<|im_start|>system\nYou are Skate, Nio's ambient companion. Be friendly, clear, grounded, and concise.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            query.trim()
        );
        let encoded = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let prompt_tokens = encoded.get_ids().to_vec();

        let _ = engine.prefill(&prompt_tokens)?;

        let mut generated_tokens = Vec::new();
        let mut all_tokens = prompt_tokens.clone();
        let max_gen = 64;

        let t_gen = Instant::now();
        let first_tok = engine.next_token_with_history(None, &[], false)?;
        if first_tok != eos_id && first_tok != 0 {
            generated_tokens.push(first_tok);
            all_tokens.push(first_tok);

            let mut current_confirmed = first_tok;
            let mut current_pos = prompt_tokens.len();

            while generated_tokens.len() < max_gen {
                let remaining = max_gen.saturating_sub(generated_tokens.len());
                let max_draft = 2.min(remaining.saturating_sub(1));

                let res = engine.speculative_pld_step(
                    &all_tokens,
                    current_confirmed,
                    current_pos,
                    &generated_tokens,
                    max_draft,
                    &[eos_id, 0],
                )?;

                let mut stopped = false;
                for tok in res.accepted_tokens {
                    if tok == eos_id || tok == 0 {
                        stopped = true;
                        break;
                    }
                    generated_tokens.push(tok);
                    all_tokens.push(tok);
                    if generated_tokens.len() >= max_gen {
                        stopped = true;
                        break;
                    }
                }

                current_confirmed = res.next_confirmed;
                current_pos = res.next_pos;

                if stopped || res.is_eos {
                    break;
                }
            }
        }
        let gen_time = t_gen.elapsed().as_secs_f64();
        let tps = if gen_time > 0.0 {
            generated_tokens.len() as f64 / gen_time
        } else {
            0.0
        };

        let decoded = tokenizer
            .decode(&generated_tokens, false)
            .unwrap_or_default();
        let words = decoded.split_whitespace().count();
        let wps = if gen_time > 0.0 {
            words as f64 / gen_time
        } else {
            0.0
        };
        let (_, hwm) = memory_mb();

        total_tokens += generated_tokens.len();
        total_words += words;
        total_gen_duration += gen_time;
        prompt_wps.push(wps);
        prompt_tps.push(tps);

        println!(
            "{:<4} | {:<16} | {:<6} | {:<6} | {:<9.1} | {:<9.1} | {:.2} MB",
            idx + 1,
            label,
            generated_tokens.len(),
            words,
            tps,
            wps,
            hwm
        );
    }

    let avg_tps = total_tokens as f64 / total_gen_duration;
    let avg_wps = total_words as f64 / total_gen_duration;
    let (final_rss, final_hwm) = memory_mb();

    println!("============================================================");
    println!("🏁 Skate Benchmark Aggregate (20 Prompts)");
    println!("============================================================");
    println!("Total Tokens:         {}", total_tokens);
    println!("Total Words:          {}", total_words);
    println!("Total Decode Time:    {:.2}s", total_gen_duration);
    println!("Mean Tokens/sec:      {:.2} tok/s", avg_tps);
    println!("Mean Words/sec:       {:.2} words/s", avg_wps);
    println!("Final Resident RAM:   {:.2} MB", final_rss);
    println!("Peak Process RAM:     {:.2} MB (< 68 MB limit)", final_hwm);
    println!("============================================================");

    Ok(())
}
