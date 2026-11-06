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
        "ZeroCopy",
        "What is zero-copy networking and how does splice/sendfile achieve it?",
    ),
    (
        "Epoll",
        "How does Linux epoll work under the hood compared to select and poll?",
    ),
    (
        "LRU",
        "Write an LRU cache implementation in Rust with O(1) get and put",
    ),
    (
        "Deadlock",
        "What causes deadlock in concurrent thread pools and how to prevent it?",
    ),
    (
        "MemoryAlignment",
        "Explain how memory alignment works in C and why it affects CPU cache",
    ),
    (
        "LSMTree",
        "Compare LSM-Tree and B-Tree for write amplification vs read latency",
    ),
    (
        "SO_KEEPALIVE",
        "What is a TCP half-open connection and how does SO_KEEPALIVE mitigate it?",
    ),
];

fn main() -> Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/mn".to_string());
    let model_path = PathBuf::from(&home).join(".nioxs/models/manta.vmicrolm-vector.gguf");
    let tok_path = PathBuf::from(&home).join(".nioxs/models/manta-tokenizer.json");

    println!("============================================================");
    println!("🦈 Manta Coding Assistant Multi-Prompt Benchmark");
    println!("============================================================");

    let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;
    engine.set_suppressed_tokens([41233, 54275, 73594]);

    let (load_rss, load_hwm) = memory_mb();
    println!(
        "Engine Initialized: Working RAM = {:.2} MB | Peak RAM = {:.2} MB\n",
        load_rss, load_hwm
    );
    println!(
        "{:<4} | {:<18} | {:<6} | {:<6} | {:<9} | {:<9} | {:<10}",
        "#", "Task", "Tokens", "Words", "Tok/s", "Words/s", "Peak RAM"
    );
    println!("-----+--------------------+--------+--------+-----------+-----------+----------");

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let mut total_tokens = 0usize;
    let mut total_words = 0usize;
    let mut total_time = 0.0f64;
    let max_gen = 64;

    for (i, (tag, prompt)) in PROMPTS.iter().enumerate() {
        engine.reset_cache();
        let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let enc = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let prompt_tokens = enc.get_ids().to_vec();

        let _ = engine.prefill(&prompt_tokens)?;

        let mut gen_tokens = Vec::new();
        let t_gen = Instant::now();

        for _ in 0..max_gen {
            let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
            if next_tok == eos_id || next_tok == 0 {
                break;
            }
            gen_tokens.push(next_tok);
            let _ =
                engine.forward_stream(&[next_tok], prompt_tokens.len() + gen_tokens.len() - 1)?;
        }

        let gen_time = t_gen.elapsed().as_secs_f64();
        let tps = if gen_time > 0.0 {
            gen_tokens.len() as f64 / gen_time
        } else {
            0.0
        };
        let decoded = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
        let words = decoded.split_whitespace().count();
        let wps = if gen_time > 0.0 {
            words as f64 / gen_time
        } else {
            0.0
        };
        let (_, cur_hwm) = memory_mb();

        total_tokens += gen_tokens.len();
        total_words += words;
        total_time += gen_time;

        println!(
            "{:<4} | {:<18} | {:<6} | {:<6} | {:<9.1} | {:<9.1} | {:.2} MB",
            i + 1,
            tag,
            gen_tokens.len(),
            words,
            tps,
            wps,
            cur_hwm
        );
    }

    let (final_rss, final_hwm) = memory_mb();
    let mean_tps = if total_time > 0.0 {
        total_tokens as f64 / total_time
    } else {
        0.0
    };
    let mean_wps = if total_time > 0.0 {
        total_words as f64 / total_time
    } else {
        0.0
    };

    println!("============================================================");
    println!("🏁 Manta Benchmark Aggregate");
    println!("============================================================");
    println!("Total Tokens:         {}", total_tokens);
    println!("Total Words:          {}", total_words);
    println!("Total Decode Time:    {:.2}s", total_time);
    println!("Mean Tokens/sec:      {:.2} tok/s", mean_tps);
    println!("Mean Words/sec:       {:.2} words/s", mean_wps);
    println!("Final Resident RAM:   {:.2} MB", final_rss);
    println!("Peak Process RAM:     {:.2} MB", final_hwm);
    println!("============================================================");

    Ok(())
}
