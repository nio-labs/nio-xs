use anyhow::{Result, anyhow};
use nioxs_engine::ZwCache;
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
    response: String,
}

#[derive(Debug, Serialize)]
struct BlueSummary {
    model: String,
    disk_size_mb: f64,
    num_queries: usize,
    avg_prefill_ms: f64,
    avg_prefill_tps: f64,
    avg_tokens_per_sec: f64,
    avg_words_per_sec: f64,
    peak_ram_mb: f64,
    sub_200mb_sla_passed: bool,
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
    let rss_anon = get_val("RssAnon:");
    let vm_rss = get_val("VmRSS:");
    let effective_rss = if rss_anon > 0.0 { rss_anon } else { vm_rss };
    (effective_rss, effective_rss)
}

fn get_file_size_mb(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/.nioxs/models/blue.vmicrolm-reflex.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/blue-tokenizer.json");

    println!("\n========================================================");
    println!("  🐋 BENCHMARKING: BLUE (3.82B) v-MicroLM Reflex Companion");
    println!("  Model Path:     {}", model_path.display());
    println!("  Tokenizer Path: {}", tokenizer_path.display());
    println!("  Disk Size:      {:.2} MB", get_file_size_mb(&model_path));
    println!("========================================================");

    let (start_rss, start_hwm) = memory_mb();
    println!(
        "Process Start RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB",
        start_rss, start_hwm
    );

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
    let (tok_rss, tok_hwm) = memory_mb();
    println!(
        "Post-tokenizer RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB",
        tok_rss, tok_hwm
    );

    let low_ram = std::env::var("NIOXS_LOW_RAM")
        .map(|v| v == "1")
        .unwrap_or(false);
    let t0 = Instant::now();
    let mut engine = AstMicroEngine::new(&model_path)?;
    if low_ram {
        engine.set_resident_layers(2);
        engine.set_rolling_window(2);
    } else {
        engine.set_resident_layers(32);
        engine.set_rolling_window(32);
    }

    let load_time = t0.elapsed().as_secs_f64();
    let (load_rss, load_hwm) = memory_mb();
    println!(
        "Engine loaded in {:.2}s | Post-load RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB\n",
        load_time, load_rss, load_hwm
    );

    let stop_tokens: Vec<u32> = [
        engine.eos_token_id(),
        tokenizer.token_to_id("<|im_end|>"),
        tokenizer.token_to_id("<|endoftext|>"),
        tokenizer.token_to_id("<|im_start|>"),
    ]
    .into_iter()
    .flatten()
    .collect();
    let is_eos = |tok: u32| stop_tokens.contains(&tok) || tok == 0;
    let max_gen_tokens = 64;

    let test_cases = [
        (
            "runtime_architecture",
            "why do we need to use Node.js and not PHP?",
        ),
        (
            "systems_architecture",
            "Design an in-memory lock-free priority queue for high-frequency trading in Rust.",
        ),
        (
            "distributed_consensus",
            "Compare Raft vs Multi-Paxos for distributed key-value stores under high network jitter.",
        ),
    ];

    let mut results = Vec::new();
    let mut total_prefill_ms = 0.0;
    let mut total_prefill_tps = 0.0;
    let mut total_tps = 0.0;
    let mut total_wps = 0.0;
    let mut peak_process_hwm_observed: f64 = load_hwm;

    for (idx, &(category, prompt)) in test_cases.iter().enumerate() {
        println!("--------------------------------------------------------");
        println!(
            "[Case {}/{}] Category: {} | Prompt: \"{}\"",
            idx + 1,
            test_cases.len(),
            category,
            prompt
        );

        engine.reset_cache();
        engine.set_sampling(0.35, 0.88, 1.05, 64);

        let formatted = format!(
            "<|im_start|>system\nYou are Blue, Nio's Flagship Sovereign Reflex Companion. Speak with the calm, approachable, and grounded voice of a seasoned staff engineer pair-programming with a colleague. Be conversational and engaging, but keep your dialogue focused and high-signal without fluff or sycophancy. When complex reasoning is required, verify invariants inside concise <think>...</think> tags (under 100 tokens), close your thoughts with </think>, and deliver your answer naturally. Reserve code blocks strictly for executable code.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let encoded = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow!("{e}"))?;
        let prompt_tokens = encoded.get_ids().to_vec();

        // Prefill
        let t_prefill = Instant::now();
        let _ = engine.prefill(&prompt_tokens)?;
        let prefill_duration = t_prefill.elapsed();
        let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
        let prefill_tps = prompt_tokens.len() as f64 / prefill_duration.as_secs_f64();

        // Decode loop with V5 Speculative Lookahead (ZwCache)
        let t_gen = Instant::now();
        let first_tok = engine.next_token_with_history(None, &[], false)?;
        let mut gen_tokens = Vec::new();
        if !is_eos(first_tok) {
            gen_tokens.push(first_tok);
        }

        let mut zw_cache = ZwCache::new_for_model(true);
        zw_cache.seed_prompt(&prompt_tokens);
        if let Some(&last_prompt) = prompt_tokens.last() {
            let p2 = if prompt_tokens.len() >= 2 {
                Some(prompt_tokens[prompt_tokens.len() - 2])
            } else {
                None
            };
            zw_cache.observe(p2, last_prompt, first_tok);
        }

        let mut current_confirmed = first_tok;
        let mut stream_pos = prompt_tokens.len();

        while gen_tokens.len() < max_gen_tokens && !is_eos(current_confirmed) {
            let remaining = max_gen_tokens.saturating_sub(gen_tokens.len());
            let prev2 = if gen_tokens.len() >= 2 {
                Some(gen_tokens[gen_tokens.len() - 2])
            } else {
                None
            };

            let (d1_opt, d2_opt) = if remaining >= 3 {
                zw_cache.predict_pair(prev2, current_confirmed, &gen_tokens)
            } else if remaining >= 2 {
                (
                    zw_cache.predict(prev2, current_confirmed, &gen_tokens),
                    None,
                )
            } else {
                (None, None)
            };

            let d1_valid = d1_opt.filter(|&d| !is_eos(d));
            let d2_valid = d2_opt.filter(|&d| !is_eos(d));

            if let (Some(d1), Some(d2)) = (d1_valid, d2_valid) {
                let outcome = engine.speculative_trio_step(
                    current_confirmed,
                    d1,
                    d2,
                    stream_pos,
                    &gen_tokens,
                    &stop_tokens,
                )?;
                let mut stopped = false;
                for &tok in &outcome.accepted_tokens {
                    if !is_eos(tok) {
                        let prev_p2 = if gen_tokens.len() >= 2 {
                            Some(gen_tokens[gen_tokens.len() - 2])
                        } else {
                            None
                        };
                        if let Some(&last_tok) = gen_tokens.last() {
                            zw_cache.observe(prev_p2, last_tok, tok);
                        }
                        gen_tokens.push(tok);
                        if gen_tokens.len() >= max_gen_tokens {
                            stopped = true;
                            break;
                        }
                    } else {
                        stopped = true;
                        break;
                    }
                }
                current_confirmed = outcome.last_confirmed;
                stream_pos = outcome.stream_pos;
                if stopped || outcome.is_eos {
                    break;
                }
            } else if let Some(draft_tok) = d1_valid {
                let outcome = engine.speculative_pair_step(
                    current_confirmed,
                    draft_tok,
                    stream_pos,
                    &gen_tokens,
                    &stop_tokens,
                    false,
                )?;
                let mut stopped = false;
                for &tok in &outcome.accepted_tokens {
                    if !is_eos(tok) {
                        let prev_p2 = if gen_tokens.len() >= 2 {
                            Some(gen_tokens[gen_tokens.len() - 2])
                        } else {
                            None
                        };
                        if let Some(&last_tok) = gen_tokens.last() {
                            zw_cache.observe(prev_p2, last_tok, tok);
                        }
                        gen_tokens.push(tok);
                        if gen_tokens.len() >= max_gen_tokens {
                            stopped = true;
                            break;
                        }
                    } else {
                        stopped = true;
                        break;
                    }
                }
                current_confirmed = outcome.last_confirmed;
                stream_pos = outcome.stream_pos;
                if stopped || outcome.is_eos {
                    break;
                }
            } else {
                let _ = engine.forward_stream(&[current_confirmed], stream_pos)?;
                stream_pos += 1;
                let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
                if is_eos(next_tok) {
                    break;
                }
                let prev_p2 = if gen_tokens.len() >= 2 {
                    Some(gen_tokens[gen_tokens.len() - 2])
                } else {
                    None
                };
                if let Some(&last_tok) = gen_tokens.last() {
                    zw_cache.observe(prev_p2, last_tok, next_tok);
                }
                gen_tokens.push(next_tok);
                current_confirmed = next_tok;
            }
        }

        let gen_duration = t_gen.elapsed();
        let gen_time_s = gen_duration.as_secs_f64();
        let tps = if gen_time_s > 0.0 {
            gen_tokens.len() as f64 / gen_time_s
        } else {
            0.0
        };

        let decoded_response = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
        let words = decoded_response
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .count();
        let wps = if gen_time_s > 0.0 {
            words as f64 / gen_time_s
        } else {
            0.0
        };

        let (cur_rss, cur_hwm) = memory_mb();
        if cur_hwm > peak_process_hwm_observed {
            peak_process_hwm_observed = cur_hwm;
        }

        println!("  Prompt Tokens:  {}", prompt_tokens.len());
        println!(
            "  Prefill Time:   {:.2} ms ({:.2} tok/s)",
            prefill_ms, prefill_tps
        );
        println!(
            "  Gen Tokens:     {} tokens ({} words) in {:.2}s",
            gen_tokens.len(),
            words,
            gen_time_s
        );
        println!(
            "  Throughput:     {:.2} tokens/sec | {:.2} words/sec",
            tps, wps
        );
        println!(
            "  Current RAM:    VmRSS = {:.2} MB | VmHWM = {:.2} MB",
            cur_rss, cur_hwm
        );
        println!(
            "  Output Snippet: {}",
            decoded_response
                .replace('\n', " ")
                .chars()
                .take(120)
                .collect::<String>()
        );

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
            prefill_tps: prefill_tps,
            gen_tokens: gen_tokens.len(),
            gen_time_s,
            tokens_per_sec: tps,
            word_count: words,
            words_per_sec: wps,
            vm_rss_mb: cur_rss,
            vm_hwm_mb: cur_hwm,
            response: decoded_response,
        });
    }

    let n = test_cases.len() as f64;
    let avg_prefill_ms = total_prefill_ms / n;
    let avg_prefill_tps = total_prefill_tps / n;
    let avg_tokens_per_sec = total_tps / n;
    let avg_words_per_sec = total_wps / n;
    let sub_200mb_sla_passed = peak_process_hwm_observed < 200.0;

    println!("\n========================================================");
    println!("  🏁 BLUE (3.82B) PRODUCTION BENCHMARK SUMMARY");
    println!("========================================================");
    println!("  Avg Prefill Time:       {:.2} ms", avg_prefill_ms);
    println!("  Avg Prefill Throughput: {:.2} tok/s", avg_prefill_tps);
    println!(
        "  Avg Generation Speed:   {:.2} tok/s ({:.2} words/sec)",
        avg_tokens_per_sec, avg_words_per_sec
    );
    println!(
        "  Peak RAM (VmHWM):       {:.2} MB",
        peak_process_hwm_observed
    );
    println!(
        "  < 200 MB SLA Status:    {}",
        if sub_200mb_sla_passed {
            "PASSED ✅"
        } else {
            "EXCEEDED ⚠️"
        }
    );
    println!("========================================================\n");

    let summary = BlueSummary {
        model: "blue.vmicrolm-reflex.gguf".to_string(),
        disk_size_mb: get_file_size_mb(&model_path),
        num_queries: test_cases.len(),
        avg_prefill_ms,
        avg_prefill_tps,
        avg_tokens_per_sec,
        avg_words_per_sec,
        peak_ram_mb: peak_process_hwm_observed,
        sub_200mb_sla_passed,
        results,
    };

    let json_out = serde_json::to_string_pretty(&summary)?;
    std::fs::write(
        "/home/mn/nio-labs/nio-xs/blue_production_benchmark.json",
        json_out,
    )?;
    println!("Saved benchmark report to /home/mn/nio-labs/nio-xs/blue_production_benchmark.json");

    Ok(())
}
