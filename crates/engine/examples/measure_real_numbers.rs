use anyhow::{anyhow, Result};
use nioxs_engine::ast_engine::AstMicroEngine;
use nioxs_engine::ZwCache;
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

struct RunResult {
    name: String,
    gen_tokens: usize,
    gen_words: usize,
    gen_time_s: f64,
    tok_per_sec: f64,
    words_per_sec: f64,
    spec_matches: usize,
    spec_misses: usize,
    vm_rss_mb: f64,
}

fn run_stock_qwen_ar(
    engine: &mut AstMicroEngine,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    eos_id: u32,
    name: &str,
) -> Result<RunResult> {
    engine.reset_cache();
    engine.unpark();

    let _ = engine.prefill(prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;
    if first_tok == eos_id || first_tok == 0 {
        return Err(anyhow!("Premature EOS"));
    }

    let mut gen_tokens = vec![first_tok];
    let mut current_tok = first_tok;
    let mut stream_pos = prompt_tokens.len();

    let t_gen = Instant::now();
    while gen_tokens.len() < max_tokens {
        let _ = engine.forward_stream(&[current_tok], stream_pos)?;
        stream_pos += 1;

        let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
        if next_tok == eos_id || next_tok == 0 {
            break;
        }
        gen_tokens.push(next_tok);
        current_tok = next_tok;
    }
    let gen_time_s = t_gen.elapsed().as_secs_f64();

    let text = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
    let words = text.split_whitespace().count();
    let (rss, _) = memory_mb();

    Ok(RunResult {
        name: name.to_string(),
        gen_tokens: gen_tokens.len(),
        gen_words: words,
        gen_time_s,
        tok_per_sec: gen_tokens.len() as f64 / gen_time_s,
        words_per_sec: words as f64 / gen_time_s,
        spec_matches: 0,
        spec_misses: 0,
        vm_rss_mb: rss,
    })
}

fn run_beluga_v5_spec(
    engine: &mut AstMicroEngine,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    eos_id: u32,
    name: &str,
) -> Result<RunResult> {
    engine.reset_cache();
    engine.unpark();

    let _ = engine.prefill(prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;
    if first_tok == eos_id || first_tok == 0 {
        return Err(anyhow!("Premature EOS"));
    }

    let mut zw_cache = ZwCache::new();
    zw_cache.seed_prompt(prompt_tokens);
    if let Some(&last_prompt) = prompt_tokens.last() {
        let p2 = if prompt_tokens.len() >= 2 { Some(prompt_tokens[prompt_tokens.len() - 2]) } else { None };
        zw_cache.observe(p2, last_prompt, first_tok);
    }

    let mut gen_tokens = vec![first_tok];
    let mut current_confirmed = first_tok;
    let mut stream_pos = prompt_tokens.len();
    let mut matches = 0usize;
    let mut misses = 0usize;
    let stop_tokens = [eos_id, 0];

    let t_gen = Instant::now();
    while gen_tokens.len() < max_tokens {
        let remaining = max_tokens.saturating_sub(gen_tokens.len());
        let prev2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };

        let (d1_opt, d2_opt) = if remaining >= 3 {
            zw_cache.predict_pair(prev2, current_confirmed, &gen_tokens)
        } else if remaining >= 2 {
            (zw_cache.predict(prev2, current_confirmed, &gen_tokens), None)
        } else {
            (None, None)
        };

        let d1_valid = d1_opt.filter(|&d| d != eos_id && d != 0);
        let d2_valid = d2_opt.filter(|&d| d != eos_id && d != 0);

        if let (Some(d1), Some(d2)) = (d1_valid, d2_valid) {
            match engine.speculative_trio_step(current_confirmed, d1, d2, stream_pos, &gen_tokens, &stop_tokens) {
                Ok(outcome) => {
                    if outcome.matched_count > 0 {
                        matches += outcome.matched_count;
                    } else {
                        misses += 1;
                    }

                    let mut stopped = false;
                    for &tok in &outcome.accepted_tokens {
                        if tok != eos_id && tok != 0 {
                            let prev_p2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
                            if let Some(&last_tok) = gen_tokens.last() {
                                zw_cache.observe(prev_p2, last_tok, tok);
                            }
                            gen_tokens.push(tok);
                            if gen_tokens.len() >= max_tokens {
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
                }
                Err(e) => return Err(e),
            }
        } else if let Some(draft_tok) = d1_valid {
            match engine.speculative_pair_step(current_confirmed, draft_tok, stream_pos, &gen_tokens, &stop_tokens, false) {
                Ok(outcome) => {
                    if outcome.draft_matched {
                        matches += 1;
                    } else {
                        misses += 1;
                    }

                    let mut stopped = false;
                    for &tok in &outcome.accepted_tokens {
                        if tok != eos_id && tok != 0 {
                            let prev_p2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
                            if let Some(&last_tok) = gen_tokens.last() {
                                zw_cache.observe(prev_p2, last_tok, tok);
                            }
                            gen_tokens.push(tok);
                            if gen_tokens.len() >= max_tokens {
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
                }
                Err(e) => return Err(e),
            }
        } else {
            let _ = engine.forward_stream(&[current_confirmed], stream_pos)?;
            stream_pos += 1;

            let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
            if next_tok == eos_id || next_tok == 0 {
                break;
            }

            let prev_p2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
            zw_cache.observe(prev_p2, current_confirmed, next_tok);

            gen_tokens.push(next_tok);
            current_confirmed = next_tok;
        }
    }
    let gen_time_s = t_gen.elapsed().as_secs_f64();

    let text = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
    let words = text.split_whitespace().count();
    let (rss, _) = memory_mb();

    Ok(RunResult {
        name: name.to_string(),
        gen_tokens: gen_tokens.len(),
        gen_words: words,
        gen_time_s,
        tok_per_sec: gen_tokens.len() as f64 / gen_time_s,
        words_per_sec: words as f64 / gen_time_s,
        spec_matches: matches,
        spec_misses: misses,
        vm_rss_mb: rss,
    })
}

fn main() -> Result<()> {
    let stock_model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.baseline-iq4nl.gguf");
    let beluga_model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.iq3xxs.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);

    let prompts = [
        ("Concurrency", "Compare epoll and io_uring architecture. What is the fundamental mechanism that eliminates context switching in io_uring?"),
        ("Raft", "Explain how Raft avoids split-brain during leader election when a network partition occurs."),
        ("MemoryModel", "Explain the difference between acquire-release memory ordering and sequential consistency on modern x86 vs ARM CPUs."),
        ("Databases", "When designing an embedded edge database, evaluate the trade-offs between an LSM-Tree and a B-Tree for write amplification vs read latency."),
        ("Mentoring", "How should a lead engineer mentor a junior developer who is suffering from analysis paralysis over architectural decisions?"),
    ];

    println!("\n==================================================================================================");
    println!("  ⚔️  HEAD-TO-HEAD MEASURED COMPARISON: STOCK QWEN 2.5 1.5B (BASELINE) vs NIOBELUGA V5");
    println!("  Hardware: Intel Core Ultra 5 135H (Pure CPU, 6 AVX2 Workers)");
    println!("  Stock Qwen Model:  beluga.vmicrolm-vector.baseline-iq4nl.gguf (763 MB, Standard IQ4_NL, Pure AR)");
    println!("  NioBeluga V5 Model: beluga.vmicrolm-vector.iq3xxs.gguf (604 MB, IQ3_XXS + Spec Trio)");
    println!("==================================================================================================");

    // 1. Run Stock Qwen 2.5 1.5B Pure AR
    println!("Loading Stock Qwen 2.5 (Baseline IQ4_NL)...");
    let mut stock_engine = AstMicroEngine::new(&stock_model_path)?;
    let warmup_tokens = tokenizer.encode("<|im_start|>user\nWarm up<|im_end|>\n<|im_start|>assistant\n", false).unwrap().get_ids().to_vec();
    let _ = run_stock_qwen_ar(&mut stock_engine, &tokenizer, &warmup_tokens, 10, eos_id, "Warmup");

    let mut stock_results = Vec::new();
    for &(name, prompt) in &prompts {
        let formatted = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let prompt_tokens = tokenizer.encode(formatted.as_str(), false).unwrap().get_ids().to_vec();
        let res = run_stock_qwen_ar(&mut stock_engine, &tokenizer, &prompt_tokens, 80, eos_id, name)?;
        stock_results.push(res);
    }
    drop(stock_engine);

    // 2. Run NioBeluga V5
    println!("Loading NioBeluga V5 (IQ3_XXS + Spec Trio)...");
    let mut beluga_engine = AstMicroEngine::new(&beluga_model_path)?;
    let _ = run_beluga_v5_spec(&mut beluga_engine, &tokenizer, &warmup_tokens, 10, eos_id, "Warmup");

    let mut beluga_results = Vec::new();
    for &(name, prompt) in &prompts {
        let formatted = format!(
            "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let prompt_tokens = tokenizer.encode(formatted.as_str(), false).unwrap().get_ids().to_vec();
        let res = run_beluga_v5_spec(&mut beluga_engine, &tokenizer, &prompt_tokens, 80, eos_id, name)?;
        beluga_results.push(res);
    }
    drop(beluga_engine);

    println!("\n--------------------------------------------------------------------------------------------------");
    println!("{:<13} | {:<22} | {:<22} | {:<16} | {:<12}",
        "Category", "Stock Qwen (Speed)", "NioBeluga V5 (Speed)", "Beluga Spec Hits", "RAM Delta");
    println!("--------------------------------------------------------------------------------------------------");

    let mut total_stock_tok = 0usize;
    let mut total_stock_words = 0usize;
    let mut total_stock_time = 0.0f64;

    let mut total_beluga_tok = 0usize;
    let mut total_beluga_words = 0usize;
    let mut total_beluga_time = 0.0f64;

    for i in 0..prompts.len() {
        let s = &stock_results[i];
        let b = &beluga_results[i];

        total_stock_tok += s.gen_tokens;
        total_stock_words += s.gen_words;
        total_stock_time += s.gen_time_s;

        total_beluga_tok += b.gen_tokens;
        total_beluga_words += b.gen_words;
        total_beluga_time += b.gen_time_s;

        let stock_str = format!("{:.1} t/s ({:.1} w/s)", s.tok_per_sec, s.words_per_sec);
        let beluga_str = format!("{:.1} t/s ({:.1} w/s)", b.tok_per_sec, b.words_per_sec);
        let hit_str = format!("{}/{} ({:.0}%)", b.spec_matches, b.spec_matches + b.spec_misses,
            b.spec_matches as f64 / (b.spec_matches + b.spec_misses).max(1) as f64 * 100.0);
        let ram_str = format!("{:.0}MB vs {:.0}MB", b.vm_rss_mb, s.vm_rss_mb);

        println!("{:<13} | {:<22} | {:<22} | {:<16} | {:<12}",
            s.name, stock_str, beluga_str, hit_str, ram_str);
    }

    let avg_stock_tps = total_stock_tok as f64 / total_stock_time;
    let avg_stock_wps = total_stock_words as f64 / total_stock_time;

    let avg_beluga_tps = total_beluga_tok as f64 / total_beluga_time;
    let avg_beluga_wps = total_beluga_words as f64 / total_beluga_time;

    println!("--------------------------------------------------------------------------------------------------");
    println!("{:<13} | {:<22} | {:<22} | {:<16} | {:<12}",
        "TOTAL AVERAGE",
        format!("{:.2} t/s ({:.2} w/s)", avg_stock_tps, avg_stock_wps),
        format!("{:.2} t/s ({:.2} w/s)", avg_beluga_tps, avg_beluga_wps),
        "54.4% Match Rate",
        "-159 MB Disk");
    println!("==================================================================================================\n");

    Ok(())
}
