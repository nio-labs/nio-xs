use anyhow::{anyhow, Result};
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

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    if !model_path.exists() {
        return Err(anyhow!("Beluga model missing at {:?}", model_path));
    }
    if !tokenizer_path.exists() {
        return Err(anyhow!("Beluga tokenizer missing at {:?}", tokenizer_path));
    }

    println!("\n========================================================");
    println!("  🐋 NIOBELUGA (1.54B) - PLD SPECULATIVE DECODING EVAL");
    println!("========================================================");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let system_prompt = "You are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions. Defer raw low-level code generation to Cuda.";

    let test_cases = [
        ("systems_architecture", "Design an in-memory lock-free priority queue for high-frequency trading in Rust."),
        ("memory_safety", "Explain the memory safety implications and cache hazards of uninitialized memory in tensor runtimes."),
        ("consensus_protocols", "Compare Raft vs Multi-Paxos for distributed key-value stores under high network jitter."),
    ];

    let max_draft_configs = [0, 2, 3]; // 0 = Pure AR, 2 = PLD 2, 3 = PLD 3

    for &max_draft in &max_draft_configs {
        let mode_label = if max_draft == 0 {
            "Baseline Pure Autoregressive (AR)".to_string()
        } else {
            format!("Speculative Prompt Lookup Decoding (PLD max_draft={max_draft})")
        };

        println!("\n>>> RUNNING MODE: {} <<<", mode_label);

        let mut total_tps = 0.0;
        let mut total_wps = 0.0;
        let mut total_drafts_proposed = 0usize;
        let mut total_drafts_accepted = 0usize;
        let mut total_steps = 0usize;
        let max_gen_tokens = 64;

        for (idx, &(category, prompt)) in test_cases.iter().enumerate() {
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
            let _prefill_time = t_prefill.elapsed();

            let mut gen_tokens = Vec::new();
            let mut all_tokens = prompt_tokens.clone();
            let mut drafts_proposed = 0usize;
            let mut drafts_accepted = 0usize;
            let mut steps = 0usize;

            let t_gen = Instant::now();

            if max_draft == 0 {
                // Pure AR
                for _ in 0..max_gen_tokens {
                    let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
                    if next_tok == eos_id || next_tok == 0 {
                        break;
                    }
                    gen_tokens.push(next_tok);
                    let _ = engine.forward_stream(&[next_tok], prompt_tokens.len() + gen_tokens.len() - 1)?;
                }
            } else {
                // PLD Speculative Decoding
                let first_tok = engine.next_token_with_history(None, &[], false)?;
                if first_tok != eos_id && first_tok != 0 {
                    gen_tokens.push(first_tok);
                    all_tokens.push(first_tok);

                    let mut current_confirmed = first_tok;
                    let mut current_pos = prompt_tokens.len();

                    while gen_tokens.len() < max_gen_tokens {
                        let remaining = max_gen_tokens.saturating_sub(gen_tokens.len());
                        let draft_budget = max_draft.min(remaining.saturating_sub(1));

                        steps += 1;
                        let res = engine.speculative_pld_step(
                            &all_tokens,
                            current_confirmed,
                            current_pos,
                            &gen_tokens,
                            draft_budget,
                            &[eos_id, 0],
                        )?;

                        // Count how many drafts were actually proposed vs accepted
                        let n_accepted = res.accepted_tokens.len();
                        if draft_budget > 0 {
                            drafts_proposed += draft_budget;
                            if n_accepted > 1 {
                                drafts_accepted += n_accepted - 1;
                            }
                        }

                        let mut stopped = false;
                        for tok in res.accepted_tokens {
                            if tok == eos_id || tok == 0 {
                                stopped = true;
                                break;
                            }
                            gen_tokens.push(tok);
                            all_tokens.push(tok);
                            if gen_tokens.len() >= max_gen_tokens {
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
            }

            let gen_duration = t_gen.elapsed();
            let gen_time_s = gen_duration.as_secs_f64();
            let tps = if gen_time_s > 0.0 { gen_tokens.len() as f64 / gen_time_s } else { 0.0 };
            let decoded = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
            let words = decoded.split_whitespace().count();
            let wps = if gen_time_s > 0.0 { words as f64 / gen_time_s } else { 0.0 };

            total_tps += tps;
            total_wps += wps;
            total_drafts_proposed += drafts_proposed;
            total_drafts_accepted += drafts_accepted;
            total_steps += steps;

            let (rss, hwm) = memory_mb();
            let accept_rate = if drafts_proposed > 0 {
                (drafts_accepted as f64 / drafts_proposed as f64) * 100.0
            } else {
                0.0
            };

            println!(
                "  [Case {}/{}] {}: {:.2} tok/s | {:.2} w/s | RAM: {:.1}MB (HWM: {:.1}MB) | drafts accepted: {}/{} ({:.1}%)",
                idx + 1, test_cases.len(), category, tps, wps, rss, hwm, drafts_accepted, drafts_proposed, accept_rate
            );
        }

        let n = test_cases.len() as f64;
        let avg_tps = total_tps / n;
        let avg_wps = total_wps / n;
        let overall_accept = if total_drafts_proposed > 0 {
            (total_drafts_accepted as f64 / total_drafts_proposed as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "  ==> Mode Summary: Avg TPS: {:.2} tok/s | Avg WPS: {:.2} w/s | Overall Draft Acceptance: {}/{} ({:.1}%) across {} steps",
            avg_tps, avg_wps, total_drafts_accepted, total_drafts_proposed, overall_accept, total_steps
        );
    }

    Ok(())
}
