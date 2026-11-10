use anyhow::{Result, anyhow};
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

fn memory_mb() -> (f64, f64, f64) {
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
    (get_val("RssAnon:"), get_val("VmRSS:"), get_val("VmHWM:"))
}

fn main() -> Result<()> {
    let model_path = std::env::var("NIOXS_BLUE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root/.nioxs/models/blue.vmicrolm-reflex.gguf"));
    let tokenizer_path = std::env::var("NIOXS_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root/.nioxs/models/blue-tokenizer.json"));

    if !model_path.exists() {
        return Err(anyhow!("Blue model missing at {:?}", model_path));
    }
    if !tokenizer_path.exists() {
        return Err(anyhow!("Blue tokenizer missing at {:?}", tokenizer_path));
    }

    println!("\n========================================================");
    println!("  🔷 NIOBLUE (3.82B) - PLD SPECULATIVE DECODING EVAL");
    println!("  P-Core Pinning: 4 Workers (CPUs 0, 2, 4, 6)");
    println!("  Memory Constraint: Strictly < 300 MB Active RAM");
    println!("========================================================");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);

    let resident_env = std::env::var("NIOXS_BLUE_RESIDENT_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3);
    let window_env = std::env::var("NIOXS_BLUE_ROLLING_WINDOW")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3);

    let t0 = Instant::now();
    let mut engine = AstMicroEngine::new(&model_path)?;
    engine.set_resident_layers(resident_env);
    engine.set_rolling_window(window_env);

    let load_time = t0.elapsed().as_secs_f64();
    let (load_anon, load_rss, load_hwm) = memory_mb();
    println!(
        "Engine loaded in {:.2}s (resident_layers={}, rolling_window={})",
        load_time, resident_env, window_env
    );
    println!(
        "Post-load RAM: RssAnon = {:.2} MB | VmRSS = {:.2} MB | VmHWM = {:.2} MB\n",
        load_anon, load_rss, load_hwm
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
        (
            "logic_reasoning",
            "What is 2 + 2? Explain the logic in one short paragraph.",
        ),
    ];

    let max_draft_configs = [0, 1, 2]; // 0 = Pure AR, 1 = PLD 1 draft, 2 = PLD 2 drafts
    let max_gen_tokens = 64;

    #[allow(dead_code)]
    struct ModeStats {
        label: String,
        avg_tps: f64,
        avg_wps: f64,
        avg_prefill_tps: f64,
        drafts_accepted: usize,
        drafts_proposed: usize,
        acceptance_pct: f64,
        peak_anon: f64,
        peak_hwm: f64,
    }

    let mut all_stats = Vec::new();

    for &max_draft in &max_draft_configs {
        let mode_label = if max_draft == 0 {
            "Baseline Pure Autoregressive (AR)".to_string()
        } else {
            format!("PLD Speculative Decoding (max_draft = {max_draft})")
        };

        println!("\n>>> EVALUATING: {} <<<", mode_label);

        let mut total_prefill_tps = 0.0;
        let mut total_tps = 0.0;
        let mut total_wps = 0.0;
        let mut total_drafts_proposed = 0usize;
        let mut total_drafts_accepted = 0usize;
        let mut mode_peak_anon = 0.0f64;
        let mut mode_peak_hwm = 0.0f64;

        for (idx, &(category, prompt)) in test_cases.iter().enumerate() {
            engine.reset_cache();
            engine.set_sampling(0.35, 0.88, 1.05, 64);

            let formatted = format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                nioxs_engine::blue_system()
            );
            let encoded = tokenizer
                .encode(formatted.as_str(), false)
                .map_err(|e| anyhow!("{e}"))?;
            let prompt_tokens = encoded.get_ids().to_vec();

            // Prefill
            let t_prefill = Instant::now();
            let _ = engine.prefill(&prompt_tokens)?;
            let prefill_duration = t_prefill.elapsed();
            let prefill_tps = prompt_tokens.len() as f64 / prefill_duration.as_secs_f64();
            total_prefill_tps += prefill_tps;

            let mut gen_tokens = Vec::new();
            let mut all_tokens = prompt_tokens.clone();
            let mut case_drafts_proposed = 0usize;
            let mut case_drafts_accepted = 0usize;

            let t_gen = Instant::now();

            if max_draft == 0 {
                // Pure AR Baseline (1 token per step)
                let first_tok = engine.next_token_with_history(None, &[], false)?;
                if !is_eos(first_tok) {
                    gen_tokens.push(first_tok);
                    let mut cur = first_tok;
                    let mut pos = prompt_tokens.len();

                    while gen_tokens.len() < max_gen_tokens {
                        let _ = engine.forward_stream(&[cur], pos)?;
                        pos += 1;
                        let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
                        if is_eos(next_tok) {
                            break;
                        }
                        gen_tokens.push(next_tok);
                        cur = next_tok;
                    }
                }
            } else {
                // Speculative PLD
                let first_tok = engine.next_token_with_history(None, &[], false)?;
                if !is_eos(first_tok) {
                    gen_tokens.push(first_tok);
                    all_tokens.push(first_tok);

                    let mut current_confirmed = first_tok;
                    let mut current_pos = prompt_tokens.len();

                    while gen_tokens.len() < max_gen_tokens && !is_eos(current_confirmed) {
                        let remaining = max_gen_tokens.saturating_sub(gen_tokens.len());
                        let draft_budget = max_draft.min(remaining.saturating_sub(1));

                        let res = engine.speculative_pld_step(
                            &all_tokens,
                            current_confirmed,
                            current_pos,
                            &gen_tokens,
                            draft_budget,
                            &stop_tokens,
                        )?;

                        let n_accepted = res.accepted_tokens.len();
                        if res.drafts_proposed > 0 {
                            case_drafts_proposed += res.drafts_proposed;
                            if n_accepted > 1 {
                                case_drafts_accepted += n_accepted - 1;
                            }
                        }

                        let mut stopped = false;
                        for tok in res.accepted_tokens {
                            if is_eos(tok) {
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
            let tps = if gen_time_s > 0.0 {
                gen_tokens.len() as f64 / gen_time_s
            } else {
                0.0
            };
            let decoded = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
            let words = decoded
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .count();
            let wps = if gen_time_s > 0.0 {
                words as f64 / gen_time_s
            } else {
                0.0
            };

            total_tps += tps;
            total_wps += wps;
            total_drafts_proposed += case_drafts_proposed;
            total_drafts_accepted += case_drafts_accepted;

            let (cur_anon, cur_rss, cur_hwm) = memory_mb();
            mode_peak_anon = mode_peak_anon.max(cur_anon);
            mode_peak_hwm = mode_peak_hwm.max(cur_hwm);

            let accept_rate = if case_drafts_proposed > 0 {
                (case_drafts_accepted as f64 / case_drafts_proposed as f64) * 100.0
            } else {
                0.0
            };

            println!(
                "  [{}/{}] {:<22}: {:5.2} tok/s | {:5.2} w/s | RAM RssAnon: {:5.1} MB (HWM: {:5.1} MB) | Drafts: {}/{} ({:.1}%)",
                idx + 1,
                test_cases.len(),
                category,
                tps,
                wps,
                cur_anon,
                cur_hwm,
                case_drafts_accepted,
                case_drafts_proposed,
                accept_rate
            );
        }

        let n = test_cases.len() as f64;
        let avg_tps = total_tps / n;
        let avg_wps = total_wps / n;
        let avg_prefill_tps = total_prefill_tps / n;
        let overall_accept = if total_drafts_proposed > 0 {
            (total_drafts_accepted as f64 / total_drafts_proposed as f64) * 100.0
        } else {
            0.0
        };

        all_stats.push(ModeStats {
            label: mode_label.clone(),
            avg_tps,
            avg_wps,
            avg_prefill_tps,
            drafts_accepted: total_drafts_accepted,
            drafts_proposed: total_drafts_proposed,
            acceptance_pct: overall_accept,
            peak_anon: mode_peak_anon,
            peak_hwm: mode_peak_hwm,
        });

        println!(
            "  --> Summary {}: Avg Generation = {:.2} tok/s ({:.2} w/s) | Draft Acc = {:.1}% ({}/{}) | Peak RssAnon = {:.1} MB",
            mode_label, avg_tps, avg_wps, overall_accept, total_drafts_accepted, total_drafts_proposed, mode_peak_anon
        );
    }

    println!("\n========================================================");
    println!("  FINAL COMPARISON & VALIDATION SUMMARY");
    println!("========================================================");
    let baseline_tps = all_stats[0].avg_tps;
    for stat in &all_stats {
        let speedup = if baseline_tps > 0.0 {
            stat.avg_tps / baseline_tps
        } else {
            1.0
        };
        println!(
            "  * {:<42} | Gen: {:5.2} tok/s ({:5.2} w/s) | Speedup: {:4.2}x | Draft Acc: {:4.1}% | Peak RssAnon: {:5.1} MB | Peak HWM: {:5.1} MB",
            stat.label, stat.avg_tps, stat.avg_wps, speedup, stat.acceptance_pct, stat.peak_anon, stat.peak_hwm
        );
    }
    println!("========================================================\n");

    Ok(())
}
