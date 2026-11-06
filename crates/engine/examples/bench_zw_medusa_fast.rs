use anyhow::{Result, anyhow};
use nioxs_engine::ast_engine::AstMicroEngine;
use std::collections::HashMap;
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

/// Zero-Weight Context Cache
struct ContextCache {
    bigram_counts: HashMap<u32, HashMap<u32, usize>>,
    trigram_counts: HashMap<(u32, u32), HashMap<u32, usize>>,
}

impl ContextCache {
    fn new() -> Self {
        Self {
            bigram_counts: HashMap::new(),
            trigram_counts: HashMap::new(),
        }
    }

    fn observe(&mut self, prev: Option<u32>, curr: u32, next: u32) {
        *self
            .bigram_counts
            .entry(curr)
            .or_default()
            .entry(next)
            .or_default() += 1;
        if let Some(p) = prev {
            *self
                .trigram_counts
                .entry((p, curr))
                .or_default()
                .entry(next)
                .or_default() += 1;
        }
    }

    fn predict(&self, prev: Option<u32>, curr: u32) -> Option<u32> {
        if let Some(p) = prev {
            if let Some(cands) = self.trigram_counts.get(&(p, curr)) {
                if let Some((&cand, _)) = cands.iter().max_by_key(|&(_, count)| *count) {
                    return Some(cand);
                }
            }
        }
        if let Some(cands) = self.bigram_counts.get(&curr) {
            if let Some((&cand, _)) = cands.iter().max_by_key(|&(_, count)| *count) {
                return Some(cand);
            }
        }
        None
    }
}

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    engine.set_resident_layers(7);
    engine.set_rolling_window(4);

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let prompt = "Explain how Raft avoids split-brain during leader election when a network partition occurs.";
    let formatted = format!(
        "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );

    let encoded = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    // Populate ContextCache from prompt
    let mut cache = ContextCache::new();
    for i in 1..prompt_tokens.len() {
        let p = if i >= 2 {
            Some(prompt_tokens[i - 2])
        } else {
            None
        };
        cache.observe(p, prompt_tokens[i - 1], prompt_tokens[i]);
    }
    if let Some(&last_p) = prompt_tokens.last() {
        let p = if prompt_tokens.len() >= 2 {
            Some(prompt_tokens[prompt_tokens.len() - 2])
        } else {
            None
        };
        cache.observe(p, last_p, first_tok);
    }

    let mut gen_tokens = vec![first_tok];
    let mut current_confirmed = first_tok;
    let mut stream_pos = prompt_tokens.len();
    let mut matches = 0usize;
    let mut misses = 0usize;
    let mut ar_steps = 0usize;

    let t_start = Instant::now();
    for step in 0..45 {
        let prev = if gen_tokens.len() >= 2 {
            Some(gen_tokens[gen_tokens.len() - 2])
        } else {
            None
        };
        let draft_opt = cache.predict(prev, current_confirmed);

        if let Some(draft_tok) = draft_opt.filter(|&d| d != eos_id && d != 0) {
            let t_step = Instant::now();
            let outcome = engine.speculative_pair_step(
                current_confirmed,
                draft_tok,
                stream_pos,
                &gen_tokens,
                &[eos_id, 0],
                false,
            )?;
            let step_dur = t_step.elapsed();

            if outcome.draft_matched {
                matches += 1;
            } else {
                misses += 1;
            }

            println!(
                "Step {:02} [SPEC] | draft={:6} ({:10?}) | accepted={:?} | matched={:5} | time={:5.1}ms",
                step,
                draft_tok,
                tokenizer.decode(&[draft_tok], false).unwrap_or_default(),
                outcome.accepted_tokens,
                outcome.draft_matched,
                step_dur.as_secs_f64() * 1000.0,
            );

            for &tok in &outcome.accepted_tokens {
                let p = if gen_tokens.len() >= 2 {
                    Some(gen_tokens[gen_tokens.len() - 2])
                } else {
                    None
                };
                if let Some(&last) = gen_tokens.last() {
                    cache.observe(p, last, tok);
                }
                gen_tokens.push(tok);
            }

            current_confirmed = outcome.last_confirmed;
            stream_pos = outcome.stream_pos;

            if outcome.is_eos {
                break;
            }
        } else {
            let t_step = Instant::now();
            engine.forward_stream(&[current_confirmed], stream_pos)?;
            stream_pos += 1;

            let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
            let step_dur = t_step.elapsed();
            ar_steps += 1;

            if next_tok == eos_id || next_tok == 0 {
                break;
            }

            let p = if gen_tokens.len() >= 2 {
                Some(gen_tokens[gen_tokens.len() - 2])
            } else {
                None
            };
            cache.observe(p, current_confirmed, next_tok);

            gen_tokens.push(next_tok);
            current_confirmed = next_tok;

            println!(
                "Step {:02} [AR]   | confirmed={:6} ({:10?}) | time={:5.1}ms",
                step,
                next_tok,
                tokenizer.decode(&[next_tok], false).unwrap_or_default(),
                step_dur.as_secs_f64() * 1000.0,
            );
        }
    }

    let elapsed = t_start.elapsed();
    let (rss_end, hwm) = memory_mb();

    let full_text = tokenizer.decode(&gen_tokens, true).unwrap_or_default();
    let words = full_text.split_whitespace().count();
    let tok_rate = gen_tokens.len() as f64 / elapsed.as_secs_f64();
    let word_rate = words as f64 / elapsed.as_secs_f64();

    println!("\n========================================================");
    println!("  RESULTS SUMMARY");
    println!("========================================================");
    println!("Tokens generated: {}", gen_tokens.len());
    println!("Words generated:  {}", words);
    println!("Total time:       {:.3} s", elapsed.as_secs_f64());
    println!(
        "Spec matches:     {} / {} ({:.1}%)",
        matches,
        matches + misses,
        matches as f64 / (matches + misses).max(1) as f64 * 100.0
    );
    println!("AR steps:         {}", ar_steps);
    println!("Token rate:       {:.2} tok/s", tok_rate);
    println!("Word rate:        {:.2} words/sec", word_rate);
    println!("End RAM (VmRSS):  {:.2} MB", rss_end);
    println!("Peak RAM (VmHWM): {:.2} MB", hwm);

    Ok(())
}
