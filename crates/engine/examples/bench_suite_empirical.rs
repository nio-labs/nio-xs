use anyhow::{anyhow, Result};
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

struct FastAdaptiveDrafter {
    bigram_counts: HashMap<u32, HashMap<u32, usize>>,
    trigram_counts: HashMap<(u32, u32), HashMap<u32, usize>>,
}

impl FastAdaptiveDrafter {
    fn new() -> Self {
        Self {
            bigram_counts: HashMap::new(),
            trigram_counts: HashMap::new(),
        }
    }

    fn seed_text(&mut self, text: &str, tokenizer: &Tokenizer, weight: usize) {
        if let Ok(enc) = tokenizer.encode(text, false) {
            let ids = enc.get_ids();
            for window in ids.windows(2) {
                *self.bigram_counts.entry(window[0]).or_default().entry(window[1]).or_default() += weight;
            }
            for window in ids.windows(3) {
                *self.trigram_counts.entry((window[0], window[1])).or_default().entry(window[2]).or_default() += weight * 2;
            }
        }
    }

    fn observe(&mut self, prev: Option<u32>, curr: u32, next: u32) {
        *self.bigram_counts.entry(curr).or_default().entry(next).or_default() += 2;
        if let Some(p) = prev {
            *self.trigram_counts.entry((p, curr)).or_default().entry(next).or_default() += 4;
        }
    }

    fn predict(&self, prev: Option<u32>, curr: u32) -> Option<u32> {
        if let Some(p) = prev {
            if let Some(cands) = self.trigram_counts.get(&(p, curr)) {
                if let Some((&cand, &cnt)) = cands.iter().max_by_key(|&(_, count)| *count) {
                    if cnt >= 2 {
                        return Some(cand);
                    }
                }
            }
        }
        if let Some(cands) = self.bigram_counts.get(&curr) {
            if cands.len() == 1 {
                let (&cand, &cnt) = cands.iter().next().unwrap();
                if cnt >= 2 {
                    return Some(cand);
                }
            } else if let Some((&cand, &cnt)) = cands.iter().max_by_key(|&(_, count)| *count) {
                if cnt >= 6 {
                    return Some(cand);
                }
            }
        }
        None
    }
}

fn run_prompt(
    engine: &mut AstMicroEngine,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_steps: usize,
) -> Result<(usize, usize, f64, f64, f64, usize, usize)> {
    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let formatted = format!(
        "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );

    let encoded = tokenizer.encode(formatted.as_str(), false).map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    let mut drafter = FastAdaptiveDrafter::new();
    // Seed prompt and reasoning idioms
    drafter.seed_text(&formatted, tokenizer, 5);
    drafter.seed_text("<think>...</think>\n1. **Core Mechanism**:\n2. **Architectural Trade-offs**:\n3. **Conclusion**:\n", tokenizer, 2);

    let mut gen_tokens = vec![first_tok];
    let mut current_confirmed = first_tok;
    let mut stream_pos = prompt_tokens.len();
    let mut spec_matches = 0usize;
    let mut spec_misses = 0usize;

    let t_start = Instant::now();
    for _ in 0..max_steps {
        let prev = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
        let draft_opt = drafter.predict(prev, current_confirmed);

        if let Some(draft_tok) = draft_opt.filter(|&d| d != eos_id && d != 0) {
            let outcome = engine.speculative_pair_step(
                current_confirmed,
                draft_tok,
                stream_pos,
                &gen_tokens,
                &[eos_id, 0],
                false,
            )?;

            if outcome.draft_matched {
                spec_matches += 1;
            } else {
                spec_misses += 1;
            }

            for &tok in &outcome.accepted_tokens {
                let p = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
                if let Some(&last) = gen_tokens.last() {
                    drafter.observe(p, last, tok);
                }
                gen_tokens.push(tok);
            }

            current_confirmed = outcome.last_confirmed;
            stream_pos = outcome.stream_pos;

            if outcome.is_eos {
                break;
            }
        } else {
            engine.forward_stream(&[current_confirmed], stream_pos)?;
            stream_pos += 1;

            let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;

            if next_tok == eos_id || next_tok == 0 {
                break;
            }

            let p = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
            drafter.observe(p, current_confirmed, next_tok);

            gen_tokens.push(next_tok);
            current_confirmed = next_tok;
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let full_text = tokenizer.decode(&gen_tokens, true).unwrap_or_default();
    let words = full_text.split_whitespace().count();
    let tok_rate = gen_tokens.len() as f64 / elapsed;
    let word_rate = words as f64 / elapsed;

    Ok((gen_tokens.len(), words, elapsed, tok_rate, word_rate, spec_matches, spec_misses))
}

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    engine.set_resident_layers(7);
    engine.set_rolling_window(4);

    let prompts = [
        ("Distributed Raft", "Explain how Raft avoids split-brain during leader election when a network partition occurs."),
        ("Systems Concurrency", "Compare epoll and io_uring architecture. What is the fundamental mechanism that eliminates context switching in io_uring?"),
        ("Storage Engine", "When designing an embedded edge database, evaluate the trade-offs between an LSM-Tree and a B-Tree for write amplification vs read latency."),
    ];

    println!("\n========================================================");
    println!("  🧠 NIOBELUGA (1.54B) - END-TO-END SPECULATIVE BENCHMARK");
    println!("========================================================");

    for (name, prompt) in &prompts {
        let (tokens, words, elapsed, tok_rate, word_rate, matches, misses) =
            run_prompt(&mut engine, &tokenizer, prompt, 50)?;
        let (rss, hwm) = memory_mb();

        println!(
            "[{:20}] Tokens: {:3} | Words: {:3} | Time: {:.2}s | Matches: {:2}/{:2} ({:4.1}%) | {:5.2} tok/s | {:5.2} words/sec | RAM: {:.1}MB (HWM: {:.1}MB)",
            name, tokens, words, elapsed, matches, matches + misses,
            matches as f64 / (matches + misses).max(1) as f64 * 100.0,
            tok_rate, word_rate, rss, hwm
        );
    }

    Ok(())
}
