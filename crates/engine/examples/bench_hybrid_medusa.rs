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

struct SpeculativeDrafter {
    // Dynamic context trie / 2-gram / 3-gram index over prompt + generated tokens
    bigrams: HashMap<u32, Vec<u32>>,
    trigrams: HashMap<(u32, u32), Vec<u32>>,
}

impl SpeculativeDrafter {
    fn new(prompt_tokens: &[u32]) -> Self {
        let mut drafter = Self {
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
        };
        drafter.update(prompt_tokens);
        drafter
    }

    fn update(&mut self, tokens: &[u32]) {
        for window in tokens.windows(2) {
            let entry = self.bigrams.entry(window[0]).or_default();
            if !entry.contains(&window[1]) {
                entry.push(window[1]);
            }
        }
        for window in tokens.windows(3) {
            let entry = self.trigrams.entry((window[0], window[1])).or_default();
            if !entry.contains(&window[2]) {
                entry.push(window[2]);
            }
        }
    }

    fn propose_draft(
        &self,
        engine: &mut AstMicroEngine,
        h: &[f32],
        history: &[u32],
    ) -> Option<u32> {
        let n = history.len();
        // 1. Try 3-gram continuation from history
        if n >= 2 {
            let key = (history[n - 2], history[n - 1]);
            if let Some(cands) = self.trigrams.get(&key) {
                if let Some(&first) = cands.last() {
                    return Some(first);
                }
            }
        }
        // 2. Try 2-gram continuation from history
        if n >= 1 {
            let key = history[n - 1];
            if let Some(cands) = self.bigrams.get(&key) {
                if let Some(&first) = cands.last() {
                    return Some(first);
                }
            }
        }
        // 3. Fallback to Medusa neural head projection
        engine.predict_medusa_draft(h, history).ok().flatten()
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

    println!(
        "Prefill done. First token: {} ({:?})",
        first_tok,
        tokenizer.decode(&[first_tok], false)
    );

    let mut drafter = SpeculativeDrafter::new(&prompt_tokens);
    drafter.update(&[first_tok]);

    let mut gen_tokens = vec![first_tok];
    let mut current_confirmed = first_tok;
    let mut stream_pos = prompt_tokens.len();
    let mut matches = 0usize;
    let mut misses = 0usize;

    let h_init = engine.scratch.x_norm.clone();
    let mut next_draft = drafter.propose_draft(&mut engine, &h_init, &gen_tokens);

    let t_start = Instant::now();
    for step in 0..40 {
        if let Some(draft_tok) = next_draft.take().filter(|&d| d != eos_id && d != 0) {
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
                "Step {:02} [PAIR] | draft={:6} ({:10?}) | accepted={:?} | matched={:5} | time={:5.1}ms",
                step,
                draft_tok,
                tokenizer.decode(&[draft_tok], false).unwrap_or_default(),
                outcome.accepted_tokens,
                outcome.draft_matched,
                step_dur.as_secs_f64() * 1000.0,
            );

            for &tok in &outcome.accepted_tokens {
                gen_tokens.push(tok);
            }
            drafter.update(&outcome.accepted_tokens);

            current_confirmed = outcome.last_confirmed;
            stream_pos = outcome.stream_pos;

            // Propose next draft using hybrid drafter
            let h_curr = engine.scratch.x_norm.clone();
            next_draft = drafter.propose_draft(&mut engine, &h_curr, &gen_tokens);

            if outcome.is_eos {
                println!("Hit EOS at step {}", step);
                break;
            }
        } else {
            let t_step = Instant::now();
            engine.forward_stream(&[current_confirmed], stream_pos)?;
            stream_pos += 1;

            let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
            let step_dur = t_step.elapsed();

            if next_tok == eos_id || next_tok == 0 {
                println!("Hit EOS at step {}", step);
                break;
            }

            gen_tokens.push(next_tok);
            drafter.update(&[next_tok]);
            current_confirmed = next_tok;

            let h_last = engine.scratch.x_norm.clone();
            next_draft = drafter.propose_draft(&mut engine, &h_last, &gen_tokens);

            println!(
                "Step {:02} [AR]   | confirmed={:6} | time={:5.1}ms",
                step,
                next_tok,
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
    println!("  HYBRID MEDUSA SPECULATIVE RESULTS");
    println!("========================================================");
    println!("Tokens generated: {}", gen_tokens.len());
    println!("Words generated:  {}", words);
    println!("Total time:       {:.3} s", elapsed.as_secs_f64());
    println!(
        "Draft matches:    {} ({:.1}%)",
        matches,
        matches as f64 / (matches + misses) as f64 * 100.0
    );
    println!("Draft misses:     {}", misses);
    println!("Token rate:       {:.2} tok/s", tok_rate);
    println!("Word rate:        {:.2} words/sec", word_rate);
    println!("End RAM (VmRSS):  {:.2} MB", rss_end);
    println!("Peak RAM (VmHWM): {:.2} MB", hwm);
    println!("Output text:\n{}", full_text);

    Ok(())
}
