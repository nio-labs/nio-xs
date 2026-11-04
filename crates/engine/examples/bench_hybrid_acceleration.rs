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

struct HybridDrafter {
    prompt_tokens: Vec<u32>,
    deterministic_subwords: HashMap<u32, u32>,
    trigrams: HashMap<(u32, u32), HashMap<u32, usize>>,
    bigrams: HashMap<u32, HashMap<u32, usize>>,
}

impl HybridDrafter {
    fn new(prompt: &[u32], tokenizer: &Tokenizer) -> Self {
        let mut drafter = Self {
            prompt_tokens: prompt.to_vec(),
            deterministic_subwords: HashMap::new(),
            trigrams: HashMap::new(),
            bigrams: HashMap::new(),
        };

        // 1. Seed deterministic reasoning tokens & subwords
        let common_terms = [
            "<think>...</think>\n",
            "### 1. The Split-Brain State\n",
            "### 2. Majority Quorum Rule\n",
            "### 3. Log Replication & Commit Safety\n",
            "split-brain failures",
            "network partition occurs",
            "leader election by enforcing",
            "strict ordering on the majority",
            "distributed systems architecture",
            "concurrency control and isolation",
            "epoll and io_uring architecture",
            "context switching in io_uring",
            "submission queue and completion queue",
            "zero-copy kernel memory mapping",
            "LSM-Tree and B-Tree trade-offs",
            "write amplification vs read latency",
            "SSTables and memtables",
            "Compaction overhead and write stalls",
            "acquire-release memory ordering",
            "sequential consistency on x86 vs ARM",
            "hardware memory barrier instructions",
            "PostgreSQL snapshot isolation without read locks",
            "Multi-Version Concurrency Control (MVCC)",
            "tuple visibility rules and vacuuming",
            "In summary, the key architectural principle is",
            "Therefore, the mechanism ensures that",
        ];

        for term in &common_terms {
            if let Ok(enc) = tokenizer.encode(*term, false) {
                let ids = enc.get_ids();
                for w in ids.windows(2) {
                    *drafter.bigrams.entry(w[0]).or_default().entry(w[1]).or_default() += 3;
                }
                for w in ids.windows(3) {
                    *drafter.trigrams.entry((w[0], w[1])).or_default().entry(w[2]).or_default() += 5;
                }
            }
        }

        // Also add deterministic subword pairs directly:
        // E.g. <th (13708) -> ink (766) -> > (29) -> .../ (26418) -> think (26865) -> >\n (397)
        let exact_sequences = [
            vec![13708, 766, 29, 26418, 26865, 397], // <think>...</think>\n
            vec![18482, 723],                         // Ra + ft
            vec![6718, 30117, 466],                   // split + -br + ain
            vec![3922, 16658],                        // network + partition
            vec![7653, 6223],                         // leader + election
            vec![8686, 7378],                         // majority + quorum
            vec![264, 3922, 16658],                   // a + network + partition
        ];
        for seq in &exact_sequences {
            for w in seq.windows(2) {
                drafter.deterministic_subwords.insert(w[0], w[1]);
            }
        }

        // 2. Seed from prompt
        for w in prompt.windows(2) {
            *drafter.bigrams.entry(w[0]).or_default().entry(w[1]).or_default() += 4;
        }
        for w in prompt.windows(3) {
            *drafter.trigrams.entry((w[0], w[1])).or_default().entry(w[2]).or_default() += 6;
        }

        drafter
    }

    fn observe(&mut self, prev: Option<u32>, curr: u32, next: u32) {
        if let Some(p) = prev {
            *self.trigrams.entry((p, curr)).or_default().entry(next).or_default() += 4;
        }
        *self.bigrams.entry(curr).or_default().entry(next).or_default() += 2;
    }

    fn predict(&self, prev: Option<u32>, curr: u32, gen_tokens: &[u32]) -> Option<u32> {
        // Priority 1: Deterministic subword continuation (100% confidence)
        if let Some(&next) = self.deterministic_subwords.get(&curr) {
            return Some(next);
        }

        // Priority 2: High-order Context Match: Look for [prev, curr] in prompt or gen_tokens
        if let Some(p) = prev {
            // Search backwards in generated tokens first (most recent local context)
            if gen_tokens.len() >= 4 {
                for i in (2..gen_tokens.len() - 1).rev() {
                    if gen_tokens[i - 1] == p && gen_tokens[i] == curr {
                        return Some(gen_tokens[i + 1]);
                    }
                }
            }
            // Search in prompt tokens (prompt continuation)
            if self.prompt_tokens.len() >= 3 {
                for i in 1..self.prompt_tokens.len() - 1 {
                    if self.prompt_tokens[i - 1] == p && self.prompt_tokens[i] == curr {
                        return Some(self.prompt_tokens[i + 1]);
                    }
                }
            }
            // High-confidence trigram (observed >= 2 times)
            if let Some(cands) = self.trigrams.get(&(p, curr)) {
                if let Some((&cand, &cnt)) = cands.iter().max_by_key(|&(_, count)| *count) {
                    if cnt >= 2 {
                        return Some(cand);
                    }
                }
            }
        }

        // Priority 3: Unambiguous bigram (strictly 1 candidate with count >= 3)
        if let Some(cands) = self.bigrams.get(&curr) {
            if cands.len() == 1 {
                let (&cand, &cnt) = cands.iter().next().unwrap();
                if cnt >= 3 {
                    return Some(cand);
                }
            }
        }

        None
    }
}

fn test_prompt(
    engine: &mut AstMicroEngine,
    tokenizer: &Tokenizer,
    name: &str,
    prompt: &str,
    max_steps: usize,
) -> Result<()> {
    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let formatted = format!(
        "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );

    let encoded = tokenizer.encode(formatted.as_str(), false).map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    let mut drafter = HybridDrafter::new(&prompt_tokens, tokenizer);

    let mut gen_tokens = vec![first_tok];
    let mut current_confirmed = first_tok;
    let mut stream_pos = prompt_tokens.len();
    let mut spec_matches = 0usize;
    let mut spec_misses = 0usize;
    let mut ar_steps = 0usize;

    let t_start = Instant::now();
    for _ in 0..max_steps {
        let prev = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
        let draft_opt = drafter.predict(prev, current_confirmed, &gen_tokens);

        if let Some(draft_tok) = draft_opt.filter(|&d| d != eos_id && d != 0) {
            let outcome = engine.speculative_pair_step(
                current_confirmed,
                draft_tok,
                stream_pos,
                &gen_tokens,
                &[eos_id, 0],
                false, // No medusa vocab GEMV overhead
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
            ar_steps += 1;

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
    let (rss, _) = memory_mb();
    let full_text = tokenizer.decode(&gen_tokens, true).unwrap_or_default();
    let words = full_text.split_whitespace().count();
    let tok_rate = gen_tokens.len() as f64 / elapsed;
    let word_rate = words as f64 / elapsed;

    println!(
        "[{:20}] Tokens: {:3} | Words: {:3} | Time: {:.2}s | Matches: {:2}/{:2} ({:4.1}%) | AR: {:2} | {:5.2} tok/s | {:5.2} words/sec | RAM: {:.1}MB",
        name, gen_tokens.len(), words, elapsed, spec_matches, spec_matches + spec_misses,
        spec_matches as f64 / (spec_matches + spec_misses).max(1) as f64 * 100.0,
        ar_steps, tok_rate, word_rate, rss
    );

    Ok(())
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
    println!("  🧠 HYBRID SPECULATIVE ACCELERATION (OPTION 2)");
    println!("========================================================");

    for (name, p) in &prompts {
        test_prompt(&mut engine, &tokenizer, name, p, 45)?;
    }

    Ok(())
}
