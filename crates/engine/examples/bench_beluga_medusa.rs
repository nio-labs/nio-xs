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

    println!("\n========================================================");
    println!("  🧠 NIOBELUGA (1.54B) - MEDUSA OPTION 2 BENCHMARK");
    println!("========================================================");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    println!("Medusa loaded: {}", engine.medusa_head.is_some());
    engine.set_resident_layers(7);
    engine.set_rolling_window(4);

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let prompt = "Explain how Raft avoids split-brain during leader election when a network partition occurs.";
    let formatted = format!(
        "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );

    let encoded = tokenizer.encode(formatted.as_str(), false).map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    // Warmup prefill
    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    println!("Prefill done. First token: {} ({:?})", first_tok, tokenizer.decode(&[first_tok], false));
    let (rss_init, _) = memory_mb();
    println!("RAM after prefill: {:.2} MB", rss_init);

    // Let's inspect diagnostic logits for first_tok's base prediction
    let logits_base = engine.diagnostic_logits()?;
    let mut indexed_base: Vec<(usize, f32)> = logits_base.into_iter().enumerate().collect();
    indexed_base.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Base Top 5 after prefill:");
    for (tok, score) in &indexed_base[..5] {
        println!("  tok {} ({:?}): {:.4}", tok, tokenizer.decode(&[*tok as u32], false), score);
    }

    // Now inspect Medusa prediction on h_init
    let h_init = engine.scratch.x_norm.clone();
    let logits_medusa = engine.medusa_diagnostic_logits(&h_init)?;
    let mut indexed_medusa: Vec<(usize, f32)> = logits_medusa.into_iter().enumerate().collect();
    indexed_medusa.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Medusa Top 5 from h_init (draft token):");
    for (tok, score) in &indexed_medusa[..5] {
        println!("  tok {} ({:?}): {:.4}", tok, tokenizer.decode(&[*tok as u32], false), score);
    }

    // Run 50 speculative steps with Medusa
    let mut gen_tokens = vec![first_tok];
    let mut current_confirmed = first_tok;
    let mut stream_pos = prompt_tokens.len();
    let mut matches = 0usize;
    let mut misses = 0usize;

    let mut next_draft = engine.predict_medusa_draft(&h_init, &gen_tokens)?;

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
                true,
            )?;
            let step_dur = t_step.elapsed();

            if outcome.draft_matched {
                matches += 1;
            } else {
                misses += 1;
            }

            println!(
                "Step {:02} [PAIR] | draft_tok={:6} | accepted={:?} | matched={:5} | step_time={:5.1}ms",
                step,
                draft_tok,
                outcome.accepted_tokens,
                outcome.draft_matched,
                step_dur.as_secs_f64() * 1000.0,
            );

            for &tok in &outcome.accepted_tokens {
                gen_tokens.push(tok);
            }
            current_confirmed = outcome.last_confirmed;
            stream_pos = outcome.stream_pos;
            next_draft = outcome.next_draft_tok;

            if outcome.is_eos {
                println!("Hit EOS at step {}", step);
                break;
            }
        } else {
            // Autoregressive fallback step
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
            current_confirmed = next_tok;

            let h_last = engine.scratch.x_norm.clone();
            next_draft = engine.predict_medusa_draft(&h_last, &gen_tokens).ok().flatten();

            println!(
                "Step {:02} [AR]   | confirmed={:6} | step_time={:5.1}ms",
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
    println!("  RESULTS SUMMARY");
    println!("========================================================");
    println!("Tokens generated: {}", gen_tokens.len());
    println!("Words generated:  {}", words);
    println!("Total time:       {:.3} s", elapsed.as_secs_f64());
    println!("Draft matches:    {} ({:.1}%)", matches, matches as f64 / (matches + misses) as f64 * 100.0);
    println!("Draft misses:     {}", misses);
    println!("Token rate:       {:.2} tok/s", tok_rate);
    println!("Word rate:        {:.2} words/sec", word_rate);
    println!("End RAM (VmRSS):  {:.2} MB", rss_end);
    println!("Peak RAM (VmHWM): {:.2} MB", hwm);
    println!("Output text:\n{}", full_text);

    Ok(())
}
