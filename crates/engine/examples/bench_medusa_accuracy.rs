use anyhow::{anyhow, Result};
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    engine.set_resident_layers(7);
    engine.set_rolling_window(4);

    let prompt = "Explain how Raft avoids split-brain during leader election when a network partition occurs.";
    let formatted = format!(
        "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );

    let encoded = tokenizer.encode(formatted.as_str(), false).map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    let mut tokens = vec![first_tok];
    let mut curr = first_tok;
    let mut pos = prompt_tokens.len();

    let mut top1_matches = 0;
    let mut top2_matches = 0;
    let mut top3_matches = 0;
    let mut top5_matches = 0;
    let mut total_eval = 0;

    println!("Diagnosing Medusa predictions over 30 steps...");

    for step in 0..30 {
        let h_current = engine.scratch.x_norm.clone();
        // Medusa logits from h_current
        let medusa_logits = engine.medusa_diagnostic_logits(&h_current)?;
        let mut indexed_medusa: Vec<(usize, f32)> = medusa_logits.into_iter().enumerate().collect();
        indexed_medusa.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Now advance AR by 1 step to get the TRUE next token
        engine.forward_stream(&[curr], pos)?;
        pos += 1;
        let true_tok = engine.next_token_with_history(None, &tokens, false)?;
        tokens.push(true_tok);
        curr = true_tok;

        let true_str = tokenizer.decode(&[true_tok], false).unwrap_or_default();
        let top1 = indexed_medusa[0].0 as u32;
        let top2 = indexed_medusa[1].0 as u32;
        let top3 = indexed_medusa[2].0 as u32;
        let top4 = indexed_medusa[3].0 as u32;
        let top5 = indexed_medusa[4].0 as u32;

        let in_top1 = true_tok == top1;
        let in_top2 = in_top1 || true_tok == top2;
        let in_top3 = in_top2 || true_tok == top3;
        let in_top5 = in_top3 || true_tok == top4 || true_tok == top5;

        if in_top1 { top1_matches += 1; }
        if in_top2 { top2_matches += 1; }
        if in_top3 { top3_matches += 1; }
        if in_top5 { top5_matches += 1; }
        total_eval += 1;

        println!(
            "Step {:02} | True: {:6} ({:12?}) | Top1: {:6} ({:12?}) | Top2: {:6} | Match: top1={} top2={} top5={}",
            step,
            true_tok,
            true_str,
            top1,
            tokenizer.decode(&[top1], false).unwrap_or_default(),
            top2,
            in_top1,
            in_top2,
            in_top5
        );
    }

    println!("\n=== ACCURACY OVER {} STEPS ===", total_eval);
    println!("Top-1 Accuracy: {} / {} ({:.1}%)", top1_matches, total_eval, top1_matches as f64 / total_eval as f64 * 100.0);
    println!("Top-2 Accuracy: {} / {} ({:.1}%)", top2_matches, total_eval, top2_matches as f64 / total_eval as f64 * 100.0);
    println!("Top-3 Accuracy: {} / {} ({:.1}%)", top3_matches, total_eval, top3_matches as f64 / total_eval as f64 * 100.0);
    println!("Top-5 Accuracy: {} / {} ({:.1}%)", top5_matches, total_eval, top5_matches as f64 / total_eval as f64 * 100.0);

    Ok(())
}
