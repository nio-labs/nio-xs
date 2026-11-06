use anyhow::{Result, anyhow};
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

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
    let tok0 = engine.next_token_with_history(None, &[], false)?;

    // Advance 1 token
    engine.forward_stream(&[tok0], prompt_tokens.len())?;
    let tok1 = engine.next_token_with_history(None, &[tok0], false)?;

    // Advance 2nd token
    engine.forward_stream(&[tok1], prompt_tokens.len() + 1)?;
    let tok2 = engine.next_token_with_history(None, &[tok0, tok1], false)?;

    println!("Token 0: {} ({:?})", tok0, tokenizer.decode(&[tok0], false));
    println!("Token 1: {} ({:?})", tok1, tokenizer.decode(&[tok1], false));
    println!("Token 2: {} ({:?})", tok2, tokenizer.decode(&[tok2], false));

    let x_raw = engine.scratch.x.clone();
    let x_norm = engine.scratch.x_norm.clone();

    let (draft_from_norm, draft_from_raw) = {
        let Some(medusa) = &engine.medusa_head else {
            return Err(anyhow!("No medusa head"));
        };
        let dim = medusa.dim;
        let mut d_norm = vec![0.0f32; dim];
        medusa.forward(&x_norm, &mut d_norm);
        let mut d_raw = vec![0.0f32; dim];
        medusa.forward(&x_raw, &mut d_raw);
        (d_norm, d_raw)
    };

    let pred_norm =
        engine.sample_token_from_x_norm(&draft_from_norm, None, &[tok0, tok1, tok2], false)?;
    let pred_raw =
        engine.sample_token_from_x_norm(&draft_from_raw, None, &[tok0, tok1, tok2], false)?;

    println!(
        "Pred from x_norm: {} ({:?})",
        pred_norm,
        tokenizer.decode(&[pred_norm], false)
    );
    println!(
        "Pred from raw x:  {} ({:?})",
        pred_raw,
        tokenizer.decode(&[pred_raw], false)
    );

    Ok(())
}
