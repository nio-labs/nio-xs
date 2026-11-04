use anyhow::{anyhow, Result};
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    let mut engine = AstMicroEngine::new(&model_path)?;

    engine.set_resident_layers(7);
    engine.set_rolling_window(4);

    let prompt = "Explain Raft.";
    let encoded = tokenizer.encode(prompt, false).map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;

    // 1. Base model logits
    let base_logits = engine.diagnostic_logits()?;
    let mut b_idx: Vec<(usize, f32)> = base_logits.into_iter().enumerate().collect();
    b_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 2. Medusa logits
    let h = engine.scratch.x_norm.clone();
    let medusa_logits = engine.medusa_diagnostic_logits(&h)?;
    let mut m_idx: Vec<(usize, f32)> = medusa_logits.into_iter().enumerate().collect();
    m_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Base Top 5 from prompt last token:");
    for (tok, s) in &b_idx[..5] {
        println!("  tok {} ({:?}): {:.4}", tok, tokenizer.decode(&[*tok as u32], false), s);
    }

    println!("\nMedusa Top 5 from prompt last token:");
    for (tok, s) in &m_idx[..5] {
        println!("  tok {} ({:?}): {:.4}", tok, tokenizer.decode(&[*tok as u32], false), s);
    }

    Ok(())
}
