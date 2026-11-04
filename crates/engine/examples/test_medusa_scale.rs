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

    let prompt = "Explain how Raft avoids split-brain.";
    let encoded = tokenizer.encode(prompt, false).map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    let norm_x: f32 = engine.scratch.x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_x_norm: f32 = engine.scratch.x_norm.iter().map(|v| v * v).sum::<f32>().sqrt();

    println!("scratch.x (unnormalized) L2 norm:      {:.4}", norm_x);
    println!("scratch.x_norm (RMS-normalized) L2 norm: {:.4}", norm_x_norm);
    println!("Ratio: {:.2}x", norm_x / norm_x_norm);

    Ok(())
}
