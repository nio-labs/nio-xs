use anyhow::{Result, anyhow};
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    engine.set_resident_layers(7);
    engine.set_rolling_window(4);

    let prompt = "Explain Raft.";
    let encoded = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    let mut t_1 = 0.0;
    let mut t_2 = 0.0;
    let mut t_3 = 0.0;

    for i in 0..5 {
        let t0 = Instant::now();
        engine.forward_batch_layer_outer(&[first_tok], prompt_tokens.len() + i * 4)?;
        t_1 += t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        engine.forward_batch_layer_outer(&[first_tok, 100], prompt_tokens.len() + i * 4 + 1)?;
        t_2 += t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        engine
            .forward_batch_layer_outer(&[first_tok, 100, 200], prompt_tokens.len() + i * 4 + 2)?;
        t_3 += t2.elapsed().as_secs_f64();
    }

    println!("Average 1 token forward: {:.2} ms", t_1 / 5.0 * 1000.0);
    println!("Average 2 token forward: {:.2} ms", t_2 / 5.0 * 1000.0);
    println!("Average 3 token forward: {:.2} ms", t_3 / 5.0 * 1000.0);

    Ok(())
}
