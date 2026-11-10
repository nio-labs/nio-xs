use anyhow::Result;
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let model_path = std::env::var_os("NIOXS_BLUE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.nioxs/models/blue.vmicrolm-reflex.gguf"));
    let mut engine = AstMicroEngine::new(&model_path)?;
    engine.set_resident_layers(32);
    engine.set_rolling_window(32);

    println!("Warming up 1 token...");
    let _ = engine.prefill(&[1])?;

    let t_vocab0 = Instant::now();
    let tok = engine.next_token_with_history(None, &[], false)?;
    println!(
        "Vocabulary sample time: {:.2} ms (token: {})",
        t_vocab0.elapsed().as_secs_f64() * 1000.0,
        tok
    );

    let t_1tok = Instant::now();
    let _ = engine.forward_stream(&[tok], 1)?;
    println!(
        "1-Token forward (32 layers IQ3_XXS, warm) time: {:.2} ms ({:.2} tok/s)",
        t_1tok.elapsed().as_secs_f64() * 1000.0,
        1.0 / t_1tok.elapsed().as_secs_f64()
    );

    let t_2tok = Instant::now();
    let _ = engine.forward_stream(&[tok, tok], 2)?;
    println!(
        "2-Token forward (32 layers IQ3_XXS, warm) time: {:.2} ms ({:.2} tok/s)",
        t_2tok.elapsed().as_secs_f64() * 1000.0,
        2.0 / t_2tok.elapsed().as_secs_f64()
    );

    let t_3tok = Instant::now();
    let _ = engine.forward_stream(&[tok, tok, tok], 4)?;
    println!(
        "3-Token forward (32 layers IQ3_XXS, warm) time: {:.2} ms ({:.2} tok/s)",
        t_3tok.elapsed().as_secs_f64() * 1000.0,
        3.0 / t_3tok.elapsed().as_secs_f64()
    );

    Ok(())
}
