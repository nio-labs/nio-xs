use anyhow::Result;
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/.nioxs/models/blue.vmicrolm-reflex.gguf");
    println!("Loading AstMicroEngine with {:?}...", model_path);
    let t0 = Instant::now();
    let mut engine = AstMicroEngine::new(&model_path)?;
    println!("Loaded engine in {:.2}s!", t0.elapsed().as_secs_f64());

    engine.set_resident_layers(2);
    engine.set_rolling_window(512);

    println!("Testing prefill on dummy tokens [1, 32002, 100, 32007]...");
    let t1 = Instant::now();
    let prompt_tokens = vec![1, 32002, 100, 32007];
    let _ = engine.prefill(&prompt_tokens)?;
    println!("Prefill finished in {:.2}ms", t1.elapsed().as_secs_f64() * 1000.0);

    let next_tok = engine.next_token_with_history(None, &[], false)?;
    println!("First generated token: {}", next_tok);

    println!("Testing 5 generation steps...");
    let mut gen_tokens = vec![next_tok];
    for step in 0..5 {
        let t_step = Instant::now();
        let _ = engine.forward_stream(&[gen_tokens[step]], prompt_tokens.len() + step)?;
        let tok = engine.next_token_with_history(None, &gen_tokens, false)?;
        gen_tokens.push(tok);
        println!("  Step {}: token {} in {:.2}ms", step, tok, t_step.elapsed().as_secs_f64() * 1000.0);
    }
    println!("🎉 AstMicroEngine verification passed for Blue (3.82B)!");
    Ok(())
}
