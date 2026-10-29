use anyhow::Result;
use nioxs_engine::{AstMicroEngine, RayEngine, skate_system};
use serde_json::{json,Value};
use std::path::Path;
use tokenizers::Tokenizer;
fn main() -> Result<()> {
 let a: Vec<_>=std::env::args().collect();
 let tok=Tokenizer::from_file(&a[2]).map_err(|e|anyhow::anyhow!("{e}"))?;
 let mut engine=AstMicroEngine::new(Path::new(&a[1]))?;
 RayEngine::validate_tokenizer(&tok,&engine.tokenizer_tokens()?)?;
 let suite:Value=serde_json::from_slice(&std::fs::read(&a[3])?)?;
 let index:usize=a[4].parse()?; let steps:usize=a[5].parse()?;
 let supplied:Option<Vec<u32>>=if let Some(path)=a.get(6) {
  let trace:Value=serde_json::from_slice(&std::fs::read(path)?)?;
  let values=trace["forced_tokens"].as_array().ok_or_else(||anyhow::anyhow!("missing forced tokens"))?;
  if values.len()!=steps { anyhow::bail!("forced token count does not match steps"); }
  Some(values.iter().map(|v|v.as_u64().and_then(|x|u32::try_from(x).ok()).ok_or_else(||anyhow::anyhow!("invalid forced token"))).collect::<Result<Vec<_>>>()?)
 } else { None };
 let question=suite["held_out"][index].as_str().ok_or_else(||anyhow::anyhow!("missing prompt"))?;
 let policy=skate_system()?;
 let text=format!("<|im_start|>system\n{policy}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n");
 let ids=tok.encode(text,false).map_err(|e|anyhow::anyhow!("{e}"))?.get_ids().to_vec();
 engine.prefill(&ids)?;
 let mut forced=Vec::new();let mut logits=Vec::new();
 for i in 0..steps { let scores=engine.diagnostic_logits()?;
  let next=if let Some(tokens)=&supplied { tokens[i] } else { scores.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).ok_or_else(||anyhow::anyhow!("empty logits"))?.0 as u32 };
  logits.push(scores);forced.push(next);engine.forward_stream(&[next],ids.len()+i)?; }
 println!("{}",json!({"prompt_tokens":ids,"forced_tokens":forced,"logits":logits}));Ok(())
}
