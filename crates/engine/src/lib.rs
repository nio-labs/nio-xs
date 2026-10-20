use anyhow::{anyhow, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokenizers::Tokenizer;

pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.tokenizer
                .decode(tokens, false)
                .map_err(|e| anyhow!("{e}"))?
        };
        self.tokens.push(token);
        let text = {
            let tokens = &self.tokens[self.prev_index..self.tokens.len()];
            self.tokenizer
                .decode(tokens, false)
                .map_err(|e| anyhow!("{e}"))?
        };
        if text.len() > prev_text.len() && !text.ends_with('\u{fffd}') {
            let c = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(c.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.tokenizer
                .decode(tokens, false)
                .map_err(|e| anyhow!("{e}"))?
        };
        let text = {
            let tokens = &self.tokens[self.prev_index..self.tokens.len()];
            self.tokenizer
                .decode(tokens, false)
                .map_err(|e| anyhow!("{e}"))?
        };
        if text.len() > prev_text.len() {
            let c = text.split_at(prev_text.len());
            Ok(Some(c.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

pub struct ModelBundle {
    pub model: ModelWeights,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

#[derive(Clone)]
pub struct RayEngine {
    bundle: Arc<Mutex<Option<ModelBundle>>>,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
}

impl RayEngine {
    pub fn new() -> Result<Self> {
        let (model_path, tokenizer_path) = Self::find_model_paths()?;
        Ok(Self {
            bundle: Arc::new(Mutex::new(None)),
            model_path,
            tokenizer_path,
        })
    }

    pub fn is_available() -> bool {
        Self::find_model_paths().is_ok()
    }

    pub fn find_model_paths() -> Result<(PathBuf, PathBuf)> {
        if let (Ok(m), Ok(t)) = (
            std::env::var("NIOXS_RAY_MODEL"),
            std::env::var("NIOXS_TOKENIZER"),
        ) {
            let mp = PathBuf::from(m);
            let tp = PathBuf::from(t);
            if mp.is_file() && tp.is_file() {
                return Ok((mp, tp));
            }
        }

        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from);

        let candidates = [
            // User cached dir
            home.as_ref()
                .map(|h| h.join(".nioxs").join("models").join("nio-ray.Q4_K_M.gguf")),
            // Current dir models
            Some(PathBuf::from("models/nio-ray.Q4_K_M.gguf")),
            Some(PathBuf::from("../models/nio-ray.Q4_K_M.gguf")),
            Some(PathBuf::from("../../models/nio-ray.Q4_K_M.gguf")),
            // nio-ideas dev path
            Some(PathBuf::from("/home/mn/nio-labs/nio-ideas/models/nio-ray.Q4_K_M.gguf")),
            Some(PathBuf::from("/home/mn/nio-labs/nio-xs/models/nio-ray.Q4_K_M.gguf")),
        ];

        let tokenizer_candidates = [
            home.as_ref()
                .map(|h| h.join(".nioxs").join("models").join("tokenizer.json")),
            Some(PathBuf::from("models/tokenizer.json")),
            Some(PathBuf::from("../models/tokenizer.json")),
            Some(PathBuf::from("../../models/tokenizer.json")),
            Some(PathBuf::from("/home/mn/nio-labs/nio-ideas/models/tokenizer.json")),
            Some(PathBuf::from("/home/mn/nio-labs/nio-xs/models/tokenizer.json")),
        ];

        let mut found_model = None;
        for c in candidates.into_iter().flatten() {
            if c.is_file() {
                found_model = Some(c);
                break;
            }
        }

        let mut found_tokenizer = None;
        for c in tokenizer_candidates.into_iter().flatten() {
            if c.is_file() {
                found_tokenizer = Some(c);
                break;
            }
        }

        match (found_model, found_tokenizer) {
            (Some(m), Some(t)) => Ok((m, t)),
            _ => Err(anyhow!(
                "NioRay model or tokenizer not found. Place nio-ray.Q4_K_M.gguf and tokenizer.json in ~/.nioxs/models/ or ./models/"
            )),
        }
    }

    async fn ensure_loaded(&self) -> Result<()> {
        let mut guard = self.bundle.lock().await;
        if guard.is_some() {
            return Ok(());
        }

        let model_path = self.model_path.clone();
        let tokenizer_path = self.tokenizer_path.clone();

        let bundle = tokio::task::spawn_blocking(move || -> Result<ModelBundle> {
            let device = Device::Cpu;
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Failed loading tokenizer: {e}"))?;

            let mut file = File::open(&model_path)
                .map_err(|e| anyhow!("Failed opening model at {:?}: {e}", model_path))?;
            let content = gguf_file::Content::read(&mut file)?;
            let model = ModelWeights::from_gguf(content, &mut file, &device)?;

            Ok(ModelBundle {
                model,
                tokenizer,
                device,
            })
        })
        .await
        .map_err(|e| anyhow!("Task spawn join error: {e}"))??;

        *guard = Some(bundle);
        Ok(())
    }

    pub fn format_prompt(raw_prompt: &str) -> String {
        if raw_prompt.contains("<|im_start|>") {
            raw_prompt.to_string()
        } else {
            format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", raw_prompt.trim())
        }
    }

    pub async fn generate_stream(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<mpsc::Receiver<String>> {
        self.ensure_loaded().await?;
        let (tx, rx) = mpsc::channel(64);

        let bundle_arc = self.bundle.clone();
        tokio::task::spawn(async move {
            let mut guard = bundle_arc.lock().await;
            let Some(bundle) = guard.as_mut() else {
                let _ = tx.send("Error: Model bundle not loaded".into()).await;
                return;
            };

            let formatted = Self::format_prompt(&prompt);
            let encoded = match bundle.tokenizer.encode(formatted, true) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(format!("Tokenizer error: {e}")).await;
                    return;
                }
            };

            let prompt_tokens = encoded.get_ids();
            if prompt_tokens.is_empty() {
                return;
            }

            let mut tos = TokenOutputStream::new(bundle.tokenizer.clone());
            let seed = rand::random::<u64>();
            let mut logits_processor = LogitsProcessor::new(
                seed,
                Some(if temperature <= 0.01 { 0.01 } else { temperature }),
                Some(top_p),
            );
            let eos_token_id = bundle.tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
            let eot_token_id = bundle.tokenizer.token_to_id("<|endoftext|>").unwrap_or(151643);
            let is_eos = |tok: u32| tok == eos_token_id || tok == eot_token_id;

            // Prefill pass at index_pos = 0 (resets KV cache)
            let input = match Tensor::new(&prompt_tokens[..], &bundle.device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(format!("Tensor error: {e}")).await;
                    return;
                }
            };

            let logits = match bundle.model.forward(&input, 0).and_then(|l| l.squeeze(0)) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(format!("Inference error: {e}")).await;
                    return;
                }
            };

            let mut next_token = match logits_processor.sample(&logits) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(format!("Sampling error: {e}")).await;
                    return;
                }
            };

            if is_eos(next_token) {
                return;
            }

            if let Ok(Some(chunk)) = tos.next_token(next_token) {
                if tx.send(chunk).await.is_err() {
                    return; // Receiver dropped, stop early
                }
            }

            // Generation loop
            for index_pos in prompt_tokens.len()..(prompt_tokens.len() + max_tokens.max(1) - 1) {
                let input = match Tensor::new(&[next_token], &bundle.device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(_) => break,
                };

                let logits = match bundle.model.forward(&input, index_pos).and_then(|l| l.squeeze(0)) {
                    Ok(l) => l,
                    Err(_) => break,
                };

                next_token = match logits_processor.sample(&logits) {
                    Ok(t) => t,
                    Err(_) => break,
                };

                if is_eos(next_token) {
                    break;
                }

                if let Ok(Some(chunk)) = tos.next_token(next_token) {
                    if tx.send(chunk).await.is_err() {
                        break;
                    }
                }
            }

            if let Ok(Some(rest)) = tos.decode_rest() {
                let _ = tx.send(rest).await;
            }
        });

        Ok(rx)
    }

    pub async fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<String> {
        let mut rx = self.generate_stream(prompt, max_tokens, temperature, top_p).await?;
        let mut result = String::new();
        while let Some(chunk) = rx.recv().await {
            result.push_str(&chunk);
        }
        Ok(result)
    }
}
