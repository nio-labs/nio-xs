use anyhow::{Result, anyhow};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, mpsc};

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
        if let Ok(model) = std::env::var("NIOXS_RAY_MODEL") {
            let model = PathBuf::from(model);
            let tokenizer = std::env::var_os("NIOXS_TOKENIZER")
                .map(PathBuf::from)
                .unwrap_or_else(|| Self::tokenizer_for_model(&model));
            if model.is_file() && tokenizer.is_file() {
                return Ok((model, tokenizer));
            }
            return Err(anyhow!(
                "Configured NioRay model or matching tokenizer is missing: {:?}, {:?}",
                model,
                tokenizer
            ));
        }
        if std::env::var_os("NIOXS_TOKENIZER").is_some() {
            return Err(anyhow!("Set NIOXS_RAY_MODEL together with NIOXS_TOKENIZER"));
        }

        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from);
        let directories = [
            home.map(|h| h.join(".nioxs").join("models")),
            Some(PathBuf::from("models")),
            Some(PathBuf::from("../models")),
            Some(PathBuf::from("../../models")),
        ];
        for directory in directories.into_iter().flatten() {
            for name in [
                "nio-ray-moe.Q4_K_M.gguf",
                "nio-ray-dev.Q4_K_M.gguf",
                "nio-ray.Q4_K_M.gguf",
            ] {
                let model = directory.join(name);
                let tokenizer = Self::tokenizer_for_model(&model);
                if model.is_file() && tokenizer.is_file() {
                    return Ok((model, tokenizer));
                }
            }
        }
        Err(anyhow!(
            "NioRay model/tokenizer pair not found. Install nio-ray-moe.Q4_K_M.gguf, nio-ray-dev.Q4_K_M.gguf with nio-ray-dev-tokenizer.json, or nio-ray.Q4_K_M.gguf with tokenizer.json, in ~/.nioxs/models/ or ./models/"
        ))
    }

    fn tokenizer_for_model(model: &std::path::Path) -> PathBuf {
        let name = match model.file_name().and_then(|s| s.to_str()) {
            Some("nio-ray-dev.Q4_K_M.gguf") => "nio-ray-dev-tokenizer.json",
            _ => "tokenizer.json",
        };
        model.with_file_name(name)
    }

    fn validate_tokenizer(tokenizer: &Tokenizer, tokens: &[gguf_file::Value]) -> Result<()> {
        let vocab = tokenizer.get_vocab(true);
        for (token, id) in &vocab {
            let expected = tokens.get(*id as usize).and_then(|v| v.to_string().ok());
            if expected.map(String::as_str) != Some(token.as_str()) {
                return Err(anyhow!(
                    "NioRay tokenizer mismatch at token ID {id}: {token:?}. Use the tokenizer shipped for this model variant."
                ));
            }
        }
        for (id, value) in tokens.iter().enumerate() {
            let token = value.to_string()?;
            // Full-vocabulary Qwen GGUFs contain unused padding entries.
            if !token.starts_with("[PAD") && vocab.get(token).copied() != Some(id as u32) {
                return Err(anyhow!(
                    "NioRay tokenizer is missing model token {token:?} at ID {id}"
                ));
            }
        }
        Ok(())
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
            let tokens = content
                .metadata
                .get("tokenizer.ggml.tokens")
                .ok_or_else(|| anyhow!("NioRay GGUF is missing its tokenizer vocabulary"))?
                .to_vec()?;
            Self::validate_tokenizer(&tokenizer, tokens)?;
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
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                raw_prompt.trim()
            )
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
                Some(if temperature <= 0.01 {
                    0.01
                } else {
                    temperature
                }),
                Some(top_p),
            );
            let eos_token_id = bundle.tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
            let eot_token_id = bundle
                .tokenizer
                .token_to_id("<|endoftext|>")
                .unwrap_or(151643);
            let is_eos = |tok: u32| tok == eos_token_id || tok == eot_token_id;

            // Prefill pass at index_pos = 0 (resets KV cache)
            let input = match Tensor::new(&prompt_tokens[..], &bundle.device)
                .and_then(|t| t.unsqueeze(0))
            {
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
                let input =
                    match Tensor::new(&[next_token], &bundle.device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => t,
                        Err(_) => break,
                    };

                let logits = match bundle
                    .model
                    .forward(&input, index_pos)
                    .and_then(|l| l.squeeze(0))
                {
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
        let mut rx = self
            .generate_stream(prompt, max_tokens, temperature, top_p)
            .await?;
        let mut result = String::new();
        while let Some(chunk) = rx.recv().await {
            result.push_str(&chunk);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;

    fn tokenizer() -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [(String::from("hello"), 0), (String::from("<|im_end|>"), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token(String::from("hello"))
            .build()
            .unwrap();
        Tokenizer::new(model)
    }

    #[test]
    fn matching_tokenizer_accepts_unused_gguf_padding() {
        let tokens = ["hello", "<|im_end|>", "[PAD2]"].map(|s| gguf_file::Value::String(s.into()));
        RayEngine::validate_tokenizer(&tokenizer(), &tokens).unwrap();
    }

    #[test]
    fn rejects_remapped_special_tokens_and_out_of_range_ids() {
        let tokens = ["<|im_end|>", "hello"].map(|s| gguf_file::Value::String(s.into()));
        assert!(RayEngine::validate_tokenizer(&tokenizer(), &tokens).is_err());
        let tokens = [gguf_file::Value::String("hello".into())];
        assert!(RayEngine::validate_tokenizer(&tokenizer(), &tokens).is_err());
    }

    #[test]
    fn rejects_missing_model_tokens() {
        let tokens = ["hello", "<|im_end|>", "world"].map(|s| gguf_file::Value::String(s.into()));
        assert!(RayEngine::validate_tokenizer(&tokenizer(), &tokens).is_err());
    }

    #[test]
    fn variant_tokenizers_stay_beside_the_model() {
        assert_eq!(
            RayEngine::tokenizer_for_model(std::path::Path::new("/cache/nio-ray-dev.Q4_K_M.gguf")),
            PathBuf::from("/cache/nio-ray-dev-tokenizer.json")
        );
        assert_eq!(
            RayEngine::tokenizer_for_model(std::path::Path::new("/cache/nio-ray.Q4_K_M.gguf")),
            PathBuf::from("/cache/tokenizer.json")
        );
    }

    #[tokio::test]
    #[ignore = "requires release assets in NIOXS_RAY_TEST_MODELS"]
    async fn release_models_load_and_prefill_with_matching_tokenizers() {
        let directory = PathBuf::from(std::env::var("NIOXS_RAY_TEST_MODELS").unwrap());
        for name in ["nio-ray.Q4_K_M.gguf", "nio-ray-dev.Q4_K_M.gguf"] {
            let model_path = directory.join(name);
            let tokenizer_path = RayEngine::tokenizer_for_model(&model_path);
            let engine = RayEngine {
                bundle: Arc::new(Mutex::new(None)),
                model_path,
                tokenizer_path,
            };
            engine.ensure_loaded().await.unwrap();
            let mut guard = engine.bundle.lock().await;
            let bundle = guard.as_mut().unwrap();
            let prompt = RayEngine::format_prompt("Write a Rust function.");
            let encoded = bundle.tokenizer.encode(prompt, true).unwrap();
            let input = Tensor::new(encoded.get_ids(), &bundle.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = bundle.model.forward(&input, 0).unwrap();
            let (_, vocabulary_size) = logits.dims2().unwrap();
            assert_eq!(
                vocabulary_size,
                if name.contains("-dev") { 32000 } else { 151936 }
            );
        }
        let mut model = File::open(directory.join("nio-ray-dev.Q4_K_M.gguf")).unwrap();
        let content = gguf_file::Content::read(&mut model).unwrap();
        let tokens = content.metadata["tokenizer.ggml.tokens"].to_vec().unwrap();
        let wrong = Tokenizer::from_file(directory.join("tokenizer.json")).unwrap();
        assert!(RayEngine::validate_tokenizer(&wrong, tokens).is_err());
    }
}
