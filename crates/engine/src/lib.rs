use anyhow::{Result, anyhow};
use candle_core::Device;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, mpsc};

pub mod ast_engine;
pub use ast_engine::{AstMicroEngine, memory_mb};

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
    pub engine: AstMicroEngine,
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
        Self::for_model("nio-ray")
    }

    pub fn for_model(model_name: &str) -> Result<Self> {
        let (model_path, tokenizer_path) = Self::find_or_pull_model_paths(model_name)?;
        Ok(Self {
            bundle: Arc::new(Mutex::new(None)),
            model_path,
            tokenizer_path,
        })
    }

    pub fn from_paths(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            bundle: Arc::new(Mutex::new(None)),
            model_path,
            tokenizer_path,
        }
    }

    pub fn is_available() -> bool {
        Self::find_model_paths().is_ok()
    }

    pub fn is_model_available(model_name: &str) -> bool {
        Self::find_model_paths_for(model_name).is_ok()
    }

    pub fn find_or_pull_model_paths(model_name: &str) -> Result<(PathBuf, PathBuf)> {
        if let Ok(paths) = Self::find_model_paths_for(model_name) {
            return Ok(paths);
        }

        let lowered = model_name.trim().to_ascii_lowercase();
        let normalized = match lowered.as_str() {
            "ray" | "nio-ray" => "nio-ray",
            "viper" | "nio-viper" => "nio-viper",
            "manta" | "nio-manta" => "nio-manta",
            "cuda" | "nio-cuda" => "nio-cuda",
            other => other,
        };

        // Try pulling via nioxs_artifacts with auto_yes = true when invoked programmatically
        if let Ok(model_path) = nioxs_artifacts::prompt_and_pull_if_needed(normalized, true) {
            let tokenizer_path = Self::tokenizer_for_model(&model_path);
            if tokenizer_path.is_file() {
                return Ok((model_path, tokenizer_path));
            }
        }

        Self::find_model_paths_for(model_name)
    }

    pub fn find_model_paths() -> Result<(PathBuf, PathBuf)> {
        Self::find_model_paths_for("nio-ray")
    }

    pub fn find_model_paths_for(model_name: &str) -> Result<(PathBuf, PathBuf)> {
        let normalized = model_name.trim().to_ascii_lowercase();
        let target_candidates: &[&str] = match normalized.as_str() {
            "manta" | "nio-manta" => &[
                "nio-manta-ast.Q4_0.gguf",
                "nio-manta.Q4_K_M.gguf",
                "nio-manta.Q4_0.gguf",
                "nio-ray-ast.Q4_0.gguf",
            ],
            "viper" | "nio-viper" => &[
                "nio-viper-ast-32k.Q4_0.gguf",
                "nio-viper-ast.Q4_0.gguf",
                "nio-viper.Q4_K_M.gguf",
            ],
            "cuda" | "nio-cuda" => &[
                "nio-cuda-ast.Q4_0.gguf",
                "nio-cuda.Q4_K_M.gguf",
                "nio-cuda.Q4_0.gguf",
            ],
            _ => &[
                "nio-ray-ast.Q4_0.gguf",
                "nio-ray-moe.Q4_K_M.gguf",
                "nio-ray-dev.Q4_K_M.gguf",
                "nio-ray.Q4_K_M.gguf",
            ],
        };

        if normalized.contains("ray") {
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
        }

        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from);
        let directories = [
            home.as_ref().map(|h| h.join(".nioxs").join("models")),
            Some(PathBuf::from("models")),
            Some(PathBuf::from("../models")),
            Some(PathBuf::from("../../models")),
            Some(PathBuf::from("../nio-ideas/models")),
            Some(PathBuf::from("../../nio-ideas/models")),
            Some(PathBuf::from("/home/mn/nio-labs/nio-ideas/models")),
            Some(PathBuf::from(
                "/home/mn/nio-labs/nio-xs-models/models/nio-cuda",
            )),
            Some(PathBuf::from(
                "/home/mn/nio-labs/nio-xs-models/models/nio-manta",
            )),
            Some(PathBuf::from(
                "/home/mn/nio-labs/nio-xs-models/models/nio-viper",
            )),
            Some(PathBuf::from(
                "/home/mn/nio-labs/nio-xs-models/models/nio-ray",
            )),
        ];

        for directory in directories.into_iter().flatten() {
            for name in target_candidates {
                let model = directory.join(name);
                let tokenizer = Self::tokenizer_for_model(&model);
                if model.is_file() && tokenizer.is_file() {
                    return Ok((model, tokenizer));
                }
            }
        }

        Err(anyhow!(
            "Model '{model_name}' or tokenizer not found. Run 'nioxs pull {model_name}' to install it into ~/.nioxs/models/"
        ))
    }

    fn tokenizer_for_model(model: &std::path::Path) -> PathBuf {
        let name = match model.file_name().and_then(|s| s.to_str()) {
            Some("nio-ray-dev.Q4_K_M.gguf") => "nio-ray-dev-tokenizer.json",
            Some(s) if s.contains("cuda") => "nio-cuda-tokenizer.json",
            Some(s) if s.contains("viper") => "nio-viper-tokenizer.json",
            _ => "tokenizer.json",
        };
        let beside = model.with_file_name(name);
        if beside.is_file() {
            return beside;
        }

        beside
    }

    pub fn validate_tokenizer(tokenizer: &Tokenizer, tokens: &[String]) -> Result<()> {
        let vocab = tokenizer.get_vocab(true);
        for (token, id) in &vocab {
            let expected = tokens.get(*id as usize).map(|s| s.as_str());
            if expected != Some(token.as_str()) {
                return Err(anyhow!(
                    "Tokenizer mismatch at token ID {id}: {token:?}. Use the tokenizer shipped for this model variant."
                ));
            }
        }
        for (id, token) in tokens.iter().enumerate() {
            // Full-vocabulary Qwen GGUFs contain unused padding entries.
            if !token.starts_with("[PAD") && vocab.get(token).copied() != Some(id as u32) {
                return Err(anyhow!(
                    "Tokenizer is missing model token {token:?} at ID {id}"
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

            let engine = AstMicroEngine::new(&model_path)
                .map_err(|e| anyhow!("Failed loading AST-Micro engine for {:?}: {e}", model_path))?;

            Ok(ModelBundle {
                engine,
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
        let mut output = String::new();
        while let Some(chunk) = rx.recv().await {
            output.push_str(&chunk);
        }
        Ok(output)
    }

    pub async fn generate_stream(
        &self,
        prompt: String,
        max_tokens: usize,
        _temperature: f64,
        _top_p: f64,
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
            let encoded = match bundle.tokenizer.encode(formatted.as_str(), false) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(format!("Tokenizer error: {e}")).await;
                    return;
                }
            };

            let prompt_tokens = encoded.get_ids().to_vec();
            if prompt_tokens.is_empty() {
                return;
            }

            bundle.engine.reset_cache();
            let hidden = match bundle.engine.forward_stream(&prompt_tokens, 0) {
                Ok(h) => h,
                Err(e) => {
                    let _ = tx.send(format!("AST-Micro prefill error: {e}")).await;
                    return;
                }
            };

            let eos_token_id = bundle.tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
            let eot_token_id = bundle
                .tokenizer
                .token_to_id("<|endoftext|>")
                .unwrap_or(151643);
            let is_eos = |tok: u32| tok == eos_token_id || tok == eot_token_id;

            let mut next_token = match bundle.engine.next_token(&hidden, None) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(format!("AST-Micro sampling error: {e}")).await;
                    return;
                }
            };

            if is_eos(next_token) {
                bundle.engine.reset_cache();
                return;
            }

            let mut tos = TokenOutputStream::new(bundle.tokenizer.clone());
            if let Ok(Some(chunk)) = tos.next_token(next_token) {
                if tx.send(chunk).await.is_err() {
                    bundle.engine.reset_cache();
                    return;
                }
            }

            // Generation loop using AST-Micro streaming with KV-cache
            let max_gen = max_tokens.max(1);
            for step in 0..(max_gen - 1) {
                let step_pos = prompt_tokens.len() + step;
                let step_hidden = match bundle.engine.forward_stream(&[next_token], step_pos) {
                    Ok(h) => h,
                    Err(_) => break,
                };

                next_token = match bundle.engine.next_token(&step_hidden, None) {
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

            bundle.engine.reset_cache();
        });

        Ok(rx)
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
        let tokens = [String::from("hello"), String::from("<|im_end|>"), String::from("[PAD2]")];
        RayEngine::validate_tokenizer(&tokenizer(), &tokens).unwrap();
    }

    #[test]
    fn rejects_remapped_special_tokens_and_out_of_range_ids() {
        let tokens = [String::from("<|im_end|>"), String::from("hello")];
        assert!(RayEngine::validate_tokenizer(&tokenizer(), &tokens).is_err());
        let tokens = [String::from("hello")];
        assert!(RayEngine::validate_tokenizer(&tokenizer(), &tokens).is_err());
    }

    #[test]
    fn rejects_missing_model_tokens() {
        let tokens = [String::from("hello"), String::from("<|im_end|>"), String::from("world")];
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
}
