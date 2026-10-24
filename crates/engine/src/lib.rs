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

struct PhraseBuffer {
    buffer: String,
    word_count: usize,
}

impl PhraseBuffer {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            word_count: 0,
        }
    }

    fn push(&mut self, chunk: &str, tx: &mpsc::Sender<String>) -> bool {
        let was_empty = self.buffer.is_empty();
        self.buffer.push_str(chunk);

        if was_empty && !chunk.trim().is_empty() {
            self.word_count = chunk.split_whitespace().count().max(1);
        } else if chunk.contains(' ') || chunk.contains('\n') {
            self.word_count += chunk.split_whitespace().count().max(1);
        }

        let should_flush = self.buffer.contains('\n')
            || self.buffer.contains("```")
            || self.buffer.contains('<')
            || self.buffer.contains('>')
            || self.word_count >= 4
            || (self.buffer.len() >= 32 && (self.buffer.ends_with(' ') || self.buffer.ends_with('\t')))
            || self.buffer.ends_with(". ")
            || self.buffer.ends_with("! ")
            || self.buffer.ends_with("? ")
            || (self.word_count >= 3 && (self.buffer.ends_with(": ") || self.buffer.ends_with(", ")));

        if should_flush && !self.buffer.is_empty() {
            let to_send = std::mem::take(&mut self.buffer);
            self.word_count = 0;
            tx.blocking_send(to_send).is_ok()
        } else {
            true
        }
    }

    fn flush(&mut self, tx: &mpsc::Sender<String>) -> bool {
        if !self.buffer.is_empty() {
            let to_send = std::mem::take(&mut self.buffer);
            self.word_count = 0;
            tx.blocking_send(to_send).is_ok()
        } else {
            true
        }
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
            "viper" | "nio-viper" => "nio-viper",
            "ray" | "nio-ray" => "nio-ray",
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
            "viper" | "nio-viper" => &[
                "nio-viper.Q4_0.gguf",
                "nio-viper-ast-32k.Q4_0.gguf",
                "nio-viper-ast.Q4_0.gguf",
                "nio-viper.Q4_K_M.gguf",
            ],
            "manta" | "nio-manta" => &[
                "niomanta_int8.onnx",
                "nio-manta.onnx",
                "nio-manta-ast.Q4_0.gguf",
                "nio-manta.Q4_K_M.gguf",
                "nio-manta.Q4_0.gguf",
                "nio-ray-ast.Q4_0.gguf",
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
            Some(s) if s.contains("viper") => "nio-viper-tokenizer.json",
            Some(s) if s.contains("cuda") => "nio-cuda-tokenizer.json",
            Some(s) if s.contains("manta") => "nio-manta-tokenizer.json",
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

    pub async fn ensure_loaded(&self) -> Result<()> {
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

    pub fn find_candidate_ngram(context: &[u32], ngram_len: usize) -> Option<u32> {
        if context.len() < ngram_len + 1 {
            return None;
        }
        let query = &context[context.len() - ngram_len..];
        let search_limit = context.len() - ngram_len;
        for i in (0..search_limit).rev() {
            if context[i..].starts_with(query) {
                let match_idx = i + ngram_len;
                if match_idx < search_limit {
                    return Some(context[match_idx]);
                }
            }
        }
        None
    }

    pub fn format_prompt_for_engine(&self, raw_prompt: &str) -> String {
        if raw_prompt.contains("<|im_start|>") {
            raw_prompt.to_string()
        } else if self.model_path.to_string_lossy().contains("viper") {
            #[cfg(target_os = "windows")]
            let os_prefix = if !raw_prompt.to_ascii_lowercase().contains("powershell") && !raw_prompt.to_ascii_lowercase().contains("cmd") {
                "powershell "
            } else {
                ""
            };
            #[cfg(not(target_os = "windows"))]
            let os_prefix = "";

            let query = format!("{}{}", os_prefix, raw_prompt.trim());
            format!(
                "<|im_start|>system\nYou are NioViper, the terminal reflex oracle MicroLM.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                query
            )
        } else {
            Self::format_prompt(raw_prompt)
        }
    }

    pub fn format_prompt(raw_prompt: &str) -> String {
        if raw_prompt.contains("<|im_start|>") {
            raw_prompt.to_string()
        } else {
            format!(
                "<|im_start|>system\nYou are Nio, an AI coding assistant for the terminal. Answer directly, concisely, and helpfully. For casual greetings, reply warmly in one sentence.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
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
        let formatted = self.format_prompt_for_engine(&prompt);
        tokio::task::spawn_blocking(move || {
            let mut guard = bundle_arc.blocking_lock();
            let Some(bundle) = guard.as_mut() else {
                let _ = tx.blocking_send("Error: Model bundle not loaded".into());
                return;
            };

            let encoded = match bundle.tokenizer.encode(formatted.as_str(), false) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Tokenizer error: {e}"));
                    return;
                }
            };

            let prompt_tokens = encoded.get_ids().to_vec();
            if prompt_tokens.is_empty() {
                return;
            }

            let hidden = match bundle.engine.prefill(&prompt_tokens) {
                Ok(h) => h,
                Err(e) => {
                    let _ = tx.blocking_send(format!("AST-Micro prefill error: {e}"));
                    return;
                }
            };

            let eos_token_id = bundle.tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
            let eot_token_id = bundle
                .tokenizer
                .token_to_id("<|endoftext|>")
                .unwrap_or(151643);
            let im_start_token_id = bundle
                .tokenizer
                .token_to_id("<|im_start|>")
                .unwrap_or(151644);
            let is_eos = |tok: u32| {
                tok == eos_token_id || tok == eot_token_id || tok == im_start_token_id
            };

            let mut next_token = match bundle.engine.next_token(&hidden, None) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.blocking_send(format!("AST-Micro sampling error: {e}"));
                    return;
                }
            };

            if is_eos(next_token) {
                bundle.engine.reset_cache();
                return;
            }

            let mut tos = TokenOutputStream::new(bundle.tokenizer.clone());
            let mut phrase_buffer = PhraseBuffer::new();

            if let Ok(Some(chunk)) = tos.next_token(next_token) {
                if chunk.contains("<|im_end|>")
                    || chunk.contains("<|im_start|>")
                    || chunk.contains("<|endoftext|>")
                {
                    bundle.engine.reset_cache();
                    return;
                }
                if !phrase_buffer.push(&chunk, &tx) {
                    bundle.engine.reset_cache();
                    return;
                }
            }

            // Generation loop using prompt-lookup speculative decoding with KV-cache
            let mut context_tokens = prompt_tokens.clone();
            context_tokens.push(next_token);

            let max_gen = max_tokens.max(1);
            let mut current_pos = prompt_tokens.len();
            while current_pos < prompt_tokens.len() + max_gen - 1 {
                let step_hidden = match bundle.engine.forward_stream(&[next_token], current_pos) {
                    Ok(h) => h,
                    Err(_) => break,
                };
                next_token = match bundle.engine.next_token(&step_hidden, None) {
                    Ok(t) => t,
                    Err(_) => break,
                };
                current_pos += 1;

                if is_eos(next_token) {
                    break;
                }

                context_tokens.push(next_token);
                if let Ok(Some(chunk)) = tos.next_token(next_token) {
                    // eprintln!("[DEBUG] pos: {}, tok: {}, chunk: {:?}", current_pos, next_token, chunk);
                    if chunk.contains("<|im_end|>")
                        || chunk.contains("<|im_start|>")
                        || chunk.contains("<|endoftext|>")
                        || chunk.contains("\nuser\n")
                        || chunk.contains("\nUser:")
                    {
                        break;
                    }
                    if !phrase_buffer.push(&chunk, &tx) {
                        break;
                    }
                }
            }

            phrase_buffer.flush(&tx);

            if let Ok(Some(rest)) = tos.decode_rest() {
                let clean = rest
                    .replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<|endoftext|>", "");
                if !clean.is_empty() {
                    let _ = tx.blocking_send(clean);
                }
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

    #[test]
    fn test_phrase_buffer_groups_words_before_flushing() {
        let (tx, mut rx) = mpsc::channel(10);
        std::thread::spawn(move || {
            let mut pb = PhraseBuffer::new();

            assert!(pb.push("Hello", &tx));
            assert!(pb.push(" world", &tx));
            assert!(pb.push(" from", &tx));
            assert!(pb.push(" NioXS", &tx));
            assert!(pb.push(" tail", &tx));
            pb.flush(&tx);
        }).join().unwrap();

        let chunk1 = rx.blocking_recv().expect("Should flush on 4 words");
        assert_eq!(chunk1, "Hello world from NioXS");

        let chunk2 = rx.blocking_recv().expect("Should flush tail");
        assert_eq!(chunk2, " tail");
    }

    #[test]
    fn test_prompt_lookup_finds_ngram() {
        let tokens = vec![10, 20, 30, 40, 50, 10, 20];
        let cand = RayEngine::find_candidate_ngram(&tokens, 2);
        assert_eq!(cand, Some(30));
    }

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

    #[tokio::test]
    async fn test_ray_generation_speed() {
        if let Ok(engine) = RayEngine::new() {
            if engine.ensure_loaded().await.is_ok() {
                let mut guard = engine.bundle.lock().await;
                if let Some(bundle) = guard.as_mut() {
                    let prompt_tokens = vec![1, 2, 3, 4, 5];
                    bundle.engine.reset_cache();
            bundle.engine.evict_all();
                    let hidden = bundle.engine.forward_stream(&prompt_tokens, 0).unwrap();
                    let tok = bundle.engine.next_token(&hidden, None).unwrap();
                    assert!(tok > 0);
                    let (rss, _hwm) = memory_mb();
                    assert!(rss < 100.0, "Resident RAM was {rss}MB, expected <100MB");
                }
            }
        }
    }
}
