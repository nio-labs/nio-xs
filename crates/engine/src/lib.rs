use anyhow::{Result, anyhow};
use candle_core::Device;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, mpsc};

pub mod gguf;
pub mod quants;
pub mod pool;
pub mod ast_engine;
pub mod hardware;
pub use ast_engine::{AstMicroEngine, memory_mb};
pub use hardware::{ActiveDevice, DevicePreference};

fn valid_draft_token(token: u32, stop_tokens: &[u32], accepts_input: impl FnOnce(u32) -> bool) -> bool {
    token != 0 && !stop_tokens.contains(&token) && accepts_input(token)
}

pub fn skate_system() -> Result<String> {
    if let Some(path) = std::env::var_os("NIOXS_SKATE_SYSTEM_FILE") {
        if let Ok(prompt) = std::fs::read_to_string(&path) {
            if !prompt.trim().is_empty() {
                return Ok(prompt.trim().to_owned());
            }
        }
    }
    let user_home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    if let Some(ref home) = user_home {
        let p = home.join(".nioxs").join("skate_policy.txt");
        if let Ok(prompt) = std::fs::read_to_string(&p) {
            if !prompt.trim().is_empty() {
                return Ok(prompt.trim().to_owned());
            }
        }
    }
    let default_policy = "You are Skate, Nio's ambient companion. Be friendly, clear, grounded, and concise. Explain everyday topics accurately and express uncertainty when you do not know. Do not write code; direct programming requests to Manta or Fin. For medical concerns recommend an appropriate qualified professional and encourage emergency help when urgent. Respond to signs of crisis with empathy and encourage immediate human support and local emergency or crisis resources where appropriate.";
    Ok(default_policy.to_owned())
}


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

const UNIVERSAL_SUBWORDS_JSON: &[u8] = include_bytes!("universal_subwords.json");
const BLUE_SUBWORDS_JSON: &[u8] = include_bytes!("blue_subwords.json");

fn static_subwords(is_blue: bool) -> &'static std::collections::HashMap<u32, u32> {
    static BLUE_SUB: std::sync::OnceLock<std::collections::HashMap<u32, u32>> = std::sync::OnceLock::new();
    static UNIV_SUB: std::sync::OnceLock<std::collections::HashMap<u32, u32>> = std::sync::OnceLock::new();
    if is_blue {
        BLUE_SUB.get_or_init(|| serde_json::from_slice(BLUE_SUBWORDS_JSON).unwrap_or_default())
    } else {
        UNIV_SUB.get_or_init(|| serde_json::from_slice(UNIVERSAL_SUBWORDS_JSON).unwrap_or_default())
    }
}

#[derive(Debug, Clone)]
pub struct ZwCache {
    trigrams: std::collections::HashMap<(u32, u32), std::collections::HashMap<u32, u32>>,
    bigrams: std::collections::HashMap<u32, std::collections::HashMap<u32, u32>>,
    is_blue: bool,
    prompt_tokens: Vec<u32>,
}

impl ZwCache {
    pub fn new() -> Self {
        Self::new_for_model(false)
    }

    pub fn new_for_model(is_blue: bool) -> Self {
        let _ = static_subwords(is_blue);
        Self {
            trigrams: std::collections::HashMap::new(),
            bigrams: std::collections::HashMap::new(),
            is_blue,
            prompt_tokens: Vec::new(),
        }
    }

    pub fn seed_prompt(&mut self, prompt: &[u32]) {
        self.prompt_tokens = prompt.to_vec();
        for w in prompt.windows(2) {
            *self.bigrams.entry(w[0]).or_default().entry(w[1]).or_default() += 4;
        }
        for w in prompt.windows(3) {
            *self.trigrams.entry((w[0], w[1])).or_default().entry(w[2]).or_default() += 4;
        }
    }

    pub fn observe(&mut self, prev2: Option<u32>, prev1: u32, next: u32) {
        let bi = self.bigrams.entry(prev1).or_default();
        *bi.entry(next).or_default() += 4;

        if let Some(p2) = prev2 {
            let tri = self.trigrams.entry((p2, prev1)).or_default();
            *tri.entry(next).or_default() += 4;
        }
    }

    pub fn predict(&self, prev2: Option<u32>, prev1: u32, gen_tokens: &[u32]) -> Option<u32> {
        if let Some(&next) = static_subwords(self.is_blue).get(&prev1) {
            return Some(next);
        }

        if let Some(p2) = prev2 {
            if gen_tokens.len() >= 4 {
                for i in (1..gen_tokens.len() - 1).rev() {
                    if gen_tokens[i - 1] == p2 && gen_tokens[i] == prev1 {
                        return Some(gen_tokens[i + 1]);
                    }
                }
            }

            if let Some(nexts) = self.trigrams.get(&(p2, prev1)) {
                let mut best_tok = 0;
                let mut best_cnt = 0;
                let mut total_cnt = 0;
                for (&tok, &cnt) in nexts {
                    total_cnt += cnt;
                    if cnt > best_cnt {
                        best_cnt = cnt;
                        best_tok = tok;
                    }
                }
                if total_cnt >= 4 && best_cnt * 5 >= total_cnt * 4 {
                    return Some(best_tok);
                }
            }
        }

        if let Some(nexts) = self.bigrams.get(&prev1) {
            let mut best_tok = 0;
            let mut best_cnt = 0;
            let mut total_cnt = 0;
            for (&tok, &cnt) in nexts {
                total_cnt += cnt;
                if cnt > best_cnt {
                    best_cnt = cnt;
                    best_tok = tok;
                }
            }
            if total_cnt >= 4 && best_cnt * 20 >= total_cnt * 17 {
                return Some(best_tok);
            }
        }

        None
    }

    pub fn predict_pair(&self, prev2: Option<u32>, prev1: u32, gen_tokens: &[u32]) -> (Option<u32>, Option<u32>) {
        if let Some(d1) = self.predict(prev2, prev1, gen_tokens) {
            let d2 = static_subwords(self.is_blue)
                .get(&d1)
                .copied()
                .or_else(|| self.predict(Some(prev1), d1, gen_tokens));
            (Some(d1), d2)
        } else {
            (None, None)
        }
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

        // Hold back any unclosed '<...' suffix so stop_position can inspect completed tags
        // before '<' is ever emitted to the stream.
        if let Some(lt_pos) = self.buffer.rfind('<') {
            if !self.buffer[lt_pos..].contains('>') {
                if lt_pos > 0 {
                    let safe_prefix = self.buffer[..lt_pos].to_string();
                    self.buffer.drain(..lt_pos);
                    self.word_count = 0;
                    return tx.blocking_send(safe_prefix).is_ok();
                }
                return true;
            }
        }

        let should_flush = self.buffer.contains('\n')
            || self.buffer.contains("```")
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
            let mut to_send = std::mem::take(&mut self.buffer);
            self.word_count = 0;
            // Strip any incomplete trailing open tag like `<end`, `<|`, or `<`
            if let Some(pos) = to_send.rfind('<') {
                if !to_send[pos..].contains('>') || stop_position(&to_send[pos..]).is_some() {
                    to_send.truncate(pos);
                }
            }
            if !to_send.is_empty() {
                return tx.blocking_send(to_send).is_ok();
            }
        }
        true
    }
}

const STOP_PHRASES: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|end",
    "<|",
    "<invariant",
    "<end",
    "</thought",
    "<thought_tag",
    "You are Skate",
    "You are a helpful AI assistant",
    "\nuser\n",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
];

fn stop_position(text: &str) -> Option<usize> {
    if let Some(pos) = STOP_PHRASES.iter().filter_map(|phrase| text.find(phrase)).min() {
        return Some(pos);
    }
    // Also intercept any trailing hallucinated XML/tag close like <endof...>, <invariant...>, <tag...>
    if let Some(open_idx) = text.rfind('<') {
        let suffix = &text[open_idx..];
        if suffix.contains('>')
            && (suffix.contains("tag")
                || suffix.contains("invariant")
                || suffix.contains("verif")
                || suffix.contains("end")
                || suffix.contains("answer")
                || suffix.starts_with("<|"))
        {
            return Some(open_idx);
        }
    }
    None
}

const MAX_STOP_PHRASE_LEN: usize = 32;

fn push_safe_chunk(
    chunk: &str,
    emitted: &mut String,
    phrase_buffer: &mut PhraseBuffer,
    tx: &mpsc::Sender<String>,
) -> bool {
    let already = emitted.len();
    emitted.push_str(chunk);

    let mut check_start = already.saturating_sub(MAX_STOP_PHRASE_LEN);
    while !emitted.is_char_boundary(check_start) {
        check_start = check_start.saturating_sub(1);
    }

    if let Some(rel_idx) = stop_position(&emitted[check_start..]) {
        let idx = check_start + rel_idx;
        if idx > already && emitted.is_char_boundary(idx) && emitted.is_char_boundary(already) {
            let safe = &emitted[already..idx];
            if !safe.is_empty() && !phrase_buffer.push(safe, tx) {
                return false;
            }
        }
        // Flush any valid completed text accumulated before the stop phrase, but drop the stop phrase itself
        phrase_buffer.flush(tx);
        if emitted.is_char_boundary(idx) {
            emitted.truncate(idx);
        }
        return false;
    }

    phrase_buffer.push(chunk, tx)
}

pub struct ModelBundle {
    pub engine: AstMicroEngine,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

#[derive(Clone)]
pub struct AstEngine {
    bundle: Arc<Mutex<Option<ModelBundle>>>,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    is_skate: bool,
    is_manta: bool,
    is_beluga: bool,
    is_blue: bool,
    device_pref: DevicePreference,
}

#[deprecated(note = "RayEngine has been renamed to AstEngine")]
pub type RayEngine = AstEngine;

impl AstEngine {
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    pub fn tokenizer_path(&self) -> &std::path::Path {
        &self.tokenizer_path
    }

    pub fn new() -> Result<Self> {
        Self::for_model("blue")
    }

    pub fn for_model(model_name: &str) -> Result<Self> {
        Self::for_model_with_device(model_name, DevicePreference::Auto)
    }

    pub fn for_model_with_device(model_name: &str, device_pref: DevicePreference) -> Result<Self> {
        let (model_path, tokenizer_path) = Self::find_or_pull_model_paths(model_name)?;
        let lowered = model_name.trim().to_ascii_lowercase();
        let is_skate = matches!(lowered.as_str(), "skate" | "nio-skate");
        let is_manta = matches!(lowered.as_str(), "manta" | "nio-manta") || model_path.to_string_lossy().contains("manta");
        let is_beluga = matches!(lowered.as_str(), "beluga" | "nio-beluga") || model_path.to_string_lossy().contains("beluga");
        let is_blue = matches!(lowered.as_str(), "blue" | "nio-blue") || model_path.to_string_lossy().contains("blue");
        Ok(Self {
            bundle: Arc::new(Mutex::new(None)),
            model_path,
            tokenizer_path,
            is_skate,
            is_manta,
            is_beluga,
            is_blue,
            device_pref,
        })
    }

    pub fn active_device(&self) -> ActiveDevice {
        ActiveDevice::detect(self.device_pref)
    }

    pub fn set_device_preference(&mut self, pref: DevicePreference) {
        self.device_pref = pref;
    }

    pub fn from_paths(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        let is_skate = model_path.to_string_lossy().contains("skate");
        let is_manta = model_path.to_string_lossy().contains("manta");
        let is_beluga = model_path.to_string_lossy().contains("beluga");
        let is_blue = model_path.to_string_lossy().contains("blue");
        Self {
            bundle: Arc::new(Mutex::new(None)),
            model_path,
            tokenizer_path,
            is_skate,
            is_manta,
            is_beluga,
            is_blue,
            device_pref: DevicePreference::Auto,
        }
    }

    pub fn is_available() -> bool {
        Self::find_model_paths().is_ok()
    }

    pub fn is_model_available(model_name: &str) -> bool {
        Self::find_model_paths_for(model_name).is_ok()
    }

    pub fn find_or_pull_model_paths(model_name: &str) -> Result<(PathBuf, PathBuf)> {
        if matches!(model_name.trim().to_ascii_lowercase().as_str(), "skate" | "nio-skate")
            && std::env::var_os("NIOXS_SKATE_MODEL").is_some()
        {
            return Self::find_model_paths_for(model_name);
        }
        if let Ok(paths) = Self::find_model_paths_for(model_name) {
            return Ok(paths);
        }

        let lowered = model_name.trim().to_ascii_lowercase();
        let normalized = match lowered.as_str() {
            "viper" | "nio-viper" => "nio-viper",
            "ray" | "nio-ray" => "nio-ray",
            "manta" | "nio-manta" => "nio-manta",
            "cuda" | "nio-cuda" => "nio-cuda",
            "beluga" | "nio-beluga" => "nio-beluga",
            "blue" | "nio-blue" => "blue",
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
            "skate" | "nio-skate" => &[
                "skate.vmicrolm-vector.gguf",
                "nio-skate.vmicrolm-vector.gguf",
            ],
            "beluga" | "nio-beluga" => &[
                "beluga.vmicrolm-vector.gguf",
                "nio-beluga.vmicrolm-vector.gguf",
            ],
            "blue" | "nio-blue" => &[
                "blue.vmicrolm-reflex.gguf",
                "nio-blue.vmicrolm-reflex.gguf",
            ],
            "viper" | "nio-viper" => &[
                "nio-viper.Q4_0.gguf",
                "nio-viper-ast-32k.Q4_0.gguf",
                "nio-viper-ast.Q4_0.gguf",
                "nio-viper.Q4_K_M.gguf",
            ],
            "manta" | "nio-manta" => &[
                "manta.vmicrolm-vector.gguf",
                "nio-manta.vmicrolm-vector.gguf",
                "manta.Q4_0.gguf",
                "nio-manta-ast.Q4_0.gguf",
                "nio-manta.Q4_K_M.gguf",
                "nio-manta.Q4_0.gguf",
                "niomanta_int8.onnx",
                "nio-manta.onnx",
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

        let model_override = match normalized.as_str() {
            "skate" | "nio-skate" => Some("NIOXS_SKATE_MODEL"),
            "beluga" | "nio-beluga" => Some("NIOXS_BELUGA_MODEL"),
            "blue" | "nio-blue" => Some("NIOXS_BLUE_MODEL"),
            "ray" | "nio-ray" => Some("NIOXS_RAY_MODEL"),
            _ => None,
        };
        if let Some(variable) = model_override {
            if let Ok(model) = std::env::var(variable) {
                let model = PathBuf::from(model);
                let tokenizer = if variable == "NIOXS_SKATE_MODEL" {
                    std::env::var_os("NIOXS_SKATE_TOKENIZER")
                } else {
                    std::env::var_os("NIOXS_TOKENIZER")
                }
                    .map(PathBuf::from)
                    .unwrap_or_else(|| if variable == "NIOXS_SKATE_MODEL" {
                        model.with_file_name("skate-tokenizer.json")
                    } else {
                        Self::tokenizer_for_model(&model)
                    });
                if model.is_file() && tokenizer.is_file() {
                    return Ok((model, tokenizer));
                }
                return Err(anyhow!(
                    "Configured model or matching tokenizer is missing: {:?}, {:?}",
                    model,
                    tokenizer
                ));
            }
        }

        let directory = nioxs_artifacts::resolve_models_dir();
        for name in target_candidates {
            let model = directory.join(name);
            let tokenizer = Self::tokenizer_for_model(&model);
            if model.is_file() && tokenizer.is_file() {
                return Ok((model, tokenizer));
            }
        }

        Err(anyhow!(
            "Model '{model_name}' or tokenizer not found. Run 'nioxs pull {model_name}' to install it into ~/.nioxs/models/"
        ))
    }

    fn tokenizer_for_model(model: &std::path::Path) -> PathBuf {
        if let Some(s) = model.file_name().and_then(|s| s.to_str()) {
            if s.contains("beluga") {
                let candidate1 = model.with_file_name("beluga-tokenizer.json");
                if candidate1.is_file() {
                    return candidate1;
                }
                let candidate2 = model.with_file_name("nio-beluga-tokenizer.json");
                if candidate2.is_file() {
                    return candidate2;
                }
            }
            if s.contains("blue") {
                let candidate1 = model.with_file_name("blue-tokenizer.json");
                if candidate1.is_file() {
                    return candidate1;
                }
                let candidate2 = model.with_file_name("nio-blue-tokenizer.json");
                if candidate2.is_file() {
                    return candidate2;
                }
            }
            if s.contains("manta") {
                let candidate1 = model.with_file_name("manta-tokenizer.json");
                if candidate1.is_file() {
                    return candidate1;
                }
                let candidate2 = model.with_file_name("nio-manta-tokenizer.json");
                if candidate2.is_file() {
                    return candidate2;
                }
            }
        }
        let name = match model.file_name().and_then(|s| s.to_str()) {
            Some(s) if s.contains("skate") => "skate-tokenizer.json",
            Some(s) if s.contains("beluga") => "beluga-tokenizer.json",
            Some(s) if s.contains("blue") => "blue-tokenizer.json",
            Some("nio-ray-dev.Q4_K_M.gguf") => "nio-ray-dev-tokenizer.json",
            Some(s) if s.contains("viper") => "nio-viper-tokenizer.json",
            Some(s) if s.contains("cuda") => "nio-cuda-tokenizer.json",
            Some(s) if s.contains("manta") => "manta-tokenizer.json",
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
        // Qwen GGUFs define padded vocab sizes (e.g. 151936) while tokenizer.json has 151665
        let debug_mode = cfg!(debug_assertions) || std::env::var_os("NIOXS_DEBUG").is_some();
        if !debug_mode {
            if vocab.len() > tokens.len() {
                return Err(anyhow!(
                    "Tokenizer vocab size {} exceeds model vocab size {}",
                    vocab.len(),
                    tokens.len()
                ));
            }
            // Sentinel check: index 0 and a sample landmark token
            if let Some(expected_token) = tokens.get(0) {
                if !expected_token.starts_with("[PAD") && vocab.get(expected_token).copied() != Some(0) {
                    return Err(anyhow!("Tokenizer mismatch at token ID 0 ({expected_token})"));
                }
            }
            return Ok(());
        }

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

    pub fn is_vector_runtime(&self) -> bool {
        false
    }

    pub async fn ensure_loaded(&self) -> Result<()> {
        if self.is_vector_runtime() {
            if !self.model_path.is_file() {
                return Err(anyhow!("Vector model file not found at {:?}", self.model_path));
            }
            return Ok(());
        }

        let mut guard = self.bundle.lock().await;
        if guard.is_some() {
            return Ok(());
        }

        let model_path = self.model_path.clone();
        let tokenizer_path = self.tokenizer_path.clone();
        let is_skate = self.is_skate;
        let is_manta = self.is_manta;
        let is_beluga = self.is_beluga;
        let is_blue = self.is_blue;

        let active_dev = self.active_device();

        let bundle = tokio::task::spawn_blocking(move || -> Result<ModelBundle> {
            let device = Device::Cpu;
            let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Failed loading tokenizer: {e}"))?;
            tokenizer.with_padding(None);

            let mut engine = AstMicroEngine::new(&model_path)
                .map_err(|e| anyhow!("Failed loading AST-Micro engine for {:?}: {e}", model_path))?;
            Self::validate_tokenizer(&tokenizer, &engine.tokenizer_tokens()?)
                .map_err(|e| anyhow!("Tokenizer does not match model: {e}"))?;

            if is_skate {
                engine.set_suppressed_tokens([3725, 47182]);
            } else if is_manta {
                engine.set_suppressed_tokens([41233, 54275, 73594]);
            } else if is_beluga {
                let gpu_layers = active_dev.recommended_gpu_layers(28);
                if gpu_layers > 0 && std::env::var_os("NIOXS_LOW_RAM").is_none() {
                    engine.set_resident_layers(gpu_layers);
                    engine.set_rolling_window(gpu_layers.max(4));
                } else {
                    engine.set_resident_layers(7);
                    engine.set_rolling_window(4);
                }
            } else if is_blue {
                let gpu_layers = active_dev.recommended_gpu_layers(32);
                if gpu_layers > 0 && std::env::var_os("NIOXS_LOW_RAM").is_none() {
                    engine.set_resident_layers(gpu_layers);
                    engine.set_rolling_window(gpu_layers.max(4));
                } else {
                    engine.set_resident_layers(2);
                    engine.set_rolling_window(2);
                }
            }

            // Immediately park worker threads so nioxs sits at 0.0% CPU when idle
            engine.park();

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

    pub fn format_prompt_for_engine(&self, raw_prompt: &str) -> Result<String> {
        Ok(if raw_prompt.contains("<|im_start|>") {
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
        } else if self.is_beluga {
            format!(
                "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions. Defer raw low-level code generation to Cuda.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                raw_prompt.trim()
            )
        } else if self.is_blue {
            format!(
                "<|im_start|>system\nYou are Blue, Nio's Flagship Sovereign Reflex Companion. Speak with the calm, approachable, and grounded voice of a seasoned staff engineer pair-programming with a colleague. Be conversational and engaging, but keep your dialogue focused and high-signal without fluff or sycophancy. When complex reasoning is required, verify invariants inside concise <think>...</think> tags (under 100 tokens), close your thoughts with </think>, and deliver your answer naturally. Reserve code blocks strictly for executable code.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                raw_prompt.trim()
            )
        } else if self.is_skate {
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                skate_system()?,
                raw_prompt.trim()
            )
        } else if self.is_manta {
            format!(
                "<|im_start|>system\nYou are Manta, Nio's Conversational Guru and DAG Architect. You specialize in systems architecture, boundary analysis, trade-off evaluation, and operational planning. Decompose technical tasks into structured implementation roadmaps using <manta_plan>, <manta_step>, and <manta_risk> tags. You operate strictly as a systems architect; cleanly defer raw code implementation to Cuda or Fin.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                raw_prompt.trim()
            )
        } else {
            Self::format_prompt(raw_prompt)
        })
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
        temperature: f64,
        top_p: f64,
    ) -> Result<mpsc::Receiver<String>> {
        self.ensure_loaded().await?;
        let (tx, rx) = mpsc::channel(64);
        let temperature = temperature.clamp(0.05, 2.0) as f32;
        let top_p = top_p.clamp(0.05, 1.0) as f32;

        let active_dev = self.active_device();
        if active_dev.is_gpu() {
            let runner_bin = std::env::var_os("NIOXS_GPU_RUNNER")
                .or_else(|| std::env::var_os("NIOXS_LLAMA_CLI"))
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("llama-cli"));

            let runner_available = if runner_bin.is_file() {
                true
            } else if let Ok(path_var) = std::env::var("PATH") {
                std::env::split_paths(&path_var).any(|d| d.join(&runner_bin).is_file())
            } else {
                false
            };

            if runner_available {
                let model_path = self.model_path.clone();
                let total_layers = if self.is_blue { 32 } else if self.is_beluga { 28 } else if self.is_manta { 24 } else { 30 };
                let gpu_layers = active_dev.recommended_gpu_layers(total_layers);
                let ngl_arg = gpu_layers.to_string();
                eprintln!("[nioxs] Hardware: {} (offloading {}/{} layers to GPU)", active_dev.name(), gpu_layers, total_layers);

                tokio::task::spawn_blocking(move || {
                    let temp_arg = format!("{temperature:.2}");
                    let top_p_arg = format!("{top_p:.2}");
                    let mut cmd = std::process::Command::new(&runner_bin);
                    cmd.args([
                        "-m", model_path.to_str().unwrap_or_default(),
                        "-p", &prompt,
                        "-n", &max_tokens.to_string(),
                        "-c", "4096",
                        "-ngl", &ngl_arg,
                        "--temp", &temp_arg,
                        "--top-p", &top_p_arg,
                        "--min-p", "0.05",
                        "--repeat-penalty", "1.15",
                        "--repeat-last-n", "64",
                        "--no-display-prompt",
                        "--simple-io",
                    ]);
                    if let Some(lib_dir) = runner_bin.parent().filter(|p| !p.as_os_str().is_empty()) {
                        let existing = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
                        let mut paths = vec![lib_dir.to_path_buf()];
                        paths.extend(std::env::split_paths(&existing));
                        if let Ok(joined) = std::env::join_paths(paths) {
                            cmd.env("LD_LIBRARY_PATH", joined);
                        }
                    }
                    cmd.stdout(std::process::Stdio::piped());
                    cmd.stderr(std::process::Stdio::null());

                    if let Ok(mut child) = cmd.spawn() {
                        if let Some(stdout) = child.stdout.take() {
                            use std::io::{BufRead, BufReader};
                            let reader = BufReader::new(stdout);
                            for line in reader.lines().flatten() {
                                if !line.is_empty() {
                                    let _ = tx.blocking_send(format!("{line}\n"));
                                }
                            }
                        }
                        let _ = child.wait();
                    }
                });
                return Ok(rx);
            }
        }

        if self.is_vector_runtime() {
            let model_path = self.model_path.clone();
            tokio::task::spawn_blocking(move || {
                let runner_bin = std::env::var_os("NIOXS_LLAMA_CLI")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("llama-cli"));

                let temp_arg = format!("{temperature:.2}");
                let top_p_arg = format!("{top_p:.2}");
                let mut cmd = std::process::Command::new(&runner_bin);
                cmd.args([
                    "-m", model_path.to_str().unwrap_or_default(),
                    "-p", &prompt,
                    "-n", &max_tokens.to_string(),
                    "-c", "512",
                    "-t", "2",
                    "-st",
                    "--temp", &temp_arg,
                    "--top-p", &top_p_arg,
                    "--min-p", "0.05",
                    "--repeat-penalty", "1.15",
                    "--repeat-last-n", "64",
                    "--no-display-prompt",
                    "--simple-io",
                ]);
                if let Some(lib_dir) = runner_bin.parent().filter(|p| !p.as_os_str().is_empty()) {
                    let existing = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
                    let mut paths = vec![lib_dir.to_path_buf()];
                    paths.extend(std::env::split_paths(&existing));
                    if let Ok(joined) = std::env::join_paths(paths) {
                        cmd.env("LD_LIBRARY_PATH", joined);
                    }
                }
                cmd.stdout(std::process::Stdio::piped());
                cmd.stderr(std::process::Stdio::null());

                if let Ok(mut child) = cmd.spawn() {
                    if let Some(stdout) = child.stdout.take() {
                        use std::io::{BufRead, BufReader};
                        let reader = BufReader::new(stdout);
                        let mut assistant_started = false;
                        for line in reader.lines().flatten() {
                            if line.contains("<|im_start|>assistant") {
                                assistant_started = true;
                                continue;
                            }
                            if assistant_started {
                                let trimmed = line.trim();
                                if trimmed.starts_with("[ Prompt:") || trimmed == "Exiting..." {
                                    break;
                                }
                                if !line.is_empty() && !trimmed.contains("<|im_end|>") {
                                    let _ = tx.blocking_send(format!("{line}\n"));
                                }
                            } else if line.starts_with('>') && !line.contains("<|im_start|>") {
                                assistant_started = true;
                            }
                        }
                    }
                    let _ = child.wait();
                }
            });
            return Ok(rx);
        }

        let bundle_arc = self.bundle.clone();
        let formatted = self.format_prompt_for_engine(&prompt)?;
        let is_beluga = self.is_beluga;
        let is_blue = self.is_blue;
        tokio::task::spawn_blocking(move || {
            let mut guard = bundle_arc.blocking_lock();
            if tx.is_closed() {
                return;
            }
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

            let repeat_penalty = if is_blue { 1.05 } else if is_beluga { 1.02 } else { 1.10 };
            bundle
                .engine
                .set_sampling(temperature, top_p, repeat_penalty, 64);

            if tx.is_closed() {
                return;
            }

            bundle.engine.unpark();

            let _hidden = match bundle.engine.prefill(&prompt_tokens) {
                Ok(h) => h,
                Err(e) => {
                    let _ = tx.blocking_send(format!("AST-Micro prefill error: {e}"));
                    bundle.engine.park();
                    return;
                }
            };

            let stop_tokens: Vec<u32> = [
                bundle.engine.eos_token_id(),
                bundle.tokenizer.token_to_id("<|im_end|>"),
                bundle.tokenizer.token_to_id("<|endoftext|>"),
                bundle.tokenizer.token_to_id("<|im_start|>"),
            ].into_iter().flatten().collect();
            let is_eos = |tok: u32| stop_tokens.contains(&tok);

            let next_token = match bundle.engine.next_token_with_history(None, &[], false) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.blocking_send(format!("AST-Micro sampling error: {e}"));
                    bundle.engine.park();
                    return;
                }
            };

            if is_eos(next_token) {
                bundle.engine.reset_cache();
                bundle.engine.park();
                return;
            }

            let mut tos = TokenOutputStream::new(bundle.tokenizer.clone());
            let mut phrase_buffer = PhraseBuffer::new();
            let mut emitted = String::new();

            if let Ok(Some(chunk)) = tos.next_token(next_token) {
                if !push_safe_chunk(&chunk, &mut emitted, &mut phrase_buffer, &tx) {
                    phrase_buffer.flush(&tx);
                    bundle.engine.reset_cache();
                    bundle.engine.park();
                    return;
                }
            }

            // Generation loop using V4 Zero-Weight Lookahead (ZW-Spec) with Asymmetric Zero-Penalty Verification
            let mut zw_cache = ZwCache::new_for_model(is_blue);
            zw_cache.seed_prompt(&prompt_tokens);
            if let Some(&last_prompt) = prompt_tokens.last() {
                let p2 = if prompt_tokens.len() >= 2 { Some(prompt_tokens[prompt_tokens.len() - 2]) } else { None };
                zw_cache.observe(p2, last_prompt, next_token);
            }

            let mut gen_tokens = vec![next_token];
            let max_gen = max_tokens.max(1);
            let mut current_confirmed = next_token;
            let mut stream_pos = prompt_tokens.len();
            let mut consecutive_misses = 0usize;

            while gen_tokens.len() < max_gen {
                if tx.is_closed() {
                    break;
                }

                let remaining = max_gen.saturating_sub(gen_tokens.len());
                let prev2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };

                // Predict draft candidates: strictly high-confidence context and subword drafter
                let (d1_opt, d2_opt) = if remaining >= 3 {
                    zw_cache.predict_pair(prev2, current_confirmed, &gen_tokens)
                } else if remaining >= 2 {
                    (zw_cache.predict(prev2, current_confirmed, &gen_tokens), None)
                } else {
                    (None, None)
                };

                // Static draft tables may come from a different tokenizer. Never
                // feed a guess outside this model's embedding vocabulary.
                let valid_draft = |d: &u32| valid_draft_token(*d, &stop_tokens, |t| bundle.engine.accepts_input_token(t));
                let d1_valid = d1_opt.filter(valid_draft);
                let d2_valid = d2_opt.filter(valid_draft);

                if let (Some(d1), Some(d2)) = (d1_valid, d2_valid) {
                    // V5 Architecture: 3-Token Speculative Step (reads weights ONCE for 3 tokens)
                    match bundle.engine.speculative_trio_step(
                        current_confirmed,
                        d1,
                        d2,
                        stream_pos,
                        &gen_tokens,
                        &stop_tokens,
                    ) {
                        Ok(outcome) => {
                            if outcome.matched_count > 0 {
                                consecutive_misses = 0;
                            } else {
                                consecutive_misses += 1;
                            }

                            let mut stopped = false;
                            for &tok in &outcome.accepted_tokens {
                                if !is_eos(tok) && tok != 0 {
                                    let prev_p2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
                                    if let Some(&last_tok) = gen_tokens.last() {
                                        zw_cache.observe(prev_p2, last_tok, tok);
                                    }
                                    gen_tokens.push(tok);

                                    if let Ok(Some(chunk)) = tos.next_token(tok) {
                                        if !push_safe_chunk(&chunk, &mut emitted, &mut phrase_buffer, &tx) {
                                            stopped = true;
                                            break;
                                        }
                                    }
                                    if gen_tokens.len() >= max_gen {
                                        stopped = true;
                                        break;
                                    }
                                } else {
                                    stopped = true;
                                    break;
                                }
                            }

                            current_confirmed = outcome.last_confirmed;
                            stream_pos = outcome.stream_pos;

                            if stopped || outcome.is_eos {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.blocking_send(format!("AST-Micro speculative error: {e}"));
                            break;
                        }
                    }
                } else if let Some(draft_tok) = d1_valid {
                    // V5 Architecture: 2-Token Speculative Step fallback
                    let predict_next_medusa = false;
                    match bundle.engine.speculative_pair_step(
                        current_confirmed,
                        draft_tok,
                        stream_pos,
                        &gen_tokens,
                        &stop_tokens,
                        predict_next_medusa,
                    ) {
                        Ok(outcome) => {
                            if outcome.draft_matched {
                                consecutive_misses = 0;
                            } else {
                                consecutive_misses += 1;
                            }

                            let mut stopped = false;
                            for &tok in &outcome.accepted_tokens {
                                if !is_eos(tok) && tok != 0 {
                                    let prev_p2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
                                    if let Some(&last_tok) = gen_tokens.last() {
                                        zw_cache.observe(prev_p2, last_tok, tok);
                                    }
                                    gen_tokens.push(tok);

                                    if let Ok(Some(chunk)) = tos.next_token(tok) {
                                        if !push_safe_chunk(&chunk, &mut emitted, &mut phrase_buffer, &tx) {
                                            stopped = true;
                                            break;
                                        }
                                    }
                                    if gen_tokens.len() >= max_gen {
                                        stopped = true;
                                        break;
                                    }
                                } else {
                                    stopped = true;
                                    break;
                                }
                            }

                            current_confirmed = outcome.last_confirmed;
                            stream_pos = outcome.stream_pos;

                            if stopped || outcome.is_eos {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.blocking_send(format!("AST-Micro speculative error: {e}"));
                            break;
                        }
                    }
                } else {
                    // Pure Autoregressive 1-token step
                    if consecutive_misses > 0 {
                        consecutive_misses -= 1;
                    }

                    if let Err(e) = bundle.engine.forward_stream(&[current_confirmed], stream_pos) {
                        let _ = tx.blocking_send(format!("AST-Micro forward error: {e}"));
                        break;
                    }
                    stream_pos += 1;

                    let next_tok = match bundle.engine.next_token_with_history(None, &gen_tokens, false) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = tx.blocking_send(format!("AST-Micro sampling error: {e}"));
                            break;
                        }
                    };

                    if is_eos(next_tok) || next_tok == 0 {
                        break;
                    }

                    let prev_p2 = if gen_tokens.len() >= 2 { Some(gen_tokens[gen_tokens.len() - 2]) } else { None };
                    zw_cache.observe(prev_p2, current_confirmed, next_tok);

                    gen_tokens.push(next_tok);
                    current_confirmed = next_tok;

                    if let Ok(Some(chunk)) = tos.next_token(next_tok) {
                        if !push_safe_chunk(&chunk, &mut emitted, &mut phrase_buffer, &tx) {
                            break;
                        }
                    }

                    if gen_tokens.len() >= max_gen {
                        break;
                    }
                }
            }

            if let Ok(Some(rest)) = tos.decode_rest() {
                let _ = push_safe_chunk(&rest, &mut emitted, &mut phrase_buffer, &tx);
            }

            phrase_buffer.flush(&tx);

            bundle.engine.reset_cache();
            bundle.engine.park();
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn speculative_drafts_must_fit_the_model_vocabulary() {
        let in_skate_vocab = |token| token < 49_152;
        assert!(valid_draft_token(49_151, &[2], in_skate_vocab));
        assert!(!valid_draft_token(56_928, &[2], in_skate_vocab));
        assert!(!valid_draft_token(2, &[2], in_skate_vocab));
        assert!(!valid_draft_token(0, &[2], in_skate_vocab));
    }

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

    #[test]
    fn test_zw_cache_predicts_bigrams_and_trigrams() {
        let mut cache = ZwCache::new();
        // Seed a sequence: A -> B -> C -> D, then A -> B -> C
        cache.observe(None, 1, 2);
        cache.observe(Some(1), 2, 3);
        cache.observe(Some(2), 3, 4);

        cache.observe(None, 1, 2);
        cache.observe(Some(1), 2, 3);

        // Bigram 1 -> 2 observed twice
        assert_eq!(cache.predict(None, 1, &[]), Some(2));
        // Trigram (1, 2) -> 3 observed twice
        assert_eq!(cache.predict(Some(1), 2, &[]), Some(3));
        // Single observation trigram (2, 3) -> 4 with 100% confidence
        assert_eq!(cache.predict(Some(2), 3, &[]), Some(4));
        // Unknown token has no prediction
        assert_eq!(cache.predict(None, 999, &[]), None);
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
        assert!(AstEngine::validate_tokenizer(&tokenizer(), &tokens).is_err());
    }

    #[test]
    fn variant_tokenizers_stay_beside_the_model() {
        assert_eq!(
            AstEngine::tokenizer_for_model(std::path::Path::new("/cache/nio-ray-dev.Q4_K_M.gguf")),
            PathBuf::from("/cache/nio-ray-dev-tokenizer.json")
        );
        assert_eq!(
            AstEngine::tokenizer_for_model(std::path::Path::new("/cache/nio-ray.Q4_K_M.gguf")),
            PathBuf::from("/cache/tokenizer.json")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_ray_generation_speed() {
        if let Ok(engine) = AstEngine::new() {
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
