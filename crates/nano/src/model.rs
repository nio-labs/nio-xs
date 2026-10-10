use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub schema_version: u32,
    pub model_version: String,
    pub feature_version: u32,
    pub algorithm: String,
    pub labels: Vec<String>,
    pub features: Vec<String>,
    pub weights: Vec<Vec<f64>>,
    pub priors: Vec<f64>,
    pub threshold: f64,
    pub margin: f64,
    pub dataset_sha256: String,
    #[serde(skip)]
    index: HashMap<String, usize>,
}

pub struct Prediction {
    pub label: String,
    pub probabilities: Vec<f64>,
}

impl Model {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > 4 * 1024 * 1024 {
            return Err("model artifact exceeds limit".into());
        }
        let mut model: Self =
            serde_json::from_slice(bytes).map_err(|_| "invalid model artifact")?;
        let labels = [
            "read_file",
            "inspect_project",
            "find_files",
            "search_code",
            "run_checks",
            "clarify",
            "unsupported",
        ];
        if model.schema_version != 1
            || model.feature_version != 1
            || model.model_version != "0.1.0-alpha.2"
            || model.algorithm != "multinomial_naive_bayes"
            || model.labels != labels
            || model.features.is_empty()
            || model.features.len() > 8192
            || model.weights.len() != labels.len()
            || model.priors.len() != labels.len()
            || !model.priors.iter().all(|v| v.is_finite())
            || !model
                .weights
                .iter()
                .all(|row| row.len() == model.features.len() && row.iter().all(|v| v.is_finite()))
            || !(0.0..=1.0).contains(&model.threshold)
            || !(0.0..=1.0).contains(&model.margin)
            || model.dataset_sha256.len() != 64
            || !model.dataset_sha256.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err("incompatible or invalid model metadata/weights".into());
        }
        model.index = model
            .features
            .iter()
            .enumerate()
            .map(|(i, word)| (word.clone(), i))
            .collect();
        if model.index.len() != model.features.len() {
            return Err("duplicate model features".into());
        }
        Ok(model)
    }
    pub fn embedded() -> Result<Self, String> {
        Self::load(include_bytes!("../../../models/nano.json"))
    }
    pub fn predict(&self, prompt: &str) -> Prediction {
        let mut counts = HashMap::<usize, usize>::new();
        for feature in features(prompt) {
            if let Some(index) = self.index.get(&feature) {
                *counts.entry(*index).or_default() += 1;
            }
        }
        let scores: Vec<f64> = self
            .priors
            .iter()
            .enumerate()
            .map(|(label, prior)| {
                prior
                    + counts
                        .iter()
                        .map(|(index, count)| self.weights[label][*index] * *count as f64)
                        .sum::<f64>()
            })
            .collect();
        let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities: Vec<f64> = scores.iter().map(|score| (score - peak).exp()).collect();
        let total: f64 = probabilities.iter().sum();
        for p in &mut probabilities {
            *p /= total;
        }
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        order.sort_by(|a, b| probabilities[*b].total_cmp(&probabilities[*a]));
        let label = if counts.is_empty()
            || probabilities[order[0]] < self.threshold
            || probabilities[order[0]] - probabilities[order[1]] < self.margin
        {
            "unsupported".to_string()
        } else {
            self.labels[order[0]].clone()
        };
        Prediction {
            label,
            probabilities,
        }
    }
}

pub fn words(prompt: &str) -> Vec<String> {
    prompt
        .to_ascii_lowercase()
        .split(|c: char| !(c.is_ascii_alphanumeric() || "_./*-".contains(c)))
        .filter(|s| !s.is_empty())
        .take(80)
        .map(str::to_string)
        .collect()
}

pub fn features(prompt: &str) -> Vec<String> {
    let original = words(prompt);
    let risk = [
        "write",
        "new",
        "delete",
        "steal",
        "bypass",
        "novel",
        "weather",
        "translate",
        "email",
        "captcha",
        "deploy",
        "fix",
        "rewrite",
        "refactor",
        "explain",
        "design",
        "recommend",
        "compare",
        "summarize",
        "debug",
        "create",
        "calculate",
        "solve",
        "research",
        "remove",
        "arbitrary",
        "private",
        "passwords",
        "credentials",
        "hack",
        "upload",
        "rm",
        "sudo",
        "web",
        "better",
    ];
    let unsafe_request = original.iter().any(|w| {
        risk.contains(&w.as_str())
            || w.starts_with('/')
            || w.starts_with("../")
            || w.starts_with("~/")
    });
    let missing = original
        .iter()
        .any(|w| ["it", "that", "thing"].contains(&w.as_str()));
    let stop = [
        "please", "now", "me", "for", "our", "the", "this", "a", "an", "to", "of", "in", "at",
        "you", "could", "would", "can", "let", "give", "get", "what", "is", "are", "us", "every",
        "my", "here", "today", "with", "using", "named",
    ];
    let aliases = [
        ("where", "locate"),
        ("which", "find"),
        ("contain", "references"),
        ("containing", "references"),
        ("occurs", "references"),
        ("occurrences", "references"),
        ("matches", "references"),
        ("look", "search"),
        ("linter", "lint"),
        ("compile", "build"),
        ("suite", "tests"),
        ("unit", "tests"),
        ("test", "tests"),
        ("filenames", "files"),
        ("filename", "files"),
        ("repository", "project"),
        ("checkout", "project"),
        ("workspace", "project"),
        ("dependencies", "manifests"),
        ("dependency", "manifest"),
        ("display", "read"),
        ("open", "read"),
        ("contents", "read"),
    ];
    let normalized: Vec<String> = original
        .iter()
        .filter(|w| !stop.contains(&w.as_str()))
        .map(|w| {
            let value = aliases
                .iter()
                .find(|(a, _)| *a == w)
                .map_or(w.as_str(), |(_, b)| *b);
            if value.contains(['.', '/', '*']) {
                "<path>".into()
            } else if value.contains('_') {
                "<symbol>".into()
            } else {
                value.into()
            }
        })
        .collect();
    let mut result = normalized.clone();
    result.extend(
        normalized
            .windows(2)
            .map(|pair| format!("b:{}:{}", pair[0], pair[1])),
    );
    if unsafe_request {
        result.extend(std::iter::repeat_n("<unsupported-risk>".into(), 8));
    }
    if missing {
        result.extend(std::iter::repeat_n("<missing>".into(), 8));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rust_matches_python_predictions_and_probabilities() {
        let model = Model::embedded().unwrap();
        let fixtures: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../../evaluation/phase2-parity.json"))
                .unwrap();
        for fixture in fixtures.as_array().unwrap() {
            let prediction = model.predict(fixture["prompt"].as_str().unwrap());
            assert_eq!(
                prediction.label, fixture["expected_label"],
                "{}",
                fixture["prompt"]
            );
            for (actual, expected) in prediction
                .probabilities
                .iter()
                .zip(fixture["probabilities"].as_array().unwrap())
            {
                assert!((actual - expected.as_f64().unwrap()).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn invalid_weights_and_versions_are_rejected() {
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../../models/nano.json")).unwrap();
        value["feature_version"] = serde_json::json!(99);
        assert!(Model::load(&serde_json::to_vec(&value).unwrap()).is_err());
        value["feature_version"] = serde_json::json!(1);
        value["weights"][0] = serde_json::json!([]);
        assert!(Model::load(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
