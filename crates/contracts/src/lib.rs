use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const PROVIDER: &str = "nioxs";
pub const MODEL_NANO: &str = "nio-nano";
pub const MODEL_HELM: &str = "nio-helm";
pub const MODEL_NOVA: &str = "nio-nova";
pub const MODEL_ATOM: &str = "nio-atom";
pub const MODEL_NEON: &str = "nio-neon";
pub const MODEL_FLOW: &str = "nio-flow";
pub const MODEL_THINKER: &str = "nio-thinker";
pub const MODEL_ECHO: &str = "nio-echo";
pub const MODEL_SAGE: &str = "nio-sage";
pub const MODEL_QUILL: &str = "nio-quill";
pub const MODEL_SCOUT: &str = "nio-scout";
/// Default public model: the task orchestrator.
pub const MODEL: &str = MODEL_FLOW;
/// Deprecated request alias; kept for existing clients, but no longer advertised.
pub const MODEL_AUTO: &str = "nio-auto";
pub const MODELS: &[&str] = &[
    MODEL,
    MODEL_NANO,
    MODEL_ECHO,
    MODEL_SAGE,
    MODEL_QUILL,
    MODEL_SCOUT,
    MODEL_THINKER,
    MODEL_NOVA,
    MODEL_ATOM,
    MODEL_NEON,
    MODEL_HELM,
];
pub const SCHEMA_VERSION: u32 = 1;
pub const MAX_REQUEST_BYTES: usize = 512 * 1024;
pub const MAX_RESULT_BYTES: usize = 32 * 1024;

pub fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub stream: bool,
}

impl ChatRequest {
    pub fn validate(&self) -> Result<usize, String> {
        if !MODELS.contains(&self.model.as_str()) && self.model != MODEL_AUTO {
            return Err(format!("unsupported model; select {}", MODELS.join(", ")));
        }
        if self.messages.is_empty() || self.messages.len() > 200 {
            return Err("messages must contain 1..200 entries".into());
        }
        if self.tools.len() > 64 {
            return Err("at most 64 tool definitions are supported".into());
        }
        for message in &self.messages {
            if !matches!(
                message.role.as_str(),
                "system" | "developer" | "user" | "assistant" | "tool"
            ) {
                return Err("unsupported message role".into());
            }
            if !message.content.is_null() && !message.content.is_string() {
                return Err("Phase 1 supports text messages only".into());
            }
            let limit = if message.role == "tool" {
                MAX_RESULT_BYTES
            } else {
                128 * 1024
            };
            if message.content.as_str().is_some_and(|s| s.len() > limit) {
                return Err("message exceeds content limit".into());
            }
            if message.role == "tool" && message.tool_call_id.is_none() {
                return Err("tool result requires tool_call_id".into());
            }
            if message.role != "assistant" && !message.tool_calls.is_empty() {
                return Err("only assistants may contain tool_calls".into());
            }
            if message.tool_calls.len() > 16 {
                return Err("too many tool calls".into());
            }
        }
        self.messages
            .iter()
            .rposition(|m| m.role == "user")
            .ok_or_else(|| "a user message is required".into())
    }

    pub fn permits_read(&self) -> bool {
        self.permits_tool("read_file")
    }

    pub fn permits_tool(&self, name: &str) -> bool {
        let choice_ok = self
            .tool_choice
            .as_ref()
            .is_none_or(|v| v.as_str() == Some("auto"));
        choice_ok
            && self
                .tools
                .iter()
                .any(|t| t["type"] == "function" && t["function"]["name"] == name)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    ToolRequired,
    Completed,
    CapabilityLimitation,
    Clarification,
    ToolFailed,
    UnknownAction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Execution {
    pub schema_version: u32,
    pub turn_id: String,
    pub provider: String,
    pub model: String,
    pub outcome: Outcome,
    pub handoff: Option<Handoff>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Handoff {
    pub reason: String,
    pub continuation_required: bool,
    pub completed_action_ids: Vec<String>,
}

// --- Decision contract -----------------------------------------------------
//
// The LLM (or any client) *proposes*; NioNano *decides*. Proposals are scored
// against intent, permissions, required arguments, evidence, cost, and prior
// outcomes. `POST /v1/decide` is advisory only: it never executes a proposal.

pub const MAX_PROPOSALS: usize = 32;
pub const MAX_PROMPT_BYTES: usize = 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub id: String,
    /// Tool or workflow name, or `answer` for a plain reply.
    pub action: String,
    #[serde(default)]
    pub arguments: Value,
    /// Evidence references (digests, paths, or URLs) backing the proposal.
    #[serde(default)]
    pub evidence: Vec<String>,
    /// Relative cost units; higher is more expensive. Used as a bounded penalty.
    #[serde(default)]
    pub cost: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PastOutcome {
    pub action: String,
    pub successes: u32,
    pub failures: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecideRequest {
    pub schema_version: u32,
    pub prompt: String,
    #[serde(default)]
    pub proposals: Vec<Proposal>,
    /// Actions permitted for this turn. Empty means "answer only".
    #[serde(default)]
    pub allowed_actions: Vec<String>,
    /// Argument names a candidate must carry to be executable.
    #[serde(default)]
    pub required_fields: Vec<String>,
    #[serde(default)]
    pub past_outcomes: Vec<PastOutcome>,
}

impl DecideRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err("unsupported schema_version".into());
        }
        if self.prompt.trim().is_empty() || self.prompt.len() > MAX_PROMPT_BYTES {
            return Err("prompt must contain 1..1024 bytes".into());
        }
        if self.proposals.len() > MAX_PROPOSALS {
            return Err("at most 32 proposals are supported".into());
        }
        let mut ids = Vec::with_capacity(self.proposals.len());
        for proposal in &self.proposals {
            if proposal.id.is_empty() || proposal.id.len() > 64 {
                return Err("proposal id must contain 1..64 bytes".into());
            }
            if proposal.action.is_empty() || proposal.action.len() > 64 {
                return Err("proposal action must contain 1..64 bytes".into());
            }
            if ids.contains(&proposal.id) {
                return Err("proposal ids must be unique".into());
            }
            ids.push(proposal.id.clone());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcome {
    Execute,
    Clarify,
    Abstain,
    Handoff,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecisionPolicy {
    pub allowed_actions: Vec<String>,
    /// Actions that appeared in a proposal but were not permitted this turn.
    pub denied: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RankedCandidate {
    pub id: String,
    pub action: String,
    pub score: u32,
    pub intent_match: bool,
    pub permitted: bool,
    pub evidence_refs: usize,
    pub missing_fields: Vec<String>,
    pub reason: String,
}

/// Attached to completions as `nioxs.decision` and returned by `/v1/decide`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Decision {
    pub schema_version: u32,
    pub outcome: DecisionOutcome,
    pub chosen: Option<RankedCandidate>,
    pub ranked: Vec<RankedCandidate>,
    pub confidence: f32,
    pub evidence: Vec<String>,
    pub policy: DecisionPolicy,
    pub abstained_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableEvent {
    pub schema_version: u32,
    pub deployment_id: String,
    pub turn_id: String,
    pub event: String,
    pub model: String,
    pub request_digest: String,
    pub action_id: Option<String>,
    pub result_digest: Option<String>,
    pub expires_at_unix: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn chat_accepts_profiles_and_rejects_unknown_models() {
        for model in MODELS {
            let request: ChatRequest = serde_json::from_value(
                json!({"model":model,"messages":[{"role":"user","content":"hello"}]}),
            )
            .unwrap();
            assert_eq!(request.validate().unwrap(), 0);
        }
        let request: ChatRequest = serde_json::from_value(
            json!({"model":"unknown","messages":[{"role":"user","content":"hello"}]}),
        )
        .unwrap();
        assert!(request.validate().is_err());
    }

    fn request(proposals: Vec<Proposal>) -> DecideRequest {
        DecideRequest {
            schema_version: SCHEMA_VERSION,
            prompt: "read README.md".into(),
            proposals,
            allowed_actions: vec!["read_file".into()],
            required_fields: vec!["path".into()],
            past_outcomes: Vec::new(),
        }
    }

    fn proposal(id: &str) -> Proposal {
        Proposal {
            id: id.into(),
            action: "read_file".into(),
            arguments: json!({"path": "README.md"}),
            evidence: Vec::new(),
            cost: 0,
        }
    }

    #[test]
    fn valid_decide_request_passes_validation() {
        assert!(request(vec![proposal("a")]).validate().is_ok());
        assert!(request(vec![]).validate().is_ok());
    }

    #[test]
    fn schema_version_and_prompt_limits_are_enforced() {
        let mut wrong = request(vec![]);
        wrong.schema_version = 2;
        assert!(wrong.validate().is_err());
        let mut blank = request(vec![]);
        blank.prompt = "   ".into();
        assert!(blank.validate().is_err());
        let mut long = request(vec![]);
        long.prompt = "x".repeat(MAX_PROMPT_BYTES + 1);
        assert!(long.validate().is_err());
    }

    #[test]
    fn duplicate_ids_and_oversized_batches_are_rejected() {
        assert!(
            request(vec![proposal("a"), proposal("a")])
                .validate()
                .is_err()
        );
        let many: Vec<Proposal> = (0..=MAX_PROPOSALS)
            .map(|index| proposal(&format!("p{index}")))
            .collect();
        assert!(request(many).validate().is_err());
    }

    #[test]
    fn unknown_request_fields_are_rejected_by_serde() {
        let value = json!({"schema_version": 1, "prompt": "read README.md", "extra": true});
        assert!(serde_json::from_value::<DecideRequest>(value).is_err());
    }
}
