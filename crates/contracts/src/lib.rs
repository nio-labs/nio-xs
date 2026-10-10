use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const PROVIDER: &str = "nioxs";
pub const MODEL: &str = "nio-nano";
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
        if self.model != MODEL {
            return Err("unsupported model; select nio-nano".into());
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
