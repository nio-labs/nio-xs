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

// --- Helm Verdict Contract ------------------------------------------------

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HelmGateVerdict {
    Go,
    ConditionalGo,
    NoGo,
    NeedsInput,
}

impl HelmGateVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Go => "go",
            Self::ConditionalGo => "conditional_go",
            Self::NoGo => "no_go",
            Self::NeedsInput => "needs_input",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct HelmVerdictContract {
    pub verdict: HelmGateVerdict,
    pub summary: String,
    #[serde(default)]
    pub decisive_evidence: Vec<String>,
    #[serde(default)]
    pub unmet_criteria: Vec<String>,
    #[serde(default)]
    pub residual_risks: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reversibility: Option<String>,
}

impl HelmVerdictContract {
    pub fn parse_from_text(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        // 1. Try to find and parse embedded JSON block or raw JSON object
        if let Some(parsed) = Self::try_parse_json(trimmed) {
            return Some(parsed);
        }

        // 2. Fall back to structured natural language / markdown extraction
        Self::try_parse_markdown(trimmed)
    }

    fn try_parse_json(text: &str) -> Option<Self> {
        // Look for ```json or ```json:helm block first
        if let Some(start_block) = text.find("```json") {
            let rest = &text[start_block..];
            if let Some(line_end) = rest.find('\n') {
                let code_content = &rest[line_end + 1..];
                if let Some(end_block) = code_content.find("```") {
                    let json_candidate = code_content[..end_block].trim();
                    if let Ok(val) = serde_json::from_str::<Value>(json_candidate) {
                        if let Some(contract) = Self::from_json_value(&val) {
                            return Some(contract);
                        }
                    }
                }
            }
        }

        // Look for naked JSON object containing "verdict"
        if let Some(start_brace) = text.find('{') {
            if let Some(end_brace) = text.rfind('}') {
                if end_brace > start_brace {
                    let candidate = &text[start_brace..=end_brace];
                    if let Ok(val) = serde_json::from_str::<Value>(candidate) {
                        if let Some(contract) = Self::from_json_value(&val) {
                            return Some(contract);
                        }
                    }
                }
            }
        }

        None
    }

    fn from_json_value(val: &Value) -> Option<Self> {
        let verdict_str = val.get("verdict")?.as_str()?;
        let verdict = Self::parse_verdict_str(verdict_str)?;
        let summary = val
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let decisive_evidence = extract_string_array(val.get("decisive_evidence"));
        let unmet_criteria = extract_string_array(val.get("unmet_criteria"));
        let residual_risks = extract_string_array(val.get("residual_risks"));
        let reversibility = val
            .get("reversibility")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        Some(Self {
            verdict,
            summary,
            decisive_evidence,
            unmet_criteria,
            residual_risks,
            reversibility,
        })
    }

    fn parse_verdict_str(s: &str) -> Option<HelmGateVerdict> {
        let normalized = s.to_ascii_lowercase().replace(['-', '_', ' '], "");
        match normalized.as_str() {
            "go" => Some(HelmGateVerdict::Go),
            "conditionalgo" => Some(HelmGateVerdict::ConditionalGo),
            "nogo" => Some(HelmGateVerdict::NoGo),
            "needsinput" => Some(HelmGateVerdict::NeedsInput),
            _ => None,
        }
    }

    fn try_parse_markdown(text: &str) -> Option<Self> {
        let lower = text.to_ascii_lowercase();
        let verdict = if lower.contains("conditional go")
            || lower.contains("conditional-go")
            || lower.contains("conditional_go")
        {
            HelmGateVerdict::ConditionalGo
        } else if lower.contains("no-go") || lower.contains("no go") || lower.contains("no_go") {
            HelmGateVerdict::NoGo
        } else if lower.contains("needs input")
            || lower.contains("needs-input")
            || lower.contains("needs_input")
        {
            HelmGateVerdict::NeedsInput
        } else {
            let has_go = text.lines().any(|line| {
                let l = line.to_ascii_lowercase();
                l.starts_with("go:")
                    || l.starts_with("go -")
                    || l.starts_with("verdict: go")
                    || l.starts_with("verdict: **go**")
                    || l.starts_with("**verdict**: go")
                    || l.starts_with("**verdict**: **go**")
                    || l == "go"
                    || l.starts_with("go ")
            }) || (lower.contains("verdict: go") || lower.contains("**go**"));

            if has_go {
                HelmGateVerdict::Go
            } else {
                return None;
            }
        };

        // Extract summary line
        let mut summary = extract_single_field(text, &["summary"]).unwrap_or_default();
        if summary.is_empty() {
            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                let l = trimmed.to_ascii_lowercase();
                if l.contains("go") || l.contains("verdict") || l.contains("input") {
                    if let Some((_, after)) = trimmed.split_once(':') {
                        let after_trimmed = after.trim().trim_matches('*').trim();
                        // Ignore if it's just repeating the verdict keyword
                        if !after_trimmed.is_empty()
                            && Self::parse_verdict_str(after_trimmed).is_none()
                        {
                            summary = after_trimmed.to_string();
                            break;
                        }
                    } else if Self::parse_verdict_str(trimmed).is_none() {
                        summary = trimmed.to_string();
                        break;
                    }
                }
            }
        }
        if summary.is_empty() {
            summary = text
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .to_string();
        }

        let decisive_evidence = extract_markdown_section(text, &["decisive evidence", "evidence"]);
        let unmet_criteria = extract_markdown_section(
            text,
            &["unmet criteria", "unmet completion criteria", "unmet"],
        );
        let residual_risks =
            extract_markdown_section(text, &["residual risks", "unresolved risks", "risks"]);
        let reversibility = extract_single_field(text, &["reversibility"]);

        Some(Self {
            verdict,
            summary,
            decisive_evidence,
            unmet_criteria,
            residual_risks,
            reversibility,
        })
    }
}

fn extract_string_array(val: Option<&Value>) -> Vec<String> {
    val.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn extract_markdown_section(text: &str, headers: &[&str]) -> Vec<String> {
    let mut items = Vec::new();
    let mut collecting = false;

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();

        let is_header = headers.iter().any(|h| {
            lower.starts_with(&format!("{h}:"))
                || lower.starts_with(&format!("**{h}**"))
                || lower.starts_with(&format!("### {h}"))
                || lower.starts_with(&format!("## {h}"))
        });

        if is_header {
            collecting = true;
            if let Some((_, after)) = trimmed.split_once(':') {
                let rest = after.trim().trim_matches('*').trim();
                if !rest.is_empty() && rest.to_ascii_lowercase() != "none" {
                    items.push(rest.to_string());
                }
            }
            continue;
        }

        if collecting {
            if trimmed.starts_with('#')
                || (trimmed.starts_with("**") && trimmed.contains(':'))
                || (trimmed.contains(':')
                    && !trimmed.starts_with("- ")
                    && !trimmed.starts_with("* "))
            {
                break;
            }
            if let Some(item) = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
            {
                let item = item.trim();
                if !item.is_empty() && item.to_ascii_lowercase() != "none" {
                    items.push(item.to_string());
                }
            }
        }
    }

    items
}

fn extract_single_field(text: &str, names: &[&str]) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        for name in names {
            if lower.contains(name) && trimmed.contains(':') {
                if let Some((_, after)) = trimmed.split_once(':') {
                    let field = after.trim().trim_matches('*').trim();
                    if !field.is_empty() {
                        return Some(field.to_string());
                    }
                }
            }
        }
    }
    None
}

// --- Atom Implementation Contract ----------------------------------------

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AtomPatchStatus {
    AppliedCleanly,
    PartiallyApplied,
    PatchOnly,
    Blocked,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AtomCheckResult {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_summary: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AtomContract {
    pub summary: String,
    pub patch_status: AtomPatchStatus,
    #[serde(default)]
    pub modified_files: Vec<String>,
    #[serde(default)]
    pub checks_executed: Vec<AtomCheckResult>,
    #[serde(default)]
    pub unverified_items: Vec<String>,
}

impl AtomContract {
    pub fn parse_from_text(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Some(parsed) = Self::try_parse_json(trimmed) {
            return Some(parsed);
        }

        Self::try_parse_markdown(trimmed)
    }

    fn try_parse_json(text: &str) -> Option<Self> {
        if let Some(start_block) = text.find("```json") {
            let rest = &text[start_block..];
            if let Some(line_end) = rest.find('\n') {
                let code_content = &rest[line_end + 1..];
                if let Some(end_block) = code_content.find("```") {
                    let json_candidate = code_content[..end_block].trim();
                    if let Ok(val) = serde_json::from_str::<Value>(json_candidate) {
                        if let Some(contract) = Self::from_json_value(&val) {
                            return Some(contract);
                        }
                    }
                }
            }
        }

        if let Some(start_brace) = text.find('{') {
            if let Some(end_brace) = text.rfind('}') {
                if end_brace > start_brace {
                    let candidate = &text[start_brace..=end_brace];
                    if let Ok(val) = serde_json::from_str::<Value>(candidate) {
                        if let Some(contract) = Self::from_json_value(&val) {
                            return Some(contract);
                        }
                    }
                }
            }
        }

        None
    }

    fn from_json_value(val: &Value) -> Option<Self> {
        let patch_status = val
            .get("patch_status")
            .and_then(|v| v.as_str())
            .and_then(
                |s| match s.to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
                    "applied_cleanly" | "clean" | "applied" => {
                        Some(AtomPatchStatus::AppliedCleanly)
                    }
                    "partially_applied" | "partial" => Some(AtomPatchStatus::PartiallyApplied),
                    "patch_only" | "proposed" => Some(AtomPatchStatus::PatchOnly),
                    "blocked" | "failed" => Some(AtomPatchStatus::Blocked),
                    _ => None,
                },
            )
            .unwrap_or(AtomPatchStatus::AppliedCleanly);

        let summary = val
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let modified_files = extract_string_array(val.get("modified_files"));
        let unverified_items = extract_string_array(val.get("unverified_items"));

        let mut checks_executed = Vec::new();
        if let Some(checks) = val.get("checks_executed").and_then(|v| v.as_array()) {
            for check in checks {
                let command = check
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if command.is_empty() {
                    continue;
                }
                let exit_code = check
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .map(|c| c as i32);
                let passed = check
                    .get("passed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(exit_code == Some(0));
                let output_summary = check
                    .get("output_summary")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                checks_executed.push(AtomCheckResult {
                    command,
                    exit_code,
                    passed,
                    output_summary,
                });
            }
        }

        Some(Self {
            summary,
            patch_status,
            modified_files,
            checks_executed,
            unverified_items,
        })
    }

    fn try_parse_markdown(text: &str) -> Option<Self> {
        let lower = text.to_ascii_lowercase();
        let patch_status = if lower.contains("partially applied") {
            AtomPatchStatus::PartiallyApplied
        } else if lower.contains("blocked") || lower.contains("could not apply") {
            AtomPatchStatus::Blocked
        } else if lower.contains("patch only")
            || lower.contains("no edit tools")
            || lower.contains("draft only")
        {
            AtomPatchStatus::PatchOnly
        } else {
            AtomPatchStatus::AppliedCleanly
        };

        let summary = extract_single_field(text, &["summary"]).unwrap_or_else(|| {
            text.lines()
                .find(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                .unwrap_or("")
                .trim()
                .to_string()
        });

        let modified_files =
            extract_markdown_section(text, &["modified files", "changed files", "files modified"]);
        let unverified_items = extract_markdown_section(
            text,
            &["unverified items", "unverified", "unverified requirements"],
        );

        Some(Self {
            summary,
            patch_status,
            modified_files,
            checks_executed: Vec::new(),
            unverified_items,
        })
    }
}

// --- Neon Review Contract ------------------------------------------------

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NeonReviewVerdict {
    Approve,
    RequestChanges,
    Comment,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NeonSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct NeonFinding {
    pub severity: NeonSeverity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
    pub description: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct NeonReviewContract {
    pub verdict: NeonReviewVerdict,
    pub summary: String,
    #[serde(default)]
    pub findings: Vec<NeonFinding>,
    pub verification_passed: bool,
}

impl NeonReviewContract {
    pub fn parse_from_text(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Some(parsed) = Self::try_parse_json(trimmed) {
            return Some(parsed);
        }

        Self::try_parse_markdown(trimmed)
    }

    fn try_parse_json(text: &str) -> Option<Self> {
        if let Some(start_block) = text.find("```json") {
            let rest = &text[start_block..];
            if let Some(line_end) = rest.find('\n') {
                let code_content = &rest[line_end + 1..];
                if let Some(end_block) = code_content.find("```") {
                    let json_candidate = code_content[..end_block].trim();
                    if let Ok(val) = serde_json::from_str::<Value>(json_candidate) {
                        if let Some(contract) = Self::from_json_value(&val) {
                            return Some(contract);
                        }
                    }
                }
            }
        }

        if let Some(start_brace) = text.find('{') {
            if let Some(end_brace) = text.rfind('}') {
                if end_brace > start_brace {
                    let candidate = &text[start_brace..=end_brace];
                    if let Ok(val) = serde_json::from_str::<Value>(candidate) {
                        if let Some(contract) = Self::from_json_value(&val) {
                            return Some(contract);
                        }
                    }
                }
            }
        }

        None
    }

    fn from_json_value(val: &Value) -> Option<Self> {
        let verdict_str = val.get("verdict")?.as_str()?;
        let verdict = match verdict_str
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "approve" | "approved" => NeonReviewVerdict::Approve,
            "request_changes" | "changes_requested" => NeonReviewVerdict::RequestChanges,
            "comment" | "neutral" => NeonReviewVerdict::Comment,
            _ => return None,
        };

        let summary = val
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let verification_passed = val
            .get("verification_passed")
            .and_then(|v| v.as_bool())
            .unwrap_or(verdict == NeonReviewVerdict::Approve);

        let mut findings = Vec::new();
        if let Some(arr) = val.get("findings").and_then(|v| v.as_array()) {
            for item in arr {
                let severity = item
                    .get("severity")
                    .and_then(|v| v.as_str())
                    .and_then(|s| match s.to_ascii_lowercase().as_str() {
                        "critical" => Some(NeonSeverity::Critical),
                        "high" => Some(NeonSeverity::High),
                        "medium" => Some(NeonSeverity::Medium),
                        "low" => Some(NeonSeverity::Low),
                        _ => Some(NeonSeverity::Info),
                    })
                    .unwrap_or(NeonSeverity::Medium);

                let file = item
                    .get("file")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let line = item
                    .get("line")
                    .and_then(|v| v.as_u64())
                    .map(|l| l as usize);
                let rule = item
                    .get("rule")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let description = item
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                findings.push(NeonFinding {
                    severity,
                    file,
                    line,
                    rule,
                    description,
                });
            }
        }

        Some(Self {
            verdict,
            summary,
            findings,
            verification_passed,
        })
    }

    fn try_parse_markdown(text: &str) -> Option<Self> {
        let lower = text.to_ascii_lowercase();
        let verdict = if lower.contains("request changes")
            || lower.contains("changes requested")
            || lower.contains("defects found")
            || lower.contains("issues identified")
        {
            NeonReviewVerdict::RequestChanges
        } else if lower.contains("approve")
            || lower.contains("approved")
            || lower.contains("no issues found")
            || lower.contains("looks good")
        {
            NeonReviewVerdict::Approve
        } else {
            NeonReviewVerdict::Comment
        };

        let summary = extract_single_field(text, &["summary"]).unwrap_or_else(|| {
            text.lines()
                .find(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                .unwrap_or("")
                .trim()
                .to_string()
        });

        let verification_passed = verdict == NeonReviewVerdict::Approve;

        Some(Self {
            verdict,
            summary,
            findings: Vec::new(),
            verification_passed,
        })
    }
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

    #[test]
    fn helm_verdict_contract_parses_json_block() {
        let text = r#"
Here is my final synthesis.
```json:helm
{
  "verdict": "conditional_go",
  "summary": "Proposal ready; implementation unverified",
  "decisive_evidence": ["cargo test passed"],
  "unmet_criteria": ["unrun browser checks"],
  "residual_risks": ["minor UI regression"],
  "reversibility": "high"
}
```
"#;
        let contract = HelmVerdictContract::parse_from_text(text).expect("should parse JSON block");
        assert_eq!(contract.verdict, HelmGateVerdict::ConditionalGo);
        assert_eq!(
            contract.summary,
            "Proposal ready; implementation unverified"
        );
        assert_eq!(contract.decisive_evidence, vec!["cargo test passed"]);
        assert_eq!(contract.unmet_criteria, vec!["unrun browser checks"]);
        assert_eq!(contract.residual_risks, vec!["minor UI regression"]);
        assert_eq!(contract.reversibility.as_deref(), Some("high"));
    }

    #[test]
    fn helm_verdict_contract_parses_prose_and_sections() {
        let text = "Conditional Go: proposal is ready; implementation is unverified";
        let contract = HelmVerdictContract::parse_from_text(text).expect("should parse prose");
        assert_eq!(contract.verdict, HelmGateVerdict::ConditionalGo);
        assert_eq!(
            contract.summary,
            "proposal is ready; implementation is unverified"
        );

        let markdown = r#"
### Final Decision
**Verdict**: No-Go
**Summary**: Required tests failed
**Decisive Evidence**:
- cargo test exited with code 101
**Unmet Criteria**:
- all tests must pass
**Residual Risks**:
- service crash on startup
**Reversibility**: low
"#;
        let contract_md =
            HelmVerdictContract::parse_from_text(markdown).expect("should parse markdown");
        assert_eq!(contract_md.verdict, HelmGateVerdict::NoGo);
        assert_eq!(contract_md.summary, "Required tests failed");
        assert_eq!(
            contract_md.decisive_evidence,
            vec!["cargo test exited with code 101"]
        );
        assert_eq!(contract_md.unmet_criteria, vec!["all tests must pass"]);
        assert_eq!(contract_md.residual_risks, vec!["service crash on startup"]);
        assert_eq!(contract_md.reversibility.as_deref(), Some("low"));
    }

    #[test]
    fn atom_contract_parses_json_and_markdown() {
        let json_text = r#"
Patch applied successfully.
```json:atom
{
  "summary": "Implemented JWT auth middleware",
  "patch_status": "applied_cleanly",
  "modified_files": ["src/auth.rs", "src/middleware.rs"],
  "checks_executed": [
    {
      "command": "cargo test test_auth",
      "exit_code": 0,
      "passed": true,
      "output_summary": "1 passed, 0 failed"
    }
  ],
  "unverified_items": ["E2E session expiry"]
}
```
"#;
        let atom = AtomContract::parse_from_text(json_text).expect("should parse atom json");
        assert_eq!(atom.summary, "Implemented JWT auth middleware");
        assert_eq!(atom.patch_status, AtomPatchStatus::AppliedCleanly);
        assert_eq!(
            atom.modified_files,
            vec!["src/auth.rs", "src/middleware.rs"]
        );
        assert_eq!(atom.checks_executed.len(), 1);
        assert_eq!(atom.checks_executed[0].command, "cargo test test_auth");
        assert!(atom.checks_executed[0].passed);
        assert_eq!(atom.unverified_items, vec!["E2E session expiry"]);

        let md_text = r#"
### Implementation Report
**Summary**: Created scaffold for redis client
**Modified Files**:
- src/redis.rs
**Unverified Items**:
- connection pooling under load
"#;
        let atom_md = AtomContract::parse_from_text(md_text).expect("should parse atom md");
        assert_eq!(atom_md.summary, "Created scaffold for redis client");
        assert_eq!(atom_md.patch_status, AtomPatchStatus::AppliedCleanly);
        assert_eq!(atom_md.modified_files, vec!["src/redis.rs"]);
        assert_eq!(
            atom_md.unverified_items,
            vec!["connection pooling under load"]
        );
    }

    #[test]
    fn neon_review_contract_parses_json_and_markdown() {
        let json_text = r#"
Review completed.
```json:neon
{
  "verdict": "request_changes",
  "summary": "Missing error handling on socket close",
  "verification_passed": false,
  "findings": [
    {
      "severity": "high",
      "file": "src/net.rs",
      "line": 42,
      "rule": "unhandled_result",
      "description": "socket.close() can fail silently leaving uncollected descriptors"
    }
  ]
}
```
"#;
        let neon = NeonReviewContract::parse_from_text(json_text).expect("should parse neon json");
        assert_eq!(neon.verdict, NeonReviewVerdict::RequestChanges);
        assert_eq!(neon.summary, "Missing error handling on socket close");
        assert!(!neon.verification_passed);
        assert_eq!(neon.findings.len(), 1);
        assert_eq!(neon.findings[0].severity, NeonSeverity::High);
        assert_eq!(neon.findings[0].file.as_deref(), Some("src/net.rs"));
        assert_eq!(neon.findings[0].line, Some(42));

        let md_text = r#"
### Review Assessment
**Verdict**: Approve
**Summary**: All changes are safe and correctly typed.
"#;
        let neon_md = NeonReviewContract::parse_from_text(md_text).expect("should parse neon md");
        assert_eq!(neon_md.verdict, NeonReviewVerdict::Approve);
        assert!(neon_md.verification_passed);
    }
}
