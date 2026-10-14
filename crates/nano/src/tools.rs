//! Typed, caller-owned tools and bounded inspection workflows.
use nioxs_contracts::ChatRequest;
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq)]
pub enum GitView {
    Diff,
    Staged,
    History,
}

impl GitView {
    pub fn command(&self) -> &'static str {
        match self {
            Self::Diff => "git --no-pager diff --no-ext-diff --no-textconv",
            Self::Staged => "git --no-pager diff --cached --no-ext-diff --no-textconv",
            Self::History => "git --no-pager log -20 --oneline",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DiagnosticsKind {
    Compiler,
    Lint,
}

impl DiagnosticsKind {
    pub fn command(&self) -> &'static str {
        match self {
            Self::Compiler => "cargo check --locked --message-format=json",
            Self::Lint => "cargo clippy --locked --message-format=json -- -D warnings",
        }
    }
}

/// Canonical schemas describe contracts, not provider-side executors. The
/// caller's advertised tools and policy determine availability each turn.
pub fn registry() -> Vec<Value> {
    let mut entries: Vec<Value> = crate::companion::coding_tool_schemas()
        .into_iter()
        .filter(|schema| matches!(schema["function"]["name"].as_str(), Some("read_file" | "find_files" | "search_code")))
        .map(|schema| json!({"schema":schema,"permission":"read","execution":"caller","result_format":"text"}))
        .collect();
    entries.push(json!({
        "schema":{"type":"function","function":{
            "name":"run_command",
            "description":"Execute a command through the caller's permission broker.",
            "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}
        }},
        "permission":"command_approval", "execution":"caller",
        "result_format":"exit code: <integer> followed by newline and command output"
    }));
    entries.push(json!({
        "schema":{"type":"function","function":{
            "name":"edit_file","description":"Replace a unique exact old_string in a caller project file; reject missing or ambiguous matches and require caller edit approval.",
            "parameters":{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["path","old_string","new_string"],"additionalProperties":false}
        }},"permission":"edit_approval","execution":"caller","result_format":"text; error prefixes indicate failure, followed by mandatory read-back verification"
    }));
    entries.push(json!({
        "schema":{"type":"function","function":{
            "name":"web_fetch","description":"Fetch a reference URL through the caller.",
            "parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"],"additionalProperties":false}
        }},"permission":"network","execution":"caller","result_format":"text"
    }));
    entries
}

pub fn workflow_registry() -> Value {
    let mut workflows = json!([
        {"name":"git_inspect","arguments":{"type":"object","properties":{"view":{"type":"string","enum":["diff","staged","history"]}},"required":["view"],"additionalProperties":false},"required_tools":["run_command"],"permission":"command_approval"},
        {"name":"diagnostics","arguments":{"type":"object","properties":{"kind":{"type":"string","enum":["compiler","lint"]}},"required":["kind"],"additionalProperties":false},"required_tools":["find_files","run_command"],"permission":"command_approval","languages":["rust"]},
        {"name":"project_tasks","arguments":{"type":"object","properties":{},"additionalProperties":false},"required_tools":["find_files"],"conditional_tools":["read_file"],"conditional_reason":"read_file is required when package.json is discovered","permission":"read","manifests":["Cargo.toml","package.json"],"executes_tasks":false}
    ]);
    for name in [
        "symbol_definitions",
        "symbol_references",
        "symbol_callers",
        "symbol_implementations",
    ] {
        workflows.as_array_mut().unwrap().push(json!({
            "name":name,
            "arguments":{"type":"object","properties":{"symbol":{"type":"string","description":"One Rust identifier"}},"required":["symbol"],"additionalProperties":false},
            "required_tools":["search_code","read_file"],"permission":"read","languages":["rust"],
            "resolution":"syntax_name_match","max_read_windows":6,"window_lines":300
        }));
    }
    for (name, required, permission) in [
        (
            "apply_exact_patch",
            vec!["read_file", "edit_file"],
            "edit_approval",
        ),
        (
            "rename_symbol",
            vec!["read_file", "edit_file"],
            "edit_approval",
        ),
        (
            "format_file",
            vec!["read_file", "run_command"],
            "command_approval",
        ),
        ("search_docs", vec!["search_code"], "read"),
        ("fetch_reference", vec!["web_fetch"], "network"),
        ("connected_workflow", vec![], "per_step"),
    ] {
        let arguments = match name {
            "apply_exact_patch" | "rename_symbol" => {
                json!({"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"],"additionalProperties":false})
            }
            "format_file" => {
                json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false})
            }
            "search_docs" => {
                json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false})
            }
            "fetch_reference" => {
                json!({"type":"object","properties":{"url":{"type":"string"}},"required":["url"],"additionalProperties":false})
            }
            _ => {
                json!({"type":"object","properties":{"steps":{"type":"array","minItems":1,"maxItems":6,"description":"Typed search/read/patch/rename/format/check/docs/fetch steps; tool requirements are validated before execution."}},"required":["steps"],"additionalProperties":false})
            }
        };
        workflows.as_array_mut().unwrap().push(json!({"name":name,"required_tools":required,"permission":permission,
            "arguments":arguments,"execution":"caller","max_actions":8,"availability":"connected workflows preflight tools required by their supplied steps"}));
    }
    workflows
}

pub fn session_capabilities(request: &ChatRequest) -> Value {
    let tools: Vec<Value> = registry()
        .into_iter()
        .map(|mut entry| {
            let name = entry["schema"]["function"]["name"].as_str().unwrap();
            entry["available"] = json!(request.permits_tool(name));
            entry
        })
        .collect();
    let mut workflows = workflow_registry();
    for workflow in workflows.as_array_mut().unwrap() {
        let missing: Vec<&str> = workflow["required_tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .filter(|name| !request.permits_tool(name))
            .collect();
        let missing = json!(missing);
        workflow["available"] = json!(missing.as_array().unwrap().is_empty());
        workflow["missing_tools"] = missing;
        if workflow["name"] == "connected_workflow" {
            workflow["available"] = Value::Null;
            workflow["availability_reason"] =
                json!("Supply steps to determine required tools and action budget.");
        }
        let conditional: Vec<&str> = workflow["conditional_tools"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|name| !request.permits_tool(name))
            .collect();
        let conditional = json!(conditional);
        workflow["missing_conditional_tools"] = conditional;
    }
    // Unknown caller tools are visible but are never automatically dispatched.
    let offered: Vec<&str> = request
        .tools
        .iter()
        .filter(|tool| tool["type"] == "function")
        .filter_map(|tool| tool["function"]["name"].as_str())
        .collect();
    json!({"tools":tools,"workflows":workflows,"caller_offered_tools":offered,"execution":"caller","result_outcomes":["tool_required","completed","capability_limitation","clarification","tool_failed","unknown_action"]})
}

/// Discovery displays scripts as data and never treats manifest contents as
/// permission to execute them. Rust commands use fixed built-in mappings.
pub fn project_tasks(
    paths: &[String],
    package: Option<&str>,
    limited: bool,
) -> Result<Value, String> {
    let mut tasks = Vec::new();
    if paths.iter().any(|p| p == "Cargo.toml") {
        for (name, command) in [
            ("check", "cargo check --locked"),
            ("test", "cargo test --locked"),
            ("build", "cargo build --locked"),
            ("lint", "cargo clippy --locked -- -D warnings"),
        ] {
            tasks.push(json!({"name":name,"command":command,"source":"Cargo.toml (fixed Rust mapping)","permission":"command_approval"}));
        }
    }
    if let Some(package) = package {
        let value: Value =
            serde_json::from_str(package).map_err(|_| "invalid or truncated package.json")?;
        if let Some(scripts) = value.get("scripts") {
            let scripts = scripts
                .as_object()
                .ok_or("package.json scripts must be an object")?;
            for (name, script) in scripts {
                let script = script
                    .as_str()
                    .ok_or("package.json script must be a string")?;
                tasks.push(json!({"name":name,"script":script,"source":"package.json","permission":"command_approval"}));
            }
        }
    }
    Ok(
        json!({"tasks":tasks,"inventory_truncated":limited,"executed":false,"supported_manifests":["Cargo.toml","package.json"]}),
    )
}
