//! Caller-owned editing and connected workflows. No provider filesystem access.
use crate::{Decision, Workflow, navigation, safe_path};
use nioxs_contracts::{ChatRequest, Outcome};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use syn::visit::{self, Visit};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Step {
    Search {
        query: String,
    },
    Read {
        path: String,
    },
    Patch {
        path: String,
        old: String,
        new: String,
    },
    Rename {
        path: String,
        old: String,
        new: String,
    },
    Format {
        path: String,
    },
    Check {
        kind: String,
    },
    Docs {
        query: String,
    },
    Fetch {
        url: String,
    },
}
impl Step {
    fn name(&self) -> &'static str {
        match self {
            Self::Search { .. } => "search",
            Self::Read { .. } => "read",
            Self::Patch { .. } => "patch",
            Self::Rename { .. } => "rename",
            Self::Format { .. } => "format",
            Self::Check { .. } => "check",
            Self::Docs { .. } => "docs",
            Self::Fetch { .. } => "fetch",
        }
    }
}

fn stop(
    w: &mut Workflow<'_>,
    receipts: &[Value],
    reason: String,
    mut outcome: Outcome,
) -> Decision {
    if receipts
        .last()
        .is_some_and(|r| r["verification"] == "pending")
    {
        outcome = Outcome::UnknownAction;
    }
    w.end(bounded_report(json!({"workflow":"connected_workflow","reason":reason,"steps":receipts,"changes_not_rolled_back":true})),outcome)
}

fn bounded_report(mut report: Value) -> String {
    if serde_json::to_string_pretty(&report).unwrap().len() > 8000 {
        for receipt in report["steps"].as_array_mut().unwrap() {
            for field in ["evidence", "diff"] {
                receipt.as_object_mut().unwrap().remove(field);
            }
        }
        report["content_truncated"] = json!(true);
    }
    if serde_json::to_string_pretty(&report).unwrap().len() > 8000 {
        if let Some(reason) = report["reason"].as_str() {
            report["reason"] = json!(crate::excerpt(reason, 2000));
        }
    }
    serde_json::to_string_pretty(&report).unwrap()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    steps: Vec<Step>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PatchSpec {
    path: String,
    old: String,
    new: String,
}

pub fn parse(prompt: &str) -> Option<Result<Vec<Step>, String>> {
    if prompt == "--workflow" {
        return Some(Err(
            "Use --workflow {\"steps\":[...]} with typed steps.".into()
        ));
    }
    if let Some(json) = prompt
        .strip_prefix("--patch ")
        .or_else(|| prompt.strip_prefix("--patch:"))
    {
        return Some(if prompt.len() > 8192 {
            Err("Patch request exceeds 8 KiB.".into())
        } else {
            serde_json::from_str::<PatchSpec>(json)
                .map(|patch| {
                    vec![Step::Patch {
                        path: patch.path,
                        old: patch.old,
                        new: patch.new,
                    }]
                })
                .map_err(|e| format!("Invalid exact patch: {e}"))
        });
    }
    if prompt == "--patch" {
        return Some(Err(
            "Use --patch {\"path\":\"file\",\"old\":\"target\",\"new\":\"replacement\"}.".into(),
        ));
    }
    if let Some(json) = prompt
        .strip_prefix("--workflow ")
        .or_else(|| prompt.strip_prefix("--workflow:"))
    {
        return Some(if prompt.len() > 8192 {
            Err("Workflow exceeds 8 KiB.".into())
        } else {
            serde_json::from_str::<Plan>(json)
                .map(|p| p.steps)
                .map_err(|e| format!("Invalid workflow: {e}"))
        });
    }
    if prompt == "read that file" || prompt == "read it" {
        return Some(Ok(vec![Step::Read {
            path: "$selected".into(),
        }]));
    }
    for (prefix, action) in [
        ("--format", 0),
        ("--rename", 1),
        ("--docs", 2),
        ("--fetch", 3),
    ] {
        if let Some(rest) = prompt.strip_prefix(prefix) {
            if !rest.is_empty() && !rest.starts_with([' ', ':', '\t']) {
                continue;
            }
            let rest = rest.trim_start_matches(':').trim();
            return Some(match action {
                0 => Ok(vec![Step::Format { path: rest.into() }]),
                1 => {
                    let words: Vec<_> = rest.split_whitespace().collect();
                    if let [old, new, path] = words.as_slice() {
                        Ok(vec![Step::Rename {
                            path: (*path).into(),
                            old: (*old).into(),
                            new: (*new).into(),
                        }])
                    } else {
                        Err("Use --rename <old> <new> <file.rs> for a file-scoped Rust syntax rename.".into())
                    }
                }
                2 => Ok(vec![Step::Docs { query: rest.into() }]),
                _ => Ok(vec![Step::Fetch { url: rest.into() }]),
            });
        }
    }
    None
}

pub fn validate(steps: &[Step], request: &ChatRequest) -> Result<(), String> {
    if steps.is_empty() || steps.len() > 6 {
        return Err("Use one to six workflow steps.".into());
    }
    let mut required = std::collections::BTreeSet::new();
    let max_actions: usize = steps
        .iter()
        .map(|step| match step {
            Step::Patch { .. } | Step::Rename { .. } | Step::Format { .. } => 3,
            Step::Check { .. } => 2,
            _ => 1,
        })
        .sum();
    if max_actions > 8 {
        return Err(
            "Workflow exceeds the eight-action budget; split it before executing edits.".into(),
        );
    }
    for step in steps {
        let path = match step {
            Step::Read { path }
            | Step::Patch { path, .. }
            | Step::Rename { path, .. }
            | Step::Format { path } => Some(path),
            _ => None,
        };
        if let Some(path) = path {
            if path != "$selected" && (!safe_path(path, false) || path == ".") {
                return Err("Use a safe project-relative file or $selected.".into());
            }
        }
        match step {
            Step::Search { query } | Step::Docs { query } => {
                if query.trim().is_empty() || query.len() > 240 {
                    return Err("Use a nonempty search of at most 240 bytes.".into());
                }
                required.insert("search_code");
            }
            Step::Read { .. } => {
                required.insert("read_file");
            }
            Step::Patch { old, new, .. } => {
                if old.is_empty() || old.len() + new.len() > 6000 {
                    return Err(
                        "Patch needs nonempty old text and at most 6 KiB replacement data.".into(),
                    );
                }
                required.extend(["read_file", "edit_file"]);
            }
            Step::Rename { old, new, path } => {
                if !navigation::valid_symbol(old)
                    || !navigation::valid_symbol(new)
                    || old == new
                    || path != "$selected" && !path.ends_with(".rs")
                {
                    return Err("Rust rename needs distinct identifiers and a .rs file.".into());
                }
                required.extend(["read_file", "edit_file"]);
            }
            Step::Format { path } => {
                if path != "$selected" && !path.ends_with(".rs") {
                    return Err("Selected-file formatting supports Rust .rs files.".into());
                }
                required.extend(["read_file", "run_command"]);
            }
            Step::Check { kind } => {
                if !["test", "build", "lint", "check"].contains(&kind.as_str()) {
                    return Err("Choose test, build, lint, or check.".into());
                }
                required.extend(["find_files", "run_command"]);
            }
            Step::Fetch { url } => {
                if !crate::is_web_url(url) {
                    return Err("Use an http(s) reference URL.".into());
                }
                required.insert("web_fetch");
            }
        }
    }
    for tool in required {
        if !request.permits_tool(tool) {
            return Err(format!(
                "Workflow requires offered {tool} under the caller's current policy; no actions were dispatched."
            ));
        }
    }
    Ok(())
}

pub fn from_intents(intents: &[crate::Intent]) -> Option<Vec<Step>> {
    let mut steps = Vec::new();
    for intent in intents {
        match intent {
            crate::Intent::Connected(nested) => steps.extend(nested.clone()),
            crate::Intent::Read(path) => steps.push(Step::Read { path: path.clone() }),
            crate::Intent::Search(query) => steps.push(Step::Search {
                query: query.clone(),
            }),
            crate::Intent::RefactorPatch { file, old, new } => steps.push(Step::Patch {
                path: file.clone(),
                old: old.clone(),
                new: new.clone(),
            }),
            crate::Intent::Checks(kind) => steps.push(Step::Check { kind: kind.clone() }),
            crate::Intent::WebFetch(url) => steps.push(Step::Fetch { url: url.clone() }),
            _ => return None,
        }
    }
    Some(steps)
}

/// Resolve only the preceding user turn's correlated tool evidence. Assistant
/// prose cannot create a selected path, and arbitrary commands invalidate it.
pub fn previous_selection(request: &ChatRequest, start: usize) -> Option<String> {
    let previous = request.messages[..start]
        .iter()
        .rposition(|m| m.role == "user")?;
    let mut selected = None;
    for (i, message) in request.messages[previous..start].iter().enumerate() {
        if message.role != "tool" {
            continue;
        }
        let call = request.messages[previous..previous + i]
            .iter()
            .rev()
            .flat_map(|m| &m.tool_calls)
            .find(|call| Some(call.id.as_str()) == message.tool_call_id.as_deref())?;
        let args: Value = serde_json::from_str(&call.function.arguments).ok()?;
        let result = message.content.as_str()?;
        if result.starts_with("Error:")
            || result.starts_with("error:")
            || result.contains("user denied")
        {
            continue;
        }
        match call.function.name.as_str() {
            "read_file" => {
                selected = args["path"]
                    .as_str()
                    .filter(|p| safe_path(p, false))
                    .map(str::to_string);
            }
            "search_code" => {
                selected = search_selection(result).ok().flatten();
            }
            "run_command" => {
                let command = args["command"].as_str().unwrap_or("");
                if ![
                    "cargo test --locked",
                    "cargo check --locked",
                    "cargo build --locked",
                    "cargo clippy --locked -- -D warnings",
                ]
                .contains(&command)
                {
                    selected = None;
                }
            }
            "edit_file" | "apply_patch" | "write_file" => {
                selected = None;
            }
            _ => {}
        }
    }
    selected
}
fn search_selection(text: &str) -> Result<Option<String>, String> {
    let result: Value = serde_json::from_str(text).map_err(|_| "Invalid search result")?;
    let mut paths = std::collections::BTreeSet::new();
    for hit in result["matches"]
        .as_array()
        .ok_or("Missing search matches")?
    {
        let path = hit["path"]
            .as_str()
            .filter(|p| safe_path(p, false))
            .ok_or("Invalid search path")?;
        paths.insert(path.to_string());
    }
    if result["scan_limited"] == true || !result["next_offset"].is_null() || paths.len() != 1 {
        return Ok(None);
    }
    Ok(paths.into_iter().next())
}
fn resolve(path: &str, selected: &Option<String>) -> Result<String, String> {
    if path == "$selected" {
        selected.clone().ok_or("Select one file explicitly; prior search evidence is absent, ambiguous, or incomplete.".into())
    } else {
        Ok(path.into())
    }
}

struct RenameVisitor<'a> {
    old: &'a str,
    new: &'a str,
    source: &'a str,
    offsets: Vec<usize>,
    ranges: Vec<(usize, usize)>,
    shadowed: bool,
    collision: bool,
    opaque: bool,
}
impl<'ast> Visit<'ast> for RenameVisitor<'_> {
    fn visit_ident(&mut self, ident: &'ast proc_macro2::Ident) {
        let name = ident.to_string();
        if name.trim_start_matches("r#") == self.new.trim_start_matches("r#") {
            self.collision = true;
        }
        if name.trim_start_matches("r#") == self.old.trim_start_matches("r#") {
            let start = ident.span().start();
            let end = ident.span().end();
            if start.line == end.line {
                let a = self.offsets[start.line - 1] + start.column;
                let b = self.offsets[end.line - 1] + end.column;
                if self.source.get(a..b) == Some(name.as_str()) {
                    self.ranges.push((a, b));
                } else {
                    // proc_macro2 columns count characters, while edits use
                    // byte offsets. Convert when non-ASCII text precedes a token.
                    let line = self.source.lines().nth(start.line - 1).unwrap_or("");
                    let a = self.offsets[start.line - 1]
                        + line
                            .char_indices()
                            .nth(start.column)
                            .map_or(line.len(), |(i, _)| i);
                    let b = self.offsets[end.line - 1]
                        + line
                            .char_indices()
                            .nth(end.column)
                            .map_or(line.len(), |(i, _)| i);
                    if self.source.get(a..b) == Some(name.as_str()) {
                        self.ranges.push((a, b));
                    } else {
                        self.opaque = true;
                    }
                }
            }
        }
    }
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        if pat.ident.to_string().trim_start_matches("r#") == self.old.trim_start_matches("r#") {
            self.shadowed = true;
        }
        visit::visit_pat_ident(self, pat);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if mac
            .tokens
            .to_string()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .any(|w| w == self.old.trim_start_matches("r#"))
        {
            self.opaque = true;
        }
        visit::visit_macro(self, mac);
    }
}
pub fn rename(source: &str, path: &str, old: &str, new: &str) -> Result<String, String> {
    let definitions =
        navigation::analyze(path, source, old, &navigation::NavigationKind::Definitions)?;
    if definitions.len() != 1 {
        return Err("File-scoped rename requires exactly one declaration of the old name.".into());
    }
    let ast = syn::parse_file(source).map_err(|e| e.to_string())?;
    let mut offsets = vec![0];
    offsets.extend(source.match_indices('\n').map(|(i, _)| i + 1));
    let mut visitor = RenameVisitor {
        old,
        new,
        source,
        offsets,
        ranges: vec![],
        shadowed: false,
        collision: false,
        opaque: false,
    };
    visitor.visit_file(&ast);
    if visitor.shadowed || visitor.collision || visitor.opaque {
        return Err("Rename refused: shadowing, a destination-name collision, or opaque macro usage needs compiler-aware review.".into());
    }
    let mut output = source.to_string();
    visitor.ranges.sort_unstable();
    visitor.ranges.dedup();
    for (a, b) in visitor.ranges.into_iter().rev() {
        output.replace_range(a..b, new);
    }
    syn::parse_file(&output).map_err(|e| format!("Renamed Rust syntax is invalid: {e}"))?;
    Ok(output)
}
fn diff(path: &str, before: &str, after: &str) -> String {
    if before == after {
        return String::new();
    }
    let mut result = format!(
        "--- a/{path}\n+++ b/{path}\n@@ -1,{} +1,{} @@\n",
        before.lines().count(),
        after.lines().count()
    );
    for line in before.lines() {
        result.push('-');
        result.push_str(line);
        result.push('\n');
    }
    if !before.ends_with('\n') {
        result.push_str("\\ No newline at end of file\n");
    }
    for line in after.lines() {
        result.push('+');
        result.push_str(line);
        result.push('\n');
    }
    if !after.ends_with('\n') {
        result.push_str("\\ No newline at end of file\n");
    }
    result
}

pub(super) fn execute(w: &mut Workflow<'_>, steps: &[Step]) -> Result<Decision, String> {
    if let Err(reason) = validate(steps, w.request) {
        return Ok(w.limitation(&reason, false));
    }
    let mut selected = previous_selection(w.request, w.start);
    let mut cache = std::collections::BTreeMap::<String, String>::new();
    let mut receipts = Vec::<Value>::new();
    macro_rules! action {
        ($name:expr,$args:expr) => {{
            match w.tool($name, $args)? {
                Ok(result) => result,
                Err(mut decision) => {
                    if decision.execution.outcome != Outcome::ToolRequired && !receipts.is_empty() {
                        decision.content = Some(bounded_report(json!({"workflow":"connected_workflow","reason":decision.content.unwrap_or_default(),"steps":receipts,"changes_not_rolled_back":true})));
                    }
                    return Ok(decision);
                }
            }
        }};
    }
    macro_rules! read {
        ($path:expr)=>{{
            let path=$path;
            if let Some(source)=cache.get(path) {source.clone()} else {
                let text=action!("read_file",json!({"path":path,"start_line":1,"line_count":300}));
                let (source,end,total)=match navigation::read_window(&text,1) {
                    Ok(window)=>window,
                    Err(reason)=>return Ok(stop(w, &receipts, format!("Editing/read coverage is unverified: {reason}. No further action was dispatched."),Outcome::Clarification)),
                };
                if end!=total||source.len()>8000 {return Ok(stop(w, &receipts, "This workflow requires a complete file of at most 300 lines and 8 KiB; use a narrower patch or an upstream editing workflow.".into(),Outcome::Clarification));}
                cache.insert(path.clone(),source.to_string());source.to_string()
            }
        }};
    }
    for step in steps {
        match step {
            Step::Search { query } | Step::Docs { query } => {
                let mut args =
                    json!({"query":query,"path":".","regex":false,"context_lines":1,"limit":30});
                if matches!(step, Step::Docs { .. }) {
                    args["glob"] = json!("**/*.md");
                }
                let result = action!("search_code", args);
                selected = search_selection(&result)?;
                receipts.push(json!({"action":step.name(),"selected_path":selected,"evidence":crate::excerpt(&result,1500)}));
            }
            Step::Read { path } => {
                let path = match resolve(path, &selected) {
                    Ok(path) => path,
                    Err(reason) => return Ok(stop(w, &receipts, reason, Outcome::Clarification)),
                };
                let source = read!(&path);
                selected = Some(path.clone());
                receipts.push(
                    json!({"action":"read","path":path,"evidence":crate::excerpt(&source,2000)}),
                );
            }
            Step::Patch { path, .. } | Step::Rename { path, .. } | Step::Format { path } => {
                let path = match resolve(path, &selected) {
                    Ok(path) => path,
                    Err(reason) => return Ok(stop(w, &receipts, reason, Outcome::Clarification)),
                };
                if matches!(step, Step::Rename { .. } | Step::Format { .. })
                    && !path.ends_with(".rs")
                {
                    return Ok(stop(
                        w,
                        &receipts,
                        "Rust formatting/renaming requires a .rs file.".into(),
                        Outcome::Clarification,
                    ));
                }
                let before = read!(&path);
                let expected = match step {
                    Step::Patch { old, new, .. } => {
                        match crate::apply_patch_to_string(&before, old, new, false) {
                            Ok((after, _)) => Some(after),
                            Err(reason) => {
                                return Ok(stop(w, &receipts, reason, Outcome::Clarification));
                            }
                        }
                    }
                    Step::Rename { old, new, .. } => match rename(&before, &path, old, new) {
                        Ok(after) => Some(after),
                        Err(reason) => {
                            return Ok(stop(w, &receipts, reason, Outcome::Clarification));
                        }
                    },
                    _ => None,
                };
                if let Some(after) = &expected {
                    if path.ends_with(".rs") && syn::parse_file(after).is_err() {
                        return Ok(stop(
                            w,
                            &receipts,
                            "Proposed replacement has invalid Rust syntax; no edit was dispatched."
                                .into(),
                            Outcome::Clarification,
                        ));
                    }
                    action!(
                        "edit_file",
                        json!({"path":path,"old_string":before,"new_string":after})
                    );
                } else {
                    action!(
                        "run_command",
                        json!({"command":format!("rustfmt --edition 2024 --config skip_children=true -- '{path}'")})
                    );
                }
                receipts.push(
                    json!({"action":"mutation_dispatched","path":path,"verification":"pending"}),
                );
                cache.clear(); // No stale read survives any successful mutation.
                let after = read!(&path);
                if expected.as_ref().is_some_and(|expected| expected != &after) {
                    return Ok(stop(
                        w,
                        &receipts,
                        format!(
                            "Mutation of {path} was reported but read-back differs from the proposed edit. No retry or validation command was dispatched."
                        ),
                        Outcome::UnknownAction,
                    ));
                }
                selected = Some(path.clone());
                if let Some(receipt) = receipts.last_mut() {
                    receipt["verification"] = json!("read_back_verified");
                }
                receipts.push(json!({"action":step.name(),"path":path,"verification":"read_back_verified","diff":crate::excerpt(&diff(&path,&before,&after),3000)}));
            }
            Step::Check { kind } => {
                let inventory = action!(
                    "find_files",
                    json!({"path":".","glob":"Cargo.toml","limit":50})
                );
                let (paths, _) = crate::files(&inventory)?;
                if !paths.iter().any(|p| p == "Cargo.toml") {
                    return Ok(stop(
                        w,
                        &receipts,
                        "Rust check/test steps require root Cargo.toml; no command was dispatched."
                            .into(),
                        Outcome::Clarification,
                    ));
                }
                let command = match kind.as_str() {
                    "test" => "cargo test --locked",
                    "build" => "cargo build --locked",
                    "lint" => "cargo clippy --locked -- -D warnings",
                    _ => "cargo check --locked",
                };
                let result = action!("run_command", json!({"command":command}));
                cache.clear();
                receipts.push(json!({"action":step.name(),"command":command,"evidence":crate::excerpt(&result,1500)}));
            }
            Step::Fetch { url } => {
                let result = action!("web_fetch", json!({"url":url}));
                receipts
                    .push(json!({"action":step.name(),"evidence":crate::excerpt(&result,2000)}));
            }
        }
    }
    let mut report = json!({"workflow":"connected_workflow","selected_path":selected,"steps":receipts,"content_truncated":false});
    while serde_json::to_string_pretty(&report).unwrap().len() > 8000 {
        let mut shrank = false;
        for receipt in report["steps"].as_array_mut().unwrap() {
            for key in ["evidence", "diff"] {
                if receipt.get(key).is_some() {
                    receipt.as_object_mut().unwrap().remove(key);
                    shrank = true;
                }
            }
        }
        report["content_truncated"] = json!(true);
        if !shrank {
            break;
        }
    }
    Ok(w.end(
        serde_json::to_string_pretty(&report).unwrap(),
        Outcome::Completed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(steps: Value) -> ChatRequest {
        serde_json::from_value(json!({"model":"nio-nano","messages":[{"role":"user","content":format!("--workflow {}",json!({"steps":steps}))}],
            "tools":crate::tools::registry().iter().map(|entry|entry["schema"].clone()).collect::<Vec<_>>()
        })).unwrap()
    }
    fn respond(request: &mut ChatRequest, decision: &Decision, result: &str) {
        let call = decision.call.as_ref().expect("caller action");
        request.messages.push(
            serde_json::from_value(json!({"role":"assistant","content":null,"tool_calls":[call]}))
                .unwrap(),
        );
        request.messages.push(
            serde_json::from_value(json!({"role":"tool","tool_call_id":call.id,"content":result}))
                .unwrap(),
        );
    }
    #[test]
    fn connected_find_inspect_fix_test_uses_caller_evidence() {
        let mut request = request(json!([
            {"action":"search","query":"old_name"},{"action":"read","path":"$selected"},
            {"action":"patch","path":"$selected","old":"old_name","new":"new_name"},
            {"action":"check","kind":"test"}
        ]));
        for (name, result) in [
            ("search_code", r#"{"matches":[{"path":"src/lib.rs"}]}"#),
            ("read_file", "Lines 1-1 of 1:\nfn old_name() {}"),
            ("edit_file", "Edited src/lib.rs"),
            ("read_file", "Lines 1-1 of 1:\nfn new_name() {}"),
            ("find_files", r#"{"files":["Cargo.toml"]}"#),
            ("run_command", "exit code: 0\n1 test passed"),
        ] {
            let decision = crate::decide(&request, "test").unwrap();
            assert_eq!(decision.call.as_ref().unwrap().function.name, name);
            if name == "edit_file" {
                let args: Value =
                    serde_json::from_str(&decision.call.as_ref().unwrap().function.arguments)
                        .unwrap();
                assert_eq!(args["old_string"], "fn old_name() {}");
                assert_eq!(args["new_string"], "fn new_name() {}");
            }
            respond(&mut request, &decision, result);
        }
        let completed = crate::decide(&request, "test").unwrap();
        assert_eq!(completed.execution.outcome, Outcome::Completed);
        assert_eq!(completed.observed.len(), 6);
        let report: Value = serde_json::from_str(&completed.content.unwrap()).unwrap();
        assert!(report["steps"].as_array().unwrap().iter().any(|step| {
            step["verification"] == "read_back_verified"
                && step["diff"]
                    .as_str()
                    .is_some_and(|diff| diff.contains("+fn new_name()"))
        }));
    }
    #[test]
    fn formatting_and_rename_are_selected_file_actions() {
        for (step, tool, after) in [
            (
                json!({"action":"format","path":"src/lib.rs"}),
                "run_command",
                "fn old_name() {}",
            ),
            (
                json!({"action":"rename","path":"src/lib.rs","old":"old_name","new":"new_name"}),
                "edit_file",
                "fn new_name() {}",
            ),
        ] {
            let mut request = request(json!([step]));
            let read = crate::decide(&request, "test").unwrap();
            respond(&mut request, &read, "Lines 1-1 of 1:\nfn old_name() {}");
            let mutation = crate::decide(&request, "test").unwrap();
            assert_eq!(mutation.call.as_ref().unwrap().function.name, tool);
            if tool == "run_command" {
                let args: Value =
                    serde_json::from_str(&mutation.call.as_ref().unwrap().function.arguments)
                        .unwrap();
                assert_eq!(
                    args["command"],
                    "rustfmt --edition 2024 --config skip_children=true -- 'src/lib.rs'"
                );
            }
            respond(
                &mut request,
                &mutation,
                if tool == "run_command" {
                    "exit code: 0\nformatted"
                } else {
                    "Edited"
                },
            );
            let verify = crate::decide(&request, "test").unwrap();
            assert_eq!(verify.call.as_ref().unwrap().function.name, "read_file");
            respond(&mut request, &verify, &format!("Lines 1-1 of 1:\n{after}"));
            assert_eq!(
                crate::decide(&request, "test").unwrap().execution.outcome,
                Outcome::Completed
            );
        }
    }
    #[test]
    fn preflight_and_ambiguous_selection_dispatch_no_edit() {
        let mut blocked = request(json!([{"action":"patch","path":"lib.rs","old":"a","new":"b"}]));
        blocked
            .tools
            .retain(|tool| tool["function"]["name"] != "edit_file");
        assert!(crate::decide(&blocked, "test").unwrap().call.is_none());
        for path in ["../outside.rs", ".env", "bad';cmd.rs"] {
            let blocked = request(json!([{"action":"format","path":path}]));
            assert!(crate::decide(&blocked, "test").unwrap().call.is_none());
        }
        let mut ambiguous = request(
            json!([{"action":"search","query":"term"},{"action":"read","path":"$selected"}]),
        );
        let search = crate::decide(&ambiguous, "test").unwrap();
        respond(
            &mut ambiguous,
            &search,
            r#"{"matches":[{"path":"a.rs"},{"path":"b.rs"}]}"#,
        );
        let clarification = crate::decide(&ambiguous, "test").unwrap();
        assert_eq!(clarification.execution.outcome, Outcome::Clarification);
        assert!(clarification.call.is_none());
        let over_budget = request(json!([
            {"action":"patch","path":"a.rs","old":"a","new":"b"},
            {"action":"patch","path":"b.rs","old":"a","new":"b"},
            {"action":"patch","path":"c.rs","old":"a","new":"b"}
        ]));
        assert!(crate::decide(&over_budget, "test").unwrap().call.is_none());
    }
    #[test]
    fn missing_downstream_permissions_and_incomplete_reads_prevent_edits() {
        let mut missing = request(json!([
            {"action":"patch","path":"lib.rs","old":"old_name","new":"new_name"},
            {"action":"check","kind":"test"}
        ]));
        missing
            .tools
            .retain(|t| t["function"]["name"] != "run_command");
        assert!(crate::decide(&missing, "test").unwrap().call.is_none());
        for result in ["Lines 1-1 of 2:\nfn old_name() {}", "unlabelled source"] {
            let mut incomplete = request(
                json!([{"action":"patch","path":"lib.rs","old":"old_name","new":"new_name"}]),
            );
            let read = crate::decide(&incomplete, "test").unwrap();
            respond(&mut incomplete, &read, result);
            let stopped = crate::decide(&incomplete, "test").unwrap();
            assert_eq!(stopped.execution.outcome, Outcome::Clarification);
            assert!(stopped.call.is_none());
        }
        let mut denied =
            request(json!([{"action":"patch","path":"lib.rs","old":"old_name","new":"new_name"}]));
        let read = crate::decide(&denied, "test").unwrap();
        respond(&mut denied, &read, "Lines 1-1 of 1:\nfn old_name() {}");
        let edit = crate::decide(&denied, "test").unwrap();
        respond(&mut denied, &edit, "Error: user denied");
        let stopped = crate::decide(&denied, "test").unwrap();
        assert_eq!(stopped.execution.outcome, Outcome::ToolFailed);
        assert!(stopped.call.is_none());
    }
    #[test]
    fn exact_patch_directive_and_natural_chaining_are_supported() {
        let patch = crate::parse(
            &crate::Model::embedded().unwrap(),
            r#"--patch {"path":"lib.rs","old":"old_name","new":"new_name"}"#,
        );
        assert!(matches!(patch, crate::Intent::Connected(_)));
        let chain = crate::parse(
            &crate::Model::embedded().unwrap(),
            "search code for target then read that file",
        );
        let crate::Intent::Plan(intents) = chain else {
            panic!("connected plan expected")
        };
        assert_eq!(from_intents(&intents).unwrap().len(), 2);
    }
    #[test]
    fn failed_report_limits_keep_partial_edit_status_and_valid_json() {
        let report = bounded_report(
            json!({"steps":[{"action":"patch","verification":"read_back_verified","diff":"x".repeat(10000)}],"reason":"failure evidence","changes_not_rolled_back":true}),
        );
        assert!(report.len() <= 8000);
        let report: Value = serde_json::from_str(&report).unwrap();
        assert_eq!(report["content_truncated"], true);
        assert_eq!(report["steps"][0]["verification"], "read_back_verified");
    }
    #[test]
    fn failed_verification_and_commands_stop_without_retries() {
        let mut request = request(
            json!([{"action":"patch","path":"lib.rs","old":"old_name","new":"new_name"},{"action":"check","kind":"test"}]),
        );
        let read = crate::decide(&request, "test").unwrap();
        respond(&mut request, &read, "Lines 1-1 of 1:\nfn old_name() {}");
        let edit = crate::decide(&request, "test").unwrap();
        respond(&mut request, &edit, "Edited");
        let verify = crate::decide(&request, "test").unwrap();
        respond(&mut request, &verify, "Lines 1-1 of 1:\nfn old_name() {}");
        let failed = crate::decide(&request, "test").unwrap();
        assert_eq!(failed.execution.outcome, Outcome::UnknownAction);
        assert!(failed.call.is_none());
        assert!(failed.content.unwrap().contains("mutation_dispatched"));
        request.messages.last_mut().unwrap().content = json!("Lines 1-1 of 1:\nfn new_name() {}");
        let inventory = crate::decide(&request, "test").unwrap();
        respond(&mut request, &inventory, r#"{"files":["Cargo.toml"]}"#);
        let test = crate::decide(&request, "test").unwrap();
        respond(&mut request, &test, "exit code: 101\nfailing test evidence");
        let failed = crate::decide(&request, "test").unwrap();
        assert_eq!(failed.execution.outcome, Outcome::ToolFailed);
        assert!(failed.call.is_none());
        assert!(failed.content.unwrap().contains("read_back_verified"));
    }
    #[test]
    fn followup_uses_correlated_evidence_and_reads_fresh_contents() {
        let mut request = request(json!([{"action":"search","query":"target"}]));
        let search = crate::decide(&request, "test").unwrap();
        respond(
            &mut request,
            &search,
            r#"{"matches":[{"path":"src/lib.rs"}]}"#,
        );
        request.messages.push(
            serde_json::from_value(json!({"role":"assistant","content":"Search completed"}))
                .unwrap(),
        );
        request.messages.push(
            serde_json::from_value(json!({"role":"user","content":"read that file"})).unwrap(),
        );
        let read = crate::decide(&request, "test").unwrap();
        assert_eq!(read.call.as_ref().unwrap().function.name, "read_file");
        request.messages.drain(0..request.messages.len() - 2);
        assert!(crate::decide(&request, "test").unwrap().call.is_none());
    }
    #[test]
    fn docs_and_reference_steps_use_offered_tools() {
        let mut request = request(
            json!([{"action":"docs","query":"routing"},{"action":"fetch","url":"https://example.com/reference"}]),
        );
        let docs = crate::decide(&request, "test").unwrap();
        let args: Value =
            serde_json::from_str(&docs.call.as_ref().unwrap().function.arguments).unwrap();
        assert_eq!(args["glob"], "**/*.md");
        respond(
            &mut request,
            &docs,
            r#"{"matches":[{"path":"README.md","excerpt":"routing evidence"}]}"#,
        );
        let fetch = crate::decide(&request, "test").unwrap();
        assert_eq!(fetch.call.as_ref().unwrap().function.name, "web_fetch");
        respond(&mut request, &fetch, "reference evidence");
        assert_eq!(
            crate::decide(&request, "test").unwrap().execution.outcome,
            Outcome::Completed
        );
    }
    #[test]
    fn rename_changes_rust_identifiers_and_preserves_comments_and_strings() {
        let source = "// target stays here\nfn target() {}\nfn caller() { target(); let text = \"target\"; }\n";
        let changed = rename(source, "src/lib.rs", "target", "renamed").unwrap();
        assert!(changed.contains("fn renamed()"));
        assert!(changed.contains("renamed();"));
        assert!(changed.contains("// target stays here"));
        assert!(changed.contains("\"target\""));
        assert!(
            rename(
                "fn target() {} fn caller(target:u8) {}",
                "lib.rs",
                "target",
                "next"
            )
            .is_err()
        );
        assert!(rename("fn target() {} fn next() {}", "lib.rs", "target", "next").is_err());
        assert!(
            rename(
                "fn target() {} fn caller() { macro_call!(target); }",
                "lib.rs",
                "target",
                "next"
            )
            .is_err()
        );
        assert!(
            rename(
                "mod a {fn target(){}} mod b {fn target(){}}",
                "lib.rs",
                "target",
                "next"
            )
            .is_err()
        );
    }
    #[test]
    fn rename_handles_raw_identifiers_and_unicode_source() {
        let source = "// café\nfn r#type() {}\nfn caller() { let text = \"é\"; r#type(); }";
        let output = rename(source, "lib.rs", "r#type", "next").unwrap();
        assert!(output.contains("fn next()"));
        assert!(output.contains("next();"));
        assert!(output.contains("\"é\""));
    }
    #[test]
    fn workflow_parsing_is_typed_and_rejects_extra_fields() {
        assert!(
            parse(r#"--workflow {"steps":[{"action":"read","path":"src/lib.rs"}]}"#)
                .unwrap()
                .is_ok()
        );
        assert!(
            parse(
                r#"--workflow {"steps":[{"action":"read","path":"src/lib.rs","command":"oops"}]}"#
            )
            .unwrap()
            .is_err()
        );
        assert!(parse("--formatting foo").is_none());
    }
}
