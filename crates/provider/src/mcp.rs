use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

pub fn handle_rpc(line: &str) -> Option<String> {
    let req: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(_) => {
            let res = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: None,
                result: None,
                error: Some(json!({
                    "code": -32700,
                    "message": "Parse error"
                })),
            };
            return serde_json::to_string(&res).ok();
        }
    };

    let id = req.id;
    match req.method.as_str() {
        "initialize" => {
            let res = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "nioxs-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                })),
                error: None,
            };
            serde_json::to_string(&res).ok()
        }
        "notifications/initialized" => None,
        "tools/list" => {
            let mut tools = vec![
                json!({
                    "name": "web_fetch",
                    "description": "Mako Web Scout: Fetch live web URL and strip 95-99% of DOM noise (scripts, ads, nav) to reduce LLM context token usage. Optional 'focus' pinpoints exact symbols or errors.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": { "type": "string", "description": "HTTP or HTTPS URL to fetch and distill" },
                            "focus": { "type": "string", "description": "Optional keyword, struct, function, or error message to surgically isolate" }
                        },
                        "required": ["url"]
                    }
                }),
                json!({
                    "name": "mako_search",
                    "description": "Mako Web Scout: Keyless live web search returning clean documentation URLs and summaries",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Search query or operator syntax" }
                        },
                        "required": ["query"]
                    }
                }),
                json!({
                    "name": "wiki_search",
                    "description": "Search Wikipedia articles and extract article excerpts",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"]
                    }
                }),
                json!({
                    "name": "github_repo",
                    "description": "Fetch GitHub repository README or overview",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "repo": { "type": "string" }
                        },
                        "required": ["repo"]
                    }
                }),
                json!({
                    "name": "package_info",
                    "description": "Inspect crates.io and npm package metadata",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" }
                        },
                        "required": ["name"]
                    }
                }),
                json!({
                    "name": "apply_patch",
                    "description": "Apply a deterministic line replacement patch",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "target": { "type": "string" },
                            "replacement": { "type": "string" }
                        },
                        "required": ["content", "target", "replacement"]
                    }
                }),
                json!({
                    "name": "scaffold_template",
                    "description": "Scaffold standard project templates (gitignore, ci, license)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "template": { "type": "string" }
                        },
                        "required": ["template"]
                    }
                }),
                json!({
                    "name": "ast_symbol_search",
                    "description": "Search workspace symbols (fn, struct, type, class, etc.) across Rust, TypeScript, Python, and Go",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"]
                    }
                }),
                json!({
                    "name": "companion_tool",
                    "description": "Execute any of the 20 local Python companion plugins (svg2favicon, jwt_decoder, regex_tester, json_yaml, etc.)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task": { "type": "string" }
                        },
                        "required": ["task"]
                    }
                }),
            ];

            // Expose each companion plugin as an individual MCP tool
            for plugin in nioxs_nano::companion::TOOL_PLUGINS {
                tools.push(json!({
                    "name": format!("plugin_{}", plugin.name),
                    "description": format!("{}: {}", plugin.title, plugin.capability),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "arguments": { "type": "string", "description": format!("Task description or CLI flags. Example: {}", plugin.example) }
                        },
                        "required": ["arguments"]
                    }
                }));
            }

            let res = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({ "tools": tools })),
                error: None,
            };
            serde_json::to_string(&res).ok()
        }
        "prompts/list" => {
            let prompts = json!([
                {
                    "name": "code_review",
                    "description": "Audit code for edge cases, performance, unwrap panics, and security flaws"
                },
                {
                    "name": "architect_design",
                    "description": "Analyze system boundaries, contracts, modularity, and trade-offs"
                },
                {
                    "name": "self_heal_diagnostics",
                    "description": "Analyze compiler error diagnostics and formulate step-by-step remediation"
                }
            ]);
            let res = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({ "prompts": prompts })),
                error: None,
            };
            serde_json::to_string(&res).ok()
        }
        "tools/call" => {
            let params = req.params.unwrap_or(json!({}));
            let tool_name = params["name"].as_str().unwrap_or("");
            let args = &params["arguments"];

            let output_text = if let Some(plugin_name) = tool_name.strip_prefix("plugin_") {
                let task_arg = args["arguments"].as_str().unwrap_or("");
                let full_task = format!("{plugin_name} {task_arg}");
                nioxs_nano::companion::dispatch_tool(&full_task).to_markdown()
            } else {
                match tool_name {
                    "web_fetch" => {
                        let url = args["url"].as_str().unwrap_or("");
                        if url.is_empty() {
                            "Error: 'url' parameter is required for web_fetch".to_string()
                        } else {
                            let fetch_res = match tokio::runtime::Handle::try_current() {
                                Ok(handle) => tokio::task::block_in_place(|| {
                                    handle.block_on(nioxs_browser::fetch_url(url))
                                }),
                                Err(_) => {
                                    tokio::runtime::Runtime::new()
                                        .map_err(|e| e.to_string())
                                        .and_then(|rt| rt.block_on(nioxs_browser::fetch_url(url)))
                                }
                            };
                            match fetch_res {
                                Ok(raw_html) => {
                                    let raw_len = raw_html.len();
                                    let clean_text = nioxs_browser::extract_text(&raw_html);
                                    let focus = args["focus"].as_str().filter(|f| !f.trim().is_empty());
                                    
                                    let (distilled_text, focus_note) = if let Some(target) = focus {
                                        let lower_target = target.to_lowercase();
                                        let paragraphs: Vec<&str> = clean_text.split("\n\n").collect();
                                        let matching_paras: Vec<&str> = paragraphs
                                            .iter()
                                            .copied()
                                            .filter(|p| p.to_lowercase().contains(&lower_target))
                                            .collect();
                                        if !matching_paras.is_empty() {
                                            (
                                                matching_paras.join("\n\n"),
                                                format!(" *(Focused on `{target}`)*"),
                                            )
                                        } else {
                                            (
                                                clean_text.chars().take(3000).collect(),
                                                format!(" *(Focus `{target}` not matched, showing top excerpt)*"),
                                            )
                                        }
                                    } else {
                                        (clean_text, String::new())
                                    };

                                    let clean_len = distilled_text.len();
                                    let saved_pct = if raw_len > 0 {
                                        (1.0 - (clean_len as f64 / raw_len as f64)) * 100.0
                                    } else {
                                        0.0
                                    };
                                    let title = nioxs_browser::extract_title(&raw_html)
                                        .unwrap_or_else(|| "Web Page".into());
                                    format!(
                                        "### 🦈 Mako Web Scout: {title}{focus_note}\n**URL:** {url}\n**Token Optimization:** {saved_pct:.1}% noise eliminated ({raw_len} bytes HTML → {clean_len} bytes clean payload)\n\n---\n\n{distilled_text}"
                                    )
                                }
                                Err(e) => format!("Error fetching URL '{url}': {e}"),
                            }
                        }
                    }
                    "mako_search" => {
                        let q = args["query"].as_str().unwrap_or("");
                        if q.is_empty() {
                            "Error: 'query' parameter is required for mako_search".to_string()
                        } else {
                            let search_res = match tokio::runtime::Handle::try_current() {
                                Ok(handle) => tokio::task::block_in_place(|| {
                                    handle.block_on(nioxs_browser::search(q, 5))
                                }),
                                Err(_) => {
                                    tokio::runtime::Runtime::new()
                                        .map_err(|e| e.to_string())
                                        .and_then(|rt| rt.block_on(nioxs_browser::search(q, 5)))
                                }
                            };
                            match search_res {
                                Ok(results) => {
                                    if results.is_empty() {
                                        format!("No search results found for '{q}'.")
                                    } else {
                                        let mut out = format!("### 🦈 Mako Search Results for: `{q}`\n\n");
                                        for (idx, r) in results.iter().enumerate() {
                                            out.push_str(&format!("{}. [{}]({})\n   {}\n\n", idx + 1, r.title, r.url, r.snippet));
                                        }
                                        out
                                    }
                                }
                                Err(e) => format!("Search error: {e}"),
                            }
                        }
                    }
                    "wiki_search" => {
                        let q = args["query"].as_str().unwrap_or("");
                        let res = match tokio::runtime::Handle::try_current() {
                            Ok(handle) => tokio::task::block_in_place(|| {
                                handle.block_on(nioxs_browser::search_wikipedia(q, 3))
                            }),
                            Err(_) => {
                                tokio::runtime::Runtime::new()
                                    .map_err(|e| e.to_string())
                                    .and_then(|rt| rt.block_on(nioxs_browser::search_wikipedia(q, 3)))
                            }
                        };
                        match res {
                            Ok(results) => {
                                let mut out = format!("### Wikipedia Results for: `{q}`\n\n");
                                for r in results {
                                    out.push_str(&format!("• [{}]({}): {}\n", r.title, r.url, r.snippet));
                                }
                                out
                            }
                            Err(e) => format!("Wikipedia error: {e}"),
                        }
                    }
                    "companion_tool" => {
                        let task = args["task"].as_str().unwrap_or("");
                        nioxs_nano::companion::dispatch_tool(task).to_markdown()
                    }
                    "ast_symbol_search" => {
                        let q = args["query"].as_str().unwrap_or("");
                        let ws_index =
                            nioxs_nano::brain::scan_workspace_index(std::path::Path::new("."));
                        let files = ws_index.find_files_for_symbol(q);
                        if files.is_empty() {
                            format!("No symbols found matching query '{q}'.")
                        } else {
                            format!(
                                "Found symbol '{q}' in:\n{}",
                                files
                                    .iter()
                                    .map(|f| format!("• `{f}`"))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            )
                        }
                    }
                    "scaffold_template" => {
                        let tpl = args["template"].as_str().unwrap_or("");
                        match nioxs_nano::scaffold_template(tpl) {
                            Some((name, content)) => format!("Scaffolded {name}:\n\n{content}"),
                            None => format!("Unknown template '{tpl}'"),
                        }
                    }
                    "apply_patch" => {
                        let content = args["content"].as_str().unwrap_or("");
                        let target = args["target"].as_str().unwrap_or("");
                        let replacement = args["replacement"].as_str().unwrap_or("");
                        match nioxs_nano::apply_patch_to_string(content, target, replacement, false)
                        {
                            Ok((patched, count)) => {
                                format!("Successfully patched {count} occurrence(s):\n\n{patched}")
                            }
                            Err(e) => format!("Patch error: {e}"),
                        }
                    }
                    _ => format!("Tool '{tool_name}' dispatched via NioXS MCP server"),
                }
            };

            let res = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({
                    "content": [
                        {
                            "type": "text",
                            "text": output_text
                        }
                    ]
                })),
                error: None,
            };
            serde_json::to_string(&res).ok()
        }
        _ => {
            let res = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(json!({
                    "code": -32601,
                    "message": "Method not found"
                })),
            };
            serde_json::to_string(&res).ok()
        }
    }
}

pub fn run_stdio_server() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = handle_rpc(&line) {
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_rpc() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let res_str = handle_rpc(req).unwrap();
        let res: Value = serde_json::from_str(&res_str).unwrap();
        assert_eq!(res["id"], 1);
        assert_eq!(res["result"]["serverInfo"]["name"], "nioxs-mcp");
    }

    #[test]
    fn test_tools_list_rpc() {
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
        let res_str = handle_rpc(req).unwrap();
        let res: Value = serde_json::from_str(&res_str).unwrap();
        assert!(res["result"]["tools"].as_array().unwrap().len() >= 5);
    }

    #[test]
    fn test_scaffold_tool_call() {
        let req = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"scaffold_template","arguments":{"template":"gitignore"}}}"#;
        let res_str = handle_rpc(req).unwrap();
        let res: Value = serde_json::from_str(&res_str).unwrap();
        let text = res["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains(".gitignore"));
    }

    /// Acceptance gate: the handshake advertises the protocol, tool capability,
    /// and a non-empty server version.
    #[test]
    fn test_initialize_advertises_protocol_and_capabilities() {
        let res: Value = serde_json::from_str(
            &handle_rpc(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(res["jsonrpc"], "2.0");
        assert_eq!(res["result"]["protocolVersion"], "2024-11-05");
        assert!(res["result"]["capabilities"]["tools"].is_object());
        assert!(
            !res["result"]["serverInfo"]["version"]
                .as_str()
                .unwrap_or_default()
                .is_empty(),
            "server version must be reported"
        );
    }

    /// A JSON-RPC notification must not produce a response.
    #[test]
    fn test_notifications_are_silent() {
        assert!(handle_rpc(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
    }

    /// The advertised tool list must contain every core tool and one entry per
    /// shipped companion plugin (so the MCP surface matches what Nano exposes).
    #[test]
    fn test_tools_list_exposes_core_tools_and_plugins() {
        let res: Value = serde_json::from_str(
            &handle_rpc(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#).unwrap(),
        )
        .unwrap();
        let names: Vec<&str> = res["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        for expected in [
            "web_fetch",
            "mako_search",
            "wiki_search",
            "github_repo",
            "package_info",
            "apply_patch",
            "scaffold_template",
            "ast_symbol_search",
            "companion_tool",
        ] {
            assert!(
                names.contains(&expected),
                "tools/list must expose '{expected}': {names:?}"
            );
        }
        for plugin in nioxs_nano::companion::TOOL_PLUGINS {
            let expected = format!("plugin_{}", plugin.name);
            assert!(
                names.contains(&expected.as_str()),
                "tools/list must expose '{expected}': {names:?}"
            );
        }
    }

    #[test]
    fn test_prompts_list_is_advertised() {
        let res: Value = serde_json::from_str(
            &handle_rpc(r#"{"jsonrpc":"2.0","id":3,"method":"prompts/list"}"#).unwrap(),
        )
        .unwrap();
        let prompts = res["result"]["prompts"].as_array().unwrap();
        assert!(prompts.iter().any(|p| p["name"] == "code_review"));
    }

    /// A tools/call must return a well-formed JSON-RPC result carrying text.
    #[test]
    fn test_apply_patch_tool_call_returns_text_result() {
        let req = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"apply_patch","arguments":{"content":"a\nb\n","target":"b","replacement":"c"}}}"#;
        let res: Value = serde_json::from_str(&handle_rpc(req).unwrap()).unwrap();
        assert_eq!(res["jsonrpc"], "2.0");
        let text = res["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("patched"), "unexpected result: {text}");
        assert!(text.contains('c'));
    }

    #[test]
    fn test_web_fetch_missing_url_returns_error() {
        let req = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"web_fetch","arguments":{}}}"#;
        let res: Value = serde_json::from_str(&handle_rpc(req).unwrap()).unwrap();
        assert_eq!(res["jsonrpc"], "2.0");
        let text = res["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("required"), "unexpected result: {text}");
    }

    #[test]
    fn test_unknown_method_returns_method_not_found() {
        let res: Value = serde_json::from_str(
            &handle_rpc(r#"{"jsonrpc":"2.0","id":5,"method":"does/not-exist"}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(res["error"]["code"], -32601);
    }

    #[test]
    fn test_malformed_json_returns_parse_error() {
        let res: Value = serde_json::from_str(&handle_rpc("not json").unwrap()).unwrap();
        assert_eq!(res["error"]["code"], -32700);
    }
}
