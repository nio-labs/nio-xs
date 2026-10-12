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
                    "description": "Fetch and extract clean readable text from a web URL",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": { "type": "string" }
                        },
                        "required": ["url"]
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
                })
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
                    "companion_tool" => {
                        let task = args["task"].as_str().unwrap_or("");
                        nioxs_nano::companion::dispatch_tool(task).to_markdown()
                    }
                    "ast_symbol_search" => {
                        let q = args["query"].as_str().unwrap_or("");
                        let ws_index = nioxs_nano::brain::scan_workspace_index(std::path::Path::new("."));
                        let files = ws_index.find_files_for_symbol(q);
                        if files.is_empty() {
                            format!("No symbols found matching query '{q}'.")
                        } else {
                            format!("Found symbol '{q}' in:\n{}", files.iter().map(|f| format!("• `{f}`")).collect::<Vec<_>>().join("\n"))
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
                        match nioxs_nano::apply_patch_to_string(content, target, replacement, false) {
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
}
