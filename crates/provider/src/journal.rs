use crate::App;
use nioxs_contracts::{Outcome, digest};
use nioxs_nano::Decision;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn essential(data: &Value) -> Value {
    let mut value = data.clone();
    if let Some(object) = value.as_object_mut() {
        for key in ["created_at_unix", "expires_at_unix", "owner"] {
            object.remove(key);
        }
    }
    value
}
pub async fn record(app: &App, decision: &mut Decision) -> Result<(), String> {
    let turn = decision.execution.turn_id.clone();
    let mut actions = app.db.records("nano_actions", &turn).await?;
    let runs = app.db.records("nano_runs", &turn).await?;
    for (step, observed) in decision.observed.iter().enumerate() {
        let claim = actions
            .iter()
            .find(|r| {
                r["data"]["action_id"] == observed.call.id
                    && r["data"]["status"] == "dispatch_intent"
            })
            .ok_or("workflow action has no durable dispatch intent")?;
        if claim["data"]["arguments_digest"] != digest(observed.call.function.arguments.as_bytes())
            || claim["data"]["deployment_id"] != app.deployment
            || claim["data"]["model_version"] != app.model.model_version
            || claim["data"]["request_digest"] != decision.request_digest
        {
            return Err("workflow_revision_conflict".into());
        }
        let data = json!({"schema_version":2,"deployment_id":app.deployment,"turn_id":turn,"action_id":observed.call.id,"revision":2*step+2,"model_version":app.model.model_version,"tool":observed.call.function.name,"arguments_digest":digest(observed.call.function.arguments.as_bytes()),"request_digest":decision.request_digest,"status":observed.status,"result_digest":observed.result_digest});
        if let Some(existing) = actions.iter().find(|r| {
            r["data"]["action_id"] == observed.call.id && r["data"]["status"] != "dispatch_intent"
        }) {
            if essential(&existing["data"]) != data {
                return Err("workflow_revision_conflict".into());
            }
        } else {
            let mut stored = data;
            stored["created_at_unix"] = json!(now());
            stored["expires_at_unix"] = json!(now() + 7 * 86400);
            actions.push(
                app.db
                    .append_record(
                        "nano_actions",
                        &format!("{}:{turn}:result:{}", app.deployment, observed.call.id),
                        &stored,
                    )
                    .await?,
            );
        }
    }
    if let Some(call) = &decision.call {
        if let Some(existing) = actions
            .iter()
            .find(|r| r["data"]["action_id"] == call.id && r["data"]["status"] == "dispatch_intent")
        {
            if existing["data"]["arguments_digest"] != digest(call.function.arguments.as_bytes())
                || existing["data"]["request_digest"] != decision.request_digest
            {
                return Err("workflow_revision_conflict".into());
            }
            if call.function.name == "run_command" {
                let observed = actions.iter().find(|r| {
                    r["data"]["action_id"] == call.id && r["data"]["status"] != "dispatch_intent"
                });
                decision.content=Some(match observed{Some(result)=>format!("This command already has a recorded {} outcome. No command was repeated. Its output is absent from the supplied history.",result["data"]["status"].as_str().unwrap_or("unknown")),None=>"This command was previously dispatched, but its outcome is unknown. Inspect the project before retrying in a new turn; no command was repeated.".into()});
                decision.call = None;
                decision.execution.outcome = match observed {
                    Some(result) if result["data"]["status"] == "failed" => Outcome::ToolFailed,
                    Some(_) => Outcome::Completed,
                    None => Outcome::UnknownAction,
                };
                decision.execution.handoff = None;
            }
        } else {
            let data = json!({"schema_version":2,"deployment_id":app.deployment,"turn_id":turn,"action_id":call.id,"revision":2*decision.step+1,"model_version":app.model.model_version,"tool":call.function.name,"arguments_digest":digest(call.function.arguments.as_bytes()),"request_digest":decision.request_digest,"status":"dispatch_intent","created_at_unix":now(),"expires_at_unix":now()+7*86400,"owner":app.instance});
            app.db
                .append_record(
                    "nano_actions",
                    &format!("{}:{turn}:claim:{}", app.deployment, decision.step),
                    &data,
                )
                .await?;
        }
    }
    let state =
        serde_json::to_value(&decision.execution.outcome).map_err(|_| "cannot encode state")?;
    let data = json!({"schema_version":2,"deployment_id":app.deployment,"turn_id":turn,"revision":2*decision.step+1,"state":state,"intent":decision.intent,"model_version":app.model.model_version,"request_digest":decision.request_digest,"completed_actions":decision.observed.iter().map(|o|o.call.id.clone()).collect::<Vec<_>>()});
    if let Some(existing) = runs
        .iter()
        .find(|r| r["data"]["revision"] == data["revision"] && r["data"]["state"] == data["state"])
    {
        if essential(&existing["data"]) != data {
            eprintln!("CONFLICT in journal.rs: existing={:?} vs data={:?}", essential(&existing["data"]), data);
            return Err("workflow_revision_conflict".into());
        }
    } else {
        let mut stored = data;
        stored["created_at_unix"] = json!(now());
        stored["expires_at_unix"] = json!(now() + 7 * 86400);
        app.db
            .append_record(
                "nano_runs",
                &format!("{}:{turn}:run:{}:{state}", app.deployment, decision.step),
                &stored,
            )
            .await?;
    }
    Ok(())
}

/// A local session turn record persisted in `.nioxs/history.jsonl`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LocalTurnRecord {
    pub turn_id: String,
    pub timestamp_unix: u64,
    pub prompt: String,
    pub intent: String,
    pub outcome: String,
    pub summary: Option<String>,
}

/// Appends a turn record to `.nioxs/history.jsonl`.
pub fn append_local_turn(root: &std::path::Path, record: &LocalTurnRecord) -> Result<(), String> {
    let dir = root.join(".nioxs");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("history.jsonl");
    let line = serde_json::to_string(record)
        .map_err(|e| format!("cannot serialize local turn: {e}"))?;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("cannot open history.jsonl: {e}"))?;
    writeln!(file, "{line}").map_err(|e| format!("cannot write history.jsonl: {e}"))
}

/// Reads recent local turns from `.nioxs/history.jsonl`.
pub fn read_local_turns(root: &std::path::Path, limit: usize) -> Vec<LocalTurnRecord> {
    let path = root.join(".nioxs/history.jsonl");
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut turns = Vec::new();
    for line in content.lines() {
        if let Ok(rec) = serde_json::from_str::<LocalTurnRecord>(line.trim()) {
            turns.push(rec);
        }
    }
    if turns.len() > limit {
        turns.split_off(turns.len() - limit)
    } else {
        turns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_turn_journaling() {
        let temp_dir = std::env::temp_dir().join(format!("nioxs-hist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);

        let rec = LocalTurnRecord {
            turn_id: "turn-123".into(),
            timestamp_unix: 1700000000,
            prompt: "--code test".into(),
            intent: "coding".into(),
            outcome: "completed".into(),
            summary: Some("Tested coding turn".into()),
        };

        assert!(append_local_turn(&temp_dir, &rec).is_ok());
        let read = read_local_turns(&temp_dir, 10);
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].turn_id, "turn-123");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
