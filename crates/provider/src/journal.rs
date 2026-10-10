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
