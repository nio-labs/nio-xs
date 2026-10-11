//! Decision modeling: NioNano scores candidate proposals and picks one, or
//! refuses. Nothing here executes a proposal — the caller dispatches only after
//! checking `outcome == execute`.
use crate::{Intent, Model, parse};
use nioxs_contracts::{
    DecideRequest, Decision, DecisionOutcome, DecisionPolicy, Outcome, RankedCandidate,
    SCHEMA_VERSION,
};
use std::collections::BTreeSet;

/// Minimum score for a candidate to be executed rather than abstained on.
const EXECUTE_THRESHOLD: u32 = 50;
/// Two permitted candidates closer than this are treated as ambiguous.
const AMBIGUITY_MARGIN: u32 = 5;
/// Action name for a plain reply that needs no tool dispatch.
const ANSWER_ACTION: &str = "answer";

/// Intents that are answered directly rather than dispatched as a tool call.
fn is_answer_intent(intent: &Intent) -> bool {
    matches!(
        intent,
        Intent::Greeting
            | Intent::Help
            | Intent::Clarify(_)
            | Intent::Plan(_)
            | Intent::Bash(_)
            | Intent::Tool(_)
            | Intent::Coding { .. }
    )
}

/// Scores every proposal against the parsed intent and returns the decision.
pub fn decide(model: &Model, request: &DecideRequest) -> Decision {
    let intent = parse(model, request.prompt.trim());
    let expected = intent.label().to_string();
    let answer_only = is_answer_intent(&intent) || expected == "unsupported";

    let mut ranked = Vec::with_capacity(request.proposals.len());
    for proposal in &request.proposals {
        let missing_fields: Vec<String> = request
            .required_fields
            .iter()
            .filter(|field| proposal.arguments.get(field.as_str()).is_none())
            .cloned()
            .collect();
        let intent_match = proposal.action == expected;
        let is_answer = proposal.action == ANSWER_ACTION;
        let permitted = is_answer
            || request
                .allowed_actions
                .iter()
                .any(|a| a == &proposal.action);

        let mut score: u32 = 0;
        let mut reasons: Vec<String> = Vec::new();

        if intent_match {
            score += 40;
            reasons.push(format!("matches intent '{expected}'"));
        } else if is_answer && answer_only {
            score += 35;
            reasons.push("answers a request that needs no action".into());
        } else if is_answer {
            score += 10;
            reasons.push("answer does not perform the requested action".into());
        } else {
            reasons.push(format!("does not match intent '{expected}'"));
        }

        if permitted {
            score += 25;
        } else {
            reasons.push(format!(
                "action '{}' is not permitted this turn",
                proposal.action
            ));
        }

        if missing_fields.is_empty() {
            score += 20;
        } else {
            reasons.push(format!(
                "missing required field(s): {}",
                missing_fields.join(", ")
            ));
        }

        if !proposal.evidence.is_empty() {
            score += (proposal.evidence.len().min(5) as u32) * 2;
            reasons.push(format!("{} evidence reference(s)", proposal.evidence.len()));
        }

        let cost = proposal.cost.min(100) / 10;
        if cost > 0 {
            reasons.push(format!("cost penalty {cost}"));
        }
        score = score.saturating_sub(cost);

        if let Some(past) = request
            .past_outcomes
            .iter()
            .find(|p| p.action == proposal.action)
        {
            let total = past.successes + past.failures;
            if total > 0 {
                let rate = past.successes as f32 / total as f32;
                score += (rate * 5.0).round() as u32;
                score = score.saturating_sub(past.failures.min(5));
                reasons.push(format!(
                    "prior outcomes {}/{} successful",
                    past.successes, total
                ));
            }
        }

        ranked.push(RankedCandidate {
            id: proposal.id.clone(),
            action: proposal.action.clone(),
            score,
            intent_match,
            permitted,
            evidence_refs: proposal.evidence.len(),
            missing_fields,
            reason: reasons.join("; "),
        });
    }

    // Deterministic ordering: score, then fewest blockers, then id.
    ranked.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then(a.missing_fields.len().cmp(&b.missing_fields.len()))
            .then(a.id.cmp(&b.id))
    });

    let denied: Vec<String> = ranked
        .iter()
        .filter(|c| !c.permitted)
        .map(|c| c.action.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let policy = DecisionPolicy {
        allowed_actions: if request.allowed_actions.is_empty() {
            vec![ANSWER_ACTION.to_string()]
        } else {
            request.allowed_actions.clone()
        },
        denied,
    };

    let top = ranked.first().cloned();
    let (outcome, abstained_reason) = decide_outcome(&expected, &ranked, top.as_ref());

    let chosen = if outcome == DecisionOutcome::Execute {
        top.clone()
    } else {
        None
    };
    let evidence = chosen
        .as_ref()
        .and_then(|c| request.proposals.iter().find(|p| p.id == c.id))
        .map(|p| p.evidence.clone())
        .unwrap_or_default();
    let confidence = top
        .as_ref()
        .map_or(0.0, |c| (c.score.min(100) as f32) / 100.0);

    Decision {
        schema_version: SCHEMA_VERSION,
        outcome,
        chosen,
        ranked,
        confidence,
        evidence,
        policy,
        abstained_reason,
    }
}

fn decide_outcome(
    expected: &str,
    ranked: &[RankedCandidate],
    top: Option<&RankedCandidate>,
) -> (DecisionOutcome, Option<String>) {
    if expected == "unsupported" {
        return (
            DecisionOutcome::Handoff,
            Some("request is outside NioNano's bounded workflows".into()),
        );
    }
    if ranked.is_empty() {
        return (
            DecisionOutcome::Abstain,
            Some("no candidate proposals were supplied".into()),
        );
    }
    let Some(top) = top else {
        return (
            DecisionOutcome::Abstain,
            Some("no candidate proposals were supplied".into()),
        );
    };
    if !top.permitted {
        return (
            DecisionOutcome::Abstain,
            Some(format!(
                "policy blocks every candidate; '{}' is not permitted",
                top.action
            )),
        );
    }
    if !top.missing_fields.is_empty() {
        return (
            DecisionOutcome::Clarify,
            Some(format!(
                "missing required field(s): {}",
                top.missing_fields.join(", ")
            )),
        );
    }
    if top.score < EXECUTE_THRESHOLD {
        return (
            DecisionOutcome::Abstain,
            Some(format!(
                "best candidate '{}' scored {} below the {EXECUTE_THRESHOLD} threshold",
                top.id, top.score
            )),
        );
    }
    if let Some(second) = ranked.get(1)
        && second.permitted
        && second.action != top.action
        && top.score - second.score < AMBIGUITY_MARGIN
    {
        return (
            DecisionOutcome::Clarify,
            Some(format!(
                "ambiguous between '{}' and '{}'",
                top.id, second.id
            )),
        );
    }
    (DecisionOutcome::Execute, None)
}

/// Summarizes the decision a workflow turn already made, so completions carry
/// the same `nioxs.decision` shape as `/v1/decide` without re-scoring.
pub fn summarize(
    intent: &str,
    outcome: &Outcome,
    dispatch: Option<(&str, &str)>,
    evidence: Vec<String>,
    reason: Option<String>,
) -> Decision {
    let (decision_outcome, default_reason) = match outcome {
        Outcome::ToolRequired | Outcome::Completed => (DecisionOutcome::Execute, None),
        Outcome::Clarification => (DecisionOutcome::Clarify, None),
        Outcome::CapabilityLimitation | Outcome::UnknownAction => (
            DecisionOutcome::Handoff,
            Some("request is outside NioNano's bounded workflows".to_string()),
        ),
        Outcome::ToolFailed => (
            DecisionOutcome::Abstain,
            Some("the dispatched action failed".to_string()),
        ),
    };
    let executed = decision_outcome == DecisionOutcome::Execute;
    let (action, id) = match dispatch {
        Some((action, _)) => (action.to_string(), format!("turn:{action}")),
        None => (ANSWER_ACTION.to_string(), "turn:answer".to_string()),
    };
    let candidate = RankedCandidate {
        id,
        action: action.clone(),
        score: if executed { 100 } else { 0 },
        // An executed turn (dispatched action or direct answer) is by definition
        // the choice this intent resolved to.
        intent_match: executed || dispatch.is_some(),
        permitted: dispatch.is_some() || executed,
        evidence_refs: evidence.len(),
        missing_fields: Vec::new(),
        reason: if executed {
            format!("workflow selected '{action}' for intent '{intent}'")
        } else {
            format!("intent '{intent}' did not dispatch '{action}'")
        },
    };
    Decision {
        schema_version: SCHEMA_VERSION,
        outcome: decision_outcome,
        chosen: executed.then_some(candidate.clone()),
        ranked: vec![candidate],
        confidence: if executed { 1.0 } else { 0.0 },
        evidence,
        policy: DecisionPolicy {
            allowed_actions: vec![action],
            denied: Vec::new(),
        },
        abstained_reason: if executed {
            None
        } else {
            reason.or(default_reason)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;
    use nioxs_contracts::Proposal;
    use serde_json::json;

    fn model() -> Model {
        Model::embedded().unwrap()
    }

    fn proposal(id: &str, action: &str, args: serde_json::Value) -> Proposal {
        Proposal {
            id: id.into(),
            action: action.into(),
            arguments: args,
            evidence: Vec::new(),
            cost: 0,
        }
    }

    fn request(prompt: &str, proposals: Vec<Proposal>) -> DecideRequest {
        DecideRequest {
            schema_version: SCHEMA_VERSION,
            prompt: prompt.into(),
            proposals,
            allowed_actions: vec!["read_file".into()],
            required_fields: vec!["path".into()],
            past_outcomes: Vec::new(),
        }
    }

    #[test]
    fn matching_permitted_candidate_executes() {
        let request = request(
            "read README.md",
            vec![proposal("a", "read_file", json!({"path": "README.md"}))],
        );
        let decision = decide(&model(), &request);
        assert_eq!(decision.outcome, DecisionOutcome::Execute);
        assert_eq!(decision.chosen.unwrap().action, "read_file");
        assert!(decision.confidence >= 0.5);
        assert!(decision.abstained_reason.is_none());
    }

    #[test]
    fn missing_required_field_clarifies_without_executing() {
        let request = request("read it", vec![proposal("a", "read_file", json!({}))]);
        let decision = decide(&model(), &request);
        assert_eq!(decision.outcome, DecisionOutcome::Clarify);
        assert!(decision.chosen.is_none());
        assert!(decision.abstained_reason.unwrap().contains("path"));
    }

    #[test]
    fn unpermitted_action_abstains_on_policy() {
        let request = request(
            "run tests",
            vec![proposal(
                "a",
                "run_command",
                json!({"command": "cargo test"}),
            )],
        );
        let decision = decide(&model(), &request);
        assert_eq!(decision.outcome, DecisionOutcome::Abstain);
        assert!(decision.chosen.is_none());
        assert_eq!(decision.policy.denied, vec!["run_command".to_string()]);
    }

    #[test]
    fn no_proposals_abstains() {
        let decision = decide(&model(), &request("read README.md", vec![]));
        assert_eq!(decision.outcome, DecisionOutcome::Abstain);
        assert!(decision.ranked.is_empty());
    }

    #[test]
    fn unsupported_request_hands_off() {
        let request = replace_prompt(
            request("read README.md", vec![]),
            "invent a brand new compiler toolchain",
        );
        let decision = decide(&model(), &request);
        assert_eq!(decision.outcome, DecisionOutcome::Handoff);
    }

    fn replace_prompt(mut request: DecideRequest, prompt: &str) -> DecideRequest {
        request.prompt = prompt.into();
        request
    }

    #[test]
    fn scoring_prefers_matching_action_over_an_answer() {
        let request = request(
            "read README.md",
            vec![
                proposal("answer", "answer", json!({})),
                proposal("read", "read_file", json!({"path": "README.md"})),
            ],
        );
        let decision = decide(&model(), &request);
        assert_eq!(decision.outcome, DecisionOutcome::Execute);
        assert_eq!(decision.chosen.unwrap().id, "read");
        assert!(decision.ranked[0].score > decision.ranked[1].score);
    }

    #[test]
    fn past_failures_lower_the_score_relative_to_the_same_proposal() {
        let baseline_request = request(
            "read README.md",
            vec![proposal("read", "read_file", json!({"path": "README.md"}))],
        );
        let baseline = decide(&model(), &baseline_request).ranked[0].score;

        let mut penalized_request = baseline_request;
        penalized_request.past_outcomes = vec![nioxs_contracts::PastOutcome {
            action: "read_file".into(),
            successes: 0,
            failures: 9,
        }];
        let penalized = decide(&model(), &penalized_request).ranked[0].score;

        assert!(
            penalized < baseline,
            "prior failures should reduce the score: {penalized} vs {baseline}"
        );
    }

    #[test]
    fn turn_summary_reports_clarification_without_a_chosen_candidate() {
        let summary = summarize(
            "read_file",
            &Outcome::Clarification,
            None,
            Vec::new(),
            Some("Which file?".into()),
        );
        assert_eq!(summary.outcome, DecisionOutcome::Clarify);
        assert!(summary.chosen.is_none());
        assert_eq!(summary.abstained_reason.as_deref(), Some("Which file?"));
    }

    #[test]
    fn turn_summary_marks_an_executed_turn_as_intent_matched() {
        let summary = summarize("greeting", &Outcome::Completed, None, Vec::new(), None);
        assert_eq!(summary.outcome, DecisionOutcome::Execute);
        let chosen = summary
            .chosen
            .expect("an executed turn has a chosen candidate");
        assert_eq!(chosen.action, "answer");
        assert!(chosen.intent_match, "an executed answer matches its intent");
        assert!(chosen.permitted);
        assert_eq!(summary.confidence, 1.0);
    }
}
