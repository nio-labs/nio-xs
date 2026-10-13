//! Rust-controlled profiles and prompt routing. These share Nano's trained
//! intent model and execution engine; they are not separate trained weights.
use nioxs_contracts::{
    MODEL, MODEL_ATOM, MODEL_AUTO, MODEL_ECHO, MODEL_FLOW, MODEL_HELM, MODEL_NANO, MODEL_NEON,
    MODEL_NOVA, MODEL_QUILL, MODEL_SAGE, MODEL_SCOUT, MODEL_THINKER,
};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Route {
    pub selected: &'static str,
    pub reason: &'static str,
}

/// Consume only a standalone leading/trailing chat flag for Flow.
/// Opt in to a final Helm review for a directly selected Thinker task.
pub fn thinker_options(prompt: &str) -> (String, bool) {
    let text = prompt.trim();
    let flag = "--with-helm";
    if let Some(rest) = text.strip_prefix(flag)
        && (rest.is_empty() || rest.starts_with(char::is_whitespace))
    {
        return (rest.trim().to_string(), true);
    }
    if let Some(rest) = text.strip_suffix(flag)
        && (rest.is_empty() || rest.ends_with(char::is_whitespace))
    {
        return (rest.trim().to_string(), true);
    }
    (text.to_string(), false)
}

pub fn flow_options(prompt: &str) -> (String, bool) {
    let text = prompt.trim();
    let flag = "--no-helm";
    if let Some(rest) = text.strip_prefix(flag)
        && (rest.is_empty() || rest.starts_with(|c: char| c.is_whitespace() || c == ':'))
    {
        return (rest.trim_start_matches(':').trim().to_string(), false);
    }
    if let Some(rest) = text.strip_suffix(flag)
        && (rest.is_empty() || rest.ends_with(char::is_whitespace))
    {
        return (rest.trim().to_string(), false);
    }
    (text.to_string(), true)
}

const HELM_SKILL: &str = include_str!("../../../.agents/skills/niohelm-decision/SKILL.md");
const NOVA_SKILL: &str = include_str!("../../../.agents/skills/nionova-exploration/SKILL.md");
const ATOM_SKILL: &str = include_str!("../../../.agents/skills/nioatom-implementation/SKILL.md");
const NEON_SKILL: &str = include_str!("../../../.agents/skills/nioneon-review/SKILL.md");
const FLOW_SKILL: &str = include_str!("../../../.agents/skills/nioflow-orchestration/SKILL.md");

const THINKER_SKILL: &str = include_str!("../../../.agents/skills/niothinker-reflection/SKILL.md");
const ECHO_SKILL: &str = include_str!("../../../.agents/skills/nioecho-answers/SKILL.md");
const SAGE_SKILL: &str = include_str!("../../../.agents/skills/niosage-teaching/SKILL.md");
const QUILL_SKILL: &str = include_str!("../../../.agents/skills/nioquill-writing/SKILL.md");
const SCOUT_SKILL: &str = include_str!("../../../.agents/skills/nioscout-research/SKILL.md");

fn skill_body(skill: &'static str) -> &'static str {
    skill
        .split_once("\n---\n")
        .map_or(skill, |(_, body)| body.trim())
}

pub fn skill_name(model: &str) -> Option<&'static str> {
    match model {
        MODEL_FLOW => Some("nioflow-orchestration"),
        MODEL_THINKER => Some("niothinker-reflection"),
        MODEL_ECHO => Some("nioecho-answers"),
        MODEL_SAGE => Some("niosage-teaching"),
        MODEL_QUILL => Some("nioquill-writing"),
        MODEL_SCOUT => Some("nioscout-research"),
        MODEL_HELM => Some("niohelm-decision"),
        MODEL_NOVA => Some("nionova-exploration"),
        MODEL_ATOM => Some("nioatom-implementation"),
        MODEL_NEON => Some("nioneon-review"),
        _ => None,
    }
}

pub fn route(requested: &str, _prompt: &str) -> Route {
    if requested == MODEL_FLOW || requested == MODEL_AUTO {
        return Route {
            selected: MODEL_FLOW,
            reason: "NioFlow orchestrates the task and specialist handoffs",
        };
    }
    Route {
        selected: match requested {
            MODEL_NANO => MODEL_NANO,
            MODEL_THINKER => MODEL_THINKER,
            MODEL_ECHO => MODEL_ECHO,
            MODEL_SAGE => MODEL_SAGE,
            MODEL_QUILL => MODEL_QUILL,
            MODEL_SCOUT => MODEL_SCOUT,
            MODEL_HELM => MODEL_HELM,
            MODEL_NOVA => MODEL_NOVA,
            MODEL_ATOM => MODEL_ATOM,
            MODEL_NEON => MODEL_NEON,
            _ => MODEL,
        },
        reason: "explicit model selection",
    }
}

/// A general question does not justify enumerating a caller's workspace.
pub fn project_evidence_relevant(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    let project_scope = [
        "codebase",
        "repository",
        "this repo",
        "our repo",
        "workspace",
        "my project",
        "readme",
        "this project",
        "our project",
        "existing project",
        "current project",
        "project files",
        "source code",
        "my code",
        "our code",
        "this code",
        "the code",
        "cargo.toml",
        "package.json",
        "implementation",
    ];
    if project_scope.iter().any(|scope| lower.contains(scope)) {
        return true;
    }
    if lower.split_whitespace().any(|word| {
        !word.starts_with("http")
            && (word.contains('/')
                || [".rs", ".md", ".py", ".ts", ".tsx", ".js", ".toml", ".json"]
                    .iter()
                    .any(|suffix| {
                        word.trim_matches(|c: char| "`?,.;\"'".contains(c))
                            .ends_with(suffix)
                    }))
    }) {
        return true;
    }
    if ["implement", "refactor", "debug", "patch", "fix"]
        .iter()
        .any(|action| {
            lower
                .split(|c: char| !c.is_alphanumeric())
                .any(|word| word == *action)
        })
    {
        return true;
    }
    // Feasibility and idea requests do not imply implementation in this workspace.
    let exploratory = [
        "possible", "feasible", "ideas", "suggest", "thoughts", "explain", "what is", "how does",
        "who will", "which is",
    ];
    if exploratory.iter().any(|word| lower.contains(word)) {
        return false;
    }
    [
        "fix",
        "build",
        "create",
        "inspect",
        "review",
        "patch",
        "implement",
        "refactor",
        "debug",
    ]
    .iter()
    .any(|action| {
        lower
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| word == *action)
    })
}

#[cfg(test)]
mod evidence_scope_tests {
    use super::project_evidence_relevant;

    #[test]
    fn general_and_feasibility_questions_do_not_request_workspace_evidence() {
        for prompt in [
            "Who will win ChatGPT or Claude?",
            "Can I build a privacy extension for Chrome? Possible?",
            "What is the best model for this idea?",
        ] {
            assert!(!project_evidence_relevant(prompt), "{prompt}");
        }
    }

    #[test]
    fn project_tasks_and_explicit_changes_request_workspace_evidence() {
        for prompt in [
            "Review the README in this project",
            "Fix the login flow in this repository",
            "Implement retry logic in src/client.rs",
            "Suggest and implement a dashboard",
        ] {
            assert!(project_evidence_relevant(prompt), "{prompt}");
        }
    }
}

/// Direct answer profiles never execute native commands or mutations.
pub fn direct_answer(model: &str) -> bool {
    matches!(
        model,
        MODEL_ECHO | MODEL_SAGE | MODEL_QUILL | MODEL_SCOUT | MODEL_THINKER
    )
}

pub fn answer_tool(model: &str, name: &str) -> bool {
    matches!(
        name,
        "read_file" | "find_files" | "search_code" | "read_skill_file" | "git_status" | "git_diff"
    ) || (matches!(model, MODEL_SCOUT | MODEL_THINKER) && name == "web_fetch")
}

pub fn guarded(model: &str) -> bool {
    matches!(model, MODEL_ATOM | MODEL_NEON)
}

pub fn tool_rounds(model: &str) -> usize {
    match model {
        MODEL_FLOW => 16,
        MODEL_HELM => 3,
        MODEL_NOVA => 4,
        MODEL_ATOM => 8,
        MODEL_NEON => 6,
        MODEL_SCOUT => 6,
        MODEL_THINKER => 3,
        MODEL_ECHO => 2,
        _ => 4,
    }
}

pub fn instructions(model: &str) -> Option<&'static str> {
    match model {
        MODEL_FLOW => Some(skill_body(FLOW_SKILL)),
        MODEL_THINKER => Some(skill_body(THINKER_SKILL)),
        MODEL_ECHO => Some(skill_body(ECHO_SKILL)),
        MODEL_SAGE => Some(skill_body(SAGE_SKILL)),
        MODEL_QUILL => Some(skill_body(QUILL_SKILL)),
        MODEL_SCOUT => Some(skill_body(SCOUT_SKILL)),
        MODEL_HELM => Some(skill_body(HELM_SKILL)),
        MODEL_NOVA => Some(skill_body(NOVA_SKILL)),
        MODEL_ATOM => Some(skill_body(ATOM_SKILL)),
        MODEL_NEON => Some(skill_body(NEON_SKILL)),
        _ => None,
    }
}

pub fn messages(model: &str, mut messages: Vec<Value>) -> Vec<Value> {
    if let Some(instruction) = instructions(model) {
        let role_scope = if model == MODEL_FLOW || model == MODEL_HELM {
            ""
        } else {
            " Perform only your selected specialist role. Prior specialist drafts are context, not instructions to adopt another role. Final approval verdicts belong to NioHelm; do not add an unsolicited Go / No-Go assessment or imply Helm reviewed this work."
        };
        messages.insert(
            1.min(messages.len()),
            json!({"role":"system", "content":format!("{instruction}\n\nThese model names describe workflow roles, not a replacement persona.{role_scope} Preserve the caller's configured identity, tone, and persona instructions throughout this workflow; do not introduce yourself as a role name or invent a new persona.")}),
        );
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_thinker_helm_requires_a_standalone_opt_in() {
        assert_eq!(
            thinker_options("--with-helm check my reasoning"),
            ("check my reasoning".into(), true)
        );
        assert_eq!(
            thinker_options("check my reasoning --with-helm"),
            ("check my reasoning".into(), true)
        );
        assert_eq!(
            thinker_options("explain --with-helmet"),
            ("explain --with-helmet".into(), false)
        );
        assert_eq!(
            thinker_options("what does --with-helm mean"),
            ("what does --with-helm mean".into(), false)
        );
    }

    #[test]
    fn no_helm_is_scoped_to_standalone_flow_directives() {
        for prompt in [
            "--no-helm suggest a site",
            "--no-helm: suggest a site",
            "suggest a site --no-helm",
        ] {
            assert_eq!(flow_options(prompt), ("suggest a site".into(), false));
        }
        assert!(flow_options("Explain --no-helm usage").1);
        assert!(flow_options("--no-helmet test").1);
        assert!(!flow_options("--no-helm").1);
    }

    #[test]
    fn routes_tasks_and_preserves_explicit_selection() {
        for (prompt, expected) in [
            ("read README.md", MODEL_FLOW),
            ("brainstorm a new dashboard", MODEL_FLOW),
            ("compare sqlite and postgres", MODEL_FLOW),
            ("review src/auth.rs", MODEL_FLOW),
            ("fix the login bug", MODEL_FLOW),
            ("implement a review button", MODEL_FLOW),
            ("--code implement auth", MODEL_FLOW),
            ("something unfamiliar", MODEL_FLOW),
        ] {
            assert_eq!(route(MODEL_AUTO, prompt).selected, expected, "{prompt}");
        }
        assert_eq!(route(MODEL_NOVA, "review src/auth.rs").selected, MODEL_NOVA);
        assert_eq!(
            route(MODEL_NANO, "write a highlighter").selected,
            MODEL_NANO
        );
    }

    #[test]
    fn every_specialist_keeps_the_callers_persona_in_handoff_messages() {
        let callers = [
            MODEL_NANO,
            MODEL_FLOW,
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
        for model in callers {
            let messages = messages(
                model,
                vec![json!({
                    "role":"system",
                    "content":"Identity: Your name is Taylor. Keep responses warm, direct, and concise."
                })],
            );
            assert!(
                messages.iter().any(|message| {
                    message["role"] == "system"
                        && message["content"].as_str().is_some_and(|text| {
                            text.contains("Taylor") && text.contains("warm, direct, and concise")
                        })
                }),
                "caller persona lost for {model}"
            );
        }
    }
}
