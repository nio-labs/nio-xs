//! Structural validation for Manta's explicit XML-plan output. Does not execute commands.
use anyhow::{Context, Result, bail, ensure};
use roxmltree::{Document, Node, ParsingOptions};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub id: String,
    pub depends_on: Vec<String>,
    pub name: String,
    pub description: String,
    pub command: String,
    pub rollback: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ValidatedPlan {
    pub steps: Vec<PlanStep>,
}

fn check_container(node: Node<'_, '_>) -> Result<()> {
    ensure!(
        node.tag_name().namespace().is_none(),
        "plan elements must not use XML namespaces"
    );
    for child in node.children() {
        ensure!(
            !child.is_pi(),
            "processing instructions are not allowed inside a plan"
        );
        if child.is_text() {
            ensure!(
                child.text().unwrap_or("").trim().is_empty(),
                "unexpected text inside <{}>",
                node.tag_name().name()
            );
        }
    }
    Ok(())
}

/// Require one complete XML plan and a nonempty, acyclic graph of complete steps.
/// This checks structure and dependencies, not shell-command correctness or safety.
pub fn validate(xml: &str) -> Result<ValidatedPlan> {
    ensure!(
        xml.len() <= 1_048_576,
        "Manta plan exceeds the 1 MiB validation limit"
    );
    let document = Document::parse_with_options(
        xml,
        ParsingOptions {
            allow_dtd: false,
            nodes_limit: 20_000,
            ..Default::default()
        },
    )
    .context("Manta plan is not a complete XML document")?;
    let root = document.root_element();
    ensure!(
        root.tag_name().name() == "manta_plan",
        "expected a <manta_plan> root"
    );
    ensure!(
        root.attributes().len() == 0,
        "<manta_plan> must not have attributes"
    );
    check_container(root)?;
    let mut steps = Vec::new();
    let mut ids = HashSet::new();
    for node in root.children().filter(Node::is_element) {
        ensure!(steps.len() < 256, "Manta plan exceeds 256 steps");
        ensure!(
            node.tag_name().name() == "step",
            "unexpected <{}> in plan",
            node.tag_name().name()
        );
        check_container(node)?;
        for attr in node.attributes() {
            ensure!(
                attr.namespace().is_none() && matches!(attr.name(), "id" | "depends_on"),
                "unexpected step attribute {}",
                attr.name()
            );
        }
        let id = node
            .attribute("id")
            .context("step is missing its id")?
            .trim();
        ensure!(
            !id.is_empty() && !id.chars().any(char::is_whitespace),
            "step id must be nonempty and contain no whitespace"
        );
        ensure!(ids.insert(id.to_owned()), "duplicate step id {id:?}");
        let depends_on: Vec<String> = node
            .attribute("depends_on")
            .context("step is missing depends_on (use an empty string for a root step)")?
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        ensure!(
            depends_on.iter().collect::<HashSet<_>>().len() == depends_on.len(),
            "step {id:?} has duplicate dependencies"
        );
        let mut fields = HashMap::new();
        for field in node.children().filter(Node::is_element) {
            let name = field.tag_name().name();
            ensure!(
                field.tag_name().namespace().is_none()
                    && matches!(name, "name" | "description" | "command" | "rollback"),
                "unexpected step field {name:?}"
            );
            ensure!(
                field.attributes().len() == 0
                    && !field.children().any(|n| n.is_element() || n.is_pi()),
                "step fields must contain text only"
            );
            let value: String = field
                .children()
                .filter(Node::is_text)
                .filter_map(|n| n.text())
                .collect();
            ensure!(
                !value.trim().is_empty(),
                "step {id:?} has an empty <{name}>"
            );
            ensure!(
                fields.insert(name, value.trim().to_owned()).is_none(),
                "duplicate <{name}> in step {id:?}"
            );
        }
        let mut take = |name: &str| {
            fields
                .remove(name)
                .with_context(|| format!("step {id:?} is missing <{name}>"))
        };
        steps.push(PlanStep {
            id: id.to_owned(),
            depends_on,
            name: take("name")?,
            description: take("description")?,
            command: take("command")?,
            rollback: take("rollback")?,
        });
    }
    ensure!(
        !steps.is_empty(),
        "Manta plan must contain at least one step"
    );
    let indices: HashMap<_, _> = steps
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id.as_str(), i))
        .collect();
    let mut indegree: Vec<_> = steps.iter().map(|s| s.depends_on.len()).collect();
    let mut outgoing = vec![Vec::new(); steps.len()];
    for (index, step) in steps.iter().enumerate() {
        for dependency in &step.depends_on {
            ensure!(
                dependency != &step.id,
                "step {:?} depends on itself",
                step.id
            );
            let &parent = indices.get(dependency.as_str()).with_context(|| {
                format!(
                    "step {:?} refers to unknown dependency {dependency:?}",
                    step.id
                )
            })?;
            outgoing[parent].push(index);
        }
    }
    let mut ready: VecDeque<_> = indegree
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| (n == 0).then_some(i))
        .collect();
    let mut visited = 0;
    while let Some(index) = ready.pop_front() {
        visited += 1;
        for &child in &outgoing[index] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                ready.push_back(child);
            }
        }
    }
    if visited != steps.len() {
        bail!("Manta plan contains a dependency cycle");
    }
    Ok(ValidatedPlan { steps })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn step(id: &str, dependencies: &str) -> String {
        format!(
            "<step id=\"{id}\" depends_on=\"{dependencies}\"><name>Deploy</name><description>Deploy the API</description><command>echo staging &amp;&amp; echo ready</command><rollback>restore previous version</rollback></step>"
        )
    }
    fn plan(steps: &str) -> String {
        format!("<manta_plan>{steps}</manta_plan>")
    }

    #[test]
    fn accepts_complete_dag_and_decodes_xml_entities() {
        let result = validate(&plan(
            &(step("production", "staging") + &step("staging", "")),
        ))
        .unwrap();
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.steps[0].depends_on, ["staging"]);
        assert_eq!(result.steps[0].command, "echo staging && echo ready");
    }

    #[test]
    fn rejects_incomplete_xml_fences_prose_and_empty_plans() {
        for xml in [
            "<manta_plan><step",
            "```xml\n<manta_plan/>\n```",
            "Here is a plan: <manta_plan/>",
            "<manta_plan/>",
            "<other/>",
        ] {
            assert!(validate(xml).is_err(), "accepted {xml}");
        }
    }

    #[test]
    fn rejects_missing_empty_duplicate_and_nested_fields() {
        for xml in [
            plan(&step("a", "").replace("<rollback>restore previous version</rollback>", "")),
            plan(&step("a", "").replace("<name>Deploy</name>", "<name> </name>")),
            plan(&step("a", "").replace(
                "<name>Deploy</name>",
                "<name>Deploy</name><name>Again</name>",
            )),
            plan(&step("a", "").replace("<name>Deploy</name>", "<name><b>Deploy</b></name>")),
            plan(&step("a", "").replace(" depends_on=\"\"", "")),
        ] {
            assert!(validate(&xml).is_err(), "accepted {xml}");
        }
    }

    #[test]
    fn rejects_bad_graphs() {
        for steps in [
            step("a", "") + &step("a", ""),
            step("a", "missing"),
            step("a", "a"),
            step("a", "b") + &step("b", "a"),
            step("a", "b b") + &step("b", ""),
        ] {
            assert!(validate(&plan(&steps)).is_err(), "accepted {steps}");
        }
    }

    #[test]
    fn rejects_unknown_markup_namespaces_and_dtd_entities() {
        for xml in [
            plan(&(step("a", "") + "<unexpected/>")),
            plan(&step("a", "").replace("id=\"a\"", "id=\"a\" extra=\"x\"")),
            "<manta_plan xmlns=\"urn:other\"/>".into(),
            "<!DOCTYPE manta_plan [<!ENTITY x 'value'>]><manta_plan>&x;</manta_plan>".into(),
        ] {
            assert!(validate(&xml).is_err(), "accepted {xml}");
        }
    }
}
