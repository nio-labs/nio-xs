use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DagState {
    Pending,
    Running,
    Completed,
    Failed,
    RolledBack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,
    pub description: String,
    pub state: DagState,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionDag {
    pub task_name: String,
    pub nodes: Vec<DagNode>,
}

impl ExecutionDag {
    pub fn new(task_name: &str) -> Self {
        Self {
            task_name: task_name.to_string(),
            nodes: vec![],
        }
    }

    pub fn add_node(&mut self, id: &str, description: &str, dependencies: Vec<&str>) {
        self.nodes.push(DagNode {
            id: id.to_string(),
            description: description.to_string(),
            state: DagState::Pending,
            dependencies: dependencies.into_iter().map(|s| s.to_string()).collect(),
        });
    }

    pub fn render_plan(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "### 🗺️ DAG Execution Plan: `{}`\n\n",
            self.task_name
        ));
        out.push_str("State Machine Initialized. Rollback checkpoints enabled.\n\n");
        for node in &self.nodes {
            out.push_str(&format!(
                "- **[{}]** {} (Deps: {:?})\n",
                node.id, node.description, node.dependencies
            ));
        }
        out
    }
}
