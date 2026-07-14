use std::collections::HashMap;
use std::sync::Arc;

use ai::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub type ToolInput = serde_json::Map<String, serde_json::Value>;

#[derive(Debug, Serialize, Deserialize)]
pub enum ToolOutcome {
    Success(serde_json::Value),
}

#[async_trait]
pub trait AgentTool: Send + Sync {
    fn schema(&self) -> &Tool;
    async fn invoke(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolOutcome, String>;
}

pub struct AgentToolRegistry {
    tools: HashMap<String, Arc<dyn AgentTool>>,
}

impl std::fmt::Debug for AgentToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AgentToolRegistry {{ tools: {:?} }}",
            self.tools.keys().collect::<Vec<&String>>()
        )
    }
}

impl AgentToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn AgentTool>) {
        self.tools.insert(tool.schema().name.clone(), tool);
    }

    /// Returns an owned handle so tool execution can be spawned off the
    /// actor's task without borrowing the registry.
    pub fn get(&self, tool_name: &str) -> Option<Arc<dyn AgentTool>> {
        self.tools.get(tool_name).cloned()
    }

    pub fn schemas(&self) -> Vec<Tool> {
        self.tools
            .values()
            .map(|tool| tool.schema().clone())
            .collect()
    }
}
