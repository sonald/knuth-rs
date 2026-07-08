use ai::Tool;
use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

pub type ToolInput = serde_json::Map<String, serde_json::Value>;

#[derive(Debug, Serialize, Deserialize)]
pub enum ToolOutcome {
    Success(serde_json::Value),
}

#[async_trait]
pub trait AgentTool {
    fn schema() -> Tool;
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutcome, String>;
}

pub struct BashTool {}

#[async_trait]
impl AgentTool for BashTool {
    fn schema() -> Tool {
        SCHEMA.clone()
    }

    async fn invoke(&self, input: ToolInput) -> Result<ToolOutcome, String> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("command is required")?;

        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .await
            .map_err(|e| e.to_string())?;
        Ok(ToolOutcome::Success(
            serde_json::json!({ "output": String::from_utf8(output.stdout).unwrap() }),
        ))
    }
}

static SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "bash".to_string(),
    description: "Execute a command in the shell".to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "The command to execute" }
        }
    }),
});
