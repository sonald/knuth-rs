use std::collections::HashMap;

use ai::Tool;
use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
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

pub struct BashTool {}

#[async_trait]
impl AgentTool for BashTool {
    fn schema(&self) -> &Tool {
        &BASH_SCHEMA
    }

    async fn invoke(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolOutcome, String> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("command is required")?;

        let mut cmd = Command::new("bash");
        let output = tokio::select! {
            _ = cancel_token.cancelled() => return Err("Command execution cancelled".to_string()),
            output = cmd.kill_on_drop(true).arg("-c").arg(command).output() => {
                output.map_err(|e| e.to_string())?
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        if output.status.success() {
            Ok(ToolOutcome::Success(
                serde_json::json!({ "output": stdout }),
            ))
        } else {
            Err(format!(
                "Command exited with {}.\nstdout:\n{}\nstderr:\n{}",
                output.status, stdout, stderr
            ))
        }
    }
}

static BASH_SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "bash".to_string(),
    description: "Execute a command in the shell".to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "The command to execute" }
        }
    }),
});

pub struct AgentToolRegistry {
    tools: HashMap<String, Box<dyn AgentTool>>,
}

impl std::fmt::Debug for AgentToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentToolRegistry {{ tools: {:?} }}", self.tools.keys().collect::<Vec<&String>>())
    }
}

impl AgentToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn AgentTool>) {  
        self.tools.insert(tool.schema().name.clone(), tool);
    }

    pub fn get(&self, tool_name: &str) -> Option<&dyn AgentTool> {
        self.tools.get(tool_name).map(Box::as_ref)
    }

    pub fn schemas(&self) -> Vec<Tool> {
        self.tools.values().map(|tool| tool.schema().clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(command: &str) -> ToolInput {
        let mut input = ToolInput::new();
        input.insert(
            "command".to_string(),
            serde_json::Value::String(command.to_string()),
        );
        input
    }

    #[tokio::test]
    async fn bash_tool_returns_stdout() {
        let outcome = BashTool {}
            .invoke(input("printf hello"), CancellationToken::new())
            .await
            .unwrap();

        match outcome {
            ToolOutcome::Success(value) => assert_eq!(value["output"], "hello"),
        }
    }

    #[tokio::test]
    async fn bash_tool_reports_exit_status_and_stderr() {
        let error = BashTool {}
            .invoke(input("printf nope >&2; exit 7"), CancellationToken::new())
            .await
            .unwrap_err();

        assert!(error.contains("exit status: 7"));
        assert!(error.contains("nope"));
    }

    #[tokio::test]
    async fn bash_tool_handles_non_utf8_output() {
        let outcome = BashTool {}
            .invoke(input("printf '\\377'"), CancellationToken::new())
            .await
            .unwrap();

        match outcome {
            ToolOutcome::Success(value) => assert_eq!(value["output"], "\u{fffd}"),
        }
    }
}
