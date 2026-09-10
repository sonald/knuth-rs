use std::time::Duration;

use ai::Tool;
use async_trait::async_trait;
use knuth_core::ToolOutcome;
use once_cell::sync::Lazy;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::{ToolCapabilities, ToolDescription, ToolError};

use super::{AgentTool, ToolInput, ToolResult, required_string};

pub struct PythonTool {}

#[async_trait]
impl AgentTool for PythonTool {
    fn schema(&self) -> &Tool {
        &PYTHON_SCHEMA
    }

    fn description(&self) -> ToolDescription {
        ToolDescription {
            id: (&PYTHON_SCHEMA.name).into(),
            introduction: None,
            capabilities: ToolCapabilities::ALL,
        }
    }

    async fn execute(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolResult, ToolError> {
        let code = required_string(&input, "code")?;

        let mut command = Command::new("python3");
        let output = tokio::select! {
            _ = cancel_token.cancelled() => {
                return Ok(ToolResult {
                    outcome: ToolOutcome::Cancelled,
                    content: "Python execution cancelled".to_string(),
                })
            },
            _ = tokio::time::sleep(Duration::from_secs(30)) => return Err(ToolError::TimeoutError(Duration::from_secs(30))),
            result = command.kill_on_drop(true).arg("-c").arg(code).output() => {
                result.map_err(|error| ToolError::Message(error.to_string()))?
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        let outcome = if output.status.success() {
            ToolOutcome::ExecSuccess
        } else {
            ToolOutcome::Error
        };

        Ok(ToolResult {
            outcome,
            content: format!(
                "Python exited with {}.\nstdout:\n{}\nstderr:\n{}",
                output.status, stdout, stderr
            ),
        })
    }
}

static PYTHON_SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "python_exec".to_string(),
    description: include_str!("descriptions/python.md").trim().to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": { "code": { "type": "string" } },
        "required": ["code"],
        "additionalProperties": false
    }),
});

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn python_returns_stdout() {
        let mut input = ToolInput::new();
        input.insert("code".to_string(), "print('python-ok')".into());

        let result = PythonTool {}
            .execute(input, CancellationToken::new())
            .await
            .unwrap();

        assert!(matches!(result.outcome, ToolOutcome::ExecSuccess));
        let content = result.content;
        assert!(content.contains("python-ok"), "content={content}");
    }
}
