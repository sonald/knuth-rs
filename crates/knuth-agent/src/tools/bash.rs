use ai::Tool;
use async_trait::async_trait;
use knuth_core::ToolOutcome;
use once_cell::sync::Lazy;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::{ToolCapabilities, ToolDescription, ToolError};

use super::{AgentTool, ToolInput, ToolResult, required_string};

pub struct BashTool {}

#[async_trait]
impl AgentTool for BashTool {
    fn schema(&self) -> &Tool {
        &BASH_SCHEMA
    }

    fn description(&self) -> ToolDescription {
        ToolDescription {
            id: (&BASH_SCHEMA.name).into(),
            introduction: None,
            capabilities: ToolCapabilities::ALL,
        }
    }

    async fn execute(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolResult, ToolError> {
        let command = required_string(&input, "command")?;

        let mut cmd = Command::new("bash");
        let output = tokio::select! {
            _ = cancel_token.cancelled() => return Ok(ToolResult {
                outcome: ToolOutcome::Cancelled,
                content: "Command execution cancelled".to_string(),
            }),
            output = cmd.kill_on_drop(true).arg("-c").arg(command).output() => {
                output.map_err(|e| ToolError::Message(e.to_string()))?
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
                "Command exited with {}.\nstdout:\n{}\nstderr:\n{}",
                output.status, stdout, stderr
            ),
        })
    }
}

static BASH_SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "bash".to_string(),
    description: include_str!("descriptions/bash.md").trim().to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "The command to execute" }
        },
        "required": ["command"],
        "additionalProperties": false
    }),
});

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
        let result = BashTool {}
            .execute(input("printf hello"), CancellationToken::new())
            .await
            .unwrap();

        assert!(matches!(result.outcome, ToolOutcome::ExecSuccess));
        let content = result.content;
        assert!(content.contains("hello"), "content={content}");
    }

    #[tokio::test]
    async fn bash_tool_reports_exit_status_and_stderr() {
        let result = BashTool {}
            .execute(input("printf nope >&2; exit 7"), CancellationToken::new())
            .await
            .unwrap();

        assert!(matches!(result.outcome, ToolOutcome::Error));
        let content = result.content;
        assert!(content.contains("exit status: 7"), "content={content}");
        assert!(content.contains("nope"), "content={content}");
    }

    #[tokio::test]
    async fn bash_tool_handles_non_utf8_output() {
        let result = BashTool {}
            .execute(input("printf '\\377'"), CancellationToken::new())
            .await
            .unwrap();

        assert!(matches!(result.outcome, ToolOutcome::ExecSuccess));
        let content = result.content;
        assert!(content.contains('\u{fffd}'), "content={content}");
    }

    #[tokio::test]
    async fn bash_tool_error_echoes_received_argument_keys() {
        let mut wrong_case = ToolInput::new();
        wrong_case.insert(
            "CMD".to_string(),
            serde_json::Value::String("ls".to_string()),
        );

        let error = BashTool {}
            .execute(wrong_case, CancellationToken::new())
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("missing required argument \"command\""),
            "message={message}"
        );
        assert!(
            message.contains("received arguments: \"CMD\" (string)"),
            "message={message}"
        );
        assert!(message.contains("case-sensitive"), "message={message}");
    }
}
