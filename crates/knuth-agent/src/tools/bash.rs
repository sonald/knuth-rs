use ai::Tool;
use async_trait::async_trait;
use once_cell::sync::Lazy;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::{AgentTool, ToolInput, ToolOutcome};

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
    description: include_str!("descriptions/bash.md").trim().to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "The command to execute" }
        }
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
