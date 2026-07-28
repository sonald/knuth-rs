use std::time::Duration;

use ai::Tool;
use async_trait::async_trait;
use knuth_core::ToolOutcome;
use once_cell::sync::Lazy;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::{AgentTool, ToolInput, ToolResult};

pub struct PythonTool {}

#[async_trait]
impl AgentTool for PythonTool {
    fn schema(&self) -> &Tool {
        &PYTHON_SCHEMA
    }

    async fn invoke(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolResult, String> {
        let code = input
            .get("code")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or("code must be a non-empty string")?;


        let mut command = Command::new("python3");
        let output = tokio::select! {
            _ = cancel_token.cancelled() => {
                return Ok(ToolResult {
                    outcome: ToolOutcome::Cancelled,
                    content: b"Python execution cancelled".to_vec(),
                })
            },
            _ = tokio::time::sleep(Duration::from_secs(30)) => return Err("Python execution timed out after 30 seconds".to_string()),
            result = command.kill_on_drop(true).arg("-c").arg(code).output() => {
                result.map_err(|error| error.to_string())?
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
        ).into_bytes(),
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
            .invoke(input, CancellationToken::new())
            .await
            .unwrap();

        assert!(matches!(result.outcome, ToolOutcome::ExecSuccess));
        let content = String::from_utf8_lossy(&result.content);
        assert!(content.contains("python-ok"), "content={content}");
    }
}
