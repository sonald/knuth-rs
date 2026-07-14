use std::time::Duration;

use ai::Tool;
use async_trait::async_trait;
use once_cell::sync::Lazy;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::{AgentTool, ToolInput, ToolOutcome};

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
    ) -> Result<ToolOutcome, String> {
        let code = input
            .get("code")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or("code must be a non-empty string")?;

        let mut command = Command::new("python3");
        let output = tokio::select! {
            _ = cancel_token.cancelled() => return Err("Python execution cancelled".to_string()),
            _ = tokio::time::sleep(Duration::from_secs(30)) => return Err("Python execution timed out after 30 seconds".to_string()),
            result = command.kill_on_drop(true).arg("-c").arg(code).output() => {
                result.map_err(|error| error.to_string())?
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
                "Python exited with {}.\nstdout:\n{}\nstderr:\n{}",
                output.status, stdout, stderr
            ))
        }
    }
}

static PYTHON_SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "python".to_string(),
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

        match result {
            ToolOutcome::Success(value) => assert_eq!(value["output"], "python-ok\n"),
        }
    }
}
