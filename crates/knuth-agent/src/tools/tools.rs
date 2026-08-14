use std::collections::HashMap;
use std::sync::Arc;

use ai::Tool;
use async_trait::async_trait;
use knuth_core::ids::ToolId;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use knuth_core::ToolOutcome;

use bitflags::bitflags;

use crate::{BashTool, EditFileTool, PythonTool, ReadFileTool, WriteFileTool};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDescription {
    pub id: ToolId,
    pub introduction: Option<String>,
    pub capabilities: ToolCapabilities,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct ToolCapabilities: u8 {
        const READ_FILE = 1 << 0;
        const WRITE_FILE = 1 << 1;
        const EXECUTE_COMMAND = 1 << 2;
        const NETWORK_ACCESS = 1 << 3;
        const ALL = Self::READ_FILE.bits() | Self::WRITE_FILE.bits() | Self::EXECUTE_COMMAND.bits() | Self::NETWORK_ACCESS.bits();
    }
}

#[derive(Debug)]
pub struct ToolResult {
    pub outcome: ToolOutcome,
    pub content: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool error: {0}")]
    Message(String),

    #[error("tool argument error: {0}")]
    ArgumentError(String),

    #[error("tool execution is timeout after {0:?}")]
    TimeoutError(Duration),

    #[error("tool not found: {0}")]
    InvalidTool(String)
}

pub type ToolInput = serde_json::Map<String, serde_json::Value>;

fn value_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn summarize_value(value: &serde_json::Value) -> String {
    let rendered = value.to_string();
    if rendered.len() > 80 {
        format!("{}...", &rendered[..80])
    } else {
        rendered
    }
}

fn missing_argument_error(input: &ToolInput, name: &str, expected: &str) -> ToolError {
    let received = input
        .iter()
        .map(|(key, value)| format!("\"{key}\" ({})", value_type_name(value)))
        .collect::<Vec<_>>()
        .join(", ");
    let received = if received.is_empty() {
        "none".to_string()
    } else {
        received
    };
    let mut message = format!(
        "missing required argument \"{name}\" (expected {expected}); received arguments: {received}"
    );
    let similar = input.keys().find(|key| {
        let key = key.to_ascii_lowercase();
        let name = name.to_ascii_lowercase();
        if key == name || key.len() < 3 {
            return false;
        }
        // catches near-misses like "cmd" for "command"
        let mut rest = name.chars();
        name.starts_with(&key) || key.chars().all(|c| rest.by_ref().any(|n| n == c))
    });
    if let Some(key) = input
        .keys()
        .find(|key| key.eq_ignore_ascii_case(name))
        .or(similar)
    {
        message.push_str(&format!(
            "; did you mean \"{key}\"? argument names are case-sensitive, use exactly \"{name}\""
        ));
    }
    ToolError::ArgumentError(message)
}

/// Reads a required non-empty string argument, producing error messages that
/// echo back what was actually received so the caller can self-correct.
pub fn required_string<'a>(input: &'a ToolInput, name: &str) -> Result<&'a str, ToolError> {
    match input.get(name) {
        Some(serde_json::Value::String(value)) if !value.is_empty() => Ok(value),
        Some(serde_json::Value::String(_)) => Err(ToolError::ArgumentError(format!(
            "argument \"{name}\" must be a non-empty string"
        ))),
        Some(value) => Err(ToolError::ArgumentError(format!(
            "argument \"{name}\" must be a string, but received {} ({})",
            value_type_name(value),
            summarize_value(value)
        ))),
        None => Err(missing_argument_error(input, name, "a non-empty string")),
    }
}

/// Same as [`required_string`], but empty strings are accepted.
pub fn required_string_allow_empty<'a>(
    input: &'a ToolInput,
    name: &str,
) -> Result<&'a str, ToolError> {
    match input.get(name) {
        Some(serde_json::Value::String(value)) => Ok(value),
        Some(value) => Err(ToolError::ArgumentError(format!(
            "argument \"{name}\" must be a string, but received {} ({})",
            value_type_name(value),
            summarize_value(value)
        ))),
        None => Err(missing_argument_error(input, name, "a string")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_argument_error_suggests_case_insensitive_prefix() {
        let mut input = ToolInput::new();
        input.insert(
            "CMD".to_string(),
            serde_json::Value::String("ls".to_string()),
        );
        let error = missing_argument_error(&input, "command", "a non-empty string");
        let message = error.to_string();
        assert!(
            message.contains("did you mean \"CMD\""),
            "message={message}"
        );
    }
}
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn schema(&self) -> &Tool;
    async fn execute(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolResult, ToolError>;

    fn description(&self) -> ToolDescription;
}

pub struct AgentToolRegistry {
    tools: HashMap<ToolId, Arc<dyn AgentTool>>,
}

impl std::fmt::Debug for AgentToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AgentToolRegistry {{ tools: {:?} }}",
            self.tools.keys().collect::<Vec<&ToolId>>()
        )
    }
}

impl AgentToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn load_default(&mut self) -> &mut Self {
        self.register(Arc::new(BashTool {}));
        self.register(Arc::new(ReadFileTool {}));
        self.register(Arc::new(WriteFileTool {}));
        self.register(Arc::new(EditFileTool {}));
        self.register(Arc::new(PythonTool {}));
        self
    }

    pub fn register(&mut self, tool: Arc<dyn AgentTool>) {
        self.tools.insert(tool.description().id.clone(), tool);
    }

    /// Returns an owned handle so tool execution can be spawned off the
    /// actor's task without borrowing the registry.
    pub fn get(&self, tool_name: &str) -> Option<Arc<dyn AgentTool>> {
        self.tools.get(&ToolId::from(tool_name)).cloned()
    }

    pub fn schemas(&self) -> Vec<Tool> {
        self.tools
            .values()
            .map(|tool| tool.schema().clone())
            .collect()
    }
}
