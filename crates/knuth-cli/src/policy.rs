use ai::{Tool, ToolCall};
use async_trait::async_trait;
use knuth_agent::{
    AgentTool, AgentToolRegistry, ToolCapabilities, ToolError, ToolInput, ToolResult,
    policy::{PolicyContext, PolicyEngineTrait},
};
use knuth_core::ToolOutcome;
use std::{str::FromStr, sync::Arc};
use tokio_util::sync::CancellationToken;
use tracing::debug;

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PolicyMode {
    #[serde(alias = "read_only", alias = "readonly")]
    ReadOnly,
    #[serde(alias = "accept_edits", alias = "acceptedits")]
    AcceptEdits,
    #[default]
    #[serde(alias = "default")]
    Auto,
    #[serde(alias = "bypass_permissions", alias = "bypass")]
    BypassPermissions,
}

impl PolicyMode {
    /// Canonical string form, used when rendering the effective config.
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyMode::ReadOnly => "read-only",
            PolicyMode::AcceptEdits => "accept-edits",
            PolicyMode::Auto => "auto",
            PolicyMode::BypassPermissions => "bypass-permissions",
        }
    }
}

impl FromStr for PolicyMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "read-only" | "read_only" | "readonly" => Ok(PolicyMode::ReadOnly),
            "accept-edits" | "accept_edits" | "acceptedits" => Ok(PolicyMode::AcceptEdits),
            "auto" | "default" => Ok(PolicyMode::Auto),
            "bypass-permissions" | "bypass_permissions" | "bypass" => {
                Ok(PolicyMode::BypassPermissions)
            }
            other => Err(format!("unknown policy mode '{other}'")),
        }
    }
}

#[derive(Debug)]
enum Decision {
    Allow,
    Deny { reason: String },
}

pub struct DefaultPolicyEngine {
    tool_registry: AgentToolRegistry,
    policy_mode: PolicyMode,
}

#[async_trait]
impl PolicyEngineTrait for DefaultPolicyEngine {
    fn schemas(&self) -> Vec<Tool> {
        self.tool_registry.schemas()
    }

    async fn execute(
        &self,
        tool_call: &ToolCall,
        cancel: CancellationToken,
        ctx: &PolicyContext,
    ) -> ToolResult {
        let Some(tool) = self.tool_registry.get(tool_call.name.as_str()) else {
            return ToolResult {
                outcome: ToolOutcome::Error,
                content: ToolError::InvalidTool(tool_call.name.clone()).to_string(),
            };
        };

        match self.decide(tool.clone(), &tool_call.arguments, ctx).await {
            Decision::Deny { reason } => {
                return ToolResult {
                    outcome: ToolOutcome::PolicyDenied,
                    content: reason,
                };
            }
            Decision::Allow => {}
        }

        let result = tool.execute(tool_call.arguments.clone(), cancel).await;

        match result {
            Ok(result) => result,
            Err(error) => ToolResult {
                outcome: ToolOutcome::Error,
                content: error.to_string(),
            },
        }
    }
}

impl DefaultPolicyEngine {
    pub fn new(tool_registry: AgentToolRegistry, policy_mode: PolicyMode) -> Self {
        Self {
            tool_registry,
            policy_mode,
        }
    }

    pub fn with_default_tools(policy_mode: PolicyMode) -> Self {
        let mut registry = AgentToolRegistry::new();
        registry.load_default();
        Self::new(registry, policy_mode)
    }

    pub fn register(&mut self, tool: Arc<dyn AgentTool>) {
        self.tool_registry.register(tool);
    }

    async fn decide(
        &self,
        tool: Arc<dyn AgentTool>,
        input: &ToolInput,
        ctx: &PolicyContext,
    ) -> Decision {
        match self.policy_mode {
            PolicyMode::ReadOnly => self.handle_readonly_mode(tool, input, ctx),
            PolicyMode::AcceptEdits => Decision::Allow,
            PolicyMode::Auto => Decision::Allow,
            PolicyMode::BypassPermissions => Decision::Allow,
        }
    }

    fn handle_readonly_mode(
        &self,
        tool: Arc<dyn AgentTool>,
        input: &ToolInput,
        ctx: &PolicyContext,
    ) -> Decision {
        let _ = input;
        let _ = ctx;

        debug!("handle_readonly_mode({:?})", tool.description());

        let danger_capabilities = ToolCapabilities::WRITE_FILE
            | ToolCapabilities::EXECUTE_COMMAND
            | ToolCapabilities::NETWORK_ACCESS;
        if tool
            .description()
            .capabilities
            .intersects(danger_capabilities)
        {
            Decision::Deny {
                reason: "read-only mode".to_string(),
            }
        } else {
            Decision::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::str::FromStr;

    #[test]
    fn policy_mode_parses_canonical_names() {
        assert_eq!(
            "read-only".parse::<PolicyMode>().unwrap().as_str(),
            "read-only"
        );
        assert_eq!(
            "accept-edits".parse::<PolicyMode>().unwrap().as_str(),
            "accept-edits"
        );
        assert_eq!("auto".parse::<PolicyMode>().unwrap().as_str(), "auto");
        assert_eq!(
            "bypass-permissions".parse::<PolicyMode>().unwrap().as_str(),
            "bypass-permissions"
        );
    }

    #[test]
    fn policy_mode_parses_aliases() {
        assert_eq!(
            "ReadOnly".parse::<PolicyMode>().unwrap().as_str(),
            "read-only"
        );
        assert_eq!("default".parse::<PolicyMode>().unwrap().as_str(), "auto");
        assert_eq!(
            "bypass".parse::<PolicyMode>().unwrap().as_str(),
            "bypass-permissions"
        );
    }

    #[test]
    fn policy_mode_rejects_unknown() {
        assert!("sideways".parse::<PolicyMode>().is_err());
    }

    #[test]
    fn policy_mode_deserializes_from_scalar() {
        let mode: PolicyMode = serde_json::from_str("\"read-only\"").unwrap();
        assert_eq!(mode.as_str(), "read-only");

        let mode: PolicyMode = serde_json::from_str("\"accept_edits\"").unwrap();
        assert_eq!(mode.as_str(), "accept-edits");
    }

    #[test]
    fn policy_mode_defaults_to_auto() {
        assert_eq!(PolicyMode::default().as_str(), "auto");
        let mode = PolicyMode::from_str("auto").unwrap();
        assert!(matches!(mode, PolicyMode::Auto));
    }

    fn ctx() -> PolicyContext {
        PolicyContext {}
    }

    fn write_call(path: &str, content: &str) -> ToolCall {
        let mut arguments = serde_json::Map::new();
        arguments.insert("path".into(), serde_json::Value::String(path.into()));
        arguments.insert("content".into(), serde_json::Value::String(content.into()));
        ToolCall {
            id: "call-1".into(),
            name: "write_file".into(),
            arguments,
            thought_signature: None,
        }
    }

    fn read_call(path: &str) -> ToolCall {
        let mut arguments = serde_json::Map::new();
        arguments.insert("path".into(), serde_json::Value::String(path.into()));
        ToolCall {
            id: "call-1".into(),
            name: "read_file".into(),
            arguments,
            thought_signature: None,
        }
    }

    fn bash_call(command: &str) -> ToolCall {
        let mut arguments = serde_json::Map::new();
        arguments.insert("command".into(), serde_json::Value::String(command.into()));
        ToolCall {
            id: "call-1".into(),
            name: "bash".into(),
            arguments,
            thought_signature: None,
        }
    }

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "knuth-cli-policy-{label}-{}.txt",
            uuid::Uuid::new_v4()
        ))
    }

    #[tokio::test]
    async fn read_only_denies_write_without_creating_file() {
        let path = temp_path("write");
        let engine = DefaultPolicyEngine::with_default_tools(PolicyMode::ReadOnly);
        let result = engine
            .execute(
                &write_call(&path.to_string_lossy(), "must not be written"),
                CancellationToken::new(),
                &ctx(),
            )
            .await;

        let existed = path.exists();
        if existed {
            let _ = std::fs::remove_file(&path);
        }

        assert_eq!(result.outcome, ToolOutcome::PolicyDenied);
        assert!(
            !existed,
            "read-only policy executed write_file against {path:?}"
        );
        assert_eq!(result.content, "read-only mode");
    }

    #[tokio::test]
    async fn read_only_denies_bash() {
        let engine = DefaultPolicyEngine::with_default_tools(PolicyMode::ReadOnly);
        let result = engine
            .execute(
                &bash_call("printf should-not-run"),
                CancellationToken::new(),
                &ctx(),
            )
            .await;

        assert_eq!(result.outcome, ToolOutcome::PolicyDenied);
        assert_eq!(result.content, "read-only mode");
    }

    #[tokio::test]
    async fn read_only_allows_read_file() {
        let path = temp_path("read");
        std::fs::write(&path, "hello from disk").unwrap();

        let engine = DefaultPolicyEngine::with_default_tools(PolicyMode::ReadOnly);
        let result = engine
            .execute(
                &read_call(&path.to_string_lossy()),
                CancellationToken::new(),
                &ctx(),
            )
            .await;
        let _ = std::fs::remove_file(&path);

        assert_eq!(result.outcome, ToolOutcome::ExecSuccess);
        assert!(
            result.content.contains("hello from disk"),
            "got {:?}",
            result.content
        );
    }

    #[tokio::test]
    async fn permissive_modes_allow_write() {
        let modes = [
            PolicyMode::AcceptEdits,
            PolicyMode::Auto,
            PolicyMode::BypassPermissions,
        ];

        for mode in modes {
            let path = temp_path(&format!("allow-{}", mode.as_str()));
            let engine = DefaultPolicyEngine::with_default_tools(mode.clone());
            let result = engine
                .execute(
                    &write_call(&path.to_string_lossy(), "ok"),
                    CancellationToken::new(),
                    &ctx(),
                )
                .await;

            let existed = path.exists();
            if existed {
                let _ = std::fs::remove_file(&path);
            }

            assert_eq!(
                result.outcome,
                ToolOutcome::ExecSuccess,
                "mode {} should allow write",
                mode.as_str()
            );
            assert!(existed, "mode {} did not write {path:?}", mode.as_str());
        }
    }

    #[tokio::test]
    async fn unknown_tool_is_an_error() {
        let engine = DefaultPolicyEngine::with_default_tools(PolicyMode::Auto);
        let result = engine
            .execute(
                &ToolCall {
                    id: "call-1".into(),
                    name: "not_a_tool".into(),
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                },
                CancellationToken::new(),
                &ctx(),
            )
            .await;

        assert_eq!(result.outcome, ToolOutcome::Error);
        assert!(
            result.content.contains("not_a_tool"),
            "got {:?}",
            result.content
        );
    }
}
