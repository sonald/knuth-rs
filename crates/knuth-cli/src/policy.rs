use ai::{Tool, ToolCall};
use async_trait::async_trait;
use knuth_agent::{
    AgentTool, AgentToolRegistry, ToolCapabilities, ToolError, ToolInput, ToolResult,
    policy::{PolicyContext, PolicyEngineTrait, PolicyMode},
};
use knuth_core::ToolOutcome;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::debug;

#[derive(Debug)]
enum Decision {
    Allow,
    Deny { reason: String },
}

pub enum PolicyHookResult {
    Accept,
    Deny { reason: String },
}

#[async_trait]
pub trait PolicyHook: Send + Sync {
    async fn before_tool_execution(
        &self,
        tool: Arc<dyn AgentTool>,
        input: &ToolInput,
        ctx: &PolicyContext,
    ) -> PolicyHookResult;

    async fn after_tool_execution(
        &self,
        tool: Arc<dyn AgentTool>,
        input: &ToolInput,
        ctx: &PolicyContext,
        result: &Result<ToolResult, ToolError>,
    );
}

pub struct DefaultPolicyEngine {
    hooks: Vec<Box<dyn PolicyHook>>,
    tool_registry: AgentToolRegistry,
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
                content: ToolError::InvalidTool(tool_call.name.clone())
                    .to_string()
                    .into_bytes(),
            };
        };

        match self.decide(tool.clone(), &tool_call.arguments, ctx).await {
            Decision::Deny { reason } => {
                return ToolResult {
                    outcome: ToolOutcome::PolicyDenied,
                    content: reason.into_bytes(),
                };
            }
            Decision::Allow => {}
        }

        let result = tool.execute(tool_call.arguments.clone(), cancel).await;
        for hook in self.hooks.iter() {
            hook.after_tool_execution(tool.clone(), &tool_call.arguments, ctx, &result)
                .await;
        }

        match result {
            Ok(result) => result,
            Err(error) => ToolResult {
                outcome: ToolOutcome::Error,
                content: error.to_string().into_bytes(),
            },
        }
    }
}

impl DefaultPolicyEngine {
    pub fn new(tool_registry: AgentToolRegistry) -> Self {
        Self {
            hooks: Vec::new(),
            tool_registry,
        }
    }

    pub fn with_default_tools() -> Self {
        let mut registry = AgentToolRegistry::new();
        registry.load_default();
        Self::new(registry)
    }

    pub fn add_hook(mut self, hook: Box<dyn PolicyHook>) -> Self {
        self.hooks.push(hook);
        self
    }

    async fn decide(
        &self,
        tool: Arc<dyn AgentTool>,
        input: &ToolInput,
        ctx: &PolicyContext,
    ) -> Decision {
        for hook in self.hooks.iter() {
            match hook.before_tool_execution(tool.clone(), input, ctx).await {
                PolicyHookResult::Deny { reason } => return Decision::Deny { reason },
                PolicyHookResult::Accept => {}
            }
        }

        match ctx.mode {
            PolicyMode::ReadOnly => self.handle_readonly_mode(tool, input, ctx),
            PolicyMode::AcceptEdits => Decision::Allow,
            PolicyMode::PlanMode => Decision::Allow,
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

    fn ctx(mode: PolicyMode) -> PolicyContext {
        PolicyContext { mode }
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

    struct AllowHook;

    #[async_trait]
    impl PolicyHook for AllowHook {
        async fn before_tool_execution(
            &self,
            _tool: Arc<dyn AgentTool>,
            _input: &ToolInput,
            _ctx: &PolicyContext,
        ) -> PolicyHookResult {
            PolicyHookResult::Accept
        }

        async fn after_tool_execution(
            &self,
            _tool: Arc<dyn AgentTool>,
            _input: &ToolInput,
            _ctx: &PolicyContext,
            _result: &Result<ToolResult, ToolError>,
        ) {
        }
    }

    struct DenyHook;

    #[async_trait]
    impl PolicyHook for DenyHook {
        async fn before_tool_execution(
            &self,
            _tool: Arc<dyn AgentTool>,
            _input: &ToolInput,
            _ctx: &PolicyContext,
        ) -> PolicyHookResult {
            PolicyHookResult::Deny {
                reason: "hook denied".into(),
            }
        }

        async fn after_tool_execution(
            &self,
            _tool: Arc<dyn AgentTool>,
            _input: &ToolInput,
            _ctx: &PolicyContext,
            _result: &Result<ToolResult, ToolError>,
        ) {
        }
    }

    #[tokio::test]
    async fn read_only_denies_write_without_creating_file() {
        let path = temp_path("write");
        let engine = DefaultPolicyEngine::with_default_tools();
        let result = engine
            .execute(
                &write_call(&path.to_string_lossy(), "must not be written"),
                CancellationToken::new(),
                &ctx(PolicyMode::ReadOnly),
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
        assert_eq!(String::from_utf8_lossy(&result.content), "read-only mode");
    }

    #[tokio::test]
    async fn read_only_denies_bash() {
        let engine = DefaultPolicyEngine::with_default_tools();
        let result = engine
            .execute(
                &bash_call("printf should-not-run"),
                CancellationToken::new(),
                &ctx(PolicyMode::ReadOnly),
            )
            .await;

        assert_eq!(result.outcome, ToolOutcome::PolicyDenied);
        assert_eq!(String::from_utf8_lossy(&result.content), "read-only mode");
    }

    #[tokio::test]
    async fn read_only_allows_read_file() {
        let path = temp_path("read");
        std::fs::write(&path, "hello from disk").unwrap();

        let engine = DefaultPolicyEngine::with_default_tools();
        let result = engine
            .execute(
                &read_call(&path.to_string_lossy()),
                CancellationToken::new(),
                &ctx(PolicyMode::ReadOnly),
            )
            .await;
        let _ = std::fs::remove_file(&path);

        assert_eq!(result.outcome, ToolOutcome::ExecSuccess);
        assert!(
            String::from_utf8_lossy(&result.content).contains("hello from disk"),
            "got {:?}",
            String::from_utf8_lossy(&result.content)
        );
    }

    #[tokio::test]
    async fn permissive_modes_allow_write() {
        let modes = [
            PolicyMode::AcceptEdits,
            PolicyMode::PlanMode,
            PolicyMode::Auto,
            PolicyMode::BypassPermissions,
        ];

        for mode in modes {
            let path = temp_path(&format!("allow-{}", mode.as_str()));
            let engine = DefaultPolicyEngine::with_default_tools();
            let result = engine
                .execute(
                    &write_call(&path.to_string_lossy(), "ok"),
                    CancellationToken::new(),
                    &ctx(mode.clone()),
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
        let engine = DefaultPolicyEngine::with_default_tools();
        let result = engine
            .execute(
                &ToolCall {
                    id: "call-1".into(),
                    name: "not_a_tool".into(),
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                },
                CancellationToken::new(),
                &ctx(PolicyMode::Auto),
            )
            .await;

        assert_eq!(result.outcome, ToolOutcome::Error);
        assert!(
            String::from_utf8_lossy(&result.content).contains("not_a_tool"),
            "got {:?}",
            String::from_utf8_lossy(&result.content)
        );
    }

    #[tokio::test]
    async fn accepting_hook_does_not_block_execution() {
        let path = temp_path("hook-allow");
        let engine = DefaultPolicyEngine::with_default_tools().add_hook(Box::new(AllowHook));
        let result = engine
            .execute(
                &write_call(&path.to_string_lossy(), "hooked"),
                CancellationToken::new(),
                &ctx(PolicyMode::Auto),
            )
            .await;

        let existed = path.exists();
        if existed {
            let _ = std::fs::remove_file(&path);
        }

        assert_eq!(result.outcome, ToolOutcome::ExecSuccess);
        assert!(existed, "accepting hook blocked write to {path:?}");
    }

    #[tokio::test]
    async fn hook_can_deny_before_execution() {
        let path = temp_path("hook");
        let engine = DefaultPolicyEngine::with_default_tools().add_hook(Box::new(DenyHook));
        let result = engine
            .execute(
                &write_call(&path.to_string_lossy(), "hooked"),
                CancellationToken::new(),
                &ctx(PolicyMode::BypassPermissions),
            )
            .await;

        let existed = path.exists();
        if existed {
            let _ = std::fs::remove_file(&path);
        }

        assert_eq!(result.outcome, ToolOutcome::PolicyDenied);
        assert!(!existed, "denied hook still wrote {path:?}");
        assert_eq!(String::from_utf8_lossy(&result.content), "hook denied");
    }
}
