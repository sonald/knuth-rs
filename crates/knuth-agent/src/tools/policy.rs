use ai::{Tool, ToolCall};
use async_trait::async_trait;
use tracing::debug;
use std::str::FromStr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use knuth_core::ToolOutcome;
use crate::{
    AgentTool, AgentToolRegistry, ToolCapabilities, ToolError, ToolResult, tools::ToolInput,
};


#[derive(Debug)]
pub enum Decision {
    Allow,
    Deny { reason: String },
}

#[derive(Clone)]
pub struct PolicyContext {
    pub mode: PolicyMode,
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PolicyMode {
    ReadOnly,
    AcceptEdits,
    #[serde(alias = "plan")]
    PlanMode,
    #[default]
    Auto,
    BypassPermissions,
}

impl PolicyMode {
    /// Canonical string form, used when rendering the effective config.
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyMode::ReadOnly => "read-only",
            PolicyMode::AcceptEdits => "accept-edits",
            PolicyMode::PlanMode => "plan",
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
            "plan" | "plan-mode" | "plan_mode" => Ok(PolicyMode::PlanMode),
            "auto" | "default" => Ok(PolicyMode::Auto),
            "bypass-permissions" | "bypass_permissions" | "bypass" => {
                Ok(PolicyMode::BypassPermissions)
            }
            other => Err(format!("unknown policy mode '{other}'")),
        }
    }
}

#[async_trait]
pub trait PolicyEngineTrait: Send + Sync {
    fn schemas(&self) -> Vec<Tool>;
    async fn execute(
        &self,
        tool_call: &ToolCall,
        cancel: CancellationToken,
        ctx: &PolicyContext,
    ) -> ToolResult;
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
                content: ToolError::InvalidTool(tool_call.name.clone()).to_string().into_bytes(),
            }
        };
        
        match self.decide(tool.clone(), &tool_call.arguments, ctx).await {
            Decision::Deny { reason } => return ToolResult { outcome: ToolOutcome::PolicyDenied, content: reason.into_bytes() },
            Decision::Allow => {}
        }

        let result = tool.execute(tool_call.arguments.clone(), cancel).await;
        for hook in self.hooks.iter() {
            hook.after_tool_execution(tool.clone(), &tool_call.arguments, ctx, &result).await;
        }

        match result {
            Ok(result) => result,
            Err(error) => ToolResult {
                outcome: ToolOutcome::Error,
                content: error.to_string().into_bytes(),
            }
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
            let result = hook.before_tool_execution(tool.clone(), &input, ctx).await;
            if let PolicyHookResult::Deny { reason } = result {
                return Decision::Deny { reason };
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


    fn handle_readonly_mode(&self, tool: Arc<dyn AgentTool>, input: &ToolInput, ctx: &PolicyContext) -> Decision {
        let _ = input;
        let _ = ctx;

        debug!("handle_readonly_mode({:?})", tool.description());

        let danger_capabilities = ToolCapabilities::WRITE_FILE | ToolCapabilities::EXECUTE_COMMAND | ToolCapabilities::NETWORK_ACCESS;
        if tool.description().capabilities.intersects(danger_capabilities) {
            Decision::Deny { reason: "read-only mode".to_string() }
        } else {
            Decision::Allow
        }
    }
}

pub struct DefaultPolicyHook;

#[cfg(test)]
mod tests {
    use super::PolicyMode;
    use std::str::FromStr;

    #[test]
    fn policy_mode_parses_canonical_names() {
        assert_eq!("read-only".parse::<PolicyMode>().unwrap().as_str(), "read-only");
        assert_eq!(
            "accept-edits".parse::<PolicyMode>().unwrap().as_str(),
            "accept-edits"
        );
        assert_eq!("plan".parse::<PolicyMode>().unwrap().as_str(), "plan");
        assert_eq!("auto".parse::<PolicyMode>().unwrap().as_str(), "auto");
        assert_eq!(
            "bypass-permissions".parse::<PolicyMode>().unwrap().as_str(),
            "bypass-permissions"
        );
    }

    #[test]
    fn policy_mode_parses_aliases() {
        assert_eq!(
            "plan-mode".parse::<PolicyMode>().unwrap().as_str(),
            "plan"
        );
        assert_eq!(
            "ReadOnly".parse::<PolicyMode>().unwrap().as_str(),
            "read-only"
        );
        assert_eq!("default".parse::<PolicyMode>().unwrap().as_str(), "auto");
    }

    #[test]
    fn policy_mode_rejects_unknown() {
        assert!("sideways".parse::<PolicyMode>().is_err());
    }

    #[test]
    fn policy_mode_deserializes_from_scalar() {
        let mode: PolicyMode = serde_json::from_str("\"plan\"").unwrap();
        assert_eq!(mode.as_str(), "plan");

        let mode: PolicyMode = serde_json::from_str("\"plan-mode\"").unwrap();
        assert_eq!(mode.as_str(), "plan");

        let mode: PolicyMode = serde_json::from_str("\"read-only\"").unwrap();
        assert_eq!(mode.as_str(), "read-only");
    }

    #[test]
    fn policy_mode_defaults_to_auto() {
        assert_eq!(PolicyMode::default().as_str(), "auto");
        let mode: PolicyMode = PolicyMode::from_str("auto").unwrap();
        assert!(matches!(mode, PolicyMode::Auto));
    }
}
