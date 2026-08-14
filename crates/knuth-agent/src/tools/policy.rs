use ai::Tool;
use async_trait::async_trait;

use crate::{
    AgentToolRegistry, ToolError, ToolResult,
    tools::{ToolDescription, ToolInput},
};

#[derive(Debug)]
pub enum Decision {
    Allow,
    AskUser { message: String },
    Deny { reason: String },
}

pub struct PolicyContext {
    pub mode: PolicyMode,
}

pub enum PolicyMode {
    ReadOnly,
    AcceptEdits,
    PlanMode,
    Auto,
    BypassPermissions,
}

#[async_trait]
trait PolicyEngineTrait: Send + Sync {
    async fn decide(
        &mut self,
        tool: &Tool,
        desc: &ToolDescription,
        input: ToolInput,
        ctx: &PolicyContext,
    ) -> Decision;

    async fn execute(
        &mut self,
        tool: &Tool,
        desc: &ToolDescription,
        input: ToolInput,
        ctx: &PolicyContext,
    ) -> Result<ToolResult, ToolError>;
}

pub enum PolicyHookResult {
    Continue,
    Stop(String),
}

#[async_trait]
pub trait PolicyHook: Send + Sync {
    async fn before_tool_execution(
        &self,
        tool: &Tool,
        desc: &ToolDescription,
        input: ToolInput,
        ctx: &PolicyContext,
    ) -> PolicyHookResult;

    async fn after_tool_execution(
        &self,
        tool: &Tool,
        desc: &ToolDescription,
        input: ToolInput,
        ctx: &PolicyContext,
        result: Result<ToolResult, ToolError>,
    );
}

pub struct PolicyEngineImpl {
    hooks: Vec<Box<dyn PolicyHook>>,
    tool_registry: AgentToolRegistry,
}

#[async_trait]
impl PolicyEngineTrait for PolicyEngineImpl {
    async fn decide(
        &mut self,
        tool: &Tool,
        desc: &ToolDescription,
        input: ToolInput,
        ctx: &PolicyContext,
    ) -> Decision {
        let Some(tool) = self.tool_registry.get(&tool.name) else {
            return Decision::Deny {
                reason: "tool not found".to_string(),
            };
        };
        return self.handle_mode(ctx.mode);
    }

    async fn execute(
        &mut self,
        tool: &Tool,
        desc: &ToolDescription,
        input: ToolInput,
        ctx: &PolicyContext,
    ) -> Result<ToolResult, ToolError> {
        let Some(tool) = self.tool_registry.get(&tool.name) else {
            return Err(ToolError::InvalidTool(tool.name.clone()));
        };

        Err(ToolError::InvalidTool("".to_string()))
    }
}

impl PolicyEngineImpl {
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

    fn handle_mode(&mut self, mode: PolicyMode) -> Decision {
        match mode {
            PolicyMode::ReadOnly => unimplemented!(),
            _ => unimplemented!(),
        }
    }
}

pub struct DefaultPolicyHook;
