use ai::{Context, ToolCall, UserContent};
use async_trait::async_trait;
use knuth_core::ids::{HookId, SessionId, ToolInvocationId};
use tokio_util::sync::CancellationToken;

use crate::{ToolInput, ToolResult};

#[derive(Debug, Clone)]
pub struct HookContext {
    pub session_id: SessionId,
    pub invocation_id: Option<ToolInvocationId>,
    pub cancel: CancellationToken,
}

#[derive(thiserror::Error, Debug)]
pub enum HookError {
    #[error("The hook was cancelled")]
    Cancelled,
    #[error("The hook timed out")]
    Timeout,
}

pub enum InputHookResult {
    Modify(UserContent),
    Reject(String),
    PassThrough,
}

#[async_trait]
pub trait InputHook: Send + Sync {
    fn id(&self) -> HookId;

    /// Called before a user message is submitted.
    async fn input(
        &self,
        ctx: &mut HookContext,
        input: UserContent,
    ) -> Result<InputHookResult, HookError>;
}

#[async_trait]
pub trait ContextHook: Send + Sync {
    fn id(&self) -> HookId;

    /// Called when a context is available.
    async fn transform(&self, ctx: &HookContext, context: Context) -> Result<Context, HookError>;
}

pub enum ToolCallDecision {
    Allow,
    Deny { reason: String },
}

pub struct BeforeToolUseResult {
    pub modified_arguments: Option<ToolInput>,
    pub permission: ToolCallDecision,
    pub additional_content: Option<UserContent>, // append as user message
}

#[async_trait]
pub trait BeforeToolUseHook: Send + Sync {
    fn id(&self) -> HookId;

    /// Called before a tool call is made.
    async fn transform(
        &self,
        ctx: &HookContext,
        tool_call: &ToolCall,
    ) -> Result<BeforeToolUseResult, HookError>;
}

pub struct ToolResultView {
    pub result: ToolResult,
}

pub struct AfterToolUseResult {
    pub modified: Option<ToolResult>,
    pub additional_content: Option<UserContent>, // append as user message
}

#[async_trait]
pub trait AfterToolUseHook: Send + Sync {
    fn id(&self) -> HookId;

    /// Called after a tool call is made.
    async fn transform(
        &self,
        ctx: &HookContext,
        current: &ToolResultView,
    ) -> Result<AfterToolUseResult, HookError>;
}

#[async_trait]
pub trait SessionStartHook: Send + Sync {
    fn id(&self) -> HookId;

    /// Called when a session starts.
    async fn observe(&self, ctx: &HookContext) -> Result<(), HookError>;
}

#[async_trait]
pub trait SessionEndHook: Send + Sync {
    fn id(&self) -> HookId;

    /// Called when a session ends.
    async fn observe(&self, ctx: &HookContext) -> Result<(), HookError>;
}
