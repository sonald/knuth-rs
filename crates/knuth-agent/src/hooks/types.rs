use std::path::PathBuf;

use ai::Context as AiContext;
use ai::{ToolCall, UserContent, UserContentBlock};
use async_trait::async_trait;
use knuth_core::ids::{HookId, SessionId, ToolInvocationId};
use tokio_util::sync::CancellationToken;

use crate::{ToolInput, ToolResult};

#[derive(Debug, Clone)]
pub struct HookContext {
    pub session_id: SessionId,
    /// available only for tool use hooks
    pub invocation_id: Option<ToolInvocationId>,
    pub workspace: PathBuf,
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

pub struct ContextView<'a> {
    pub snapshot: &'a AiContext,
}

#[derive(Debug)]
pub enum ContextEditData {
    ToolResultEdit(Vec<UserContentBlock>),
    UserMessageEdit(UserContent),
}

#[derive(Debug)]
pub struct ContextEdit {
    pub message_id: usize,
    pub edit: ContextEditData,
}

#[derive(Default, Debug)]
pub struct ContextPatch {
    pub edits: Vec<ContextEdit>,
    pub hints: Vec<UserContent>,
}

/// Called before messages are submitted to the model.
/// The context is the history of the conversation so far.
///
/// FIXME: this allows editting in the middle of the conversation,
/// which invalidates the prefix cache. should I allow this?
#[async_trait]
pub trait ContextHook: Send + Sync {
    fn id(&self) -> HookId;

    async fn transform(
        &self,
        ctx: &HookContext,
        view: &ContextView<'_>,
    ) -> Result<ContextPatch, HookError>;
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
    /// The call that was executed, after before-tool-use rewrites.
    pub tool_call: ToolCall,
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
