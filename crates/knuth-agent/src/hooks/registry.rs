use crate::{
    ToolResult,
    hooks::{
        AfterToolUseHook, BeforeToolUseHook, ContextHook, HookContext, HookError, InputHook,
        SessionEndHook, SessionStartHook, ToolCallDecision, ToolResultView,
    },
};
use ai::{ToolCall, UserContent};
use std::sync::Arc;

pub struct HookRegistry {
    input_hooks: Vec<Arc<dyn InputHook>>,
    context_hooks: Vec<Arc<dyn ContextHook>>,
    before_tool_use_hooks: Vec<Arc<dyn BeforeToolUseHook>>,
    after_tool_use_hooks: Vec<Arc<dyn AfterToolUseHook>>,
    session_start_hooks: Vec<Arc<dyn SessionStartHook>>,
    session_end_hooks: Vec<Arc<dyn SessionEndHook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            input_hooks: Vec::new(),
            context_hooks: Vec::new(),
            before_tool_use_hooks: Vec::new(),
            after_tool_use_hooks: Vec::new(),
            session_start_hooks: Vec::new(),
            session_end_hooks: Vec::new(),
        }
    }

    pub fn on_input(&mut self, hook: Arc<dyn InputHook>) {
        self.input_hooks.push(hook);
    }

    pub fn on_context(&mut self, hook: Arc<dyn ContextHook>) {
        self.context_hooks.push(hook);
    }

    pub fn on_before_tool_use(&mut self, hook: Arc<dyn BeforeToolUseHook>) {
        self.before_tool_use_hooks.push(hook);
    }

    pub fn on_after_tool_use(&mut self, hook: Arc<dyn AfterToolUseHook>) {
        self.after_tool_use_hooks.push(hook);
    }

    pub fn on_session_start(&mut self, hook: Arc<dyn SessionStartHook>) {
        self.session_start_hooks.push(hook);
    }

    pub fn on_session_end(&mut self, hook: Arc<dyn SessionEndHook>) {
        self.session_end_hooks.push(hook);
    }

    pub async fn before_tool_use(
        &self,
        ctx: &HookContext,
        mut tool_call: ToolCall,
    ) -> Result<BeforeToolUseData, HookError> {
        let mut additional_hints = vec![];

        for hook in self.before_tool_use_hooks.iter() {
            let result = tokio::select! {
                _ = ctx.cancel.cancelled() => {
                    return Err(HookError::Cancelled);
                }
                result = hook.transform(ctx, &tool_call) => {
                    result?
                }
            };

            match result.permission {
                ToolCallDecision::Deny { .. } => {
                    return Ok(BeforeToolUseData {
                        tool_call,
                        permission: result.permission,
                        additional_hints: vec![],
                    });
                }
                ToolCallDecision::Allow => {
                    if let Some(modified_arguments) = result.modified_arguments {
                        tool_call.arguments = modified_arguments;
                    }

                    if let Some(additional_content) = result.additional_content {
                        additional_hints.push(additional_content);
                    }
                }
            }
        }

        Ok(BeforeToolUseData {
            tool_call,
            permission: ToolCallDecision::Allow,
            additional_hints,
        })
    }

    pub async fn after_tool_use(
        &self,
        ctx: &HookContext,
        tool_result: ToolResult,
    ) -> Result<AfterToolUseData, HookError> {
        let mut additional_hints = vec![];
        let mut view = ToolResultView {
            result: tool_result,
        };

        for hook in self.after_tool_use_hooks.iter() {
            let result = tokio::select! {
                _ = ctx.cancel.cancelled() => {
                    return Err(HookError::Cancelled);
                }
                result = hook.transform(ctx, &view) => {
                    result?
                }
            };

            if let Some(modified) = result.modified {
                view.result = modified;
            }

            if let Some(additional_content) = result.additional_content {
                additional_hints.push(additional_content);
            }
        }

        Ok(AfterToolUseData {
            modified: view.result,
            additional_hints,
        })
    }
}

pub struct BeforeToolUseData {
    pub tool_call: ToolCall,
    pub permission: ToolCallDecision,
    pub additional_hints: Vec<UserContent>,
}

pub struct AfterToolUseData {
    pub modified: ToolResult,
    pub additional_hints: Vec<UserContent>,
}
