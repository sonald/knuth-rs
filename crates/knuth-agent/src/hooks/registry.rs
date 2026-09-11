use crate::{
    ToolResult,
    hooks::{
        AfterToolUseHook, BeforeToolUseHook, ContextHook, ContextPatch, ContextView, HookContext,
        HookError, InputHook, SessionEndHook, SessionStartHook, ToolCallDecision, ToolResultView,
    },
};
use ai::{Context, ToolCall, UserContent};
use std::sync::Arc;

#[derive(Default)]
pub struct HookRegistry {
    input_hooks: Vec<Arc<dyn InputHook>>,
    context_hooks: Vec<Arc<dyn ContextHook>>,
    before_tool_use_hooks: Vec<Arc<dyn BeforeToolUseHook>>,
    after_tool_use_hooks: Vec<Arc<dyn AfterToolUseHook>>,
    session_start_hooks: Vec<Arc<dyn SessionStartHook>>,
    session_end_hooks: Vec<Arc<dyn SessionEndHook>>,
}

pub enum Hook {
    Input(Arc<dyn InputHook>),
    Context(Arc<dyn ContextHook>),
    BeforeToolUse(Arc<dyn BeforeToolUseHook>),
    AfterToolUse(Arc<dyn AfterToolUseHook>),
    SessionStart(Arc<dyn SessionStartHook>),
    SessionEnd(Arc<dyn SessionEndHook>),
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, hook: Hook) {
        match hook {
            Hook::Input(hook) => self.input_hooks.push(hook),
            Hook::Context(hook) => self.context_hooks.push(hook),
            Hook::BeforeToolUse(hook) => self.before_tool_use_hooks.push(hook),
            Hook::AfterToolUse(hook) => self.after_tool_use_hooks.push(hook),
            Hook::SessionStart(hook) => self.session_start_hooks.push(hook),
            Hook::SessionEnd(hook) => self.session_end_hooks.push(hook),
        }
    }

    pub async fn context(
        &self,
        ctx: &HookContext,
        view: &ContextView<'_>,
    ) -> Result<ContextPatch, HookError> {
        let mut merged = ContextPatch::default();

        for hook in self.context_hooks.iter() {
            let patch = tokio::select! {
                _ = ctx.cancel.cancelled() => {
                    return Err(HookError::Cancelled);
                }
                result = hook.transform(ctx, view) => {
                    result?
                }
            };

            merged.edits.extend(patch.edits);
            merged.hints.extend(patch.hints);
        }
        Ok(merged)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolResult;
    use crate::hooks::{AfterToolUseResult, BeforeToolUseResult, HookContext, ToolCallDecision};
    use ai::ToolCall;
    use async_trait::async_trait;
    use knuth_core::{
        ToolOutcome,
        ids::{HookId, SessionId, ToolInvocationId},
    };
    use serde_json::Map;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio_util::sync::CancellationToken;

    fn ctx(cancel: CancellationToken) -> HookContext {
        HookContext {
            session_id: SessionId::new(),
            invocation_id: Some(ToolInvocationId::new()),
            cancel,
        }
    }

    fn bash_call(command: &str) -> ToolCall {
        let mut arguments = Map::new();
        arguments.insert(
            "command".into(),
            serde_json::Value::String(command.to_string()),
        );
        ToolCall {
            id: "call-1".into(),
            name: "bash".into(),
            arguments,
            thought_signature: None,
        }
    }

    struct RewriteCommandHook {
        id: HookId,
        to: String,
    }

    #[async_trait]
    impl BeforeToolUseHook for RewriteCommandHook {
        fn id(&self) -> HookId {
            self.id
        }

        async fn transform(
            &self,
            _ctx: &HookContext,
            _tool_call: &ToolCall,
        ) -> Result<BeforeToolUseResult, HookError> {
            let mut arguments = Map::new();
            arguments.insert("command".into(), serde_json::Value::String(self.to.clone()));
            Ok(BeforeToolUseResult {
                modified_arguments: Some(arguments),
                permission: ToolCallDecision::Allow,
                additional_content: None,
            })
        }
    }

    struct DenyHook {
        id: HookId,
        reason: String,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl BeforeToolUseHook for DenyHook {
        fn id(&self) -> HookId {
            self.id
        }

        async fn transform(
            &self,
            _ctx: &HookContext,
            _tool_call: &ToolCall,
        ) -> Result<BeforeToolUseResult, HookError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(BeforeToolUseResult {
                modified_arguments: None,
                permission: ToolCallDecision::Deny {
                    reason: self.reason.clone(),
                },
                additional_content: None,
            })
        }
    }

    struct CountingBeforeHook {
        id: HookId,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl BeforeToolUseHook for CountingBeforeHook {
        fn id(&self) -> HookId {
            self.id
        }

        async fn transform(
            &self,
            _ctx: &HookContext,
            _tool_call: &ToolCall,
        ) -> Result<BeforeToolUseResult, HookError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(BeforeToolUseResult {
                modified_arguments: None,
                permission: ToolCallDecision::Allow,
                additional_content: None,
            })
        }
    }

    struct PrefixResultHook {
        id: HookId,
        prefix: String,
    }

    #[async_trait]
    impl AfterToolUseHook for PrefixResultHook {
        fn id(&self) -> HookId {
            self.id
        }

        async fn transform(
            &self,
            _ctx: &HookContext,
            current: &ToolResultView,
        ) -> Result<AfterToolUseResult, HookError> {
            Ok(AfterToolUseResult {
                modified: Some(ToolResult {
                    outcome: current.result.outcome.clone(),
                    content: format!("{}{}", self.prefix, current.result.content),
                }),
                additional_content: None,
            })
        }
    }

    struct PendingBeforeHook {
        id: HookId,
    }

    #[async_trait]
    impl BeforeToolUseHook for PendingBeforeHook {
        fn id(&self) -> HookId {
            self.id
        }

        async fn transform(
            &self,
            _ctx: &HookContext,
            _tool_call: &ToolCall,
        ) -> Result<BeforeToolUseResult, HookError> {
            std::future::pending::<()>().await;
            unreachable!("pending future resolved")
        }
    }

    #[tokio::test]
    async fn empty_registry_allows_the_original_call() {
        let registry = HookRegistry::new();
        let data = registry
            .before_tool_use(&ctx(CancellationToken::new()), bash_call("printf hi"))
            .await
            .unwrap();

        assert!(matches!(data.permission, ToolCallDecision::Allow));
        assert_eq!(data.tool_call.arguments["command"], "printf hi");
        assert!(data.additional_hints.is_empty());
    }

    #[tokio::test]
    async fn before_hooks_can_rewrite_arguments() {
        let mut registry = HookRegistry::new();
        registry.register(Hook::BeforeToolUse(Arc::new(RewriteCommandHook {
            id: HookId::new(),
            to: "first".into(),
        })));
        registry.register(Hook::BeforeToolUse(Arc::new(RewriteCommandHook {
            id: HookId::new(),
            to: "second".into(),
        })));

        let data = registry
            .before_tool_use(&ctx(CancellationToken::new()), bash_call("original"))
            .await
            .unwrap();

        assert_eq!(data.tool_call.arguments["command"], "second");
    }

    #[tokio::test]
    async fn deny_stops_later_before_hooks() {
        let later_calls = Arc::new(AtomicUsize::new(0));
        let mut registry = HookRegistry::new();
        registry.register(Hook::BeforeToolUse(Arc::new(DenyHook {
            id: HookId::new(),
            reason: "nope".into(),
            calls: Arc::new(AtomicUsize::new(0)),
        })));
        registry.register(Hook::BeforeToolUse(Arc::new(CountingBeforeHook {
            id: HookId::new(),
            calls: Arc::clone(&later_calls),
        })));

        let data = registry
            .before_tool_use(&ctx(CancellationToken::new()), bash_call("printf hi"))
            .await
            .unwrap();

        assert!(matches!(
            data.permission,
            ToolCallDecision::Deny { reason } if reason == "nope"
        ));
        assert_eq!(later_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn after_hooks_can_rewrite_the_result() {
        let mut registry = HookRegistry::new();
        registry.register(Hook::AfterToolUse(Arc::new(PrefixResultHook {
            id: HookId::new(),
            prefix: "hooked:".into(),
        })));

        let data = registry
            .after_tool_use(
                &ctx(CancellationToken::new()),
                ToolResult {
                    outcome: ToolOutcome::ExecSuccess,
                    content: "ok".into(),
                },
            )
            .await
            .unwrap();

        assert_eq!(data.modified.content, "hooked:ok");
        assert_eq!(data.modified.outcome, ToolOutcome::ExecSuccess);
    }

    #[tokio::test]
    async fn before_hooks_surface_cancellation() {
        let mut registry = HookRegistry::new();
        registry.register(Hook::BeforeToolUse(Arc::new(PendingBeforeHook {
            id: HookId::new(),
        })));
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = registry
            .before_tool_use(&ctx(cancel), bash_call("printf hi"))
            .await;

        assert!(matches!(result, Err(HookError::Cancelled)));
    }
}
