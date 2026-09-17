use std::collections::HashMap;

use ai::UserContent;
use async_trait::async_trait;
/// default hooks for the knuth cli
use knuth_agent::{
    ToolInput,
    hooks::{AfterToolUseHook, AfterToolUseResult, HookContext, HookError, ToolResultView},
};
use knuth_core::ids::HookId;
use tokio::sync::Mutex;
use tracing::debug;

/// for debug purpose, log the tool count and warn if it exceeds the threshold
pub struct ToolCountHook {
    id: HookId,
    tool_count: Mutex<HashMap<String, usize>>,
    threshold: usize,
}

#[async_trait]
impl AfterToolUseHook for ToolCountHook {
    fn id(&self) -> HookId {
        self.id
    }

    async fn transform(
        &self,
        _: &HookContext,
        current: &ToolResultView,
    ) -> Result<AfterToolUseResult, HookError> {
        if self.threshold == 0 {
            return Ok(AfterToolUseResult {
                modified: None,
                additional_content: None,
            });
        }

        let mut tool_count = self.tool_count.lock().await;
        tool_count
            .entry(current.tool_call.name.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);

        let total_count = tool_count.values().sum::<usize>();
        if total_count % self.threshold == 0 && total_count > 0 {
            debug!(
                "-------------tool use count------------------\nTool count: {:#?}",
                tool_count
            );
        }

        Ok(AfterToolUseResult {
            modified: None,
            additional_content: None,
        })
    }
}


impl ToolCountHook {
    pub fn new(threshold: usize) -> Self {
        Self {
            id: HookId::new(),
            tool_count: Mutex::new(HashMap::new()),
            threshold,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct ToolCallStatKey {
    pub name: String,
    pub arguments: ToolInput,
}

pub struct RepeatedToolCallWarning {
    id: HookId,
    threshold: usize,
    stats: Mutex<HashMap<ToolCallStatKey, usize>>,
}

#[async_trait]
impl AfterToolUseHook for RepeatedToolCallWarning {
    fn id(&self) -> HookId {
        self.id
    }

    async fn transform(
        &self,
        _: &HookContext,
        current: &ToolResultView,
    ) -> Result<AfterToolUseResult, HookError> {
        let mut stats = self.stats.lock().await;
        let key = ToolCallStatKey {
            name: current.tool_call.name.clone(),
            arguments: current.tool_call.arguments.clone(),
        };
        let count = *stats
            .entry(key.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);

        if count >= self.threshold {
            let warning = format!(
                "You have called {} {} times with exactly the same arguments. \
                    This is likely a mistake. Please check your code and try again.",
                key.name, count
            );

            debug!("{}", warning);
            return Ok(AfterToolUseResult {
                modified: None,
                additional_content: Some(UserContent::Text(warning)),
            });
        }

        Ok(AfterToolUseResult {
            modified: None,
            additional_content: None,
        })
    }
}

impl RepeatedToolCallWarning {
    pub fn new(threshold: usize) -> Self {
        Self {
            id: HookId::new(),
            threshold: threshold,
            stats: Mutex::new(HashMap::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai::ToolCall;
    use knuth_agent::ToolResult;
    use knuth_core::{ToolOutcome, ids::SessionId};
    use serde_json::Map;
    use std::path::PathBuf;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    fn context() -> HookContext {
        HookContext {
            session_id: SessionId::new(),
            invocation_id: None,
            workspace: PathBuf::from("."),
            cancel: CancellationToken::new(),
        }
    }

    fn bash_result(command: &str) -> ToolResultView {
        let mut arguments = Map::new();
        arguments.insert("command".into(), serde_json::Value::String(command.into()));
        ToolResultView {
            tool_call: ToolCall {
                id: "call-1".into(),
                name: "bash".into(),
                arguments,
                thought_signature: None,
            },
            result: ToolResult {
                outcome: ToolOutcome::ExecSuccess,
                content: "ok".into(),
            },
        }
    }

    /// A hook that re-locks its own mutex inside `transform` deadlocks the
    /// session: `after_tool_use` awaits the hook inline, so a hang here wedges
    /// the whole turn. Every call must return.
    ///
    /// The log line is only reached on every `threshold`-th call, so the loop
    /// has to run past the first threshold to exercise it.
    #[tokio::test]
    async fn tool_count_hook_does_not_deadlock_on_log_threshold() {
        let hook = ToolCountHook::new(2);
        let ctx = context();

        for call in 0..4 {
            let view = bash_result("ls");
            tokio::time::timeout(Duration::from_secs(1), hook.transform(&ctx, &view))
                .await
                .unwrap_or_else(|_| panic!("transform deadlocked on call {}", call + 1))
                .unwrap();
        }
    }

    /// Same as above, but with `RUST_LOG=debug` in effect. `tracing` only
    /// evaluates a macro's arguments once the callsite is enabled, so the
    /// deadlock hides unless a debug subscriber is installed.
    #[tokio::test]
    async fn tool_count_hook_does_not_deadlock_when_debug_logging_is_on() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        let hook = ToolCountHook::new(2);
        let ctx = context();

        for call in 0..4 {
            let view = bash_result("ls");
            tokio::time::timeout(Duration::from_secs(1), hook.transform(&ctx, &view))
                .await
                .unwrap_or_else(|_| panic!("transform deadlocked on call {}", call + 1))
                .unwrap();
        }
    }
}
