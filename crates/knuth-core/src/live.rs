use ai::AssistantMessage;
use serde::{Deserialize, Serialize};

use crate::events::short_string;
use crate::ids::*;

/// Ephemeral progress notifications for subscribers.
///
/// Live events are fan-out only: they are never appended to the event store,
/// never contribute to the hash chain, and never feed the conversation
/// projection. Everything they report is either a partial view of state that a
/// durable [`crate::AgentEvent`] records in full, or presentation detail that
/// no replay needs. Dropping every live event must leave a session replayable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum LiveEvent {
    AssistantMessageTextStarted {
        step_id: StepId,
        content_index: usize,
    },
    AssistantMessageTextDelta {
        step_id: StepId,
        content_index: usize,
        delta: String,
    },
    AssistantMessageTextCompleted {
        step_id: StepId,
        content_index: usize,
        text_content: String,
        assistant_message: AssistantMessage,
    },

    AssistantMessageThinkingStarted {
        step_id: StepId,
        content_index: usize,
    },
    AssistantMessageThinkingDelta {
        step_id: StepId,
        content_index: usize,
        delta: String,
    },
    AssistantMessageThinkingCompleted {
        step_id: StepId,
        content_index: usize,
        content: String,
    },

    ToolExecutionStarted {
        step_id: StepId,
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
    },
    ToolExecutionUpdated {
        step_id: StepId,
        tool_call_id: String,
        delta: String,
    },
    ToolExecutionEnded {
        step_id: StepId,
        tool_call_id: String,
        tool_name: String,
        result: String,
    },
}

impl LiveEvent {
    /// Returns the variant name of the event, e.g. `"AssistantMessageTextDelta"`.
    pub fn name(&self) -> &'static str {
        match self {
            LiveEvent::AssistantMessageTextStarted { .. } => "AssistantMessageTextStarted",
            LiveEvent::AssistantMessageTextDelta { .. } => "AssistantMessageTextDelta",
            LiveEvent::AssistantMessageTextCompleted { .. } => "AssistantMessageTextCompleted",
            LiveEvent::AssistantMessageThinkingStarted { .. } => "AssistantMessageThinkingStarted",
            LiveEvent::AssistantMessageThinkingDelta { .. } => "AssistantMessageThinkingDelta",
            LiveEvent::AssistantMessageThinkingCompleted { .. } => {
                "AssistantMessageThinkingCompleted"
            }
            LiveEvent::ToolExecutionStarted { .. } => "ToolExecutionStarted",
            LiveEvent::ToolExecutionUpdated { .. } => "ToolExecutionUpdated",
            LiveEvent::ToolExecutionEnded { .. } => "ToolExecutionEnded",
        }
    }

    pub fn step_id(&self) -> StepId {
        match self {
            LiveEvent::AssistantMessageTextStarted { step_id, .. }
            | LiveEvent::AssistantMessageTextDelta { step_id, .. }
            | LiveEvent::AssistantMessageTextCompleted { step_id, .. }
            | LiveEvent::AssistantMessageThinkingStarted { step_id, .. }
            | LiveEvent::AssistantMessageThinkingDelta { step_id, .. }
            | LiveEvent::AssistantMessageThinkingCompleted { step_id, .. }
            | LiveEvent::ToolExecutionStarted { step_id, .. }
            | LiveEvent::ToolExecutionUpdated { step_id, .. }
            | LiveEvent::ToolExecutionEnded { step_id, .. } => *step_id,
        }
    }
}

impl std::fmt::Display for LiveEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveEvent::AssistantMessageTextStarted {
                step_id,
                content_index,
            } => {
                write!(
                    f,
                    "AssistantMessageTextStarted(step_id={}, #{content_index})",
                    step_id.short()
                )
            }
            LiveEvent::AssistantMessageTextDelta {
                step_id,
                content_index,
                delta,
            } => {
                write!(
                    f,
                    "AssistantMessageTextDelta(step_id={}, #{content_index}, delta={})",
                    step_id.short(),
                    short_string(delta)
                )
            }
            LiveEvent::AssistantMessageTextCompleted {
                content_index,
                text_content,
                ..
            } => {
                write!(
                    f,
                    "AssistantMessageTextCompleted(#{content_index}, text_content={})",
                    short_string(text_content)
                )
            }
            LiveEvent::AssistantMessageThinkingStarted {
                step_id,
                content_index,
            } => {
                write!(
                    f,
                    "AssistantMessageThinkingStarted(step_id={}, #{content_index})",
                    step_id.short()
                )
            }
            LiveEvent::AssistantMessageThinkingDelta {
                step_id,
                content_index,
                delta,
            } => {
                write!(
                    f,
                    "AssistantMessageThinkingDelta(step_id={}, #{content_index}, delta={delta:?})",
                    step_id.short(),
                )
            }
            LiveEvent::AssistantMessageThinkingCompleted {
                step_id,
                content_index,
                content,
            } => {
                write!(
                    f,
                    "AssistantMessageThinkingCompleted(step_id={}, #{content_index}, content={})",
                    step_id.short(),
                    short_string(content)
                )
            }
            LiveEvent::ToolExecutionStarted {
                step_id,
                tool_call_id,
                tool_name,
                arguments,
            } => {
                write!(
                    f,
                    "ToolExecutionStarted(step_id={}, tool_call_id={tool_call_id}, tool_name={tool_name}, arguments={arguments:?})",
                    step_id.short(),
                )
            }
            LiveEvent::ToolExecutionUpdated {
                step_id,
                tool_call_id,
                delta,
            } => {
                write!(
                    f,
                    "ToolExecutionUpdated(step_id={}, tool_call_id={tool_call_id}, delta={delta:?})",
                    step_id.short(),
                )
            }
            LiveEvent::ToolExecutionEnded {
                step_id,
                tool_call_id,
                tool_name,
                result,
            } => {
                write!(
                    f,
                    "ToolExecutionEnded(step_id={}, tool_call_id={tool_call_id}, tool_name={tool_name}, result={})",
                    step_id.short(),
                    short_string(result)
                )
            }
        }
    }
}
