use ai::{AssistantMessage, UserContent};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStepEndReason {
    Success,
    Length,
    ToolUse,
    Error(String),
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionEndReason {
    Success,
    Error,
    Cancelled,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserMessageIntent {
    Normal,
    Steer,
    Followup,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    SessionStarted {
        session_id: Uuid,
    },
    SessionEnded {
        reason: SessionEndReason,
    },

    SystemPromptSet {
        prompt: String,
    },

    AgentTurnStarted {
        turn_id: Uuid,
    },

    AgentTurnEnded {
        turn_id: Uuid,
    },

    ModelStepStarted {
        step_id: Uuid,
    },

    ModelStepEnded {
        step_id: Uuid,
        reason: ModelStepEndReason,
        assistant_message: Option<AssistantMessage>,
    },

    UserMessageCommitted {
        content: UserContent,
        intent: UserMessageIntent,
    },

    AssistantMessageTextStarted {
        content_index: usize,
    },
    AssistantMessageTextDelta {
        content_index: usize,
        delta: String,
    },
    AssistantMessageTextCompleted {
        content_index: usize,
        text_content: String,
        assistant_message: AssistantMessage,
    },

    AssistantMessageThinkingStarted {
        content_index: usize,
    },
    AssistantMessageThinkingDelta {
        content_index: usize,
        delta: String,
    },
    AssistantMessageThinkingCompleted {
        content_index: usize,
        content: String,
    },

    ToolExecutionStarted {
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
    },
    ToolExecutionUpdated {
        tool_call_id: String,
        delta: String,
    },
    ToolExecutionEnded {
        tool_call_id: String,
        tool_name: String,
        result: String,
    },

    ErrorOccurred {
        message: String,
        details: Option<serde_json::Value>,
    },
}

impl AgentEvent {
    /// Returns the variant name of the event, e.g. `"SessionStarted"`.
    pub fn name(&self) -> &'static str {
        match self {
            AgentEvent::SessionStarted { .. } => "SessionStarted",
            AgentEvent::SessionEnded { .. } => "SessionEnded",
            AgentEvent::AgentTurnStarted { .. } => "AgentTurnStarted",
            AgentEvent::AgentTurnEnded { .. } => "AgentTurnEnded",
            AgentEvent::ModelStepStarted { .. } => "ModelStepStarted",
            AgentEvent::ModelStepEnded { .. } => "ModelStepEnded",
            AgentEvent::UserMessageCommitted { .. } => "UserMessageCommitted",
            AgentEvent::AssistantMessageTextStarted { .. } => "AssistantMessageTextStarted",
            AgentEvent::AssistantMessageTextDelta { .. } => "AssistantMessageTextDelta",
            AgentEvent::AssistantMessageTextCompleted { .. } => "AssistantMessageTextCompleted",
            AgentEvent::AssistantMessageThinkingStarted { .. } => "AssistantMessageThinkingStarted",
            AgentEvent::AssistantMessageThinkingDelta { .. } => "AssistantMessageThinkingDelta",
            AgentEvent::AssistantMessageThinkingCompleted { .. } => {
                "AssistantMessageThinkingCompleted"
            }
            AgentEvent::ToolExecutionStarted { .. } => "ToolExecutionStarted",
            AgentEvent::ToolExecutionUpdated { .. } => "ToolExecutionUpdated",
            AgentEvent::ToolExecutionEnded { .. } => "ToolExecutionEnded",
            AgentEvent::ErrorOccurred { .. } => "ErrorOccurred",
            AgentEvent::SystemPromptSet { .. } => "SystemPromptSet",
        }
    }
}

impl Hash for AgentEvent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Some variants carry `serde_json::Value`/`Map`, which don't implement
        // `Hash`. Hash a canonical serialized form so the whole event content
        // contributes deterministically (used for the integrity hash chain).
        match serde_json::to_vec(self) {
            Ok(bytes) => bytes.hash(state),
            // Serialization of an `AgentEvent` never fails in practice; fall
            // back to the variant name to keep `Hash` total.
            Err(_) => self.name().hash(state),
        }
    }
}

/// Formats a UUID showing only its trailing hex digits.
///
/// IDs in this crate are generated with `Uuid::now_v7`, which packs a
/// millisecond timestamp into the leading bits. Events emitted close
/// together therefore share a long, uninformative common prefix; only the
/// tail carries enough entropy to tell IDs apart at a glance in logs.
fn short_uuid(id: &Uuid) -> impl std::fmt::Display {
    let s = id.simple().to_string();
    format!("…{}", &s[s.len() - 8..])
}

fn short_string(s: &str) -> impl std::fmt::Display {
    if s.len() <= 32 {
        return s.to_owned();
    }
    format!(
        "…{}",
        s.chars()
            .rev()
            .take(32)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    )
}

impl std::fmt::Display for AgentEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentEvent::SystemPromptSet { prompt } => {
                write!(f, "SystemPromptSet(prompt={})", short_string(prompt))
            }
            AgentEvent::SessionStarted { session_id } => {
                write!(f, "SessionStarted(session_id={})", short_uuid(session_id))
            }
            AgentEvent::SessionEnded { reason } => {
                write!(f, "SessionEnded(reason={reason:?})")
            }
            AgentEvent::AgentTurnStarted { turn_id } => {
                write!(f, "AgentTurnStarted(turn_id={})", short_uuid(turn_id))
            }
            AgentEvent::AgentTurnEnded { turn_id } => {
                write!(f, "AgentTurnEnded(turn_id={})", short_uuid(turn_id))
            }
            AgentEvent::ModelStepStarted { step_id } => {
                write!(f, "ModelStepStarted(step_id={})", short_uuid(step_id))
            }
            AgentEvent::ModelStepEnded { step_id, reason, .. } => {
                write!(f, "ModelStepEnded(step_id={}, reason={reason:?})", short_uuid(step_id))
            }
            AgentEvent::UserMessageCommitted {
                intent, ..
            } => {
                write!(
                    f,
                    "UserMessageCommitted(intent={intent:?})",
                )
            }
            AgentEvent::AssistantMessageTextStarted { content_index } => {
                write!(
                    f,
                    "AssistantMessageTextStarted(#{content_index})",
                )
            }
            AgentEvent::AssistantMessageTextDelta { content_index, delta } => {
                write!(
                    f,
                    "AssistantMessageTextDelta(#{content_index}, delta={})",
                    short_string(delta)
                )
            }
            AgentEvent::AssistantMessageTextCompleted {
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
            AgentEvent::AssistantMessageThinkingStarted { content_index } => {
                write!(
                    f,
                    "AssistantMessageThinkingStarted(#{content_index})",
                )
            }
            AgentEvent::AssistantMessageThinkingDelta { content_index, delta } => {
                write!(
                    f,
                    "AssistantMessageThinkingDelta(#{content_index}, delta={delta:?})",
                )
            }
            AgentEvent::AssistantMessageThinkingCompleted {
                content_index,
                content,
            } => {
                write!(
                    f,
                    "AssistantMessageThinkingCompleted(#{content_index}, content={})",
                    short_string(content)
                )
            }
            AgentEvent::ToolExecutionStarted { tool_call_id, tool_name, arguments } => {
                write!(f, "ToolExecutionStarted(tool_call_id={tool_call_id}, tool_name={tool_name}, arguments={arguments:?})")
            }
            AgentEvent::ToolExecutionUpdated {
                tool_call_id,
                delta,
            } => {
                write!(
                    f,
                    "ToolExecutionUpdated(tool_call_id={tool_call_id}, delta={delta:?})"
                )
            }
            AgentEvent::ToolExecutionEnded {
                tool_call_id,
                tool_name,
                result,
            } => {
                write!(
                    f,
                    "ToolExecutionEnded(tool_call_id={tool_call_id}, tool_name={tool_name}, result={result})"
                )
            }
            AgentEvent::ErrorOccurred { message, details } => {
                write!(f, "ErrorOccurred(message={message:?}, details={details:?})")
            }
        }
    }
}
