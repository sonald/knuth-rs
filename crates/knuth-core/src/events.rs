use ai::{AssistantMessage, UserContent};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnEndReason {
    Success,
    Error,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionEndReason {
    Success,
    Error,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnOutcome {
    pub turn_id: Uuid,
    pub reason: TurnEndReason,
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

    AgentTurnStarted {
        turn_id: Uuid,
    },

    AgentTurnEnded {
        turn_id: Uuid,
        reason: TurnEndReason,
        assistant_message: Option<AssistantMessage>,
    },

    UserMessageCommitted {
        message_id: Uuid,
        content: UserContent,
        intent: UserMessageIntent,
    },

    AssistantMessageStarted {
        message_id: Uuid,
    },
    AssistantMessageTextDelta {
        message_id: Uuid,
        delta: String,
    },
    AssistantMessageCompleted {
        message_id: Uuid,
        text_content: String,
        assistant_message: AssistantMessage,
    },

    AssistantMessageThinkingStarted {
        message_id: Uuid,
    },
    AssistantMessageThinkingDelta {
        message_id: Uuid,
        delta: String,
    },
    AssistantMessageThinkingCompleted {
        message_id: Uuid,
        content: String,
    },

    ToolCallRequested {
        tool_call_id: String,
        name: String,
        #[serde(default)]
        arguments: serde_json::Map<String, serde_json::Value>,
    },
    ToolExecutionStarted {
        tool_call_id: String,
    },
    ToolExecutionUpdated {
        tool_call_id: String,
        delta: String,
    },
    ToolExecutionEnded {
        tool_call_id: String,
        result: serde_json::Value,
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
            AgentEvent::UserMessageCommitted { .. } => "UserMessageCommitted",
            AgentEvent::AssistantMessageStarted { .. } => "AssistantMessageStarted",
            AgentEvent::AssistantMessageTextDelta { .. } => "AssistantMessageTextDelta",
            AgentEvent::AssistantMessageCompleted { .. } => "AssistantMessageCompleted",
            AgentEvent::AssistantMessageThinkingStarted { .. } => "AssistantMessageThinkingStarted",
            AgentEvent::AssistantMessageThinkingDelta { .. } => "AssistantMessageThinkingDelta",
            AgentEvent::AssistantMessageThinkingCompleted { .. } => {
                "AssistantMessageThinkingCompleted"
            }
            AgentEvent::ToolCallRequested { .. } => "ToolCallRequested",
            AgentEvent::ToolExecutionStarted { .. } => "ToolExecutionStarted",
            AgentEvent::ToolExecutionUpdated { .. } => "ToolExecutionUpdated",
            AgentEvent::ToolExecutionEnded { .. } => "ToolExecutionEnded",
            AgentEvent::ErrorOccurred { .. } => "ErrorOccurred",
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

impl std::fmt::Display for AgentEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentEvent::SessionStarted { session_id } => {
                write!(f, "SessionStarted(session_id={})", short_uuid(session_id))
            }
            AgentEvent::SessionEnded { reason } => {
                write!(f, "SessionEnded(reason={reason:?})")
            }
            AgentEvent::AgentTurnStarted { turn_id } => {
                write!(f, "AgentTurnStarted(turn_id={})", short_uuid(turn_id))
            }
            AgentEvent::AgentTurnEnded { turn_id, reason , .. } => {
                write!(
                    f,
                    "AgentTurnEnded(turn_id={}, reason={reason:?})",
                    short_uuid(turn_id)
                )
            }
            AgentEvent::UserMessageCommitted {
                message_id, intent, ..
            } => {
                write!(
                    f,
                    "UserMessageCommitted(message_id={}, intent={intent:?})",
                    short_uuid(message_id)
                )
            }
            AgentEvent::AssistantMessageStarted { message_id } => {
                write!(
                    f,
                    "AssistantMessageStarted(message_id={})",
                    short_uuid(message_id)
                )
            }
            AgentEvent::AssistantMessageTextDelta { message_id, delta } => {
                write!(
                    f,
                    "AssistantMessageTextDelta(message_id={}, delta={delta:?})",
                    short_uuid(message_id)
                )
            }
            AgentEvent::AssistantMessageCompleted {
                message_id,
                text_content,
                ..
            } => {
                write!(
                    f,
                    "AssistantMessageCompleted(message_id={}, text_content={text_content:?})",
                    short_uuid(message_id)
                )
            }
            AgentEvent::AssistantMessageThinkingStarted { message_id } => {
                write!(
                    f,
                    "AssistantMessageThinkingStarted(message_id={})",
                    short_uuid(message_id)
                )
            }
            AgentEvent::AssistantMessageThinkingDelta { message_id, delta } => {
                write!(
                    f,
                    "AssistantMessageThinkingDelta(message_id={}, delta={delta:?})",
                    short_uuid(message_id)
                )
            }
            AgentEvent::AssistantMessageThinkingCompleted {
                message_id,
                content,
            } => {
                write!(
                    f,
                    "AssistantMessageThinkingCompleted(message_id={}, content={content:?})",
                    short_uuid(message_id)
                )
            }
            AgentEvent::ToolCallRequested {
                tool_call_id,
                name,
                arguments,
            } => {
                write!(
                    f,
                    "ToolCallRequested(tool_call_id={tool_call_id}, name={name}, arguments={arguments:?})"
                )
            }
            AgentEvent::ToolExecutionStarted { tool_call_id } => {
                write!(f, "ToolExecutionStarted(tool_call_id={tool_call_id})")
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
                result,
            } => {
                write!(
                    f,
                    "ToolExecutionEnded(tool_call_id={tool_call_id}, result={result})"
                )
            }
            AgentEvent::ErrorOccurred { message, details } => {
                write!(f, "ErrorOccurred(message={message:?}, details={details:?})")
            }
        }
    }
}
