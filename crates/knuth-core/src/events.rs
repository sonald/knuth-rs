use ai::{AssistantMessage, UserContent};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::ids::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolOutcome {
    ExecSuccess,
    Denied,
    Cancelled,
    Interrupted,
    Error,
}

impl ToolOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self, ToolOutcome::ExecSuccess)
    }
}

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

/// The durable session history.
///
/// Every variant is appended to the event store, joins the hash chain, and is
/// replayable: the conversation sent to the model on the next step is derived
/// from these events alone. Streaming progress that only matters to a live
/// observer belongs in [`crate::LiveEvent`] instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    SessionStarted {
        session_id: SessionId,
    },
    SessionEnded {
        reason: SessionEndReason,
    },

    SystemPromptSet {
        prompt: String,
    },

    // only one agent turn can be active at a time
    AgentTurnStarted {
        turn_id: TurnId,
    },

    AgentTurnEnded {
        turn_id: TurnId,
    },

    ModelStepStarted {
        step_id: StepId,
    },

    ModelStepEnded {
        step_id: StepId,
        reason: ModelStepEndReason,
        assistant_message: Option<AssistantMessage>,
    },

    UserMessageCommitted {
        message_id: MessageId, // for idempotency
        content: UserContent,
        intent: UserMessageIntent,
    },

    ErrorOccurred {
        message: String,
        details: Option<serde_json::Value>,
    },

    ToolExecutionRequested {
        invocation_id: ToolInvocationId,
        step_id: StepId,
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
    },

    /// Terminal record of the invocation opened by `ToolExecutionRequested`;
    /// carries the payload that becomes the tool-result message in the
    /// conversation.
    ToolResultReceived {
        invocation_id: ToolInvocationId,
        tool_call_id: String,
        tool_name: String,
        outcome: ToolOutcome,
        content: Vec<u8>,
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
            AgentEvent::ToolExecutionRequested { .. } => "ToolExecutionRequested",
            AgentEvent::ToolResultReceived { .. } => "ToolResultReceived",
            AgentEvent::ErrorOccurred { .. } => "ErrorOccurred",
            AgentEvent::SystemPromptSet { .. } => "SystemPromptSet",
        }
    }

    pub fn event_type(&self) -> &'static str {
        match self {
            AgentEvent::SessionStarted { .. } => "session.started",
            AgentEvent::SessionEnded { .. } => "session.ended",
            AgentEvent::AgentTurnStarted { .. } => "turn.started",
            AgentEvent::AgentTurnEnded { .. } => "turn.ended",
            AgentEvent::ModelStepStarted { .. } => "model_step.started",
            AgentEvent::ModelStepEnded { .. } => "model_step.ended",
            AgentEvent::UserMessageCommitted { .. } => "user_message.committed",
            AgentEvent::ErrorOccurred { .. } => "error.occurred",
            AgentEvent::SystemPromptSet { .. } => "system_prompt.set",
            AgentEvent::ToolExecutionRequested { .. } => "tool.execution_requested",
            AgentEvent::ToolResultReceived { .. } => "tool.result_received",
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

pub(crate) fn short_string(s: &str) -> impl std::fmt::Display {
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
                write!(f, "SessionStarted(session_id={})", session_id.short())
            }
            AgentEvent::SessionEnded { reason } => {
                write!(f, "SessionEnded(reason={reason:?})")
            }
            AgentEvent::AgentTurnStarted { turn_id } => {
                write!(f, "AgentTurnStarted(turn_id={})", turn_id.short())
            }
            AgentEvent::AgentTurnEnded { turn_id } => {
                write!(f, "AgentTurnEnded(turn_id={})", turn_id.short())
            }
            AgentEvent::ModelStepStarted { step_id } => {
                write!(f, "ModelStepStarted(step_id={})", step_id.short())
            }
            AgentEvent::ModelStepEnded {
                step_id, reason, ..
            } => {
                write!(
                    f,
                    "ModelStepEnded(step_id={}, reason={reason:?})",
                    step_id.short()
                )
            }
            AgentEvent::UserMessageCommitted { intent, .. } => {
                write!(f, "UserMessageCommitted(intent={intent:?})",)
            }
            AgentEvent::ToolExecutionRequested {
                invocation_id,
                step_id,
                tool_call_id,
                tool_name,
                ..
            } => {
                write!(
                    f,
                    "ToolExecutionRequested(invocation_id={}, step_id={}, tool_call_id={tool_call_id}, tool_name={tool_name})",
                    invocation_id.short(),
                    step_id.short()
                )
            }
            AgentEvent::ToolResultReceived {
                invocation_id,
                tool_call_id,
                tool_name,
                outcome,
                content,
            } => {
                write!(
                    f,
                    "ToolResultReceived(invocation_id={}, tool_call_id={tool_call_id}, tool_name={tool_name}, outcome={outcome:?}, content={})",
                    invocation_id.short(),
                    short_string(&String::from_utf8_lossy(content))
                )
            }
            AgentEvent::ErrorOccurred { message, details } => {
                write!(f, "ErrorOccurred(message={message:?}, details={details:?})")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_tool_events_carry_stable_event_types() {
        let invocation_id = ToolInvocationId::new();
        let requested = AgentEvent::ToolExecutionRequested {
            invocation_id,
            step_id: StepId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            arguments: serde_json::Map::new(),
        };
        let received = AgentEvent::ToolResultReceived {
            invocation_id,
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            outcome: ToolOutcome::ExecSuccess,
            content: b"ok".to_vec(),
        };

        assert_eq!(requested.event_type(), "tool.execution_requested");
        assert_eq!(received.event_type(), "tool.result_received");
    }

    #[test]
    fn agent_event_round_trips_through_serde() {
        let event = AgentEvent::ToolResultReceived {
            invocation_id: ToolInvocationId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            outcome: ToolOutcome::Error,
            content: b"boom".to_vec(),
        };

        let json = serde_json::to_string(&event).unwrap();
        let decoded: AgentEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.name(), "ToolResultReceived");
    }
}
