use ai::{AssistantMessage, UserContent};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::ids::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolOutcome {
    ExecSuccess,
    Denied,
    Cancelled,
    Interrupted,
    Error,
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
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
    },

    ToolResultReceived {
        outcome: ToolOutcome,
        content: Vec<u8>,
    },

    // Live events
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
            AgentEvent::ToolExecutionRequested { .. } => "ToolExecutionRequested",
            AgentEvent::ToolResultReceived { .. } => "ToolResultReceived",
            AgentEvent::ErrorOccurred { .. } => "ErrorOccurred",
            AgentEvent::SystemPromptSet { .. } => "SystemPromptSet",
        }
    }

    pub fn is_durable(&self) -> bool {
        matches! {
            self,
            AgentEvent::SessionStarted { .. } |
            AgentEvent::SessionEnded { .. } |
            AgentEvent::AgentTurnStarted { .. } |
            AgentEvent::AgentTurnEnded { .. } |
            AgentEvent::ModelStepStarted { .. } |
            AgentEvent::ModelStepEnded { .. } |
            AgentEvent::UserMessageCommitted { .. } |
            AgentEvent::ErrorOccurred { .. } |
            AgentEvent::SystemPromptSet { .. } |
            AgentEvent::ToolExecutionRequested { .. } |
            AgentEvent::ToolResultReceived { .. }
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

            AgentEvent::AssistantMessageTextStarted { .. } => "live",
            AgentEvent::AssistantMessageTextDelta { .. } => "live",
            AgentEvent::AssistantMessageTextCompleted { .. } => "live",
            AgentEvent::AssistantMessageThinkingStarted { .. } => "live",
            AgentEvent::AssistantMessageThinkingDelta { .. } => "live",
            AgentEvent::AssistantMessageThinkingCompleted { .. } => "live",
            AgentEvent::ToolExecutionEnded { .. } => "live",
            AgentEvent::ToolExecutionStarted { .. } => "live",
            AgentEvent::ToolExecutionUpdated { .. } => "live",
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
            AgentEvent::AssistantMessageTextStarted {
                step_id,
                content_index,
            } => {
                write!(
                    f,
                    "AssistantMessageTextStarted(step_id={}, #{content_index})",
                    step_id.short()
                )
            }
            AgentEvent::AssistantMessageTextDelta {
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
            AgentEvent::AssistantMessageThinkingStarted {
                step_id,
                content_index,
            } => {
                write!(
                    f,
                    "AssistantMessageThinkingStarted(step_id={}, #{content_index})",
                    step_id.short()
                )
            }
            AgentEvent::AssistantMessageThinkingDelta {
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
            AgentEvent::AssistantMessageThinkingCompleted {
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
            AgentEvent::ToolExecutionStarted {
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
            AgentEvent::ToolExecutionUpdated {
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
            AgentEvent::ToolExecutionRequested {
                invocation_id,
                tool_call_id,
                tool_name,
                ..
            } => {
                write!(
                    f,
                    "ToolExecutionRequested(invocation_id={invocation_id}, tool_call_id={tool_call_id}, tool_name={tool_name})"
                )
            }
            AgentEvent::ToolResultReceived { outcome, content } => {
                write!(f, "ToolResultReceived(outcome={outcome:?}, content={content:?})")
            }
            AgentEvent::ToolExecutionEnded {
                step_id,
                tool_call_id,
                tool_name,
                result,
            } => {
                write!(
                    f,
                    "ToolExecutionEnded(step_id={}, tool_call_id={tool_call_id}, tool_name={tool_name}, result={result})",
                    step_id.short(),
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
    fn durable_tool_events_are_distinct_from_live_progress() {
        let requested = AgentEvent::ToolExecutionRequested {
            invocation_id: ToolInvocationId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            arguments: serde_json::Map::new(),
        };
        let completed = AgentEvent::ToolResultReceived {
            outcome: ToolOutcome::ExecSuccess,
            content: b"ok".to_vec(),
        };
        let live_ended = AgentEvent::ToolExecutionEnded {
            step_id: StepId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            result: "ok".to_string(),
        };
        let progress = AgentEvent::AssistantMessageTextDelta {
            step_id: StepId::new(),
            content_index: 0,
            delta: "partial".to_string(),
        };

        assert!(requested.is_durable());
        assert_eq!(requested.event_type(), "tool.execution_requested");
        assert!(completed.is_durable());
        assert_eq!(completed.event_type(), "tool.result_received");
        assert!(!live_ended.is_durable());
        assert_eq!(live_ended.event_type(), "live");
        assert!(!progress.is_durable());
        assert_eq!(progress.event_type(), "live");
    }
}
