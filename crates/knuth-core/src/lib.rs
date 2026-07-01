//! Core primitives for the Knuth project.

use ai::{ StopReason, Usage, UserContent};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;
use std::pin::Pin;
use std::task::{Context, Poll};
use futures_core::Stream;
use std::result::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use std::fmt::Debug;
use tracing::debug;



#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub enum UserMessageIntent {
    Normal,
    Steer,
    Followup
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
    },
    UserMessageReceived {
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
        content: String,
        usage: Usage,
        stop_reason: StopReason,
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
            AgentEvent::UserMessageReceived { .. } => "UserMessageReceived",
            AgentEvent::AssistantMessageStarted { .. } => "AssistantMessageStarted",
            AgentEvent::AssistantMessageTextDelta { .. } => "AssistantMessageTextDelta",
            AgentEvent::AssistantMessageCompleted { .. } => "AssistantMessageCompleted",
            AgentEvent::AssistantMessageThinkingStarted { .. } => "AssistantMessageThinkingStarted",
            AgentEvent::AssistantMessageThinkingDelta { .. } => "AssistantMessageThinkingDelta",
            AgentEvent::AssistantMessageThinkingCompleted { .. } => "AssistantMessageThinkingCompleted",
            AgentEvent::ToolCallRequested { .. } => "ToolCallRequested",
            AgentEvent::ToolExecutionStarted { .. } => "ToolExecutionStarted",
            AgentEvent::ToolExecutionUpdated { .. } => "ToolExecutionUpdated",
            AgentEvent::ToolExecutionEnded { .. } => "ToolExecutionEnded",
            AgentEvent::ErrorOccurred { .. } => "ErrorOccurred",
        }
    }
}

impl std::fmt::Display for AgentEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentEvent::SessionStarted { session_id } => {
                write!(f, "SessionStarted(session_id={session_id})")
            }
            AgentEvent::SessionEnded { reason } => {
                write!(f, "SessionEnded(reason={reason:?})")
            }
            AgentEvent::AgentTurnStarted { turn_id } => {
                write!(f, "AgentTurnStarted(turn_id={turn_id})")
            }
            AgentEvent::AgentTurnEnded { turn_id, reason } => {
                write!(f, "AgentTurnEnded(turn_id={turn_id}, reason={reason:?})")
            }
            AgentEvent::UserMessageReceived { message_id, intent, .. } => {
                write!(f, "UserMessageReceived(message_id={message_id}, intent={intent:?})")
            }
            AgentEvent::AssistantMessageStarted { message_id } => {
                write!(f, "AssistantMessageStarted(message_id={message_id})")
            }
            AgentEvent::AssistantMessageTextDelta { message_id, delta } => {
                write!(f, "AssistantMessageTextDelta(message_id={message_id}, delta={delta:?})")
            }
            AgentEvent::AssistantMessageCompleted { message_id, content, usage, stop_reason } => {
                write!(
                    f,
                    "AssistantMessageCompleted(message_id={message_id}, content={content:?}, usage={usage:?}, stop_reason={stop_reason:?})"
                )
            }
            AgentEvent::AssistantMessageThinkingStarted { message_id } => {
                write!(f, "AssistantMessageThinkingStarted(message_id={message_id})")
            }
            AgentEvent::AssistantMessageThinkingDelta { message_id, delta } => {
                write!(f, "AssistantMessageThinkingDelta(message_id={message_id}, delta={delta:?})")
            }
            AgentEvent::AssistantMessageThinkingCompleted { message_id, content } => {
                write!(f, "AssistantMessageThinkingCompleted(message_id={message_id}, content={content:?})")
            }
            AgentEvent::ToolCallRequested { tool_call_id, name, arguments } => {
                write!(f, "ToolCallRequested(tool_call_id={tool_call_id}, name={name}, arguments={arguments:?})")
            }
            AgentEvent::ToolExecutionStarted { tool_call_id } => {
                write!(f, "ToolExecutionStarted(tool_call_id={tool_call_id})")
            }
            AgentEvent::ToolExecutionUpdated { tool_call_id, delta } => {
                write!(f, "ToolExecutionUpdated(tool_call_id={tool_call_id}, delta={delta:?})")
            }
            AgentEvent::ToolExecutionEnded { tool_call_id, result } => {
                write!(f, "ToolExecutionEnded(tool_call_id={tool_call_id}, result={result})")
            }
            AgentEvent::ErrorOccurred { message, details } => {
                write!(f, "ErrorOccurred(message={message:?}, details={details:?})")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEvent {
    pub event: AgentEvent,
    pub timestamp: DateTime<Utc>,
    pub stream_seq: u64, // monotonically increasing sequence number for the stream (session)
    pub id: Uuid,
    pub hash: String, // used for integrity checks and auditability
    pub parent_hash: Option<String>, // form a hash chain
}

pub struct AgentSubscription {
    rx: mpsc::Receiver<AgentEvent>,
    pub id: Uuid,
}

impl AgentSubscription {
    pub fn new(rx: mpsc::Receiver<AgentEvent>) -> Self {
        Self { rx, id: Uuid::now_v7() }
    }
}

impl Stream for AgentSubscription {
    type Item = AgentEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EventStoreError {
    #[error("Failed to append event: {0}")]
    AppendFailed(String),
    #[error("Failed to range events: {0}")]
    RangeFailed(String),
}

#[async_trait]
pub trait EventStore: Send + Sync + Debug {
    async fn append(&self, event: AgentEvent) -> Result<StoredEvent, EventStoreError>;
    async fn range(&self, from_seq: u64, limit: usize) -> Result<Vec<StoredEvent>, EventStoreError>;
}

#[derive(Debug)]
pub struct InMemoryEventStore {
    events: Arc<Mutex<Vec<StoredEvent>>>,
}

/// use for debug now 
impl InMemoryEventStore {
    pub fn new() -> Self {
        Self { events: Arc::new(Mutex::new(Vec::new())) }
    }
}

#[async_trait]
impl EventStore for InMemoryEventStore {
    async fn append(&self, event: AgentEvent) -> Result<StoredEvent, EventStoreError> {
        let mut events = self.events.lock().await;

        let stored_event = StoredEvent {
            event: event,
            timestamp: Utc::now(),
            stream_seq: events.len() as u64,
            id: Uuid::now_v7(),
            hash: Uuid::now_v7().to_string(),
            parent_hash: None,
        };
        debug!("InMemoryEventStore: Appending event: seq {}, parent: {:?}", stored_event.stream_seq, stored_event.parent_hash);

        events.push(stored_event.clone());
        Ok(stored_event)
    }

    async fn range(&self, from_seq: u64, limit: usize) -> Result<Vec<StoredEvent>, EventStoreError> {
        let events = self.events.lock().await.clone();
        Ok(events.iter().skip(from_seq as usize).take(limit).cloned().collect())
    }
}