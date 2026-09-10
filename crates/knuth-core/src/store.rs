use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use sha2::Sha256;
use std::fmt::Debug;
use std::pin::Pin;
use std::result::Result;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tracing::debug;
use uuid::Uuid;

use crate::events::*;
use crate::live::LiveEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEvent {
    pub event: AgentEvent,
    pub timestamp: DateTime<Utc>,
    pub stream_seq: u64, // monotonically increasing sequence number for the stream (session)
    pub id: Uuid,
    pub hash: String,                // used for integrity checks and auditability
    pub parent_hash: Option<String>, // form a hash chain
}

#[derive(Debug, Serialize)]
struct StoredEventHashInput<'a> {
    event: &'a AgentEvent,
    timestamp: DateTime<Utc>,
    stream_seq: u64,
    id: Uuid,
    parent_hash: &'a Option<String>,
}

impl<'a> From<&'a StoredEvent> for StoredEventHashInput<'a> {
    fn from(stored_event: &'a StoredEvent) -> Self {
        Self {
            event: &stored_event.event,
            timestamp: stored_event.timestamp,
            stream_seq: stored_event.stream_seq,
            id: stored_event.id,
            parent_hash: &stored_event.parent_hash,
        }
    }
}

impl StoredEventHashInput<'_> {
    pub fn digest(&self) -> Result<String, EventStoreError> {
        let bytes = serde_json::to_vec(self)?;
        let hash = Sha256::digest(&bytes);
        Ok(format!("{:x}", hash))
    }
}

impl StoredEvent {
    pub fn new(
        event: AgentEvent,
        stream_seq: u64,
        parent_event: Option<&StoredEvent>,
    ) -> Result<Self, EventStoreError> {
        let mut stored_event = StoredEvent {
            event: event,
            timestamp: Utc::now(),
            stream_seq: stream_seq,
            id: Uuid::now_v7(),
            hash: "".to_string(),
            parent_hash: parent_event.map(|event| event.hash.clone()),
        };

        stored_event.hash = StoredEventHashInput::from(&stored_event).digest()?;

        Ok(stored_event)
    }
}

fn short_hash(s: &str) -> String {
    format!("{}", s.chars().take(8).collect::<String>())
}

impl std::fmt::Display for StoredEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#{}: {}: ts: {}, hash: {}, parent_hash: {}",
            self.stream_seq,
            self.event,
            self.timestamp,
            short_hash(&self.hash),
            self.parent_hash
                .as_deref()
                .map(|h| short_hash(h))
                .unwrap_or_default()
        )
    }
}

/// What a subscriber observes: the durable log interleaved with the ephemeral
/// progress stream.
///
/// The two arms differ in guarantees, not just in payload. `Durable` events are
/// persisted and sequenced, so a subscriber can resume from `stream_seq`.
/// `Live` events are best-effort and unsequenced; a slow subscriber may miss
/// them entirely.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "event")]
pub enum SessionEvent {
    Durable(StoredEvent),
    Live(LiveEvent),
}

impl SessionEvent {
    pub fn as_durable(&self) -> Option<&StoredEvent> {
        match self {
            SessionEvent::Durable(stored) => Some(stored),
            SessionEvent::Live(_) => None,
        }
    }

    pub fn as_live(&self) -> Option<&LiveEvent> {
        match self {
            SessionEvent::Live(event) => Some(event),
            SessionEvent::Durable(_) => None,
        }
    }
}

impl std::fmt::Display for SessionEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionEvent::Durable(stored) => write!(f, "{stored}"),
            SessionEvent::Live(event) => write!(f, "live: {event}"),
        }
    }
}

pub struct AgentSubscription {
    rx: mpsc::Receiver<SessionEvent>,
    pub id: Uuid,
}

impl AgentSubscription {
    pub fn new(rx: mpsc::Receiver<SessionEvent>) -> Self {
        Self {
            rx,
            id: Uuid::now_v7(),
        }
    }
}

impl Stream for AgentSubscription {
    type Item = SessionEvent;

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
    #[error("Failed to digest event: {0}")]
    DigestFailed(#[from] serde_json::Error),
}

#[async_trait]
pub trait EventStore: Send + Sync + Debug {
    async fn append(&self, event: AgentEvent) -> Result<StoredEvent, EventStoreError>;
    async fn range(&self, from_seq: u64, limit: usize)
    -> Result<Vec<StoredEvent>, EventStoreError>;
    async fn verify_hash_chain(&self, events: &[StoredEvent]) -> Result<(), EventStoreError> {
        let mut expected_parent_hash: Option<String> = None;

        for event in events {
            if event.parent_hash != expected_parent_hash {
                return Err(EventStoreError::RangeFailed(format!(
                    "Broken hash chain at seq {}: expected parent {:?}, got {:?}",
                    event.stream_seq, expected_parent_hash, event.parent_hash,
                )));
            }

            let expected_hash = StoredEventHashInput::from(event).digest()?;

            if event.hash != expected_hash {
                return Err(EventStoreError::RangeFailed(format!(
                    "Invalid event hash at seq {}: expected {}, got {}",
                    event.stream_seq, expected_hash, event.hash,
                )));
            }

            expected_parent_hash = Some(event.hash.clone());
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct InMemoryEventStore {
    events: Arc<Mutex<Vec<StoredEvent>>>,
}

/// use for debug now
impl InMemoryEventStore {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl EventStore for InMemoryEventStore {
    async fn append(&self, event: AgentEvent) -> Result<StoredEvent, EventStoreError> {
        let mut events = self.events.lock().await;

        let parent_event = events.last();

        let stored_event = StoredEvent::new(event, events.len() as u64, parent_event)?;

        debug!(
            "seq {}, hash: {}, parent: {:?}",
            stored_event.stream_seq,
            &stored_event.hash[..8],
            stored_event.parent_hash.as_deref().map(|h| &h[..8])
        );

        events.push(stored_event.clone());
        Ok(stored_event)
    }

    async fn range(
        &self,
        from_seq: u64,
        limit: usize,
    ) -> Result<Vec<StoredEvent>, EventStoreError> {
        let events = self.events.lock().await.clone();
        Ok(events
            .iter()
            .skip(from_seq as usize)
            .take(limit)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::SessionId;

    #[tokio::test]
    async fn append_builds_a_verifiable_hash_chain() {
        let store = InMemoryEventStore::new();
        store
            .append(AgentEvent::SessionStarted {
                session_id: SessionId::new(),
            })
            .await
            .unwrap();
        store
            .append(AgentEvent::SystemPromptSet {
                prompt: "be brief".to_string(),
            })
            .await
            .unwrap();

        let events = store.range(0, 16).await.unwrap();
        store.verify_hash_chain(&events).await.unwrap();

        assert_eq!(events[0].stream_seq, 0);
        assert_eq!(events[1].stream_seq, 1);
        assert_eq!(events[1].parent_hash.as_ref(), Some(&events[0].hash));
        assert!(events[0].parent_hash.is_none());
    }

    #[tokio::test]
    async fn verify_hash_chain_rejects_tampered_hash() {
        let store = InMemoryEventStore::new();
        store
            .append(AgentEvent::SystemPromptSet {
                prompt: "p".to_string(),
            })
            .await
            .unwrap();
        store
            .append(AgentEvent::SystemPromptSet {
                prompt: "q".to_string(),
            })
            .await
            .unwrap();

        let mut events = store.range(0, 16).await.unwrap();
        events[1].hash = "deadbeef".to_string();

        let error = store.verify_hash_chain(&events).await.unwrap_err();
        assert!(
            error.to_string().contains("Invalid event hash"),
            "got {error}"
        );
    }

    #[test]
    fn session_event_distinguishes_durable_from_live() {
        let durable = SessionEvent::Durable(
            StoredEvent::new(
                AgentEvent::SystemPromptSet {
                    prompt: "p".to_string(),
                },
                0,
                None,
            )
            .unwrap(),
        );
        let live = SessionEvent::Live(LiveEvent::AssistantMessageTextDelta {
            step_id: crate::ids::StepId::new(),
            content_index: 0,
            delta: "x".to_string(),
        });

        assert!(durable.as_durable().is_some());
        assert!(durable.as_live().is_none());
        assert!(live.as_live().is_some());
        assert!(live.as_durable().is_none());
    }
}
