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

pub struct AgentSubscription {
    rx: mpsc::Receiver<AgentEvent>,
    pub id: Uuid,
}

impl AgentSubscription {
    pub fn new(rx: mpsc::Receiver<AgentEvent>) -> Self {
        Self {
            rx,
            id: Uuid::now_v7(),
        }
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
