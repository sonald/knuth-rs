use ai::{Message, ToolResultMessage, ToolResultRole, UserContentBlock, UserMessage, UserRole};
use knuth_core::{
    AgentEvent, AgentSubscription, EventStore, EventStoreError, ModelStepEndReason, StoredEvent,
};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tracing::debug;

/// Append-only session log.
///
/// Owns the three things that must stay in sync on every event: durable
/// storage (`EventStore`), the conversation projection used to assemble model
/// context, and fan-out to live subscribers. `commit` is the only write path,
/// so the three views cannot drift apart.
#[derive(Debug)]
pub struct EventLog {
    store: Box<dyn EventStore>,
    subscriptions: Vec<mpsc::Sender<StoredEvent>>,
    conversation: ConversationState,
}

impl EventLog {
    pub fn new(store: Box<dyn EventStore>) -> Self {
        Self {
            store,
            subscriptions: Vec::new(),
            conversation: ConversationState::default(),
        }
    }

    pub async fn commit(&mut self, event: AgentEvent) -> Result<StoredEvent, EventStoreError> {
        let stored = self.store.append(event).await?;
        self.conversation.apply_event(stored.clone());
        self.notify_subscriptions(stored.clone());
        Ok(stored)
    }

    pub fn subscribe(&mut self, buffer: usize) -> AgentSubscription {
        let (tx, rx) = mpsc::channel(buffer);
        self.subscriptions.push(tx);
        AgentSubscription::new(rx)
    }

    pub fn system_prompt(&self) -> &str {
        &self.conversation.system_prompt
    }

    pub fn messages(&self) -> &[Message] {
        &self.conversation.messages
    }

    pub(crate) fn conversation(&self) -> &ConversationState {
        &self.conversation
    }

    fn notify_subscriptions(&mut self, event: StoredEvent) {
        self.subscriptions
            .retain_mut(|s| match s.try_send(event.clone()) {
                Ok(_) => true,
                Err(TrySendError::Full(e)) => {
                    debug!(
                        "EventLog: subscription is full, dropping it (event: {})",
                        e
                    );
                    false
                }
                Err(TrySendError::Closed(e)) => {
                    debug!("EventLog: subscription closed, dropping it (event: {})", e);
                    false
                }
            });
    }
}

/// Conversation projection rebuilt by replaying events; the messages sent to
/// the model on the next step are exactly what this projection contains.
#[derive(Debug, Default)]
pub(crate) struct ConversationState {
    messages: Vec<Message>,
    system_prompt: String,
}

impl ConversationState {
    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    fn apply_event(&mut self, event: StoredEvent) {
        match event.event {
            AgentEvent::ModelStepEnded {
                reason,
                assistant_message: Some(assistant_message),
                ..
            } if matches!(
                reason,
                ModelStepEndReason::Success
                    | ModelStepEndReason::Length
                    | ModelStepEndReason::ToolUse
            ) =>
            {
                self.add_message(Message::Assistant(assistant_message));
            }
            AgentEvent::UserMessageCommitted { content, .. } => {
                self.add_message(Message::User(UserMessage {
                    role: UserRole::User,
                    content: content,
                    timestamp: event.timestamp.timestamp(),
                }));
            }
            AgentEvent::ToolExecutionEnded {
                tool_call_id,
                tool_name,
                result,
            } => {
                debug!("append ToolExecutionEnded event to conversation state");

                let content = vec![UserContentBlock::text(result)];
                self.add_message(Message::ToolResult(ToolResultMessage {
                    role: ToolResultRole::ToolResult,
                    tool_call_id,
                    tool_name,
                    content,
                    details: None,
                    is_error: false,
                    timestamp: event.timestamp.timestamp(),
                }));
            }
            AgentEvent::SystemPromptSet { prompt } => {
                self.system_prompt = prompt;
            }
            _ => {}
        }
    }
}

impl std::fmt::Display for ConversationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConversationState (system_prompt: {} chars, {} messages)",
            self.system_prompt.len(),
            self.messages.len()
        )?;
        for (i, message) in self.messages.iter().enumerate() {
            write!(f, "\n  [{i}] {message}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use knuth_core::{InMemoryEventStore, UserMessageIntent};

    fn mk_log() -> EventLog {
        EventLog::new(Box::new(InMemoryEventStore::new()))
    }

    #[tokio::test]
    async fn commit_projects_into_conversation() {
        let mut log = mk_log();

        log.commit(AgentEvent::SystemPromptSet {
            prompt: "be brief".to_string(),
        })
        .await
        .unwrap();
        log.commit(AgentEvent::UserMessageCommitted {
            content: ai::UserContent::Blocks(vec![UserContentBlock::text("hi")]),
            intent: UserMessageIntent::Normal,
        })
        .await
        .unwrap();

        assert_eq!(log.system_prompt(), "be brief");
        assert_eq!(log.messages().len(), 1);
    }

    #[tokio::test]
    async fn commit_fans_out_to_subscribers() {
        let mut log = mk_log();
        let mut sub = log.subscribe(4);

        let stored = log
            .commit(AgentEvent::SystemPromptSet {
                prompt: "p".to_string(),
            })
            .await
            .unwrap();

        let received = sub.next().await.expect("subscriber should receive event");
        assert_eq!(received.stream_seq, stored.stream_seq);
        assert_eq!(received.hash, stored.hash);
    }

    #[tokio::test]
    async fn dropped_subscriber_is_pruned() {
        let mut log = mk_log();
        let sub = log.subscribe(4);
        drop(sub);

        log.commit(AgentEvent::SystemPromptSet {
            prompt: "p".to_string(),
        })
        .await
        .unwrap();

        assert!(log.subscriptions.is_empty());
    }
}
