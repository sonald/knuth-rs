use ai::{Message, ToolResultMessage, ToolResultRole, UserContentBlock, UserMessage, UserRole};
use knuth_core::{
    AgentEvent, AgentSubscription, EventStore, EventStoreError, LiveEvent, ModelStepEndReason,
    SessionEvent, StoredEvent,
};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tracing::debug;

/// Append-only session log.
///
/// Owns the three things that must stay in sync on every durable event:
/// storage (`EventStore`), the conversation projection used to assemble model
/// context, and fan-out to subscribers. `commit` is the only write path, so the
/// three views cannot drift apart.
///
/// Live events take the separate, weaker [`EventLog::publish_live`] path: they
/// reach subscribers only, so they can never influence storage or the
/// conversation.
#[derive(Debug)]
pub struct EventLog {
    store: Box<dyn EventStore>,
    subscriptions: Vec<mpsc::Sender<SessionEvent>>,
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
        self.notify_subscriptions(SessionEvent::Durable(stored.clone()));
        Ok(stored)
    }

    /// Fans a progress notification out to subscribers without persisting it.
    pub fn publish_live(&mut self, event: LiveEvent) {
        self.notify_subscriptions(SessionEvent::Live(event));
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

    fn notify_subscriptions(&mut self, event: SessionEvent) {
        self.subscriptions
            .retain_mut(|s| match s.try_send(event.clone()) {
                Ok(_) => true,
                Err(TrySendError::Full(e)) => {
                    debug!("EventLog: subscription is full, dropping it (event: {})", e);
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
            AgentEvent::ToolResultReceived {
                tool_call_id,
                tool_name,
                outcome,
                content,
                ..
            } => {
                // Error details go into the content so every provider sees them,
                // and the flag is set too for providers that carry it natively
                // (e.g. Anthropic's tool_result.is_error).
                let is_error = !outcome.is_success();
                let result_text = if is_error {
                    format!("Tool call failed with details: \n{}", content)
                } else {
                    content
                };

                let content = vec![UserContentBlock::text(result_text)];
                self.add_message(Message::ToolResult(ToolResultMessage {
                    role: ToolResultRole::ToolResult,
                    tool_call_id,
                    tool_name,
                    content,
                    details: None,
                    is_error,
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
    use knuth_core::{
        InMemoryEventStore, ToolOutcome, UserMessageIntent,
        ids::{MessageId, StepId, ToolInvocationId},
    };

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
            message_id: MessageId::new(),
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
        let received = received.as_durable().expect("committed events are durable");
        assert_eq!(received.stream_seq, stored.stream_seq);
        assert_eq!(received.hash, stored.hash);
    }

    #[tokio::test]
    async fn live_events_reach_subscribers_without_touching_the_log() {
        let mut log = mk_log();
        let mut sub = log.subscribe(4);

        log.publish_live(LiveEvent::AssistantMessageTextDelta {
            step_id: StepId::new(),
            content_index: 0,
            delta: "partial".to_string(),
        });

        let received = sub.next().await.expect("subscriber should receive event");
        assert!(matches!(
            received.as_live(),
            Some(LiveEvent::AssistantMessageTextDelta { delta, .. }) if delta == "partial"
        ));
        assert!(
            log.messages().is_empty(),
            "live events must not enter the conversation"
        );
        assert!(
            log.store.range(0, 16).await.unwrap().is_empty(),
            "live events must not be persisted"
        );
    }

    #[tokio::test]
    async fn tool_result_projects_into_conversation_with_error_flag() {
        let mut log = mk_log();

        log.commit(AgentEvent::ToolResultReceived {
            invocation_id: ToolInvocationId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            outcome: ToolOutcome::Error,
            content: "boom".to_string(),
        })
        .await
        .unwrap();

        match log.messages() {
            [Message::ToolResult(message)] => {
                assert_eq!(message.tool_call_id, "call-1");
                assert!(message.is_error);
                match &message.content[0] {
                    UserContentBlock::Text(text) => assert!(
                        text.text.contains("boom"),
                        "error details should stay in the content, got {:?}",
                        text.text
                    ),
                    other => panic!("expected a text content block, got {other:?}"),
                }
            }
            other => panic!("expected a single tool result message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn successful_tool_result_projects_without_error_prefix() {
        let mut log = mk_log();

        log.commit(AgentEvent::ToolResultReceived {
            invocation_id: ToolInvocationId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            outcome: ToolOutcome::ExecSuccess,
            content: "ok".to_string(),
        })
        .await
        .unwrap();

        match log.messages() {
            [Message::ToolResult(message)] => {
                assert!(!message.is_error);
                match &message.content[0] {
                    UserContentBlock::Text(text) => assert_eq!(text.text, "ok"),
                    other => panic!("expected a text content block, got {other:?}"),
                }
            }
            other => panic!("expected a single tool result message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn policy_denied_tool_result_is_projected_as_error() {
        let mut log = mk_log();

        log.commit(AgentEvent::ToolResultReceived {
            invocation_id: ToolInvocationId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "write_file".to_string(),
            outcome: ToolOutcome::PolicyDenied,
            content: "read-only mode".to_string(),
        })
        .await
        .unwrap();

        match log.messages() {
            [Message::ToolResult(message)] => {
                assert!(message.is_error);
                match &message.content[0] {
                    UserContentBlock::Text(text) => {
                        assert!(text.text.contains("read-only mode"), "got {:?}", text.text)
                    }
                    other => panic!("expected a text content block, got {other:?}"),
                }
            }
            other => panic!("expected a single tool result message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn proposed_and_observed_tool_events_do_not_enter_conversation() {
        let mut log = mk_log();
        let invocation_id = ToolInvocationId::new();

        log.commit(AgentEvent::ToolCallProposed {
            invocation_id,
            step_id: StepId::new(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            arguments: serde_json::Map::new(),
        })
        .await
        .unwrap();
        log.commit(AgentEvent::ToolExecutionObserved {
            invocation_id,
            tool_call_id: "call-1".to_string(),
        })
        .await
        .unwrap();

        assert!(
            log.messages().is_empty(),
            "pipeline bookkeeping events must not become model messages"
        );
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
